# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from itertools import chain
from functools import partial
import logging
from typing import Optional, Sequence, Union, List, Tuple

import numpy as np
import torch

from .swapmatic import DefaultDType, coupling_map_to_distance_matx
from .swapmatic import coupling_map_to_swaps_topo, swaps_layer
from .math import coo_mat_vectorize, symm_sawtooth, apply_perm_to_coo
from .math import STORAGE_DENSE, torch_detach_np
from .math import reduce_np_matmul, reduce_level_np_matmul
from .errors import arg_type_error, invalid_arg
from qiskit.transpiler import CouplingMap


logger = logging.getLogger(__name__)
OptTensor = Optional[torch.Tensor]
NDArray = np.ndarray


def _prepare_circ_adj(adj: NDArray, *, n, dtype) -> torch.Tensor:
    """Prepare the adjacency matrices.

    Vectorize the adjacency matrix of a circuit's layer.
    The adjacency matrix is provided in coordinates form.
    """
    adj = np.asarray(adj)
    if not (adj.ndim == 2 and adj.shape[0] == 2):
        raise ValueError(
            invalid_arg("adj", expected="coordinates matrix (as numpy array)", actual=str(adj))
        )

    adj = coo_mat_vectorize(
        adj, np.repeat(1, adj.shape[1]), (n, n), dtype=dtype, storage=STORAGE_DENSE
    )
    return torch.flatten(adj)


def linear_decay(n: int, *, dtype=None) -> torch.Tensor:
    """Step decay function for CostFactory.hw_cost."""
    return 1 - torch.arange(n, dtype=dtype) / n


class CostFactory:
    """A structure for the computation of the components of the cost function.

    Attributes:
        params: The current tensor of params. Their initialization is performed
        through the method `init_params`.
    """

    def __init__(
        self,
        *,
        swap_layers: int = 1,
        dtype=None,
        swap_cost_tradeoff: float = np.inf,
    ):
        """
        Args:
            swap_layers: The number of swap layers for each circuit layer.
            dtype: Optional dtype (PyTorch) for the tensors created by this
            object.
            dtype: Torch dtype for the resulting tensor(s).
            swap_cost_tradeoff: Experimental tradeoff between two objectives, when np.inf
            the L2 objective is the only one influencing the optimization.
        """
        self.dtype = dtype or DefaultDType
        self.circ_adjs_v = None
        self.hwadjc_v = None
        self.swaps_topo = None
        self.swap_layers = swap_layers
        self.params = None

        self.swap_cost_tradeoff = swap_cost_tradeoff

    def _get_num_qubits(self) -> int:
        assert self.hwadjc_v is not None
        return int(np.sqrt(torch.numel(self.hwadjc_v)))

    def init_params(self, *, seed=None, theta_scale=1.0, swaps_depth: Optional[int] = None):
        """Initialize the parameters for the optimization.

        Initialize the object attribute `params` using the structure of
        the circuit given as a sequence of adjacency matrices (see method `init_data`).
        This method requires init_data to be invoked before.

        Args:
            seed: The seed for the random number generator.
            theta_scale: The scale for the randomized thetas.
            swaps_depth: The depth for the swap layers, note it can be different
            from the number of circuit layers so we can have a part of the circuit
            on which the swaps are not acting, but still considered in the cost
            function. When None (default) the depth is set to the number of circuit
            layers (or horizon when using the rolling horizon strategy).

        Raises:
            ValueError: Error raised when the effective swaps_depth is non-positive.
            Also error when 'init_data' is not invoked before this method.
        """
        if self.circ_adjs_v is None:
            raise ValueError("Status not initialized, invoke init_data before")

        rng = None
        if seed is not None:
            rng = torch.Generator()
            rng.manual_seed(seed)

        args = dict(dtype=self.dtype, generator=rng)
        # Init depth of the swapped circuit. Note that in lookahead mode
        # we can have a part of the circuit on which the swaps are not acting.
        adjs_layers = self.circ_adjs_v.shape[1]
        swaps_depth = swaps_depth or adjs_layers
        swaps_depth = min(swaps_depth, adjs_layers)
        assert swaps_depth > 0

        params_per_layer = self._get_num_qubits() - 1
        if self.swaps_topo is not None:
            params_per_layer = sum(map(len, self.swaps_topo))
        theta = torch.rand(swaps_depth, self.swap_layers, params_per_layer, **args)
        logger.debug(f"Init theta, shape={tuple(theta.shape)}")
        theta = theta * float(theta_scale)
        theta.requires_grad_(True)
        self.params = theta

    def init_data(
        self,
        circ_adjs_coo: Sequence[NDArray],
        *,
        cmap: CouplingMap,
        swaps_gen_cmap: Optional[CouplingMap] = None,
    ):
        """Init the cost factory with the input data.

        This method should be called before the method `init_params`.

        Args:
            circ_adjs_coo: A list of adjacency matrices in coordinates form (as Numpy array).
            cmap: A Qiskit coupling map.
            swaps_gen_cmap: An optional (default None), separate coupling map
            to be used for the permutation generating swaps.
            When None, its value is set internally to the value of the argument `cmap`.

        Raises:
            ValueError: Error raised when the arguments expected as CouplingMap
            do not match the required type.
        """
        # TODO Validate adjs and cmap. Possibly pass the param 'n'
        # in this function instead of the constructor.
        swaps_gen_cmap = cmap if swaps_gen_cmap is None else swaps_gen_cmap
        if not isinstance(cmap, CouplingMap):
            raise ValueError(
                arg_type_error(arg="cmap", expected_type=CouplingMap, actual_type=type(cmap))
            )
        if not isinstance(swaps_gen_cmap, CouplingMap):
            raise ValueError(
                arg_type_error(
                    arg="swaps_gen_cmap",
                    expected_type=CouplingMap,
                    actual_type=type(swaps_gen_cmap),
                )
            )
        if cmap.size() != swaps_gen_cmap.size():
            raise ValueError("Incompatible coupling maps")
        if cmap.size() < 2:
            raise ValueError(invalid_arg("cmap", "qubit count greater than 1", cmap.size()))

        args = dict(n=cmap.size(), dtype=self.dtype)
        circ_adjs_v = [_prepare_circ_adj(adj, **args) for adj in circ_adjs_coo]
        if len(circ_adjs_v) == 0:
            raise ValueError(
                invalid_arg("circ_adjs_coo", expected="at least one layer", actual="(empty)")
            )
        circ_adjs_v = torch.stack(circ_adjs_v).T

        swaps_topo = coupling_map_to_swaps_topo(swaps_gen_cmap)

        # Prepare hardware (special) distance matrix.
        # TODO Note that this is invoked at each time window.
        hwadjc_v = coupling_map_to_distance_matx(cmap, vec=True)
        hwadjc_v = torch.tensor(hwadjc_v, dtype=self.dtype).view(1, -1)

        self.circ_adjs_v = circ_adjs_v
        self.hwadjc_v = hwadjc_v
        self.swaps_topo = swaps_topo
        self.params = None  # Reset params if the obj was previously being init.

    def _get_params(self, params: OptTensor = None) -> torch.Tensor:
        if self.params is None:
            raise ValueError("Invalid status, data not initialized")
        return self.params if params is None else params

    def hw_cost(self, params: OptTensor = None, *, split=False, decay=None) -> torch.Tensor:
        """Computer the hw cost value.

        The function assumes that data and parameters are initialized unless
        they are passed diretly, see 'init_params' and 'init_data' for more information.

        Args:
            params: The parameters for the model or None to use self.params.
            split: When True (default False), return a vector where each element
            is the cost value for each layer.
            decay: Function (optional) for calculating the weights to be applied to
            each layer (used to implement the adaptive feasability).

        Returns: A tensor (PyTorch) (scalar or vector depending on the argument split)
        containing the value(s) of the hardware cost. When a vector is returned,
        each element refers to the cost function for each layer.

        Raises:
            ValueError: Error when data is not initialized.
        """
        theta = self._get_params(params)
        cadjs_v, hwadjc_v = self.circ_adjs_v, self.hwadjc_v
        if cadjs_v.shape[1] != theta.shape[0]:
            raise ValueError(f"Invalid parameters shape")

        # With 'sublayers=True' we remove all matrix-matrix multiplications.
        args = dict(
            topology=self.swaps_topo,
            n=self._get_num_qubits(),
            tpow2=True,
            sublayers=True,
            ident_term=False,
        )
        # Loop over circuit and swap layers
        cadjs_v1 = [None] * theta.shape[0]
        for c in range(theta.shape[0]):
            # Apply the permutation layers for each adjacency matrix.
            for k in range(theta.shape[1]):
                for m in swaps_layer(theta[c, k], **args):
                    cadjs_v = torch.sparse.addmm(cadjs_v, m, cadjs_v)
            cadjs_v1[c] = cadjs_v[:, 0:1]
            cadjs_v = cadjs_v[:, 1:]
        cadjs_v, cadjs_v1 = torch.concat(cadjs_v1, dim=1), None

        # Pre-mul by the vectorized complementary adjacency matrix
        # corresponding to the hw connectivity.
        cadjs_v = torch.flatten(hwadjc_v @ cadjs_v)

        # Costs decay and aggregation
        if decay is not None:
            decay_v = decay(len(cadjs_v), dtype=cadjs_v.dtype)
            cadjs_v = cadjs_v * decay_v
        return cadjs_v if split else torch.sum(cadjs_v)

    def compute_valid_frame_sz(self, params: OptTensor = None) -> int:
        """Compute the valid frame size w.r.t. the hardware constraints.

        Args:
            params: Optional tensor (PyTorch) for the thetas. When None,
            the parameters are taken from the current object.

        Returns: An integer representing the number of valid
        layers.

        """
        # Don't apply decay here
        costs = self.hw_cost(params, split=True)
        with torch.no_grad():
            costs = (costs >= (1.0 - 0.001)).type(costs.dtype)
            i = int(torch.argmax(costs))
            if i == 0:
                return len(costs) if costs[0] < 0.5 else 0
        return i

    def eval_permutations(self, params: OptTensor = None, *, compose=True):
        """Obtain the swaps corresponding to the given parameters.

        Args:
            params: Optional tensor (PyTorch) for the thetas. When None,
            the parameters are taken from the current object.
            compose: When True (default) each layer is given as a single permutation
            matrix otherwise layers are represented a lists of swap permutations.

        Returns: A list or a list of lists of permutation matrices (as Numpy arrays).

        """
        theta = self._get_params(params)
        args = dict(
            n=self._get_num_qubits(),
            tpow2=False,
            mode="np",
            sublayers=True,
            topology=self.swaps_topo,
        )
        ret = []
        for c in range(theta.shape[0]):
            # Note not using the ident_term=False strategy here.
            layer = [swaps_layer(theta[c, k], **args) for k in range(theta.shape[1])]
            layer = chain(*layer)
            layer = list(map(_round_perm, layer))[::-1]
            layer = reduce_np_matmul(layer) if compose else layer
            ret.append(layer)
        return ret

    def sample_exact_params(self, *, n: int = 1, seed=None) -> Tuple[NDArray]:
        """Sample exact params around the current parameters.

        The function samples vector of parameters that are "close" to
        the given one. The sampled parameters have a structure that
        determine a single permutation per sample. Note that in general
        a set of parameters can determine a convex combination of permutations.

        Args:
            n: The number of samples to be generated.
            When n=1, the closest parametrization is returned.
            seed: The seed for the random number generator.

        Returns: A tuple of tensors (Numpy array), where the number of
        elements depends on the argument `n`.

        Raises:
            ValueError: Error when the number of samples is not valid.
        """
        if n < 1:
            raise ValueError(f"Invalid number of samples, requires n>=1, found: {n}")
        params = self._get_params()
        params = torch_detach_np(params)
        if n == 1:
            # Obtain the closest int mul of pi/2 parametrization.
            params = _theta_to_p(params, round=True)
            params = (_p_to_activation_arg(params),)
        else:
            params = _sample_exact(params, n=n, seed=seed)
            params = tuple(params)
        return params

    def swap_cost(self, params: OptTensor = None, *, skip_head_layers: int = 1) -> torch.Tensor:
        """Compute the swap cost.

        Args:
            params: Optional tensor (PyTorch) for the thetas. When None,
            the parameters are taken from the current object.
            skip_head_layers: The number of layers (default 1) to be excluded
            (starting from the head) from the cost calculation.

        Returns: A tensor (PyTorch) containing the value of the swap cost.

        """
        tradeoff = self.swap_cost_tradeoff
        m = self._get_params(params)
        if not isinstance(m, torch.Tensor):
            m = torch.as_tensor(m)

        m = m[skip_head_layers:] / (np.pi / 2)
        if not np.isfinite(tradeoff):
            # TODO Split into columns when we have the diff term active?
            return _diff_norm(m)

        m1 = m[:, :, :-1] - m[:, :, 1:]
        if np.isclose(tradeoff, 0):
            return _diff_norm(m1)

        v = _diff_norm(m) * tradeoff
        v = v + _diff_norm(m1)
        return v


def _diff_norm(m: torch.Tensor, *, l1_approx=False, l1_approx_beta=0.01) -> torch.Tensor:
    """Differentiable norms.

    Norms available are L2 squared and a differentiable
    approximation of L1.
    """
    m = torch.square(m)
    if l1_approx:
        # m = torch.sqrt(m + l1_approx_beta**2) - l1_approx_beta
        m = torch.sqrt(m + l1_approx_beta)
    return torch.sum(m)


def _p_to_activation_arg(p: NDArray) -> NDArray:
    """Convert an array of probabilities to angles.

    This is used in the sampling procedure.
    """
    p = np.asarray(p, dtype=float)
    return p * (np.pi / 2)


def _theta_to_p(v: NDArray, *, round=False) -> NDArray:
    """Angles to probabilities.

    Convert an array of angles to probabilities according to
    whether an angle is closer to an even or odd integer multiple
    of pi/2.
    """
    v = np.asarray(v) / (np.pi / 2)
    v = symm_sawtooth(v)
    return np.round(v) if round else v


def _sample_exact(v: NDArray, *, n: int = 1, seed=None) -> NDArray:
    """Sample exact permutations "close" to the given thetas."""
    v = _theta_to_p(v)
    rng = np.random.default_rng(seed)
    p = rng.uniform(size=(n,) + v.shape)
    return _p_to_activation_arg(p < v)


def _round_perm(m: NDArray) -> NDArray:
    """Round and validate a permutation matrix."""
    m = np.asarray(m)
    assert m.ndim == 2 and len(set(m.shape)) == 1
    m = m.round().astype(int)
    assert np.all(m >= 0) and np.all(m <= 1)
    for k in range(2):
        assert np.all(np.sum(m, axis=k) == np.ones(len(m)))
    return m


class EarlyStopping:
    """A structure for the handling the early stopping mechanism."""

    def __init__(self, abs_delta_thr: float, patience: int = 1) -> None:
        """
        Args:
            abs_delta_thr: Threshold for the minimal variation between
            two samples to be considered as activity.
            patience: The number of samples of non activity to wait before
            triggering the stop signal.
        """
        self.abs_delta_thr = abs_delta_thr
        self.patience = patience
        self.reset()

    def reset(self):
        """Reset the state of the current object."""
        self.prev_v = np.inf
        self.counter = 0

    def __call__(self, v):
        """Update the state with the given value.

        Args:
            v: The new value for updating the status.

        Returns: True when the stopping mechanism triggers.
        """
        v = float(v)
        abs_delta = np.abs(v - self.prev_v)
        self.prev_v = v
        if abs_delta < self.abs_delta_thr:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


_GD_ALGOS = {
    "adagrad": torch.optim.Adagrad,
    "rmsprop": torch.optim.RMSprop,
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
}


def _is_tensor_with_grad(v) -> bool:
    """Check tensor with grad.

    Check whether the input value is a Torch tensor with a non-null
    gradient.
    """
    return isinstance(v, torch.Tensor) and (v.grad is not None)


def _get_effective_swap_layers(config: Union[str, int, None], *, n: int) -> int:
    """Obtain the effective number of swap layers.

    Given the user config for the number of swap layers and the number
    of qubits of the current circuit, expand the number of swap layers
    to its effective value (the user may provide a string identifying the
    strategy).
    """
    config = config or "log_2"
    if isinstance(config, int):
        if config <= 0:
            raise ValueError(f"Invalid swap layers value: {swaps_layer}")
        return config
    if config == "sqrt":
        return int(np.ceil(np.sqrt(n)))
    if config == "log_2":
        return int(np.ceil(np.log2(n)))
    raise ValueError(f"Unrecognized swap layers config: {swaps_layer}")


class GlobalKnitter:
    """A structure for managing the knitter algorithm."""

    def __init__(self) -> None:
        self.stats = None

        # Configuration for the optimizer.
        self.gd_config = dict(algo="adagrad", lr=1.0, lr_decay=0.1)

        # Configuration fot the early stopping mechanism.
        self.early_stop_config = dict(abs_delta_thr=1e-2, patience=5)

        # Initial value for the Lagrange multiplier related to the
        # hardware cost function.
        self.hw_cost_lam0 = 0.01

        # Scale for the random initialization of the thetas.
        self.theta_init_scale = np.pi / 8

        self.swaps_depth = None
        self.cost_f = None

        # Function for calculating the hw cost decay coefficients.
        self.hw_cost_decay = None

        self.cost_f_config = None

    def reset(self):
        """Reset the object status (cost function and stats)."""
        self.stats = None
        self.cost_f = None

    def _get_cost_f_config(self):
        config = dict()
        if self.cost_f_config is not None:
            config.update(self.cost_f_config)
        return config

    def _get_gd_kwparams(self):
        return dict((k, v) for k, v in self.gd_config.items() if k != "algo")

    def _get_gd_algo(self):
        """Get optimizer class.

        Get the class of the optimiser according to the
        user configuration.
        """
        algo = self.gd_config["algo"]
        return _GD_ALGOS[algo]

    def _get_optim_params(self, cost_f, *args):
        ret = (cost_f.params,)
        return ret + args

    def __call__(
        self,
        circ_adjs,
        *,
        cmap: CouplingMap,
        swaps_gen_cmap: Optional[CouplingMap] = None,
        swap_layers=None,
        max_optim_steps: int = 30,
        seed=None,
        swap_cost_skip_layers: int = 0,
    ) -> List[dict]:
        """Run the knitter on the given sequence of adjacency matrices.

            Args:
                circ_adjs: A list of adjacency matrices in coordinates form.
                cmap: The coupling map for the hardware qubits.
                swaps_gen_cmap: Optional coupling map defining the generators for the
                permutations (swaps). When None, the value from `cmap` is used.
                swap_layers: The number of swap layers or the name of the mechanism for
                its automatic calculation. Supported mechanisms are: 'log_2' and 'sqrt'.
                swap_cost_skip_layers: The number of layers (default 1) to be excluded
                (starting from the head) from the swap cost calculation.
                max_optim_steps: The maximum number of steps for the optimizer.
                seed: Seed for the random number generator.

        Returns: A list of dictionaries each representing a record for a solution.

        """
        swap_layers = _get_effective_swap_layers(swap_layers, n=cmap.size())
        logger.debug(f"Effective swap_layers={swap_layers}")

        # Init cost function
        cost_f = CostFactory(
            swap_layers=swap_layers,
            **self._get_cost_f_config(),
        )
        cost_f.init_data(circ_adjs, cmap=cmap, swaps_gen_cmap=swaps_gen_cmap)
        cost_f.init_params(
            seed=seed, theta_scale=self.theta_init_scale, swaps_depth=self.swaps_depth
        )

        # Lagrange multiplier variable
        lm1 = torch.tensor(float(self.hw_cost_lam0), requires_grad=True)
        optim_vars = [lm1]  # Additional optimization variables

        optimizer = self._get_gd_algo()
        optimizer = optimizer(
            self._get_optim_params(cost_f, *optim_vars), **self._get_gd_kwparams()
        )
        l_monitor = EarlyStopping(**self.early_stop_config)
        hw_cost_decay = self.hw_cost_decay

        for k in range(max_optim_steps):
            optimizer.zero_grad()

            # Prepare hardware and swap costs
            hw_cost = cost_f.hw_cost(decay=hw_cost_decay)
            _b = swap_cost_skip_layers
            swap_cost = cost_f.swap_cost(skip_head_layers=_b)

            # Lagrangian
            l = lm1 * hw_cost + swap_cost
            l.backward()
            if _is_tensor_with_grad(lm1):
                # Gradient ascent on the dual variable
                lm1.grad = -lm1.grad
            optimizer.step()

            logger.debug(f"Optim step, hw_cost={float(hw_cost)}, swap_cost={float(swap_cost)}")
            # Another interesting merit function could be:
            # torch.sum(torch.square(cost_f.params.grad)).
            # See Nocedal & Wright book, stopping criteria for constrained optim.
            if l_monitor(float(swap_cost) + float(hw_cost)):
                logger.debug(f"Early stopping trigger at step {k}")
                break
        logger.info(f"Optim completed, hw_cost={float(hw_cost)}, swap_cost={float(swap_cost)}")

        # Obtain exact solution(s)
        _b = swap_cost_skip_layers
        recs = self._prepare_solution_recs(cost_f, seed=seed, swap_cost_skip_layers=_b)
        self.stats = dict(
            param_count=torch.numel(cost_f.params),
            swap_layers=int(swap_layers),
            optim_steps=(k + 1),
        )
        self.cost_f = cost_f
        return recs

    def _prepare_solution_recs(
        self, cost_f: CostFactory, *, n: int = 1, seed=None, swap_cost_skip_layers: int = 0
    ) -> List[dict]:
        """Sample solutions are prepare the corresponding records."""
        recs = []
        with torch.no_grad():
            samples = cost_f.sample_exact_params(n=n, seed=seed)
            for params in samples:
                perms = cost_f.eval_permutations(params, compose=False)
                swap_cost = cost_f.swap_cost(params, skip_head_layers=swap_cost_skip_layers)
                valid_frame_sz = cost_f.compute_valid_frame_sz(params)
                rec = dict(perms=perms, valid_frame_sz=valid_frame_sz, swap_cost=float(swap_cost))
                recs.append(rec)
        return recs


def _best_solution(recs, *, allow_unfeasible: bool = False):
    """Select the best solution.

    Policy for the selection of the best solution among
    multiple trials on the same circuit section.

    Returns: A list of permutations.
    """
    assert len(recs) > 0
    # Maximise valid_frame_sz while minimise swap_cost.
    recs = sorted(recs, key=lambda v: (-v["valid_frame_sz"], v["swap_cost"]))
    rec = recs[0]
    p, sz = rec["perms"], int(rec["valid_frame_sz"])
    # Note that valid_frame_sz can be greater than the number of layers
    # of permutations because of the lookahead.

    if allow_unfeasible:
        if sz == 0:
            logger.info("No local valid solution found but allow_unfeasible enabled.")
        p = p[:sz] if sz > 0 else p
    else:
        p = p[:sz]
    return p


def _is_perfect_solution_rec(rec: dict, *, horizon: int) -> bool:
    return rec["valid_frame_sz"] == horizon and np.isclose(rec["swap_cost"], 0)


def _is_valid_solution_rec(rec: dict) -> bool:
    return rec["valid_frame_sz"] > 0


class NoSolutionFoundError(Exception):
    """Exception raised when the algorithm fails."""

    pass


def rh_knitter(
    circ_adjs,
    *,
    cmap: CouplingMap,
    knitter: Optional[GlobalKnitter] = None,
    horizon: int = 2,
    hw_cost_decay=None,
    allow_unfeasible: bool = False,
    swap_cost_skip_layers: int = 1,
    restart: int = 5,
    restart1: Optional[int] = None,
    seed=None,
    **kwargs,
):
    """Run the rolling horizon mechanism over the knitter.

    Args:
        circ_adjs:
        cmap: A Qiskit coupling map.
        knitter: Optional (default None), pre-initialized GlobalKnitter.
        horizon: The horizon for the rolling horizon mechanism.
        It must be a positive integer.
        hw_cost_decay: An optional (default None) positive decaying function
        for the hardware cost.
        allow_unfeasible: When True (default False), the procedure does not
        raise an exception when it fails to converge.
        This is meant for debugging.
        swap_cost_skip_layers: The number of head layers (default 1) to be excluded
        from the swap cost. Valid values for this parameter are either 0 or 1.
        restart: The number of restarts (default 5) for the optimizer running on each
        sub-circuit consisting of up to a horizon number of layers.
        restart1: An optional additional restart (default None) which must be greater or
        equal than the other argument restart. This is used to run some additional attempts
        when the mechanism fails to find any feasible solution.
        seed: The seed for the random number generator.

    Returns: A list of lists of permutation matrices (as Numpy arrays). Each list of permutations
    represent the decomposition of a permutation as swaps, to be applied before a certain layer.

    Raises:
        NoSolutionFoundError: Error raised when the solver fails to find a solution
        whithin the maximum number of attempts.
        ValueError: Error in case of invalid parameters.
    """
    seed = 23948784 if seed is None else int(seed)
    if not (isinstance(restart, int) and restart > 0):
        raise ValueError(invalid_arg(arg="restart", expected="int greater than 0", actual=restart))
    restart1 = restart1 or restart
    if not (isinstance(restart1, int) and restart1 >= restart):
        raise ValueError(
            invalid_arg(
                arg="restart1", expected=f"int not less than {restart} (restart)", actual=restart1
            )
        )
    if swap_cost_skip_layers not in {0, 1}:
        raise ValueError(
            invalid_arg(
                arg="swap_cost_skip_layers",
                expected="value from set {0, 1}",
                actual=swap_cost_skip_layers,
            )
        )
    if not isinstance(cmap, CouplingMap):
        raise ValueError(
            arg_type_error(arg="cmap", expected_type=CouplingMap, actual_type=type(cmap))
        )

    circ_adjs = list(circ_adjs)
    if cmap.size() <= 2:
        # Case num qubits less than 3.
        return [[np.eye(cmap.size())]] * len(circ_adjs)

    knitter = knitter or GlobalKnitter()
    knitter.hw_cost_decay = hw_cost_decay
    perms = []
    frame_start = 0
    retry_horizon = 0  # Shorter horizon in case of failure.
    while len(circ_adjs) != 0:
        logger.info(f"Loop restart, frame_start={frame_start}")
        knitter.reset()
        knitter.user_history_rec = dict(frame_start=frame_start)
        local_recs = []
        for k in range(restart1):
            if k == restart:
                if any(map(_is_valid_solution_rec, local_recs)):
                    break
                else:
                    logger.info("No valid solutions, extending trials")

            # Invoke the GlobalKnitter on a section of the circuit.
            skipl = swap_cost_skip_layers if (frame_start == 0) else 0
            recs = knitter(
                circ_adjs[: (retry_horizon or horizon)],
                cmap=cmap,
                swap_cost_skip_layers=skipl,
                seed=seed,
                **kwargs,
            )
            if k == 0:
                logger.info(f"Knitter stats: {knitter.stats}")
            local_recs.extend(recs)
            if any(
                map(partial(_is_perfect_solution_rec, horizon=(retry_horizon or horizon)), recs)
            ):
                break
            # Update seed so next iteration uses another predictable
            # seed.
            seed += 1000

        p = _best_solution(
            local_recs,
            allow_unfeasible=allow_unfeasible,
        )
        # Check number of layers for the adaptive solution.
        if len(p) == 0:
            if retry_horizon > 0:
                raise NoSolutionFoundError(f"No valid solution found for layer {len(perms)}")
            else:
                logger.info("Re-run trial with short horizon")
                retry_horizon = 1
                continue
        retry_horizon = 0
        assert 0 < len(p) <= horizon

        perms.extend(p)
        frame_start += len(p)
        circ_adjs = circ_adjs[len(p) :]
        p = reduce_level_np_matmul(list(reversed(p)), 0)
        circ_adjs = [apply_perm_to_coo(p, c) for c in circ_adjs]
    return perms
