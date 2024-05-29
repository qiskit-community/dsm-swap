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

from multiprocessing.sharedctypes import Value
from .math import torch_detach_np, torch_sparse_mat_tp, sparse_eye
from .errors import arg_type_error, invalid_arg
import numpy as np
import networkx as nx
import torch
from qiskit.transpiler import CouplingMap
from functools import reduce, partial
from itertools import chain
from typing import Tuple, Union


TorchOrNpTensor = Union[np.ndarray, torch.Tensor]
DefaultDType = torch.float32


def sswaps_coo(c, v=None) -> Tuple[np.ndarray, TorchOrNpTensor]:
    """
    Given a list of tuples representing the swaps, prepare the configuration
    for the coo layered matrix construction. Note that the coordinates for
    the identity component are not generated.
    The returned values are meant to be used directly by the function
    `torch.sparse_coo_tensor`. Moreover the swaps must be disjoint, that is
    the integer values contained in the tuples must be unique across the entire
    list.

    Args:
        c: A list of tuples each representing a swap. Note that tuples
        of the form (i, i) must be excluded (we need (i, j) with i not equals to j).
        For performance reason there is no check for such condition.
        v: Optional vector of values to be associated to each swap.
        the dimension of the 1D vector must match the size of the
        list of tuples.

    Returns: A two-dimensional array containing the sparse coordinates and
    a vector of values associated to each index.

    """
    c = np.array(c, dtype=int)
    c = np.atleast_2d(c)
    assert c.ndim == 2 and c.shape[1] == 2
    idxs = np.expand_dims(np.arange(len(c)), -1)
    c = np.concatenate([idxs, c], axis=-1)  # (k, i, j)
    cx = c[:, [0, 2, 1]]  # (k, j, i)
    cii = c[:, [0, 1, 1]]  # (k, i, i)
    cjj = c[:, [0, 2, 2]]  # (k, j, j)
    if v is None:
        v = np.repeat([1, 1, -1, -1], len(c))
    else:
        v = torch.flatten(v)
        assert len(v) == len(c)
        v = v.repeat(4)
        v = v * torch.tensor(np.repeat([1, 1, -1, -1], len(c)), dtype=v.dtype)
    return np.concatenate([c, cx, cii, cjj]).T, v


def multieye_coo(shape) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare the coordinates, values, and a shape for the preparation of a tensor containing layered
    identity matrices. The resulting tensor has three dimension the first being the index of
    the matrix.

    Args:
        shape: A shape corresponding to the resulting matrix.

    Returns: A two-dimensional array containing the sparse coordinates, a one-filled array of
        values, and a shape. The values are used directly to create a coo tensor.

    """
    shape = tuple(shape)
    if len(shape) != 3 or shape[1] != shape[2]:
        raise ValueError("Invalid shape, expected structured as layers of square matrices")
    c, n = shape[0:2]
    m = np.repeat(np.arange(n), 2).reshape((-1, 2))  # (i, i)
    m = np.repeat(m, c, axis=0)
    idx = np.repeat([np.arange(c)], n, axis=0)
    idx = np.reshape(idx, (-1, 1))
    ret = np.concatenate([idx, m], axis=1).T, np.repeat(1, c * n), (c, n, n)
    return ret


def _psswap(m):
    assert m.ndim == 3
    iden = sparse_eye(m.shape[1], dtype=m.dtype)
    m1 = torch.stack([torch_sparse_mat_tp(l, l) for l in m])
    m2 = torch.stack([torch_sparse_mat_tp(iden, l) for l in m])
    m3 = torch.stack([torch_sparse_mat_tp(l, iden) for l in m])
    return m1 + m2 + m3


def _activation(v):
    # Equivalent to: return torch.square(torch.sin(v)).
    # Note that by using cos form we avoid squaring so there
    # should be a slight advantage in term of gradient calculation.
    return (1 - torch.cos(2 * v)) / 2


def sswaps(swaps, theta, *, n, compose=False, dtype=None, tpow2=False, ident_term=True):
    """
    Given a list of tuples representing the swaps, prepare a sparse tensor
    in which each layer is a doubly stochastic matrix resulting from the
    convex combination of the identity and the corresponding swap.
    The convex combination is mediated by means of the parameters theta.

    Args:
        swaps: List of tuples representing the swaps.
        theta: Vector of angles to be used to tune the doubly stochastic matrices.
        n: Number of qubits.
        compose: When True (default False), return the composition of the doubly stochastic matrices.
        dtype: Torch dtype for the resulting tensor(s).
        tpow2:
        ident_term: When False (default True), remove the identity component
        from the resulting matrix. This is used to reduce improve the performance
        of the subsequent matrix multiplications.

    Returns:

    """
    dtype = dtype or DefaultDType
    theta = torch.flatten(torch.as_tensor(theta, dtype=dtype))
    if len(theta) != len(swaps):
        raise ValueError("Expected same number of thetas and swaps")

    theta = _activation(theta)
    m = torch.sparse_coo_tensor(*sswaps_coo(swaps, theta), (len(swaps), n, n), dtype=dtype)
    m = _psswap(m) if tpow2 else m
    if ident_term:
        iden = torch.sparse_coo_tensor(*multieye_coo(m.shape), dtype=dtype)
        m = iden + m
    else:
        assert not compose
    return reduce(torch.sparse.mm, m) if compose else list(m)


def _layer_prep_final(mats, mode):
    if mode == "np":
        if isinstance(mats, tuple):
            mats = map(partial(torch_detach_np, dense=True), mats)
            mats = tuple(mats)
        else:
            mats = torch_detach_np(mats, dense=True)
    return mats


def _theta_slices(topology):
    # Prepare slices for array theta
    theta_sl = np.array(list(map(len, topology)))
    theta_sl = np.stack([np.concatenate([[0], np.cumsum(theta_sl)[:-1]]), np.cumsum(theta_sl)]).T
    theta_sl = list(map(lambda v: slice(*v), theta_sl))
    assert len(topology) == len(theta_sl)
    return theta_sl


_SWAPS_LAYER_MODES = {"torch_sparse", "np"}


def swaps_layer(
    theta,
    *,
    topology,
    n,
    dtype=None,
    tpow2=False,
    mode="torch_sparse",
    sublayers=False,
    ident_term=True,
):
    """
    Prepare the doubly stochastic matrices for the smooth swaps.

    Args:
        theta:
        topology: A list of tuples where each tuple represent a swap.
        n: The number of qubits on the current circuit.
        dtype: The data type for the PyTorch structures.
        tpow2: Enable the PSSWAP form, see paper for details.
        mode:
        sublayers: When True (default False) the layers of swaps are not composed into
        a single matrix.
        ident_term: Activates the same flag on the function `sswaps`.

    Returns: A list of matrices.

    """
    if mode not in _SWAPS_LAYER_MODES:
        raise ValueError(
            invalid_arg(arg="mode", expected=f"value from set {_SWAPS_LAYER_MODES}", actual=mode)
        )
    if not (isinstance(n, int) and n > 0):
        raise ValueError(invalid_arg(arg="n", expected=f"int greater than 0", actual=n))
    if not ident_term:
        # Force decomposed form when identity term for swaps
        # is disabled.
        assert sublayers
    dtype = dtype or DefaultDType
    topology = list(topology)
    theta = torch.as_tensor(theta, dtype=dtype)

    args = dict(n=n, compose=ident_term, dtype=dtype, tpow2=tpow2, ident_term=ident_term)
    theta_sl = _theta_slices(topology)
    mats = [sswaps(layer, theta.__getitem__(sl), **args) for layer, sl in zip(topology, theta_sl)]
    mats = list(chain(*mats)) if not ident_term else mats
    mats = tuple(mats[::-1]) if sublayers else reduce(torch.sparse.mm, mats)
    return _layer_prep_final(mats, mode=mode)


def coupling_map_to_distance_matx(cmap: CouplingMap, *, vec=False) -> np.ndarray:
    """
    Prepare a special distance matrix for the optimization.

    Args:
        cmap: The coupling map for which the dictance matrix
        is to be calculated.
        vec: When True (default False), return the vectorization
        (row major) of the output matrix.

    Returns: A numpy array containing either the distance matrix
    of the vectorization (row major) of the latter.

    """
    if not isinstance(cmap, CouplingMap):
        raise ValueError(
            arg_type_error(arg="cmap", expected_type=CouplingMap, actual_type=type(cmap))
        )

    m = np.where(cmap.distance_matrix <= 1, 0, cmap.distance_matrix - 1)
    return m.flatten() if vec else m


def coupling_map_to_swaps_topo(cmap: CouplingMap):
    """
    Prepare the swaps topology for the optimization from
    a coupling map.

    Args:
        cmap: The coupling map.

    Returns: A list of tuples corresponding to the mapping
    of the swaps.

    """
    if not isinstance(cmap, CouplingMap):
        raise ValueError(
            arg_type_error(arg="cmap", expected_type=CouplingMap, actual_type=type(cmap))
        )

    # TODO If cmap is line => skip edge coloring.

    # Obtain edge coloring by applying vertex coloring to the
    # line graph corresponding to the input graph.
    g = nx.Graph()
    g.add_edges_from(cmap.get_edges())
    g = nx.line_graph(g)
    colors = nx.coloring.greedy_color(g, strategy="largest_first")

    topo = [[] for _ in range(max(colors.values()) + 1)]
    for k, v in colors.items():
        topo[v].append(tuple(np.sort(k)))
    return topo
