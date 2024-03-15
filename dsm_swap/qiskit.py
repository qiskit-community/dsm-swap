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

import logging
from typing import Optional

from qiskit.transpiler import PassManagerConfig, PassManager, TranspilerError
from qiskit.transpiler import TransformationPass
from qiskit.transpiler.preset_passmanagers import common
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin

from dsm_swap.optim import rh_knitter, linear_decay, NoSolutionFoundError
from dsm_swap.utils import dag_decompose

logger = logging.getLogger(__name__)


class DSMSwap(TransformationPass):
    """DSM-SWAP algorithm which maps a logical circuit onto a physical topology by the insertion
    of SWAP gates."""

    def __init__(
        self, coupling_map, max_optim_steps=30, horizon=2, restarts=5, swap_layers=None, seed=None
    ):
        """
        Args:
            coupling_map: A Qiskit coupling map representing the connectivity.
            max_optim_steps: The maximum number of steps (default 30) for the optimizer.
            horizon: The horizon for the rolling horizon mechanism.
            restart: The number of restarts (default 5) for the optimizer running on each
            sub-circuit consisting of up to a horizon number of layers.
            swap_layers: The number of swap layers for each circuit layer. When None, this is
            set to the ceiling of the log_2 of the number of qubits.
            seed: Seed for the random number generator.
        """
        super().__init__()
        self.coupling_map = coupling_map
        self.max_optim_steps = max_optim_steps
        self.horizon = horizon
        self.restarts = restarts
        self.swap_layers = swap_layers
        self.seed = seed

    def run(self, dag):
        """Run the DSM-SWAP algorithm.

        Args:
            dag (DAGCircuit): DAG to map.

        Returns:
            DAGCircuit:

        Raises:
            TranspilerError: Error when number of virtual and physical qubits do not match.
        """
        if self.coupling_map is None:
            return dag

        if len(dag.qubits) != self.coupling_map.size():
            raise TranspilerError(
                "DSM-SWAP requires the number of virtual and physical qubits to be the same"
            )

        original_dag = dag
        try:
            dagd = dag_decompose(dag)
            layer_perms = rh_knitter(
                dagd.get_adjs(mode="coo"),
                max_optim_steps=self.max_optim_steps,
                seed=self.seed,
                horizon=self.horizon,
                restart=self.restarts,
                cmap=self.coupling_map,
                hw_cost_decay=linear_decay,
                swap_layers=self.swap_layers,
            )
            dag, final_layout = dagd.compose(layer_perms)
            self.property_set["final_layout"] = final_layout
            return dag
        except NoSolutionFoundError:
            logger.warning("DSM-SWAP convergence failure.")
            return original_dag


class DSMSwapPassManager(PassManagerStagePlugin):
    """DSM Swap pass manager plugin."""

    def pass_manager(
        self, pass_manager_config: PassManagerConfig, optimization_level: Optional[int] = None
    ) -> PassManager:
        """Build routing stage PassManager."""
        seed_transpiler = pass_manager_config.seed_transpiler
        target = pass_manager_config.target
        coupling_map = pass_manager_config.coupling_map
        backend_properties = pass_manager_config.backend_properties
        vf2_call_limit = common.get_vf2_limits(
            optimization_level,
            pass_manager_config.layout_method,
            pass_manager_config.initial_layout,
        )
        if optimization_level == 0:
            routing_pass = DSMSwap(coupling_map, restarts=5)

            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                seed_transpiler=seed_transpiler,
                use_barrier_before_measurement=True,
            )
        if optimization_level == 1:
            routing_pass = DSMSwap(coupling_map, restarts=5)

            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map,
                vf2_call_limit=vf2_call_limit,
                backend_properties=backend_properties,
                seed_transpiler=seed_transpiler,
                check_trivial=True,
                use_barrier_before_measurement=True,
            )

        if optimization_level == 2:
            routing_pass = DSMSwap(coupling_map, restarts=5)

            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                vf2_call_limit=vf2_call_limit,
                backend_properties=backend_properties,
                seed_transpiler=seed_transpiler,
                use_barrier_before_measurement=True,
            )
        if optimization_level == 3:
            routing_pass = DSMSwap(coupling_map, restarts=10)

            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                vf2_call_limit=vf2_call_limit,
                backend_properties=backend_properties,
                seed_transpiler=seed_transpiler,
                use_barrier_before_measurement=True,
            )
        raise TranspilerError(f"Invalid optimization level specified: {optimization_level}")
