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
"""All reference/integration tests for the algorithm."""
import json

from qiskit import QuantumCircuit
from qiskit.circuit.library import QuantumVolume
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import CouplingMap

from dsm_swap.optim import rh_knitter, linear_decay
from dsm_swap.utils import dag_decompose
from test.dsm_swap_test_case import DSMSwapTestCase


class TestRHKnitter(DSMSwapTestCase):
    """All reference/integration tests for the algorithm."""

    def setUp(self) -> None:
        super().setUp()

        # reference circuit
        self.reference_qc = QuantumCircuit.from_qasm_file(
            self.get_resource_path("test_circuit_reference.qasm", "test_data")
        )

        # dataset of test circuits
        with open(self.get_resource_path("test_dataset.json", "test_data")) as f:
            self.dataset = json.load(f)

    def test_reference(self):
        """
        A reference test for the algorithm.
        """
        # configuration
        max_optim_steps = 30
        seed = 38492314298
        swap_layers = "log_2"
        horizon = 2
        restarts = 5

        cmap = CouplingMap.from_ring(self.reference_qc.num_qubits, bidirectional=True)
        dag = circuit_to_dag(self.reference_qc)
        dagd = dag_decompose(dag)

        layer_perms = rh_knitter(
            dagd.get_adjs(mode="coo"),
            max_optim_steps=max_optim_steps,
            seed=seed,
            horizon=horizon,
            restart=restarts,
            cmap=cmap,
            hw_cost_decay=linear_decay,
            swap_layers=swap_layers,
        )

        optimized_dag, _ = dagd.compose(layer_perms)
        # todo: 18 -> 15
        self.assertEqual(optimized_dag.depth(), 15)
        # todo: 19 -> 14
        self.assertEqual(optimized_dag.count_ops()["swap"], 14)

        swaps = self._get_swaps(optimized_dag)

        self.assertListEqual(
            swaps,
            [
                [0, 1],
                [1, 2],
                [5, 6],
                [4, 5],
                [5, 6],
                [0, 1],
                [0, 7],
                [6, 7],
                [3, 4],
                [4, 5],
                [2, 3],
                [5, 6],
                [0, 1],
                [0, 7],
            ],
        )

    def test_dataset(self):
        """Test the algorithm on the test circuits from the test_data directory."""
        for conf in self.dataset:
            seed = conf["seed"]

            # configuration of the algorithm
            num_qubits = conf["num_qubits"]
            qv_depth = conf["qv_depth"]
            horizon = conf["horizon"]
            max_optim_steps = conf["max_optim_steps"]
            restart = conf["restart"]
            swap_layers = conf["swap_layers"]

            # results
            depth = conf["depth"]
            swaps = conf["swaps"]

            with self.subTest(f"num_qubits/qv_depth/horizon {num_qubits}/{qv_depth}/{horizon}"):
                qc = QuantumVolume(num_qubits, qv_depth, seed).decompose()
                self._test_configuration(
                    qc,
                    seed,
                    max_optim_steps,
                    horizon,
                    restart,
                    swap_layers,
                    depth,
                    swaps,
                )

    def _test_configuration(
        self, qc, seed, max_optim_steps, horizon, restart, swap_layers, depth, swaps
    ):
        cmap = CouplingMap.from_ring(qc.num_qubits, bidirectional=True)
        dag = circuit_to_dag(qc)
        dagd = dag_decompose(dag)

        layer_perms = rh_knitter(
            dagd.get_adjs(mode="coo"),
            max_optim_steps=max_optim_steps,
            seed=seed,
            horizon=horizon,
            restart=restart,
            cmap=cmap,
            hw_cost_decay=linear_decay,
            swap_layers=swap_layers,
        )

        optimized_dag, _ = dagd.compose(layer_perms)

        self.assertEqual(optimized_dag.depth(), depth)
        self.assertEqual(optimized_dag.count_ops()["swap"], len(swaps))
        self.assertListEqual(self._get_swaps(optimized_dag), swaps)

    def _get_swaps(self, dag):
        """Extract swap operations and their qubit indices into a list."""
        return list(
            map(
                lambda node: [node.qargs[0]._index, node.qargs[1]._index],
                filter(lambda node: node.name == "swap", dag.op_nodes()),
            )
        )
