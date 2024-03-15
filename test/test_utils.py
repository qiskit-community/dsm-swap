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
"""Tests of the utility functions."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.dagcircuit import DAGCircuit

from dsm_swap import dag_decompose
from dsm_swap.utils import DecomposedDAG
from test.dsm_swap_test_case import DSMSwapTestCase


class TestDecomposeDAGFunction(DSMSwapTestCase):
    """Tests of the decompose_dag utility function."""

    def setUp(self) -> None:
        super().setUp()

    def test_dag_decompose_q1(self):
        """Test on a one qubit circuit."""
        qc = QuantumCircuit(1)
        ddag = dag_decompose(qc)
        self._test_decomposed_dag(ddag, 1)
        self.assertEqual(dag_to_circuit(ddag.layers[0]), qc)

    def test_dag_decompose_q2(self):
        """Test on a two qubit circuit."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.cx(1, 0)
        ddag = dag_decompose(qc)
        self._test_decomposed_dag(ddag, 1)
        self.assertEqual(dag_to_circuit(ddag.layers[0]), qc)

    def test_dag_decompose_simple(self):
        """Test on a circuit that does not have two qubit gates."""
        qc = QuantumCircuit(4)
        qc.h(range(4))
        ddag = dag_decompose(qc)

        self._test_decomposed_dag(ddag, 1)
        np.testing.assert_array_equal(ddag.get_adjs()[0], np.zeros((4, 4)))

        self.assertEqual(dag_to_circuit(ddag.layers[0]), qc)

    def test_dag_decompose_with_measure(self):
        """Test on a circuit with measure operations."""
        qc = QuantumCircuit(4)
        qc.measure_all()
        ddag = dag_decompose(qc)

        self._test_decomposed_dag(ddag, 1)
        np.testing.assert_array_equal(ddag.get_adjs()[0], np.zeros((4, 4)))

        self.assertEqual(dag_to_circuit(ddag.layers[0]), qc)

    def test_dag_decompose_one_layer(self):
        """Test on a circuit that has only one layer."""
        qc = QuantumCircuit(4)
        qc.cx(0, 3)
        qc.cx(1, 2)
        ddag = dag_decompose(qc)

        self._test_decomposed_dag(ddag, 1)
        np.testing.assert_array_equal(ddag.get_adjs()[0], np.rot90(np.eye(4)))

        self.assertEqual(dag_to_circuit(ddag.layers[0]), qc)

    def test_dag_decompose_two_layers(self):
        """Test on a circuit that has two layers."""
        qc = QuantumCircuit(4)
        qc.cx(0, 3)
        qc.cx(1, 2)
        qc.cx(0, 1)
        ddag = dag_decompose(qc)

        self._test_decomposed_dag(ddag, 2)

        # test both matrices
        np.testing.assert_array_equal(ddag.get_adjs()[0], np.rot90(np.eye(4)))
        second_adj = np.zeros((4, 4))
        second_adj[0, 1] = 1
        second_adj[1, 0] = 1
        np.testing.assert_array_equal(ddag.get_adjs()[1], second_adj)

    def test_dag_decompose_zz(self):
        """Basic test on a `ZZFeatureMap` circuit."""
        qc = ZZFeatureMap(4).decompose()
        ddag = dag_decompose(qc)
        # too many layers, we don't test adjacency matrices
        self._test_decomposed_dag(ddag, 19)

    def test_dag_decompose_ending_1q(self):
        """
        Test that one-qubit gates in the end of the circuit are appended to a layer with
        two-qubit gates.
        """
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.h([0, 1])
        ddag = dag_decompose(qc)
        self._test_decomposed_dag(ddag, 1)

    def _test_decomposed_dag(self, ddag, num_layers):
        self.assertIsInstance(ddag, DecomposedDAG)
        self.assertIsNotNone(ddag.layers)
        self.assertEqual(len(ddag.layers), num_layers)
        for i in range(num_layers):
            self.assertIsInstance(ddag.layers[i], DAGCircuit)

        self.assertIsNotNone(ddag.swaps)
        self.assertEqual(len(ddag.swaps), num_layers)
        for i in range(num_layers):
            self.assertEqual(len(ddag.swaps[i]), 0)

        adjacency_matrices = ddag.get_adjs()
        self.assertIsNotNone(adjacency_matrices)
        self.assertEqual(len(adjacency_matrices), num_layers)

    def test_dag_decompose_exceptions(self):
        """Test with wrong parameters."""
        with self.assertRaises(ValueError):
            dag_decompose(None)

        with self.assertRaises(ValueError):
            dag_decompose(ZZFeatureMap(4))


class TestDecomposedDAGClass(DSMSwapTestCase):
    """Tests of the DecomposedDAG class."""

    def setUp(self) -> None:
        super().setUp()

    def test_more_swaps_than_layers(self):
        """Test composing circuit with more swap layers than layers in the circuit."""
        qc = QuantumCircuit(4)
        qc.x(0)
        qc.y(1)
        qc.z(2)
        qc.h(3)
        dag = circuit_to_dag(qc)

        ddag = DecomposedDAG(layers=[dag])
        swaps = [
            [np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])],
            [
                np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
            ],
        ]
        dag, layout = ddag.compose(swaps, first_layer_to_layout=False)
        qc_actual = dag_to_circuit(dag)

        # test layout
        self.assertEqual(layout.get_physical_bits()[0]._index, 1)
        self.assertEqual(layout.get_physical_bits()[1]._index, 0)
        self.assertEqual(layout.get_physical_bits()[2]._index, 3)
        self.assertEqual(layout.get_physical_bits()[3]._index, 2)

        # test resulting circuit
        qc_expected = QuantumCircuit(4)
        qc_expected.x(0)
        qc_expected.y(1)
        qc_expected.swap(2, 3)
        qc_expected.swap(0, 1)
        qc_expected.h(2)
        qc_expected.z(3)
        self.assertEqual(qc_actual, qc_expected)

    def test_decomposed_dag_exceptions(self):
        """Test DecomposeDAG with wrong parameters."""
        with self.assertRaises(ValueError):
            _ = DecomposedDAG(layers=[])

        with self.assertRaises(ValueError):
            qc = QuantumCircuit(1)
            dag = circuit_to_dag(qc)
            _ = DecomposedDAG(layers=[dag], swaps=[(), ()])  # () * 2 does not work

        with self.assertRaises(ValueError):
            qc = QuantumCircuit(4)
            qc.h(range(4))
            ddag = dag_decompose(qc)

            ddag.get_adjs(mode="unknown")

        with self.assertRaises(ValueError):
            qc = QuantumCircuit(4)
            qc.h(range(4))
            ddag = dag_decompose(qc)

            ddag.compose(swaps=[(0, 1), (1, 2), (2, 3)])
