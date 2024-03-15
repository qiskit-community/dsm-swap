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

from itertools import chain, count
from .math import (
    adj_mat_to_coo,
    swap_mat_decompose,
)
from .errors import arg_type_error
from qiskit import QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag
from qiskit.circuit.library import SwapGate
from qiskit.transpiler.layout import Layout
from qiskit.circuit import Measure
import numpy as np
from typing import Collection, Union, Tuple, List


DAGLayers = Collection[DAGCircuit]
SWAPSTupleForm = List[Tuple[int, int]]


def _get_targets(rec, *, expected=None):
    """Get targets.

    Given the qargs from a gate, obtain and validate the target qubits.
    """
    t = [int(q._index) for q in rec]
    if expected is not None and len(t) != expected:
        raise ValueError(f"Expected {expected} target qubits, found: {rec}")
    return t


def _get_swap_tuple_form(swap) -> SWAPSTupleForm:
    """Swap to tuple form.

    Normalize a given swap representation to tuple form, that is
    a tuple containing the two swapped indices.
    The function always returns a list of tuples because a
    representation can contain multiple swaps (e.g. permutation matrix).
    """
    if isinstance(swap, np.ndarray):
        return swap_mat_decompose(swap)
    if swap is None or (swap[0] == swap[1]):
        return []
    return [tuple(swap)]


def _chain_swap_tuple_forms(swaps) -> SWAPSTupleForm:
    """Chain swaps.

    Take a list of swaps into tuple form (see `_get_swap_tuple_form`)
    and concatenate the lists of tuples.
    """
    swaps = [_get_swap_tuple_form(s) for s in swaps]
    swaps = list(chain(*swaps))
    return swaps


def _swaps_apply(swaps: SWAPSTupleForm, v):
    """Apply swaps in tuple form to an array."""
    for swap in swaps:
        a, b = swap
        v[a], v[b] = v[b], v[a]


def _dag_push_swap(dag: DAGCircuit, targets: Tuple[int, int]):
    """Append a Swap gate acting on the given targets, to the input DAG."""
    qargs = dag.qubits[targets[0]], dag.qubits[targets[1]]
    # Take qargs to list (from tuple) due to possible Qiskit bug
    qargs = list(qargs)
    dag.apply_operation_back(SwapGate(), qargs=qargs)


class DecomposedDAG:
    def __init__(self, layers: DAGLayers, swaps=None) -> None:
        """
        Args:
            layers: A list of objects of type DAGCircuit. These represent the
            layering of the circuit.
            swaps: An optional list (default None) of swaps that precede each
            (DAG) layer. Each swap is encoded as either a permutation matrix
            or a tuple of two elements (the indices to be swapped).

        Raises:
            ValueError: Error in case of empty layers list or
            inconsistency between number of swap layers and circuit layers.
        """
        layers = list(layers)
        if len(layers) == 0:
            raise ValueError("At least one layer is required")
        self.layers = layers
        self.swaps = [tuple()] * len(layers) if swaps is None else list(swaps)
        if len(self.swaps) != len(self.layers):
            raise ValueError("Inconsistency between swaps layers and circuit layers")

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits."""
        return self.layers[0].num_qubits

    @property
    def size(self) -> int:
        """Return the number layers of this decomposition."""
        return len(self.layers)

    def compose(self, swaps=None, *, first_layer_to_layout=True) -> Tuple[DAGCircuit, Layout]:
        """Compose swaps.

        Compose the layers represented by the current object with the given swaps
        and produce a DAG circuit alongside the final layout.

        Args:
            swaps: An optional list (default None) of swaps that precede each
                (DAG) layer. Swaps is a list of lists of permutation matrices (as Numpy arrays).
                Each list of permutations represent the decomposition of a permutation as swaps, to be
                applied before a certain layer.
            first_layer_to_layout: When True (default) the first the swaps for
                the first layer are converted to initial layout without adding gates
                to the final circuit.

        Returns: A tuple containing the final DAG circuit and the final layout.

        Raises:
            ValueError: Error when the number of swaps layers is incompatible with the
            number of circuit layers.
        """
        swaps = swaps or self.swaps
        dag = _dag_empty_copy(self.layers[0])
        # Allow additional swaps layer at the end.
        if len(swaps) not in {len(self.layers), len(self.layers) + 1}:
            raise ValueError("Incompatible number of swaps layers")
        swapsl_acc = []
        for i, layer, swapsl in zip(count(), self.layers, swaps):
            swapsl = _chain_swap_tuple_forms(swapsl)
            if i <= 0 and first_layer_to_layout:
                # First layer swaps contribute to layout
                pass
            else:
                for swap in swapsl:
                    _dag_push_swap(dag, swap)
            swapsl_acc = swapsl + swapsl_acc
            qubits = list(dag.qubits)
            _swaps_apply(swapsl_acc, qubits)
            dag.compose(layer, inplace=True, qubits=qubits)

        if len(swaps) > len(self.layers):
            swapsl = _chain_swap_tuple_forms(swaps[-1])
            swapsl_acc = swapsl + swapsl_acc
            for swap in swapsl:
                _dag_push_swap(dag, swap)

        # Prepare final layout
        qubits = list(dag.qubits)
        _swaps_apply(swapsl_acc, qubits)
        final_layout = Layout({q: i for i, q in enumerate(qubits)})
        return dag, final_layout

    def avg_layer_density(self) -> float:
        """Compute the average layer density.

        The layer density is a measure of the concentration of two-qubits
        gates for each circuit layer. The measure goes from 0 to 1, for example
        Quantum Volume circuits are characterized by a density 1.
        """
        val = map(np.count_nonzero, self.get_adjs())
        val = np.mean(list(val)) // 2
        val = val / (self.num_qubits() // 2)
        return val

    def get_adjs(self, mode="mat") -> List[np.ndarray]:
        """Get adjacency matrices.

        Prepare the adjacency matrices corresponding to the
        layering represented by the current object.

        Args:
            mode: Either 'mat' (default) or 'coo' for selecting
            the storage format of the matrices. The option 'mat'
            produces a Numpy array whereas the value 'coo' a list
            of coordinates (as a Numpy array).

        Returns: Either a list of matrices or coordinates (both as Numpy array).

        Raises:
            ValueError: Error when mode is not recognized. Also error raised when
            the number of qubits upon which a gate acts is not supported.
        """
        if mode not in {"mat", "coo"}:
            raise ValueError(f"Unrecognized mode: {mode}")

        adjs = []
        for layer in self.layers:
            n = layer.num_qubits()
            m = np.zeros((n, n), dtype=int)
            for g in layer.op_nodes(include_directives=False):
                if isinstance(g.op, Measure):
                    continue
                assert g.op.num_clbits == 0
                if g.op.num_qubits == 1:
                    continue
                elif g.op.num_qubits == 2:
                    t = np.array(_get_targets(g.qargs, expected=2))
                    assert len(t) == 2
                    m[(t, t[::-1])] = 1
                else:
                    raise ValueError(f"Unrecognized gate: {g.op}")
            adjs.append(m)
        if mode == "coo":
            adjs = list(map(adj_mat_to_coo, adjs))
        return adjs


def _dag_count_gates(dag: DAGCircuit, min_q: int) -> int:
    """Count the number of gates acting on at least the given number of qubits."""
    assert isinstance(dag, DAGCircuit)
    v = dag.op_nodes(include_directives=False)
    v = [len(g.qargs) >= min_q for g in v]
    return int(np.sum(v))


def dag_decompose(dag: Union[DAGCircuit, QuantumCircuit]) -> DecomposedDAG:
    """Decompose a DAG circuit into layers of commuting two qubit gates.

    Args:
        dag: Either a QuantumCircuit or a DAGCircuit.

    Returns: An object of type `DecomposedDAG`.

    Raises:
        ValueError: Error when the input circuit is not a DAGCircuit nor
        a QuantumCircuit. Error when the input circuit has gates not in the
        U(2) nor U(4) classes.
    """
    if isinstance(dag, QuantumCircuit):
        dag = circuit_to_dag(dag)
    if not isinstance(dag, DAGCircuit):
        raise ValueError(arg_type_error(arg="dag", expected_type=DAGCircuit, actual_type=type(dag)))

    if dag.num_qubits() <= 2:
        return DecomposedDAG([dag])

    if _dag_count_gates(dag, 3):
        raise ValueError("Only U(2) and U(4) gates supported.")
    layers = []
    prev_layer = None  # Composition of single qubit layers
    for layer in dag.layers():
        layer = layer["graph"]
        if _dag_count_gates(layer, 2) == 0:
            if prev_layer is None:
                prev_layer = layer
            else:
                prev_layer = prev_layer.compose(layer, inplace=False)
            continue
        if prev_layer is not None:
            layer = prev_layer.compose(layer, inplace=False)
            prev_layer = None
        layers.append(layer)

    if prev_layer is not None:
        if len(layers) > 0:
            layers[-1] = layers[-1].compose(prev_layer, inplace=False)
        else:
            layers = [prev_layer]

    # layers = ([_dag_empty_copy(layers[0])] * header_empty_layers) + layers
    return DecomposedDAG(layers)


def _dag_empty_copy(dag: DAGCircuit) -> DAGCircuit:
    """DAG copy empty.

    Create an empty clone of the given DAG. The function handles the
    changing API.
    """
    if hasattr(dag, "copy_empty_like"):
        return dag.copy_empty_like()
    return dag._copy_circuit_metadata()
