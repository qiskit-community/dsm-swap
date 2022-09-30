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

import json
import time
from typing import List, Mapping

from qiskit import QuantumCircuit
from qiskit.circuit.library import QuantumVolume
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import CouplingMap

from dsm_swap.optim import rh_knitter, linear_decay
from dsm_swap.utils import dag_decompose


def apply_dsm_swap(qc, *, seed, max_optim_steps, horizon, restart, swap_layers):
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
    depth = optimized_dag.depth()

    num_swaps = optimized_dag.count_ops().get("swap", 0)

    # extract swap operations and their qubit indices into a list
    swaps = list(
        map(
            lambda node: [node.qargs[0].index, node.qargs[1].index],
            filter(lambda node: node.name == "swap", optimized_dag.op_nodes()),
        )
    )

    return qc, depth, num_swaps, swaps


def generate_dataset():
    seed = 38492314298
    circuits: List[QuantumCircuit] = []
    results: List[Mapping] = []
    max_optim_steps = 10
    restart = 3
    swap_layers = "log_2"
    for num_qubits in [4, 5, 6]:
        for qv_depth in range(5, 8):
            for horizon in [2, 3]:
                qc = QuantumVolume(num_qubits, qv_depth, seed).decompose()

                start = time.time()
                try:
                    qc, depth, num_swaps, swaps = apply_dsm_swap(
                        qc,
                        seed=seed,
                        max_optim_steps=max_optim_steps,
                        horizon=horizon,
                        restart=restart,
                        swap_layers=swap_layers,
                    )
                except ValueError as e:
                    print(
                        f"Circuit {num_qubits}/{qv_depth}/{horizon} raised an error: {e}, skipped"
                    )
                    continue
                elapsed = time.time() - start
                if num_swaps < 1:
                    print(
                        f"Circuit {num_qubits}/{qv_depth}/{horizon} is skipped, elapsed {elapsed:.2f}"
                    )
                    continue
                print(f"Circuit {num_qubits}/{qv_depth}/{horizon} is added, elapsed {elapsed:.2f}")
                circuits.append(qc)
                results.append(
                    dict(
                        seed=seed,
                        num_qubits=num_qubits,
                        qv_depth=qv_depth,
                        max_optim_steps=max_optim_steps,
                        horizon=horizon,
                        restart=restart,
                        swap_layers=swap_layers,
                        depth=depth,
                        num_swaps=num_swaps,
                        swaps=swaps,
                    )
                )

                seed += 1000

    # todo: do we need circuits as qasm files?
    for i, qc in enumerate(circuits):
        qc.qasm(filename=f"test_circuit_{i}.qasm")
    with open("test_dataset.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    generate_dataset()
