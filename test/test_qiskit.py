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
"""Test Qiskit Plugin."""

from qiskit import QuantumCircuit
from qiskit.transpiler import PassManagerConfig, CouplingMap

from dsm_swap import DSMSwapPassManager
from test.dsm_swap_test_case import DSMSwapTestCase


class TestQiskitPlugin(DSMSwapTestCase):
    """Test Qiskit Plugin."""

    def setUp(self) -> None:
        super().setUp()

        # reference circuit
        self.reference_qc = QuantumCircuit.from_qasm_file(
            self.get_resource_path("test_circuit_reference.qasm", "test_data")
        )

    def test_dsm_swap_plugin(self):
        """Basic plugin test"""
        cmap = CouplingMap.from_ring(self.reference_qc.num_qubits, bidirectional=True)
        pm_config = PassManagerConfig(
            coupling_map=cmap, layout_method="sabre", seed_transpiler=38492314298
        )
        plugin = DSMSwapPassManager()
        pm = plugin.pass_manager(pm_config, optimization_level=0)

        tqc = pm.run(self.reference_qc)
        # todo: 17 replaced with 14
        self.assertEqual(tqc.depth(), 14)
        # todo: 17 replaced with 14
        self.assertEqual(tqc.count_ops()["swap"], 14)
