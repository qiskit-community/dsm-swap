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

import numpy as np

from dsm_swap.swapmatic import sswaps

if __name__ == "__main__":
    all_params = [
        # compose, tpow2, ident_term
        [False, False, False],
        [False, False, True],
        [False, True, False],
        [False, True, True],
        # [True, False, False],
        [True, False, True],
        # [True, True, False],
        [True, True, True],
    ]

    all_swaps = [([0, 1], [1, 2]), ([0, 1], [1, 2], [2, 0]), ([0, 1], [1, 2], [2, 3])]
    all_values = [[0, 0], [np.pi / 4, np.pi / 4, np.pi / 4], [np.pi / 2, np.pi / 2, np.pi / 2]]

    sswap_dataset = []
    for compose, tpow2, ident_term in all_params:
        for swaps, values in zip(all_swaps, all_values):
            num_qubits = np.max(swaps) + 1
            result = sswaps(
                swaps, values, n=num_qubits, compose=compose, tpow2=tpow2, ident_term=ident_term
            )
            if compose:
                result = result.to_dense().tolist()
            else:
                result = [item.to_dense().tolist() for item in result]
            sswap_dataset.append(
                dict(
                    compose=compose,
                    tpow2=tpow2,
                    ident_term=ident_term,
                    swaps=swaps,
                    values=values,
                    expected_result=result,
                )
            )

    with open("test_sswap_dataset.json", "w") as f:
        json.dump(sswap_dataset, f, indent=2)
