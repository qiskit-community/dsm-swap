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
"""Tests for the plot module functions."""

import numpy as np
from ddt import ddt, unpack, idata
from itertools import product
from dsm_swap.plot import np_cumulative_mm, wire_paths
from test.dsm_swap_test_case import DSMSwapTestCase


@ddt
class TestPlot(DSMSwapTestCase):
    """Tests for the plot module functions."""

    def setUp(self) -> None:
        super().setUp()
        np.random.seed(123456)

    @staticmethod
    def _cmp_mats(mats1, mats2):
        """Compare two lists of matrices."""
        return all(map(lambda mm: np.allclose(mm[0], mm[1]), zip(mats1, mats2)))

    def test_np_cumulative_mm(self):
        """Tests plot.np_cumulative_mm function."""
        pauli_x = np.eye(2)[::-1]
        pauli_x

        mats = [np.eye(2)] * 3
        ret = np_cumulative_mm(mats)
        self.assertTrue(self._cmp_mats(mats, ret))

        mats = [np.eye(2), pauli_x]
        ret = np_cumulative_mm(mats)
        self.assertTrue(self._cmp_mats(mats, ret))

        mats = [pauli_x] * 3
        ret = np_cumulative_mm(mats)
        mats = [pauli_x, np.eye(2), pauli_x]
        self.assertTrue(self._cmp_mats(mats, ret))

    @idata(product([2, 3, 4], [1, 2, 3]))
    @unpack
    def test_wire_paths_grid(self, n=3, layers=3):
        m = np.tile(np.expand_dims(np.arange(n), 0), (layers, 1))
        grid = wire_paths(m)[1]
        self.assertTrue(grid.shape == (layers, 2, n))

        self.assertTrue(
            np.allclose(grid[:, 0, :] - np.expand_dims(np.linspace(0, 1, layers + 1)[1:], -1), 0)
        )
        self.assertTrue(np.allclose(grid[:, 1, :] - np.linspace(0, 1, n), 0))

    def test_wire_paths_exceptions(self):
        with self.assertRaises(ValueError):
            wire_paths(np.array(1))
        with self.assertRaises(ValueError):
            wire_paths(np.array([[]]))
