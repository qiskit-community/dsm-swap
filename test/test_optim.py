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
"""Tests for the optim functions."""
from itertools import product

import numpy as np
import torch
from ddt import ddt, unpack, data, idata
from qiskit.transpiler import CouplingMap

from dsm_swap.optim import linear_decay, EarlyStopping, CostFactory, rh_knitter
from test.dsm_swap_test_case import DSMSwapTestCase


@ddt
class TestOptim(DSMSwapTestCase):
    """Tests for the optim functions."""

    def setUp(self) -> None:
        super().setUp()
        np.random.seed(123456)

    @idata(product([linear_decay], [25], [torch.float, torch.double]))
    @unpack
    def test_decay_fun(self, fun, max_n, dtype):
        """Test a generic decay function for the hw cost."""
        for n in range(1, max_n):
            v = fun(n, dtype=dtype)
            self.assertIsInstance(v, torch.Tensor)
            self.assertTrue(v.dtype == dtype)
            v = v.numpy()
            # Positive values.
            self.assertTrue(np.all(v > 0))
            # Monotonic descreasing.
            self.assertTrue(np.all((v[:-1] - v[1:]) >= 0))

    @idata(product([0.1, 0.01], [2, 8, 32]))
    @unpack
    def test_early_stopping(self, thr=0.01, patience=2):
        """Test early stopping mechanism."""
        obj = EarlyStopping(abs_delta_thr=thr, patience=patience)
        # Solution for equation (1/n - 1/(n+1)) < thr.
        n = (-1 + np.sqrt(1 + 4 / np.abs(thr))) / 2
        n = int(np.ceil(n))
        vec = 1 / np.arange(1, n + patience + 1)
        vec = np.array([obj(v) for v in vec])
        self.assertTrue(np.all(~vec[:-1]))
        self.assertTrue(vec[-1])

    @data(2, 3, 4, 5)
    def test_costf_costs_1(self, n):
        """Test CostFactory.hw/swap_cost and init."""
        dtype = torch.float
        tzero = torch.as_tensor(0, dtype=dtype)
        with torch.no_grad():
            c = CostFactory(swap_layers=1, dtype=dtype)
            c.init_data([np.array([[], []])], cmap=CouplingMap.from_line(n))
            c.init_params()

            c.params.fill_(np.pi / 2)
            self.assertTrue(torch.numel(c.params) == (n - 1))
            self.assertTrue(c.circ_adjs_v.shape == (n**2, 1))
            self.assertTrue(torch.allclose(c.circ_adjs_v, tzero))
            self.assertTrue(
                torch.allclose(c.swap_cost(skip_head_layers=0), torch.as_tensor(n - 1, dtype=dtype))
            )
            self.assertTrue(torch.allclose(c.swap_cost(skip_head_layers=1), tzero))
            self.assertTrue(torch.allclose(c.hw_cost(), tzero))

            c.params.fill_(0)
            self.assertTrue(torch.allclose(c.swap_cost(skip_head_layers=0), tzero))
            self.assertTrue(torch.allclose(c.hw_cost(), tzero))

    def test_costf_init_expections(self):
        """Test CostFactory init."""
        c = CostFactory(swap_layers=1, dtype=torch.float)
        with self.assertRaises(ValueError):
            c.init_data([np.array([[], []])], cmap=CouplingMap.from_line(1))

        with self.assertRaises(ValueError):
            # Expected exception since the init_data failed and
            # init_params depends on the latter.
            c.init_params()
        with self.assertRaises(ValueError):
            c.hw_cost()

        with self.assertRaises(ValueError):
            c.init_data([], cmap=CouplingMap.from_line(2))

        with self.assertRaises(ValueError):
            c.init_data(
                [np.array([[], []])],
                cmap=CouplingMap.from_line(2),
                swaps_gen_cmap=CouplingMap.from_line(3),
            )

        c.init_data([np.array([[], []])], cmap=CouplingMap.from_line(2))
        c.init_params()
        with self.assertRaises(ValueError):
            c.hw_cost(torch.zeros(3, 1, 1))

    def test_costf_init(self):
        """Test CostFactory init."""
        c = CostFactory(swap_layers=1, dtype=torch.float)
        c.init_data([np.array([[], []])] * 2, cmap=CouplingMap.from_line(2))
        c.init_params(swaps_depth=0)
        self.assertTrue(c.params.shape[0] == 2)
        c.init_params(swaps_depth=1)
        self.assertTrue(c.params.shape[0] == 1)

    def test_costf_eval_perms(self):
        """Test CostFactory.eval_permutations."""
        dtype = torch.float
        with torch.no_grad():
            c = CostFactory(swap_layers=1, dtype=dtype)
            c.init_data([np.array([[], []])], cmap=CouplingMap.from_line(2))
            c.init_params()
            c.params.fill_(np.pi / 2)
            self.assertTrue(len(c.eval_permutations()) == 1)
            self.assertTrue(
                np.all(c.eval_permutations()[0] == np.array([[0, 1], [1, 0]], dtype=int))
            )
            c.params.fill_(0)
            self.assertTrue(np.all(c.eval_permutations()[0] == np.eye(2, dtype=int)))

    @data(2, 3, 4, 5)
    def test_costf_sample_params(self, n):
        """Test the mechanics of CostFactory.sample_exact_params."""
        dtype = torch.float
        with torch.no_grad():
            c = CostFactory(swap_layers=1, dtype=dtype)
            c.init_data([np.array([[], []])], cmap=CouplingMap.from_line(n))
            c.init_params()
            c.params.fill_(np.pi / 2 * 0.9)
            with self.assertRaises(ValueError):
                c.sample_exact_params(n=0)
            self.assertIsInstance(c.sample_exact_params()[0], np.ndarray)
            for k in range(1, 4):
                self.assertTrue(len(c.sample_exact_params(n=k)) == k)
            self.assertTrue(np.allclose(c.sample_exact_params()[0] / (np.pi / 2), 1))

    def test_rh_knitter_expections(self):
        """Test rh_knitter exceptions."""
        with self.assertRaises(ValueError):
            rh_knitter([], cmap=CouplingMap.from_line(3), restart=0)
        with self.assertRaises(ValueError):
            rh_knitter([], cmap=CouplingMap.from_line(3), restart=3, restart1=2)
        with self.assertRaises(ValueError):
            rh_knitter([], cmap=CouplingMap.from_line(3), swap_cost_skip_layers=2)
        with self.assertRaises(ValueError):
            # Invalid param swap_layers.
            rh_knitter([np.array([[], []])], cmap=CouplingMap.from_line(3), swap_layers=-1)
        with self.assertRaises(ValueError):
            # Invalid adj matrix.
            rh_knitter([np.array([1])], cmap=CouplingMap.from_line(3))

    @data(3, 4)
    def test_rh_knitter_layers_int(self, n):
        ret = rh_knitter([np.array([[], []])], cmap=CouplingMap.from_line(n), swap_layers=n)
        self.assertTrue(len(ret) == 1)
        self.assertTrue(len(ret[0]) == n * 2)
        self.assertTrue(all(map(lambda v: np.allclose(v, np.eye(n)), ret[0])))

    @data(3, 4)
    def test_rh_knitter_layers_sqrt(self, n):
        ret = rh_knitter([np.array([[], []])], cmap=CouplingMap.from_line(n), swap_layers="sqrt")
        self.assertTrue(len(ret) == 1)
        self.assertTrue(len(ret[0]) == int(np.ceil(np.sqrt(n))) * 2)
        self.assertTrue(all(map(lambda v: np.allclose(v, np.eye(n)), ret[0])))

    @data(3, 4)
    def test_rh_knitter_layers_log_2(self, n):
        ret = rh_knitter([np.array([[], []])], cmap=CouplingMap.from_line(n), swap_layers="log_2")
        self.assertTrue(len(ret) == 1)
        self.assertTrue(len(ret[0]) == int(np.ceil(np.log2(n))) * 2)
        self.assertTrue(all(map(lambda v: np.allclose(v, np.eye(n)), ret[0])))

    def test_rh_knitter_basics(self):
        """Test rh_knitter basics."""
        for k in range(1, 4):
            ret = rh_knitter([], cmap=CouplingMap.from_line(k))
            self.assertTrue(len(ret) == 0)

        for k, n in product(range(1, 3), range(1, 3)):
            ret = rh_knitter([np.array([[], []])] * k, cmap=CouplingMap.from_line(n))
            self.assertTrue(len(ret) == k)
