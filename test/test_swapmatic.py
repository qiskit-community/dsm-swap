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
"""Tests for the dsm_swap functions."""
import json
from itertools import chain, product

import numpy as np
import torch
from ddt import ddt, unpack, data, idata
from qiskit.transpiler import CouplingMap

from dsm_swap.swapmatic import (
    sswaps_coo,
    multieye_coo,
    sswaps,
    swaps_layer,
    coupling_map_to_distance_matx,
    coupling_map_to_swaps_topo,
)
from test.dsm_swap_test_case import DSMSwapTestCase


@ddt
class TestSwapmatic(DSMSwapTestCase):
    """Tests for the dsm_swap functions."""

    def setUp(self) -> None:
        super().setUp()
        np.random.seed(123456)

        with open(self.get_resource_path("test_sswap_dataset.json", "test_data")) as f:
            self.sswap_dataset = json.load(f)

    @data(
        # swaps, values, expected indices, expected values
        [([0, 1]), None, [[0, 0, 0, 0], [0, 1, 0, 1], [1, 0, 0, 1]], [1, 1, -1, -1]],
        [([0, 1]), torch.tensor([2]), [[0, 0, 0, 0], [0, 1, 0, 1], [1, 0, 0, 1]], [2, 2, -2, -2]],
        [
            ([0, 1], [1, 2]),
            None,
            [[0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 2, 0, 1, 1, 2], [1, 2, 0, 1, 0, 1, 1, 2]],
            [1, 1, 1, 1, -1, -1, -1, -1],
        ],
    )
    @unpack
    def test_sswaps_coo_predefined(self, swaps, in_values, expected_indices, expected_values):
        """
        Test preparation of the configuration for the coo layered matrix construction on a
        predefined set of inputs.
        """
        out_indices, out_values = sswaps_coo(c=swaps, v=in_values)

        np.testing.assert_array_equal(out_indices, expected_indices)
        np.testing.assert_array_equal(out_values, expected_values)

    def test_sswaps_coo_randomized(self):
        """
        Test preparation of the configuration for the coo layered matrix construction on a
        random set of input.
        """
        for _ in range(5):
            num_swaps = np.random.randint(1, 10)
            swaps = self._generate_swaps(num_swaps)

            in_values = torch.rand(num_swaps)

            indices, out_values = sswaps_coo(c=swaps, v=in_values)
            # test indices shape
            np.testing.assert_array_equal(indices.shape, [3, 4 * num_swaps])
            # test that indices contain values from the swap array
            np.testing.assert_array_equal(np.unique(indices).sort(), np.unique(swaps).sort())

            # basic tests of the indices structure
            np.testing.assert_array_equal(indices[0, :num_swaps], np.arange(0, num_swaps))
            np.testing.assert_array_equal(indices[1, :num_swaps], swaps[:, 0])
            np.testing.assert_array_equal(indices[2, :num_swaps], swaps[:, 1])

            # test values shape
            self.assertEqual(out_values.shape[0], 4 * num_swaps)

            # test last num_swap values
            np.testing.assert_array_equal(out_values[-num_swaps:], -1 * in_values)

    def test_multieye_coo_randomized(self):
        """
        Test preparation the coordinates for the preparation of a tensor containing layered
        identity matrices.
        """
        for _ in range(5):
            dim1 = np.random.randint(1, 5)
            dim2 = np.random.randint(1, 5)

            shape = (dim1, dim2, dim2)

            indices, values, size = multieye_coo(shape)

            np.testing.assert_array_equal(indices.shape, (3, shape[0] * shape[1]))
            for i in range(shape[1]):
                np.testing.assert_array_equal(
                    indices[0, i * shape[0] : (i + 1) * shape[0]], np.arange(0, shape[0])
                )
                np.testing.assert_array_equal(
                    indices[1, i * shape[0] : (i + 1) * shape[0]], np.full(shape[0], i)
                )
                np.testing.assert_array_equal(
                    indices[2, i * shape[0] : (i + 1) * shape[0]], np.full(shape[0], i)
                )

            np.testing.assert_equal(np.unique(values), [1])
            np.testing.assert_array_equal(size, shape)

    def test_multieye_coo_exceptions(self):
        """Test preparation the coordinates for the preparation of a tensor containing layered
        identity matrices raises an exception when wrong parameters are passed."""
        with self.assertRaises(ValueError):
            multieye_coo((2, 2))

        with self.assertRaises(ValueError):
            multieye_coo((1, 2, 3))

    def test_sswaps(self):
        """Test of the preparation of a sparse tensor or sswaps."""
        for i, conf in enumerate(self.sswap_dataset):
            with self.subTest(f"SSWAP conf {i}"):
                compose = conf["compose"]
                tpow2 = conf["tpow2"]
                ident_term = conf["ident_term"]
                swaps = conf["swaps"]
                values = conf["values"]
                expected_result = conf["expected_result"]
                num_qubits = np.max(swaps) + 1
                result = sswaps(
                    swaps, values, n=num_qubits, compose=compose, tpow2=tpow2, ident_term=ident_term
                )

                if compose:
                    result = result.to_dense()
                else:
                    result = [item.to_dense().tolist() for item in result]

                np.testing.assert_array_equal(result, expected_result)

    @data(
        # compose, tpow2, ident_term
        [False, False, False],
        [False, False, True],
        [False, True, False],
        [False, True, True],
        # [True, False, False],
        [True, False, True],
        # [True, True, False],
        [True, True, True],
    )
    @unpack
    def test_sswaps_randomized(self, compose, tpow2, ident_term):
        """Randomized test of the preparation of a sparse tensor or sswaps."""
        for _ in range(5):
            num_swaps = np.random.randint(1, 10)
            swaps = self._generate_swaps(num_swaps)

            values = torch.rand(num_swaps)

            num_qubits = np.max(swaps) + 1
            result = sswaps(
                swaps, values, n=num_qubits, compose=compose, tpow2=tpow2, ident_term=ident_term
            )

            if compose:
                result = result.to_dense()
                self._test_dsm(result)
                self._test_dsm_shape(result, num_qubits, tpow2)
            else:
                self._test_dsm_list(result, num_qubits, tpow2, ident_term, "torch_sparse")

    def test_sswaps_exceptions(self):
        """Test of the preparation of a sparse tensor or sswaps when wrong parameters are passed."""
        swaps = [(0, 1), (1, 2)]
        values = torch.zeros(3)
        with self.assertRaises(ValueError):
            sswaps(swaps, values, n=3)

    @idata(
        # tpow2, sublayers, ident_term, mode
        product([True, False], [True, False], [True, False], ["torch_sparse", "np"])
    )
    @unpack
    def test_swaps_layer_randomized(self, tpow2, sublayers, ident_term, mode):
        """Randomized test preparation the doubly stochastic matrices for the smooth swaps."""
        if not sublayers and sublayers == ident_term:
            self.skipTest("Both sublayers and ident_term equal to False are not supported")
        for _ in range(5):
            num_layers = np.random.randint(1, 5)
            topology = []
            num_thetas = 0
            for _ in range(num_layers):
                num_swaps = np.random.randint(1, 5)
                num_thetas += num_swaps
                layer_swaps = self._generate_swaps(num_swaps).tolist()
                topology.append(layer_swaps)

            theta = np.random.random(num_thetas)
            num_qubits = np.max(list(chain(*topology))) + 1

            result = swaps_layer(
                theta,
                topology=topology,
                n=int(num_qubits),
                tpow2=tpow2,
                mode=mode,
                sublayers=sublayers,
                ident_term=ident_term,
            )
            if sublayers:
                self._test_dsm_list(result, num_qubits, tpow2, ident_term, mode)
            else:
                self._test_dsm(result)
                self._test_dsm_shape(result, num_qubits, tpow2)

    def test_swaps_layer_exceptions(self):
        """
        Test preparation the doubly stochastic matrices for the smooth swaps when wrong parameters
        are passed.
        """
        with self.assertRaises(ValueError):
            swaps_layer(np.zeros(2), topology=[(0, 1)], n=2, mode="torch")

        with self.assertRaises(ValueError):
            swaps_layer(np.zeros(2), topology=[(0, 1)], n=0)

    def _test_dsm_list(self, matrices, num_qubits, tpow2, ident_term, mode):
        """Test if a list of tensors contains DSM matrices."""
        # convert a list of sparse tensors to dense and test them individually
        if mode == "torch_sparse":
            matrices = [item.to_dense().numpy() for item in matrices]
        for matrix in matrices:
            if not ident_term:
                # if identity term is not set the matrix is a DSM up to an identity matrix
                matrix += np.eye(len(matrix))
            self._test_dsm(matrix)
            self._test_dsm_shape(matrix, num_qubits, tpow2)

    def _test_dsm_shape(self, dsm, num_qubits, tpow2):
        """Test the shape of the DSM matrix."""
        if tpow2:
            np.testing.assert_array_equal(dsm.shape, [num_qubits**2, num_qubits**2])
        else:
            np.testing.assert_array_equal(dsm.shape, [num_qubits, num_qubits])

    def _generate_swaps(self, num_swaps):
        """Generates random swaps between ``2 * num_swaps`` qubits, all swaps are unique."""
        swaps = np.arange(2 * num_swaps)
        np.random.shuffle(swaps)
        swaps = swaps.reshape(-1, 2)
        return swaps

    def _test_dsm(self, dsm):
        """Test if a matrix is a doubly stochastic matrix"""
        if isinstance(dsm, torch.Tensor):
            dsm = dsm.to_dense().numpy()
        sum0 = np.sum(dsm, 0)
        sum1 = np.sum(dsm, 1)

        np.testing.assert_array_almost_equal(sum0, np.ones(len(sum0)))
        np.testing.assert_array_almost_equal(sum1, np.ones(len(sum1)))

    @data(
        [CouplingMap.from_line(4), [[0, 0, 1, 2], [0, 0, 0, 1], [1, 0, 0, 0], [2, 1, 0, 0]]],
        [CouplingMap.from_ring(4), [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]],
        [
            CouplingMap.from_grid(2, 3),
            [
                [0, 0, 1, 0, 1, 2],
                [0, 0, 0, 1, 0, 1],
                [1, 0, 0, 2, 1, 0],
                [0, 1, 2, 0, 0, 1],
                [1, 0, 1, 0, 0, 0],
                [2, 1, 0, 1, 0, 0],
            ],
        ],
    )
    @unpack
    def test_coupling_map_to_distance_matx(self, cmap, expected):
        """Test computation of distances on coupling maps."""
        distances = coupling_map_to_distance_matx(cmap)
        np.testing.assert_array_equal(distances, expected)

    def test_coupling_map_to_distance_matx_exceptions(self):
        """Test computation of distances on coupling maps when wrong parameters are passed."""
        with self.assertRaises(ValueError):
            coupling_map_to_distance_matx(None)

    @data(
        [CouplingMap.from_line(4), [[(1, 2)], [(0, 1), (2, 3)]]],
        [CouplingMap.from_ring(4), [[(0, 3), (1, 2)], [(2, 3), (1, 0)]]],   # 0, 1
        [
            CouplingMap.from_grid(2, 3),
            [[(1, 4), (0, 3), (2, 5)], [(3, 4), (0, 1)], [(1, 2), (4, 5)]],
        ],
    )
    @unpack
    def test_coupling_map_to_swaps_topo(self, cmap, expected):
        """Test preparation of swap topology."""
        topo = coupling_map_to_swaps_topo(cmap)
        try:
            self.assertListEqual(topo, expected)
        except Exception as e:
            raise e

    def test_coupling_map_to_swaps_topo_exceptions(self):
        """Test preparation of swap topology when wrong parameters are passed."""
        with self.assertRaises(ValueError):
            coupling_map_to_swaps_topo(None)
