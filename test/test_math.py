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

"""Tests for the "math" module functions."""

import json
import numpy as np
from typing import Tuple, List
import torch
from ddt import ddt  # , unpack, data
import dsm_swap.math as ma
from test.dsm_swap_test_case import DSMSwapTestCase


@ddt
class TestMath(DSMSwapTestCase):
    """Tests for the "math" module functions."""

    tol = float(np.sqrt(np.finfo(np.float64).eps))
    num_repeat = 100  # number of repetitions in a test

    def setUp(self) -> None:
        super().setUp()
        np.random.seed(123456)

        with open(self.get_resource_path("test_sswap_dataset.json", "test_data")) as f:
            self.sswap_dataset = json.load(f)

    @staticmethod
    def _rand_sparse(
        *, nrows: int = 0, ncols: int = 0, duplicates: bool = False
    ) -> Tuple[torch.Tensor, Tuple[int, int, np.ndarray, np.ndarray]]:
        """Generates a sparse matrix optionally with duplicates."""
        data_type = torch.complex64
        nrows = np.random.randint(15, 50) if nrows <= 0 else nrows
        ncols = np.random.randint(15, 50) if ncols <= 0 else ncols
        nnz = min(nrows, ncols, (nrows * ncols) // 10)
        idx = np.vstack(
            [  # no duplications:
                np.random.permutation(nrows)[:nnz],
                np.random.permutation(ncols)[:nnz],
            ]
        )
        val = np.random.rand(nnz)

        if duplicates:
            idx = np.repeat(idx, 2, axis=1)
            val = 0.5 * np.repeat(val, 2)  # twice many, twice smaller

        # Generate a dense target vector (to compare against).
        sp_mat = torch.sparse_coo_tensor(
            indices=idx, values=val, size=(nrows, ncols), dtype=data_type
        )
        return sp_mat, (nrows, ncols, idx, val)

    @staticmethod
    def _rand_perm(dim: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generates permutation matrix from randomly permuted identity one."""
        # return np.take(np.eye(dim, dtype=int), np.random.permutation(dim), axis=1)
        perm = np.random.permutation(dim)
        pmat = np.zeros((dim, dim), dtype=int)
        pmat[perm, np.arange(dim)] = 1
        return pmat, perm

    @staticmethod
    def _sp2dense(a: torch.Tensor) -> np.ndarray:
        """
        Converts torch tensor to a dense numpy array.
        # https://stackoverflow.com/questions/49768306/pytorch-tensor-to-numpy-array
        """
        assert isinstance(a, torch.Tensor)
        return ma.torch_detach_np(a, dense=True)

    def test_torch_sparse_mat_tp(self):
        """Tests math.torch_sparse_mat_tp() function."""
        t = torch.zeros(1)
        self.assertTrue(torch.eq(ma.torch_sparse_mat_tp(t, None), t))
        self.assertTrue(torch.eq(ma.torch_sparse_mat_tp(None, t), t))
        self.assertTrue(ma.torch_sparse_mat_tp(None, None) is None)

        for _rep_no in range(self.num_repeat):
            a_nr, a_nc = np.random.randint(15, 50), np.random.randint(15, 50)
            b_nr, b_nc = a_nc, np.random.randint(15, 50)
            a, _ = self._rand_sparse(nrows=a_nr, ncols=a_nc, duplicates=True)
            b, _ = self._rand_sparse(nrows=b_nr, ncols=b_nc, duplicates=True)
            ab = ma.torch_sparse_mat_tp(a, b)  # , coalesce=True)  # always =True !!! FIXME
            self.assertTrue(
                np.allclose(
                    np.kron(self._sp2dense(a), self._sp2dense(b)),
                    self._sp2dense(ab),
                    atol=self.tol,
                    rtol=self.tol,
                )
            )

    def test_sparse_eye(self):
        """Tests math.sparse_eye() function."""
        for data_type in [torch.int, torch.float, torch.complex64, None]:
            for n in range(1, 20):
                th_eye = ma.sparse_eye(n, dtype=data_type)
                np_eye = np.eye(n)
                self.assertTrue(
                    np.allclose(np_eye, self._sp2dense(th_eye), atol=self.tol, rtol=self.tol)
                )

    def test_coo_mat_vectorize(self):
        """Tests math.coo_mat_vectorize() function."""
        data_type = torch.complex64
        for _rep_no in range(self.num_repeat):
            target, (nrows, ncols, idx, val) = self._rand_sparse(duplicates=False)
            target = self._sp2dense(target).reshape(-1, 1)

            # Test target against different output representations.
            vec1 = ma.coo_mat_vectorize(
                idx,
                val,
                (nrows, ncols),
                dtype=data_type,
                storage=ma.STORAGE_DENSE,
            )
            self.assertTrue(vec1.shape == target.shape)
            self.assertTrue(np.allclose(vec1, target, atol=self.tol, rtol=self.tol))

            vec2 = self._sp2dense(
                ma.coo_mat_vectorize(
                    idx,
                    val,
                    (nrows, ncols),
                    dtype=data_type,
                    storage=ma.STORAGE_SPARSE,
                )
            )
            self.assertTrue(vec2.shape == target.shape)
            self.assertTrue(np.allclose(vec2, target, atol=self.tol, rtol=self.tol))

            idx2, val2, size2 = ma.coo_mat_vectorize(
                idx,
                val,
                (nrows, ncols),
                dtype=data_type,
                storage=ma.STORAGE_SPARSE_SPEC,
            )
            vec3 = self._sp2dense(
                torch.sparse_coo_tensor(indices=idx2, values=val2, size=size2, dtype=data_type)
            )
            self.assertTrue(vec3.shape == target.shape)
            self.assertTrue(np.allclose(vec3, target, atol=self.tol, rtol=self.tol))

            # Expected failure: coo_mat_vectorize() does not handle duplicates.
            target, (nrows, ncols, idx, val) = self._rand_sparse(duplicates=True)
            target = self._sp2dense(target).reshape(-1, 1)
            vec4 = ma.coo_mat_vectorize(
                idx,
                val,
                (nrows, ncols),
                dtype=data_type,
                storage=ma.STORAGE_DENSE,
            )
            self.assertTrue(vec4.shape == target.shape)
            self.assertFalse(np.allclose(vec4, target, atol=self.tol, rtol=self.tol))

    def test_symm_sawtooth(self):
        """Tests math.symm_sawtooth() function."""
        x = np.linspace(0, 1)
        for k in range(-4, 5):
            self.assertTrue(np.allclose(ma.symm_sawtooth(x + 2 * k) - x, 0))
            self.assertTrue(np.allclose(ma.symm_sawtooth(x + (2 * k + 1)) - x[::-1], 0))

    def test_onelinesym_to_perm_mat(self):
        """Tests math.onelinesym_to_perm_mat() function."""
        for _rep_no in range(self.num_repeat):
            dim = np.random.randint(15, 50)
            perm = np.random.permutation(dim)
            pmat = np.zeros((dim, dim), dtype=int)
            pmat[perm, np.arange(dim)] = 1
            self.assertTrue(np.all(pmat == ma.onelinesym_to_perm_mat(perm)))

    def test_perm_mat_to_onelinesym(self):
        """Tests math.perm_mat_to_onelinesym() function."""
        for _rep_no in range(self.num_repeat):
            for mode in ["v", "dict"]:
                dim = np.random.randint(15, 50)
                pmat, perm = self._rand_perm(dim)
                if mode == "v":
                    self.assertTrue(np.all(perm == ma.perm_mat_to_onelinesym(pmat)))
                else:
                    d = ma.perm_mat_to_onelinesym(pmat, mode=mode)
                    self.assertTrue(isinstance(d, dict))
                    self.assertTrue(np.all(np.arange(dim) == list(d.keys())))
                    self.assertTrue(np.all(perm == list(d.values())))

    def test_adj_mat_to_coo(self):
        """Tests math.adj_mat_to_coo() function."""
        for _rep_no in range(self.num_repeat):
            sp, _ = self._rand_sparse(duplicates=True)
            sp = sp.coalesce()
            idx1 = np.asarray(sp.indices())
            idx2 = ma.adj_mat_to_coo(self._sp2dense(sp))
            self.assertTrue(np.all(idx1 == idx2))

    def test_apply_perm_to_coo(self):
        """Tests math.apply_perm_to_coo() function."""
        for _rep_no in range(self.num_repeat):
            dim = np.random.randint(15, 50)
            sp = self._sp2dense(self._rand_sparse(nrows=dim, ncols=dim, duplicates=True)[0])
            idx = np.vstack(np.nonzero(sp))
            self.assertTrue(idx.ndim == 2 and idx.shape[0] == 2)
            pmat, _ = self._rand_perm(dim)
            sp_perm = pmat @ sp @ pmat.T
            idx_perm = ma.apply_perm_to_coo(pmat, idx)
            vals1 = sp[idx[0], idx[1]]
            vals2 = sp_perm[idx_perm[0], idx_perm[1]]
            self.assertTrue(np.all(np.equal(vals1, vals2)))

    def test_reduce_np_matmul(self):
        """Tests math.reduce_np_matmul() function."""
        mat_train_length = 7
        for _rep_no in range(self.num_repeat):
            for transp in [False, True]:
                # As a single matrix.
                mats = np.random.rand(np.random.randint(3, 9), np.random.randint(3, 9))
                res = ma.reduce_np_matmul(mat_or_mats=mats, transp=transp)
                self.assertTrue(np.all(np.equal(mats, res.T if transp else res)))

                # As a list of matrices.
                prev_dim = np.random.randint(15, 50)
                mats = list()
                reduced = None
                for _mat_no in range(mat_train_length):
                    dim = np.random.randint(15, 50)
                    if transp:
                        mats.append(np.random.rand(dim, prev_dim))
                        reduced = mats[-1].T.copy() if reduced is None else reduced @ mats[-1].T
                    else:
                        mats.append(np.random.rand(prev_dim, dim))
                        reduced = mats[-1].copy() if reduced is None else reduced @ mats[-1]
                    prev_dim = dim
                    reduced2 = ma.reduce_np_matmul(mats, transp=transp)
                    self.assertTrue(np.allclose(reduced, reduced2, atol=self.tol, rtol=self.tol))

                # As a stack of matrices.
                dim = np.random.randint(15, 50)
                mats = np.zeros((mat_train_length, dim, dim))
                reduced = None
                for _mat_no in range(mat_train_length):
                    new_mat = np.random.rand(dim, dim)
                    mats[_mat_no, :, :] = new_mat
                    if transp:
                        reduced = new_mat.T.copy() if reduced is None else reduced @ new_mat.T
                    else:
                        reduced = new_mat.copy() if reduced is None else reduced @ new_mat
                    reduced2 = ma.reduce_np_matmul(mats[: _mat_no + 1], transp=transp)
                    self.assertTrue(np.allclose(reduced, reduced2, atol=self.tol, rtol=self.tol))

    def test_swap_mat_decompose(self):
        """Tests math.swap_mat_decompose() function."""

        def _sort_tuple_list(_x: List[Tuple]) -> np.ndarray:
            """Sorts a list of index pairs for comparison."""
            _x = np.sort(np.asarray(_x), axis=1)  # sort each index pair
            return _x[np.argsort(_x[:, 0])]  # sort by the first index in pair

        for _rep_no in range(self.num_repeat):
            dim = np.random.randint(15, 50)
            # Generate a random swap matrix.
            perm = np.random.permutation(dim)
            idx1 = perm[0 : dim // 4]
            idx2 = perm[idx1.size : 2 * idx1.size]
            perm = np.arange(dim)
            perm[idx1] = idx2
            perm[idx2] = idx1
            swap_mat = np.take(np.eye(dim, dtype=int), perm, axis=1)
            self.assertTrue(np.all(np.equal(np.sum(swap_mat, axis=0), np.ones(dim))))
            self.assertTrue(np.all(np.equal(np.sum(swap_mat, axis=1), np.ones(dim))))
            self.assertTrue(np.all(np.equal(swap_mat @ swap_mat, np.eye(dim, dtype=int))))
            # Get the lists of swapped entries.
            list1 = _sort_tuple_list([(a, b) for a, b in zip(idx1, idx2)])
            list2 = _sort_tuple_list(ma.swap_mat_decompose(swap_mat))
            self.assertTrue(np.all(np.equal(list1, list2)))

    def test_reduce_level_np_matmul(self):
        """Tests math.reduce_level_np_matmul() function."""
        for level in [-1, 2, 3]:
            with self.assertRaises(ValueError):
                ma.reduce_level_np_matmul([np.empty(0)], level=level)

        dim = np.random.randint(15, 50)
        mat1 = np.random.rand(dim, dim)
        mat2 = np.random.rand(dim, dim)
        mat3 = np.random.rand(dim, dim)
        res = ma.reduce_level_np_matmul([mat1, mat2, mat3], level=0)
        self.assertTrue(np.allclose(mat1 @ mat2 @ mat3, res, atol=self.tol, rtol=self.tol))

    def test_torch_detach_np(self):
        """Tests math.torch_detach_np() function."""
        self.assertTrue(ma.torch_detach_np(None) is None)
        self.assertTrue(ma.torch_detach_np(torch.zeros(7)).shape == (7,))
