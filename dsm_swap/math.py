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

from .errors import invalid_arg
from functools import reduce
from typing import Union, List
import numpy as np
import torch


# Storage classes for matrices.
STORAGE_DENSE = "dense"
STORAGE_SPARSE = "sparse"
STORAGE_SPARSE_SPEC = "sparse-spec"


def _coo_tp(a, b, shape_b) -> np.ndarray:
    """
    Helper function for the creation of the sparse coordinates
    for the sparse tensor product.
    """
    a, b = map(np.asarray, (a, b))
    shape_b = np.atleast_1d(shape_b).flatten()
    shape_b = np.expand_dims(shape_b, -1)
    a = np.repeat(a, b.shape[1], axis=1) * shape_b
    b = np.tile(b, a.shape[1] // b.shape[1])
    return a + b


def _val_repeat_mul(a, b):
    """
    Helper function for the creation of sparse tensor
    products.
    """
    a = a.repeat_interleave(len(b))
    b = b.repeat(len(a) // len(b))
    return a * b


def torch_sparse_mat_tp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Given sparse matrices (PyTorch) A, B obtain the tensor product A (x) B.

    Args:
        a: A Torch sparse tensor (2D).
        b: A Torch sparse tensor (2D).
        coalesce: When True the method coalesce() is invoked on the
        input tensors prior to the computation.

    Returns: A sparse tensor (PyTorch) containing the tensor product of
    the inputs.

    """
    if a is None or b is None:
        return b if a is None else a

    a, b = a.coalesce(), b.coalesce()
    a_idxs, b_idxs = a.indices(), b.indices()
    a_idxs, b_idxs = map(np.asarray, (a_idxs, b_idxs))
    idxs = _coo_tp(a_idxs, b_idxs, shape_b=b.shape)
    val = _val_repeat_mul(a.values(), b.values())
    shape = np.atleast_1d(a.shape) * np.atleast_1d(b.shape)
    return torch.sparse_coo_tensor(idxs, val, size=tuple(shape), dtype=a.dtype)


def sparse_eye(n, dtype=None) -> torch.Tensor:
    """
    Create a sparse identity matrix of size n x n.

    Args:
        n: The number of rows/cols or the resulting matrix.

    Returns: A sparse tensor (PyTorch) containing an identity matrix
    of the requested size.

    """
    idxs = np.repeat([np.arange(n)], 2, axis=0)
    return torch.sparse_coo_tensor(idxs, np.ones(idxs.shape[1]), dtype=dtype)


def coo_mat_vectorize(c, v, shape, *, dtype=None, storage=STORAGE_SPARSE_SPEC):
    """
    Produce a sparse vector corresponding to the vectorization (row major)
    of the given matrix in coordinates form.
    The vectorization is based on the rows, that is rows of the
    input matrix are stacked to form a vector.
    The first three parameter have exactly the same semantic as the
    parameters of `torch.sparse_coo_tensor`.

    Args:
        c: The array of coordinates (as Numpy array).
        v: The array of values corresponding to each coordinate.
        shape: The shape of the input tensor (as it would be in dense form).
        dtype: The optional data type for the resulting (PyTorch) tensor.
        storage: The output storage format spec, this case be either 'dense', 'sparse'
        or 'sparse-spec'. The 'sparse-spec' option, produces a tuple of three objects that
        can be passed directly as parameters for the function `torch.sparse_coo_tensor`.

    Returns: A tensor representing the vectorized form of the input, the format
    depends on the `storage` parameter.
    """
    assert len(shape) == 2
    c = np.asarray(c)
    c = np.stack([c[0] * shape[1] + c[1], np.repeat(0, c.shape[1])])
    shape = (np.product(shape), 1)

    if storage == STORAGE_SPARSE_SPEC:
        return c, v, shape

    if storage == STORAGE_SPARSE:
        return torch.sparse_coo_tensor(c, v, shape, dtype=dtype)

    assert storage == STORAGE_DENSE
    # The code below does not admit duplicate entries in the index array "c"
    # for the best performance. Duplication is actually not happening for now,
    # just beware of future development.
    m = torch.zeros(shape, dtype=dtype)
    m[c] = torch.as_tensor(v, dtype=dtype)
    return m
    # Alternative solution handles duplicates correctly, but it is slow:
    # return torch.sparse_coo_tensor(c, v, shape, dtype=dtype).to_dense()


def symm_sawtooth(x):
    """
    A symmetric (tooth) sawtooth function.

    Args:
        x: The value(s) (possibly as Numpy array) at which the function
        is evaluated.

    Returns: The value(s) of the function.

    """
    x = np.abs(x)
    r = np.modf(x)
    r = r[0], r[1] % 2
    return np.abs(r[0] - r[1])


def onelinesym_to_perm_mat(p) -> np.ndarray:
    """
    Obtain the inverse effect to the function perm_mat_to_onelinesym.

    Args:
        p: The input one line symbol vector.

    Returns: The permutation matrix (Numpy) corresponding to the
    given one line symbol.

    """
    p = np.asarray(p, dtype=int)
    assert p.ndim == 1
    return np.eye(np.max(p) + 1)[p].T


def perm_mat_to_onelinesym(m: np.ndarray, *, mode="v") -> Union[dict, np.ndarray]:
    """
    Given a permutation matrix obtain the corresponding
    one line symbol. Note that the inverse of such operation
    is 'np.eye(n)[p].T' (see onelinesym_to_perm_mat).

    Args:
        m: The input permutation matrix.
        mode: Either 'v' for vector form or 'dict' for the dictionary
        form of the output.

    Returns: The one line symbol in dictionary or array form.

    """
    m = np.asarray(m, dtype=int).T
    c = m.nonzero()
    assert np.all(c[0] == np.arange(len(m)))
    if mode == "dict":
        return {int(v[0]): int(v[1]) for v in zip(*c)}
    return c[1]


def adj_mat_to_coo(m: np.ndarray) -> np.ndarray:
    """
    Transform an adjacency matrix to coordinates form.

    Args:
        m: The adjacency matrix (as Numpy array).

    Returns: A matrix of non-zero coordinates (as Numpy array).

    """
    m = np.asarray(m)
    assert m.ndim == 2
    m = np.nonzero(m)
    return np.stack(m).reshape((2, -1))


def apply_perm_to_coo(p: np.ndarray, coo: np.ndarray) -> np.ndarray:
    """
    Given a matrix of coordinates applies the given permutation P
    in a way that corresponds to the action PAP^T.

    Args:
        p: The permutation matrix (as Numpy array).
        coo: The array of coordinates to be transformed.

    Returns: An array of coordinates having the same shape as
    the parameter `coo`.

    """
    p = perm_mat_to_onelinesym(p)
    coo = np.asarray(coo, dtype=int)
    assert coo.ndim == 2 and coo.shape[0] == 2
    return np.array([np.take(p, coo[k]) for k in range(2)])


def reduce_np_matmul(mat_or_mats, *, transp=False) -> np.ndarray:
    """
    Reduce for matmul for Numpy matrices (as arrays).

    Args:
        mat_or_mats: A matrix (as Numpy array) or a list of matrices.
        transp: Compose the transpose of the input matrices, note the order
        is not changed.

    Returns: A matrix (as Numpy array) corresponding to the composition.

    """
    mats = mat_or_mats
    if isinstance(mats, np.ndarray):
        if mats.ndim == 2:
            return mats.T if transp else mats
        assert mats.ndim == 3
    if transp:
        return reduce(lambda a, b: np.matmul(a, b.T), mats[1:], mats[0].T)
    return reduce(np.matmul, mats[1:], mats[0])


def swap_mat_decompose(swap: np.ndarray) -> List[tuple]:
    """
    Decompose a permutation matrix P containing swaps (so P^2=I)
    into a list of swapped elements. Note that the condition P^2=I
    implies that the swaps are disjoint.
    In case of identity an empty list is returned.

    Args:
        swap: A permutation matrix as a numpy array.

    Returns: A list of tuples, each specifying the indices of the two
    elements swapped by the permutation.

    """
    swap = np.asarray(swap)
    swap = np.nonzero(swap * (1 - np.eye(len(swap))))
    swap = np.sort(np.stack(swap).T, axis=1)
    swap = set(tuple(v) for v in swap)
    return list(swap)


def reduce_level_np_matmul(mats, level=0, *, transp=False):
    """
    Multi-level reduce for matmul for Numpy matrices (as arrays).

    Args:
        mats: A list or list of lists of matrices (as Numpy array).
        level: The level of composition. When level is 0, return a
        single matrix corresponding to the composition of all matrices. When level
        is 1 compose the lists to obtain a list of matrices.
        transp: Compose the transpose of the input matrices, note the order
        is not changed.

    Returns: A matrix (as Numpy array) or a list of matrices.

    """
    if level not in {0, 1}:
        raise ValueError(invalid_arg(arg="level", expected="value from set {0, 1}", actual=level))
    mats = [reduce_np_matmul(mat, transp=transp) for mat in mats]
    if level == 0:
        mats = reduce_np_matmul(mats, transp=transp)
    return mats


def torch_detach_np(tensor: Union[None, torch.Tensor], *, dense=False) -> np.ndarray:
    """
    Detach a (PyTorch) tensor and convert it to numpy.
    Sparse tensors are supported but the storage is dense on the numpy side.

    Args:
        tensor: The input tensor (PyTorch).
        dense: When True the method `to_dense` is invoked on the input
        tensor after detaching. This must be used when the input tensor
        is sparse.

    Returns: A Numpy array corresponding to the input tensor.

    """
    if tensor is None:
        return None
    tensor = tensor.detach() if tensor.requires_grad else tensor
    return tensor.to_dense().numpy() if dense else tensor.numpy()
