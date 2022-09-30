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

from .math import perm_mat_to_onelinesym
from .math import reduce_level_np_matmul
from .errors import invalid_arg
import numpy as np
from matplotlib import patches
from matplotlib.path import Path


def np_cumulative_mm(mats):
    mats = np.asarray(mats)
    n = len(mats)
    ret = [mats[0]] * n
    for k1, k2 in (np.stack(np.triu_indices(n - 1)) + 1).T:
        ret[k2] = mats[k1] @ ret[k2]
    return np.array(ret)


def wire_paths(m):
    """
    Given a matrix representing the position of wire j at time step i,
    prepare the drawing paths.
    """
    m = np.asarray(m)
    if m.ndim != 2 or np.any(np.asarray(m.shape) == 0):
        raise ValueError(invalid_arg("m", expected="nonempty matrix (as Numpy array)", actual=m))

    m = np.vstack([np.arange(m.shape[1]), m])
    hs, vs = 1.0 / (np.array(m.shape) - 1)
    paths, grid = [], []
    for j in range(m.shape[1]):
        path = m[:, j]
        path = np.stack([np.arange(len(path)), path])
        path = path * np.vstack([hs, vs])
        path1 = np.empty((2, 3 * path.shape[1] - 2), dtype=path.dtype)
        p = path + np.vstack([hs / 2, 0])
        path1[:, 1::3] = p[:, :-1]
        path1[:, 0::3] = path
        p = path - np.vstack([hs / 2, 0])
        path1[:, 2::3] = p[:, 1:]

        codes = [Path.MOVETO] + [Path.CURVE4] * (path1.shape[1] - 1)
        paths.append(Path(path1.T, codes))
        grid.append(path[:, 1:])

    grid = np.stack(grid)
    grid = np.moveaxis(grid, 2, 0)
    grid = np.swapaxes(grid, 1, 2)
    # Grid shape: (time, coo xy, value)
    return paths, grid


def plot_circ(perms, adjs_coo, ax):
    """Plot a braid circuit for the circuit with swaps.

    Args:
        perms: A list of permutations for each layer.
        adjs_coo: A list of adjacency matrices in coordinate form. Each
        represents a circuit layer.
        ax: The Matplotlib axes.
    """
    perms = reduce_level_np_matmul(perms, 1)

    ax.patch.set_linewidth("1")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    # Accumulate the permutations
    wires = np_cumulative_mm(perms)
    wires = map(perm_mat_to_onelinesym, wires)
    wires = np.array(list(wires))
    if len(wires[0]) < 2:
        # At least two wires required
        return

    wires, grid = wire_paths(wires)
    for path in wires:
        path = patches.PathPatch(path, facecolor="none", lw=2, edgecolor="black")
        ax.add_patch(path)

    if adjs_coo is None:
        return
    path = []
    for adj, layer_coo in zip(adjs_coo, grid):
        adj = adj.T
        adj = adj[adj[:, 0] < adj[:, 1]]
        for i1, i2 in adj:
            i1 = layer_coo[0][i1], layer_coo[1][i1]
            i2 = layer_coo[0][i2], layer_coo[1][i2]
            path.append(i1)
            path.append(i2)
    path1 = Path(path, [Path.MOVETO, Path.LINETO] * (len(path) // 2))
    path1 = patches.PathPatch(path1, facecolor="none", lw=2, edgecolor="r", alpha=0.75)
    ax.add_patch(path1)

    bullets = np.array(path).T
    ax.scatter(bullets[0], bullets[1], color="r", s=64)

    props = dict(boxstyle="round", facecolor="white")
    ys = np.sort(grid[0][1])
    for idx, y in enumerate(ys):
        ax.text(0.0, y, f"$q_{{{idx}}}$", bbox=props, verticalalignment="center", fontsize=12)
