"""Two-dimensional KKT polytopes for the noiseless dispatch study."""

from itertools import combinations

import numpy as np

from .data import HIGH, LOW
from .dispatch import INTERIOR, LOWER, UPPER


def generator_vertices(x, status, price, low, high):
    """Return vertices of one unit's boxed set of exact-KKT parameters."""
    matrix = np.vstack((np.eye(2), -np.eye(2)))
    rhs = np.r_[high, -low]
    lower = status == LOWER
    upper = status == UPPER
    if lower.any():
        matrix = np.vstack((matrix, [-1, 0]))
        rhs = np.r_[rhs, -price[lower].max()]
    if upper.any():
        matrix = np.vstack((matrix, [1, x[upper][0]]))
        rhs = np.r_[rhs, price[upper].min()]
    interior = status == INTERIOR
    design = np.column_stack((np.ones(interior.sum()), x[interior]))
    if interior.any():
        base, _, rank, _ = np.linalg.lstsq(design, price[interior], rcond=1e-12)
        if not np.allclose(design @ base, price[interior], rtol=0, atol=1e-9):
            raise ValueError("inconsistent interior records")
        if rank == 2:
            if np.any(matrix @ base > rhs + 1e-9):
                raise ValueError("interior solution violates a KKT inequality")
            return base[None, :]
        direction = np.array([-x[interior][0], 1.0])
        slopes = matrix @ direction
        slack = rhs - matrix @ base
        positive, negative = slopes > 1e-12, slopes < -1e-12
        start, stop = np.max(slack[negative] / slopes[negative]), np.min(
            slack[positive] / slopes[positive])
        if start > stop + 1e-9 or np.any(slack[~(positive | negative)] < -1e-9):
            raise ValueError("empty KKT line segment")
        return base + np.array([start, stop])[:, None] * direction

    vertices = []
    for i, j in combinations(range(len(rhs)), 2):
        indices = [i, j]
        pair = matrix[indices]
        if abs(np.linalg.det(pair)) > 1e-12:
            point = np.linalg.solve(pair, rhs[indices])
            if np.all(matrix @ point <= rhs + 1e-9):
                vertices.append(point)
    if not vertices:
        raise ValueError("empty KKT polytope")
    return np.asarray(vertices)


def exact_kkt_geometry(obs):
    return tuple(generator_vertices(obs.dispatch[:, i], obs.status[:, i], obs.price,
                                    LOW[2*i:2*i+2], HIGH[2*i:2*i+2]) for i in range(2))


def admissible_diameter(vertices):
    return np.sqrt(sum(np.square(points[:, None] - points[None, :]).sum(axis=-1).max()
                       for points in vertices))


def exact_kkt_project(anchor, obs, vertices):
    """Project onto the product of two convex polytopes."""
    from .kkt import residual

    anchor = np.asarray(anchor, dtype=float).reshape(2, 2)
    errors = residual(anchor, obs)
    estimate = []
    for i, (point, polygon) in enumerate(zip(anchor, vertices)):
        if (np.max(np.abs(errors[:, i])) <= 1e-12
                and np.all(point >= LOW[2*i:2*i+2]) and np.all(point <= HIGH[2*i:2*i+2])):
            estimate.append(point)
            continue
        candidates = list(polygon)
        # Edges are among these chords; a convex polygon contains every chord.
        for left, right in combinations(polygon, 2):
            edge = right - left
            squared = edge @ edge
            if squared > 0:
                candidates.append(left + np.clip((point - left) @ edge / squared, 0, 1) * edge)
        candidates = np.asarray(candidates)
        estimate.append(candidates[np.argmin(np.square(candidates - point).sum(axis=1))])
    return np.asarray(estimate).ravel()
