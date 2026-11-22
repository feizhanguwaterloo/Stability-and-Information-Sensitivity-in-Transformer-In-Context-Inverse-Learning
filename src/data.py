import numpy as np

from .dispatch import CAPACITY, INTERIOR, Observations, solve_dispatch

LOW = np.array([1.0, 0.8, 1.0, 0.8])
HIGH = np.array([1.65, 2.0, 1.65, 2.0])
PRICE_SCALE = 3.65


def split_tasks(seed, counts):
    """Draw distinct tasks from separate random streams."""
    streams = np.random.SeedSequence(seed).spawn(len(counts))
    return tuple(np.random.default_rng(stream).uniform(LOW, HIGH, size=(count, 4))
                 for stream, count in zip(streams, counts))


def sample_context(theta, size, rng, regime="uniform"):
    if regime == "uniform":
        return solve_dispatch(theta, rng.uniform(0, CAPACITY.sum(), size))

    a, b = np.asarray(theta).reshape(2, 2).T
    knots = np.sort(np.r_[a, a + b * CAPACITY])
    # Convert marginal-price breakpoints to demand breakpoints.
    ends = np.clip((knots[:, None] - a) / b, 0, CAPACITY).sum(axis=1)
    starts, stops = ends[:-1], ends[1:]
    valid = stops - starts > 1e-10
    midpoints = (starts[valid] + stops[valid]) / 2
    intervals = np.column_stack((starts[valid], stops[valid]))
    statuses = solve_dispatch(theta, midpoints).status
    both_interior = np.all(statuses == INTERIOR, axis=1)
    interior = intervals[both_interior]
    boundary = intervals[~both_interior]

    if regime == "interior-rich":
        demands = rng.uniform(*interior[0], size=size)
    elif regime == "boundary-heavy":
        # Keep the same generator at the same bound throughout this context.
        interval = boundary[rng.integers(len(boundary))]
        demands = rng.uniform(*interval, size=size)
    elif regime == "mixed":
        n_interior = size // 2
        widths = np.diff(boundary, axis=1).ravel()
        chosen = boundary[rng.choice(len(boundary), size - n_interior, p=widths / widths.sum())]
        demands = np.r_[rng.uniform(*interior[0], size=n_interior),
                        rng.uniform(chosen[:, 0], chosen[:, 1])]
        rng.shuffle(demands)
    else:
        raise ValueError(f"unknown observation regime: {regime}")
    return solve_dispatch(theta, demands)


def observation_features(obs: Observations):
    """Encode only market observations, using fixed physical scales."""
    statuses = np.eye(3)[obs.status].reshape(-1, 6)
    return np.column_stack((obs.demand / CAPACITY.sum(), obs.dispatch / CAPACITY,
                            obs.price / PRICE_SCALE, statuses)).astype(np.float32)
