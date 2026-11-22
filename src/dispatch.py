from typing import NamedTuple

import numpy as np

LOWER, INTERIOR, UPPER = 0, 1, 2
CAPACITY = np.ones(2)


class Observations(NamedTuple):
    demand: np.ndarray
    dispatch: np.ndarray
    price: np.ndarray
    status: np.ndarray

    def take(self, indices):
        return Observations(*(field[indices] for field in self))


def solve_dispatch(theta, demand, capacity=CAPACITY):
    """Solve the two-unit ED for a scalar or array of demands."""
    cost = np.asarray(theta, dtype=float).reshape(2, 2)
    capacity = np.asarray(capacity, dtype=float)
    demand = np.asarray(demand, dtype=float)
    a, b = cost.T
    if not np.all(np.isfinite(cost)) or np.any(b <= 0):
        raise ValueError("costs must be finite with positive quadratic coefficients")
    if capacity.shape != (2,) or not np.all(np.isfinite(capacity)) or np.any(capacity <= 0):
        raise ValueError("expected two finite positive capacities")
    if not np.all(np.isfinite(demand)) or np.any((demand < 0) | (demand > capacity.sum())):
        raise ValueError("demand is outside the feasible interval")

    # Substitute x2 = d - x1 and minimize the resulting scalar quadratic.
    lo = np.maximum(0, demand - capacity[1])
    hi = np.minimum(capacity[0], demand)
    x1_free = (a[1] - a[0] + b[1] * demand) / b.sum()
    x1 = np.clip(x1_free, lo, hi)
    dispatch = np.stack((x1, demand - x1), axis=-1)
    status = np.where(dispatch <= 1e-12, LOWER,
                      np.where(dispatch >= capacity - 1e-12, UPPER, INTERIOR))
    marginal = a + b * dispatch
    # A feasible price lies between all upper-unit and all lower-unit marginal costs.
    price_lo = np.max(np.where(status == UPPER, marginal, -np.inf), axis=-1)
    price_hi = np.min(np.where(status == LOWER, marginal, np.inf), axis=-1)
    interior = status == INTERIOR
    interior_price = np.sum(np.where(interior, marginal, 0), axis=-1)
    interior_price /= np.maximum(1, interior.sum(axis=-1))
    # Use the midpoint when both price endpoints exist, otherwise the finite endpoint.
    bound_price = np.where(np.isfinite(price_lo) & np.isfinite(price_hi),
                           (np.where(np.isfinite(price_lo), price_lo, 0)
                            + np.where(np.isfinite(price_hi), price_hi, 0)) / 2,
                           np.where(np.isfinite(price_lo), price_lo, price_hi))
    price = np.where(interior.any(axis=-1), interior_price, bound_price)
    return Observations(demand, dispatch, price, status)
