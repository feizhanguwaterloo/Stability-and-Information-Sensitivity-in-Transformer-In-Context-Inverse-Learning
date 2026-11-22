import numpy as np

from .data import HIGH, LOW
from .dispatch import INTERIOR
from .kkt import loss_and_gradient


def kkt_correct(anchor, observations, gamma):
    """Minimize the boxed, strongly convex KKT objective and check stationarity."""
    from scipy.optimize import minimize

    anchor = np.asarray(anchor, dtype=float)
    if anchor.shape != LOW.shape or not np.all(np.isfinite(anchor)):
        raise ValueError("expected a finite four-parameter anchor")
    if not np.isfinite(gamma) or gamma < 0 or len(observations.demand) == 0:
        raise ValueError("gamma must be finite and nonnegative and context must be nonempty")

    def objective(theta):
        loss, gradient = loss_and_gradient(theta, observations)
        delta = theta - anchor
        return 0.5 * (delta @ delta) + gamma * loss, delta + gamma * gradient

    result = minimize(objective, np.clip(anchor, LOW, HIGH), jac=True, method="L-BFGS-B",
                      bounds=list(zip(LOW, HIGH)), options={"ftol": 1e-15, "gtol": 1e-9})
    gradient = objective(result.x)[1]
    projected = result.x - np.clip(result.x - gradient, LOW, HIGH)
    # Line search may stop at floating-point precision despite satisfying stationarity.
    feasible = np.all((LOW <= result.x) & (result.x <= HIGH))
    stationary = np.linalg.norm(projected, ord=np.inf) <= 5e-6
    if not (feasible and stationary):
        raise RuntimeError(f"KKT correction failed: {result.message}, residual {projected}")
    return result.x


def interior_ridge(prior, observations, gamma):
    """Box-constrained prior-regularized fit using interior equalities only."""
    from scipy.optimize import lsq_linear

    result = np.asarray(prior, dtype=float).reshape(2, 2).copy()
    scale = np.sqrt(gamma / len(observations.demand))
    for i in range(2):
        mask = observations.status[:, i] == INTERIOR
        if mask.any():
            x = observations.dispatch[mask, i]
            design = np.vstack((np.eye(2), scale * np.column_stack((np.ones(len(x)), x))))
            target = np.r_[result[i], scale * observations.price[mask]]
            fit = lsq_linear(design, target, bounds=(LOW[2*i:2*i+2], HIGH[2*i:2*i+2]),
                             method="bvls")
            if not fit.success:
                raise RuntimeError(f"interior ridge failed: {fit.message}")
            result[i] = fit.x
    return result.ravel()
