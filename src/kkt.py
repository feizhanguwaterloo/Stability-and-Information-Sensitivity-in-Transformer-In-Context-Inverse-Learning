import numpy as np

from .dispatch import INTERIOR, LOWER, UPPER


def residual(theta, observations):
    """Signed marginal-cost residual, with satisfied inequalities set to zero."""
    a, b = np.asarray(theta).reshape(2, 2).T
    gap = a + b * observations.dispatch - observations.price[..., None]
    return np.where(observations.status == LOWER, np.minimum(gap, 0),
                    np.where(observations.status == UPPER, np.maximum(gap, 0), gap))


def loss_and_gradient(theta, observations):
    """Average over records of the sum of the two per-generator losses."""
    errors = residual(theta, observations)
    gradient = np.stack((errors.mean(axis=0),
                         (errors * observations.dispatch).mean(axis=0)), axis=1)
    return 0.5 * np.square(errors).sum(axis=1).mean(), gradient.ravel()


def equality_gram(observations):
    """Return the per-unit interior Gram matrices, normalized by context size."""
    design = np.stack((np.ones_like(observations.dispatch), observations.dispatch), axis=-1)
    return np.einsum("mgi,mgj,mg->gij", design, design,
                     observations.status == INTERIOR) / len(design)
