import unittest

import numpy as np
from scipy.optimize import minimize

from src.data import HIGH, LOW, sample_context, split_tasks
from src.dispatch import INTERIOR, LOWER, UPPER, Observations, solve_dispatch
from src.inverse import interior_ridge, kkt_correct
from src.kkt import loss_and_gradient, residual


class DispatchTests(unittest.TestCase):
    def test_analytic_against_numerical_program(self):
        rng = np.random.default_rng(75)
        for theta in split_tasks(97, [12])[0]:
            a, b = theta.reshape(2, 2).T
            for d in np.r_[0, rng.uniform(0, 2, 8), 2]:
                obs = solve_dispatch(theta, d)
                result = minimize(
                    lambda x: a @ x + 0.5 * b @ x**2,
                    np.full(2, d / 2), jac=lambda x: a + b * x,
                    bounds=[(0, 1)] * 2,
                    constraints={"type": "eq", "fun": lambda x: x.sum() - d,
                                 "jac": lambda x: np.ones(2)},
                    method="SLSQP", options={"ftol": 1e-12},
                )
                self.assertTrue(result.success, result.message)
                np.testing.assert_allclose(obs.dispatch, result.x, atol=2e-7, rtol=0)
                np.testing.assert_allclose(residual(theta, obs), 0, atol=1e-12)

    def test_degenerate_price_interval(self):
        theta = [1, 1, 5, 1]
        obs = solve_dispatch(theta, 1)
        np.testing.assert_array_equal(obs.status, [UPPER, LOWER])
        self.assertEqual(obs.price, 3.5)
        np.testing.assert_allclose(residual(theta, obs), 0)

    def test_context_regimes_and_disjoint_tasks(self):
        train, valid, test = split_tasks(97, [16, 6, 8])
        self.assertEqual(len({tuple(row) for row in np.vstack((train, valid, test))}), 30)
        rng = np.random.default_rng(9)
        for theta in test:
            for regime, expected in [("interior-rich", 16), ("mixed", 8), ("boundary-heavy", 0)]:
                obs = sample_context(theta, 16, rng, regime)
                self.assertEqual(np.all(obs.status == INTERIOR, axis=1).sum(), expected)
                np.testing.assert_allclose(residual(theta, obs), 0, atol=1e-12)


class InverseTests(unittest.TestCase):
    def test_bound_signs_and_gradient(self):
        theta = np.array([1.2, 1.1, 1.6, 1.9])
        obs = Observations(np.array([0.4, 1.6]), np.array([[0, 0.4], [1, 0.6]]),
                           np.array([1.5, 2.0]), np.array([[LOWER, INTERIOR], [UPPER, INTERIOR]]))
        np.testing.assert_allclose(residual(theta, obs)[:, 0], [-0.3, 0.3])
        perturb = 1e-6 * np.eye(4)
        numeric = [(loss_and_gradient(theta + e, obs)[0] -
                    loss_and_gradient(theta - e, obs)[0]) / 2e-6 for e in perturb]
        np.testing.assert_allclose(loss_and_gradient(theta, obs)[1], numeric, atol=1e-8)

    def test_correction_interior_exact_formula(self):
        theta = np.array([1.2, 1.1, 1.6, 1.9])
        anchor = (LOW + HIGH) / 2
        obs = sample_context(theta, 32, np.random.default_rng(5), "interior-rich")
        gamma = 30
        fitted = kkt_correct(anchor, obs, gamma)
        expected = []
        for i, x in enumerate(obs.dispatch.T):
            design = np.column_stack((np.ones(len(x)), x))
            j = slice(2*i, 2*i+2)
            expected.extend(theta[j] + np.linalg.solve(
                np.eye(2) + gamma * design.T @ design / len(x), anchor[j] - theta[j]))
        np.testing.assert_allclose(fitted, expected, atol=1e-7)
        np.testing.assert_allclose(interior_ridge(anchor, obs, gamma), expected, atol=1e-7)
        np.testing.assert_allclose(kkt_correct(theta, obs, gamma), theta, atol=1e-10)

    def test_mixed_correction_decreases_loss_and_respects_box(self):
        theta = np.array([1.2, 1.1, 1.6, 1.9])
        anchor = np.array([1.4, 1.9, 1.05, 0.85])
        obs = sample_context(theta, 32, np.random.default_rng(5), "mixed")
        fitted = kkt_correct(anchor, obs, 20)
        self.assertTrue(np.all((LOW <= fitted) & (fitted <= HIGH)))
        self.assertLess(loss_and_gradient(fitted, obs)[0], loss_and_gradient(anchor, obs)[0])
        np.testing.assert_allclose(kkt_correct(anchor, obs, 0), anchor)


if __name__ == "__main__":
    unittest.main()
