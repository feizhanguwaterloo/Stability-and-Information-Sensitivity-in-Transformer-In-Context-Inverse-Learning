import unittest

import numpy as np

from src.data import observation_features
from src.kkt import equality_gram, residual
from src.leverage import crossed_backgrounds, hat_leverage, ridge_estimate, theorem_pair
from src.study import replace


class LeverageTests(unittest.TestCase):
    def test_dispatch_stress_pair_is_valid_and_identifying(self):
        for m in (4, 64):
            theta, (first, second) = theorem_pair(m)
            for a, b in zip(first, second):
                np.testing.assert_allclose(a[:-1], b[:-1], atol=1e-14)
            for target, obs in zip(theta, (first, second)):
                np.testing.assert_allclose(residual(target, obs), 0, atol=1e-14)
                self.assertGreater(np.linalg.eigvalsh(equality_gram(obs)).min(), 0)
                self.assertAlmostEqual(hat_leverage(obs), 1)
                np.testing.assert_allclose(ridge_estimate(obs, target, 0.01), target, atol=1e-12)
            np.testing.assert_allclose(first.price[-1], 2.4, atol=1e-14)
            np.testing.assert_allclose(second.price[-1], 2.64, atol=1e-14)

    def test_crossed_backgrounds_hold_focal_features_fixed(self):
        theta = [1.1, 1.3, 1.6, 1.8]
        designs = list(crossed_backgrounds(theta, 16))
        for index in (0, 2):
            _, _, high, new_high = designs[index]
            _, _, low, new_low = designs[index + 1]
            np.testing.assert_array_equal(observation_features(high)[-1],
                                          observation_features(low)[-1])
            np.testing.assert_array_equal(observation_features(new_high),
                                          observation_features(new_low))
            self.assertAlmostEqual(high.demand[:-1].mean(), low.demand[:-1].mean())
            self.assertGreater(hat_leverage(high), hat_leverage(low))
            np.testing.assert_allclose(residual(theta, replace(high, -1, new_high)), 0, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
