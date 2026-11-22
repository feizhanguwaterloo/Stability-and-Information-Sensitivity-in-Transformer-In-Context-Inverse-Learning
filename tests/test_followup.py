import json
import unittest
from pathlib import Path

import numpy as np

from src.kkt import equality_gram, residual
from src.study_followup import identifying_family, identifying_path, noisy_contexts


class FollowupTests(unittest.TestCase):
    def test_pair_endpoints_and_single_price_replacement_path(self):
        config = json.loads(Path("configs/followup.json").read_text(encoding="utf-8"))
        task_a, task_b, demands = identifying_family(config, np.random.default_rng(42))
        np.testing.assert_allclose(np.linalg.norm(task_b - task_a, axis=1), config["target_distance"])
        for count in (1, 4, 8):
            for left, right, demand in zip(task_a, task_b, demands):
                path = identifying_path(left, right, demand, 16, count)
                self.assertEqual(len(path), count + 1)
                for start, end in zip(path, path[1:]):
                    np.testing.assert_array_equal(start.dispatch, end.dispatch)
                    self.assertEqual(np.count_nonzero(start.price - end.price), 1)
                self.assertEqual(np.count_nonzero(path[0].price - path[-1].price), count)
                np.testing.assert_allclose(np.abs(path[0].price[-count:] - path[-1].price[-count:]),
                                           config["price_distance"], atol=1e-12)
                for theta, obs in ((left, path[0]), (right, path[-1])):
                    np.testing.assert_allclose(residual(theta, obs), 0, atol=1e-12)
                    self.assertGreater(np.linalg.eigvalsh(equality_gram(obs)).min(), 0)

    def test_noise_preserves_design(self):
        config = json.loads(Path("configs/followup.json").read_text(encoding="utf-8"))
        left, right, demands = identifying_family(config, np.random.default_rng(62))
        obs, _ = identifying_path(left[0], right[0], demands[0], 16, 1)
        draws = np.linspace(-1, 1, 16)[None, :]
        perturbed, = noisy_contexts([obs], draws, .04)
        np.testing.assert_array_equal(equality_gram(perturbed), equality_gram(obs))
        np.testing.assert_allclose(perturbed.price - obs.price, .04 * draws[0])


if __name__ == "__main__":
    unittest.main()
