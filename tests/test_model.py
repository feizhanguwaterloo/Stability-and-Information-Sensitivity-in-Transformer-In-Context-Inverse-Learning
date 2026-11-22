import unittest

import numpy as np
import torch

from src.data import observation_features, sample_context
from src.model import SetTransformer


class ModelTests(unittest.TestCase):
    def test_permutation_invariance_and_variable_lengths(self):
        torch.manual_seed(7)
        model = SetTransformer().eval()
        rng = np.random.default_rng(7)
        theta = [1.2, 1.1, 1.6, 1.9]
        for size in (4, 32):
            obs = sample_context(theta, size, rng)
            records = torch.from_numpy(observation_features(obs))
            with torch.inference_mode():
                expected = model(records[None, :, :])
                actual = model(records[rng.permutation(size)][None, :, :])
            self.assertEqual(tuple(actual.shape), (1, 4))
            torch.testing.assert_close(actual, expected, rtol=0, atol=3e-7)


if __name__ == "__main__":
    unittest.main()
