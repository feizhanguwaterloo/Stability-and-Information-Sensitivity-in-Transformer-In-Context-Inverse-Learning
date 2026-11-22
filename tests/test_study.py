import unittest

import numpy as np
import torch

from src.data import LOW, HIGH, observation_features, sample_context
from src.dispatch import solve_dispatch
from src.identified import admissible_diameter, exact_kkt_geometry, exact_kkt_project
from src.kkt import residual
from src.model import NormControlledTransformer, attention_operators
from src.study_info import context_design, information


class StudyTests(unittest.TestCase):
    def test_attention_operator_projection(self):
        cap = 0.7
        model = NormControlledTransformer(cap)
        model.project_attention()
        for weight in attention_operators(model):
            self.assertLessEqual(float(torch.linalg.matrix_norm(weight.detach(), ord=2)),
                                 cap + 2e-6)

    def test_exact_kkt_projection_for_point_line_and_polygon(self):
        theta = np.array([1.1, 1.1, 1.6, 1.8])
        prior = (LOW + HIGH) / 2
        interior = sample_context(theta, 16, np.random.default_rng(7), "interior-rich")
        boundary = solve_dispatch(theta, np.linspace(.02, .2, 16))
        full = exact_kkt_geometry(interior)
        self.assertEqual(admissible_diameter(full), 0)
        np.testing.assert_allclose(exact_kkt_project(prior, interior, full), theta, atol=1e-10)
        partial = exact_kkt_geometry(boundary)
        self.assertGreater(admissible_diameter(partial), 0.5)
        partial_fit = exact_kkt_project(prior, boundary, partial)
        np.testing.assert_allclose(partial_fit[:2], theta[:2], atol=1e-10)
        np.testing.assert_allclose(residual(partial_fit, boundary), 0, atol=1e-10)
        one = interior.take(slice(1))
        line_fit = exact_kkt_project(prior, one, exact_kkt_geometry(one))
        np.testing.assert_allclose(residual(line_fit, one), 0, atol=1e-10)

    def test_interventions_match_displacement_and_change_information(self):
        theta = [1.1, 1.1, 1.6, 1.8]
        conditions, variants, refs = context_design(theta, 16, np.random.default_rng(7))
        well, near = conditions["well-conditioned"], conditions["nearly-redundant"]
        self.assertAlmostEqual(well.demand.mean(), near.demand.mean())
        self.assertGreater(information(well), 100 * information(near))
        high, redundant = "high-replacement", "redundant-replacement"
        high_features = observation_features(variants[high]) - observation_features(refs[high])
        low_features = observation_features(variants[redundant]) - observation_features(refs[redundant])
        np.testing.assert_allclose(np.linalg.norm(high_features), np.linalg.norm(low_features),
                                   atol=2e-7)
        self.assertEqual(len(variants[high].demand), 16)
        self.assertEqual(len(variants["drop-high"].demand), 15)
        self.assertLess(information(variants[high]), information(variants[redundant]))


if __name__ == "__main__":
    unittest.main()
