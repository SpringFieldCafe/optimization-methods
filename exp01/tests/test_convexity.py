import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import lab01_convexity as lab


class ConvexityTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.x = rng.normal(size=(13, 2))
        self.y = rng.normal(size=(13, 2))
        self.theta = rng.uniform(size=13)

    def test_batch_shape_and_endpoints(self):
        self.assertEqual(lab.convex_combination(self.x, self.y, self.theta).shape, (13, 2))
        np.testing.assert_array_equal(lab.convex_combination(self.x, self.y, np.zeros(13)), self.y)
        np.testing.assert_array_equal(lab.convex_combination(self.x, self.y, np.ones(13)), self.x)

    def test_quadratic_matches_scalar_polynomial(self):
        expected = np.array([1.5 * a * a + a * b + b * b for a, b in self.x])
        np.testing.assert_allclose(lab._quadratic(self.x), expected)
        self.assertEqual(lab._quadratic(self.x).shape, (13,))

    def test_quadratic_residual_identity(self):
        delta = self.x - self.y
        expected = np.array([-0.5 * t * (1 - t) * (3 * a * a + 2 * a * b + 2 * b * b) for (a, b), t in zip(delta, self.theta)])
        np.testing.assert_allclose(lab.jensen_residual(lab._quadratic, self.x, self.y, self.theta), expected, atol=1e-14)

    def test_explicit_saddle_counterexample(self):
        x, y = np.array([[0.0, 1.0]]), np.array([[0.0, -1.0]])
        np.testing.assert_allclose(lab.jensen_residual(lab._saddle, x, y, np.array([0.5])), [1.0])

    def test_annulus_midpoint_outside(self):
        midpoint = lab.convex_combination(np.array([[0.0, 1.0]]), np.array([[0.0, -1.0]]), np.array([0.5]))
        self.assertLess(np.linalg.norm(midpoint), 0.6)

    def test_semidefinite_flat_direction(self):
        function = lambda x: 0.5 * x[:, 0] ** 2
        x, y = np.array([[0.0, 1.0]]), np.array([[0.0, -1.0]])
        np.testing.assert_array_equal(lab.jensen_residual(function, x, y, np.array([0.5])), [0.0])

    def test_indefinite_negative_curvature(self):
        function = lambda x: 0.5 * (x[:, 0] ** 2 - 0.2 * x[:, 1] ** 2)
        x, y = np.array([[0.0, 1.0]]), np.array([[0.0, -1.0]])
        np.testing.assert_allclose(lab.jensen_residual(function, x, y, np.array([0.5])), [0.1])


if __name__ == "__main__":
    unittest.main()
