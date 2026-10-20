from itertools import combinations
import os
import unittest

# Configure MKL before NumPy loads, matching the solver entry point.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np

from optimize_reference import solve_cardinality


class OptimizerTests(unittest.TestCase):
    def test_bounds_against_small_exhaustive_problem(self):
        rng = np.random.default_rng(10)
        returns = rng.normal(0, .01, size=(100, 8))
        covariance = np.cov(returns, rowvar=False)
        for k in (2, 3):
            optimum = min(
                float(np.sqrt(252 * covariance[np.ix_(subset, subset)].mean()))
                for subset in combinations(range(8), k)
            )
            for method in ("direct", "factorized", "diagonal"):
                with self.subTest(k=k, formulation=method):
                    result = solve_cardinality(
                        covariance, k, list(range(k)), seconds=10, formulation=method
                    )
                    self.assertLessEqual(result["annual_vol_lower_bound"], optimum + 1e-7)
                    self.assertGreaterEqual(result["annual_vol_upper_bound"], optimum - 1e-7)
                    self.assertAlmostEqual(result["annual_vol_upper_bound"], optimum, places=6)
                    self.assertEqual(len(result["selected_indices"]), k)
                    self.assertTrue(result["certified_to_requested_gap"])


if __name__ == "__main__":
    unittest.main()
