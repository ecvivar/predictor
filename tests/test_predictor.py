import sys, os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.predictor import Predictor
from backend.utils import dixon_coles_adjustment, poisson_distribution, monte_carlo_simulation

class TestForecastEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Initialize predictor once for all tests."""
        cls.predictor = Predictor()

    def test_elo_loader(self):
        """Test that elite teams have high Elo ratings."""
        elo = cls = self.predictor.elo_ratings
        self.assertIn("Argentina", elo)
        self.assertIn("Spain", elo)
        self.assertGreater(elo["Argentina"], 1800, "Argentina Elo should be > 1800")
        self.assertGreater(elo["Spain"], 1800, "Spain Elo should be > 1800")

    def test_rival_tier_classification(self):
        """Test Elo tier boundary classifications."""
        s = self.predictor
        # Via opponent analysis internal thresholds
        matches = s.matches
        # Check tier logic directly by examining stage_4 results
        result = s.predict("Argentina", "Brazil", {"neutral": True})
        s4 = result.get("stage_4", {})
        self.assertIn("team_a", s4)
        tiers = s4["team_a"].get("tiers", {})
        self.assertIn("elite", tiers)
        self.assertIn("strong", tiers)
        self.assertIn("medium", tiers)
        self.assertIn("weak", tiers)

    def test_dixon_coles_matrix_normalized(self):
        """Test that Dixon-Coles matrix sums to 1.0 (probability conservation)."""
        lam_a, lam_b = 1.5, 1.2
        dc = dixon_coles_adjustment(lam_a, lam_b, rho=-0.06, max_goals=6)
        total = sum(dc[i][j] for i in range(6) for j in range(6))
        self.assertAlmostEqual(total, 1.0, places=3,
            msg=f"Dixon-Coles matrix should sum to 1.0, got {total:.6f}")

    def test_monte_carlo_probabilities_sum(self):
        """Test that Monte Carlo win/draw/loss probabilities sum to 100%."""
        dc = dixon_coles_adjustment(1.4, 1.1, rho=-0.06, max_goals=6)
        result = monte_carlo_simulation(dc, iterations=10000)
        p = result["probabilities"]
        total = p["win_a"] + p["draw"] + p["win_b"]
        self.assertAlmostEqual(total, 1.0, places=3,
            msg=f"Monte Carlo probs should sum to 1.0, got {total:.6f}")

    def test_full_prediction_structure(self):
        """Test that a full prediction returns all 16 stages."""
        result = self.predictor.predict("France", "England", {"neutral": False})
        for i in range(1, 17):
            self.assertIn(f"stage_{i}", result,
                msg=f"stage_{i} missing from prediction output")

    def test_confidence_level_range(self):
        """Test that model confidence is between 0 and 100."""
        result = self.predictor.predict("Argentina", "Brazil", {"neutral": True})
        conf = result.get("stage_16", {}).get("nivel_confianza", -1)
        self.assertGreaterEqual(conf, 0, "Confidence should be >= 0")
        self.assertLessEqual(conf, 100, "Confidence should be <= 100")
        self.assertGreater(conf, 50, "For top teams, confidence should be > 50")

    def test_lambda_values_reasonable(self):
        """Test that expected goals lambdas are in realistic range."""
        result = self.predictor.predict("Spain", "Germany", {"neutral": True})
        s8 = result.get("stage_8", {})
        la, lb = s8.get("lambda_a", 0), s8.get("lambda_b", 0)
        self.assertGreater(la, 0.5, f"Lambda A should be > 0.5, got {la}")
        self.assertLess(la, 4.0, f"Lambda A should be < 4.0, got {la}")
        self.assertGreater(lb, 0.5, f"Lambda B should be > 0.5, got {lb}")
        self.assertLess(lb, 4.0, f"Lambda B should be < 4.0, got {lb}")

if __name__ == "__main__":
    unittest.main(verbosity=2)
