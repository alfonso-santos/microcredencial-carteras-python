import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from challenge import evaluate, leaderboard, parse_tickers, publish_leaderboard, record_submission


class ChallengeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.returns = pd.DataFrame(rng.normal(0, .01, (100, 8)), columns=list("ABCDEFGH"))
        self.config = {
            "competition_id": "test", "min_k": 2, "max_k": 6, "score_anchor_k": 3,
            "annualization": 252, "vol_limit": 1.0, "vol_tolerance": 1e-12,
            "reference_status": "provisional",
        }
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "register.json"

    def tearDown(self):
        self.temp.cleanup()

    def test_normalization_and_duplicates(self):
        self.assertEqual(parse_tickers("b; a\nC"), ["A", "B", "C"])
        for value in ("a,A", "", ["A", ""], [1, 2]):
            with self.assertRaises(ValueError):
                parse_tickers(value)

    def test_invalid_universe_and_size(self):
        for value in ("A", "A,B,Z", "A,B,C,D,E,F,G"):
            with self.assertRaises(ValueError):
                evaluate(value, self.returns, self.config)

    def test_volatility_and_grades(self):
        result = evaluate("A,B,C", self.returns, self.config)
        self.assertAlmostEqual(result["annual_vol"], self.returns[list("ABC")].mean(axis=1).std(ddof=1) * np.sqrt(252))
        self.assertEqual(result["technical_grade"], 10)
        self.assertEqual(evaluate("A,B,C,D,E,F", self.returns, self.config)["technical_grade"], 0)
        self.assertTrue(evaluate("A,B", self.returns, self.config)["improves_reference"])
        limited = {**self.config, "vol_limit": result["annual_vol"] - 1e-5}
        self.assertFalse(evaluate("A,B,C", self.returns, limited)["qualifies"])
        self.assertEqual(evaluate("A,B,C", self.returns, limited)["points"], 0)
        edge = {**self.config, "vol_limit": result["annual_vol"]}
        self.assertTrue(evaluate("A,B,C", self.returns, edge)["qualifies"])

    def test_idempotency_and_best_submission(self):
        record_submission("id1", "Alias", "A,B,C,D", self.returns, self.config, self.path)
        record_submission("id1", "Alias", "A,B,C", self.returns, self.config, self.path)
        duplicate = record_submission("id1", "Alias", "C,A,B", self.returns, self.config, self.path)
        self.assertTrue(duplicate["duplicate"])
        self.assertEqual(len(json.loads(self.path.read_text())["entries"]), 2)
        board = leaderboard(self.path, self.returns, self.config)
        self.assertEqual(len(board), 1)
        self.assertEqual(board.iloc[0]["K"], 3)

    def test_ties_and_public_privacy(self):
        for sid, name in [("private1", "<script>alert(1)</script>"), ("private2", "Two")]:
            record_submission(sid, name, "A,B,C", self.returns, self.config, self.path)
        board = leaderboard(self.path, self.returns, self.config)
        self.assertEqual(board["rank"].tolist(), [1, 1])
        output = Path(self.temp.name) / "ranking.html"
        publish_leaderboard(board, self.config, output)
        text = output.read_text(encoding="utf-8")
        self.assertNotIn("private1", text)
        self.assertNotIn("<script>alert(1)</script>", text)
        self.assertIn("&lt;script&gt;", text)

    def test_recompute_scores_and_version(self):
        record_submission("id1", "One", "A,B,C", self.returns, self.config, self.path)
        register = json.loads(self.path.read_text())
        register["entries"][0]["points"] = -999
        self.path.write_text(json.dumps(register))
        self.assertEqual(leaderboard(self.path, self.returns, self.config).iloc[0]["points"], 100)
        with self.assertRaises(ValueError):
            leaderboard(self.path, self.returns, {**self.config, "competition_id": "other"})

    def test_empty_ranking_and_nonqualifying(self):
        self.assertTrue(leaderboard(self.path, self.returns, self.config).empty)
        config = {**self.config, "vol_limit": 1e-6}
        record_submission("id", "Alias", "A,B,C", self.returns, config, self.path)
        board = leaderboard(self.path, self.returns, config)
        self.assertTrue(pd.isna(board.iloc[0]["rank"]))
        self.assertEqual(board.iloc[0]["status"], "No cumple el umbral")
        output = Path(self.temp.name) / "unqualified.html"
        publish_leaderboard(board, config, output)
        self.assertNotIn("&lt;NA&gt;", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
