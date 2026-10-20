"""Verify generated packages without adding fake students to the real register."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import zipfile

import pandas as pd

from challenge import load_official_data, evaluate, record_submission, leaderboard, publish_leaderboard

ROOT = Path(__file__).resolve().parent


def main():
    latest = json.loads((ROOT / "dist" / "latest.json").read_text())
    student = Path(latest["student_notebook"]).parent
    teacher = Path(latest["instructor_notebook"]).parent
    returns, config = load_official_data(teacher / "data")
    student_returns, student_config = load_official_data(student / "data")
    assert config == student_config and returns.equals(student_returns)
    evidence = json.loads((teacher / "private" / "calibration_evidence.json").read_text())
    solutions = evidence["frontier"]
    k_anchor = config["score_anchor_k"]
    qualifying = [row for row in solutions if row["upper"] <= config["vol_limit"] + config["vol_tolerance"]]
    assert min(row["K"] for row in qualifying) == k_anchor
    for row in solutions:
        result = evaluate(row["tickers"], returns, config)
        assert abs(result["annual_vol"] - row["upper"]) < 1e-12
    winner = min(qualifying, key=lambda row: (row["K"], row["upper"]))
    with tempfile.TemporaryDirectory() as directory:
        register = Path(directory) / "register.json"
        receipt = record_submission("test-private-id", "Test alias", winner["tickers"], returns, config, register)
        assert receipt["technical_grade"] == 10
        assert record_submission(
            "test-private-id", "Test alias", list(reversed(winner["tickers"])), returns, config, register
        )["duplicate"]
        board = leaderboard(register, returns, config)
        assert len(board) == 1 and board.iloc[0]["rank"] == 1
        html = publish_leaderboard(board, config, Path(directory) / "ranking.html").read_text(encoding="utf-8")
        assert "test-private-id" not in html
        assert ",".join(winner["tickers"]) not in html
    with zipfile.ZipFile(student.parent / "alumnos.zip") as bundle:
        assert bundle.testzip() is None
        names = bundle.namelist()
        assert not any("private" in name or "profesor" in name or "optimizer" in name for name in names)
        notebook = json.loads(bundle.read("competicion_alumnos.ipynb"))
        assert all(not cell.get("outputs") for cell in notebook["cells"])
        text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        assert "improve_swaps" not in text and "simulated_annealing" not in text
        assert "selected_tickers = []" in text
    register = teacher / "private" / "submissions.json"
    assert not register.exists() or not json.loads(register.read_text())["entries"]
    print(f"PASS: {len(solutions)} reference portfolios, scoring, persistence, privacy and student ZIP separation.")
    print(f"Best qualifying reference: K={k_anchor}, volatility={winner['upper']:.8%}")
    print(f"Certification: {evidence['reference_status']}")


if __name__ == "__main__":
    main()
