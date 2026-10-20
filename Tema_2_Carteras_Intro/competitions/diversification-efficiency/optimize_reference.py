"""Bound the equal-weight cardinality frontier with SCIP; never infer a proof from a timeout."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

# Avoid loading a second OpenMP runtime when SCIP is used with a Conda MKL NumPy.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np
import pandas as pd
from pyscipopt import Model, quicksum
from scipy.linalg import eigvalsh
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parent


def solve_cardinality(covariance, k, initial, seconds=20, gap=1e-6, seed=20261020, formulation="direct"):
    n = len(covariance)
    # Percentage-volatility squared keeps objective coefficients away from tiny scales.
    q = np.asarray(covariance) * (252 * 10_000 / k**2)
    model = Model(f"equal_weight_k_{k}")
    model.hideOutput()
    model.setParam("limits/time", float(seconds))
    model.setParam("limits/gap", float(gap))
    model.setParam("numerics/feastol", 1e-8)
    model.setParam("randomization/randomseedshift", int(seed))
    model.setParam("parallel/maxnthreads", 1)
    z = [model.addVar(vtype="B", name=f"asset_{i}") for i in range(n)]
    initial = sorted(map(int, initial))
    initial_value = float(q[np.ix_(initial, initial)].sum())
    t = model.addVar(lb=0, ub=initial_value + 1e-6, name="annual_variance_percent_squared")
    model.addCons(quicksum(z) == k)
    factors, transformed = None, None
    if formulation in ("factorized", "diagonal"):
        # Keep continuous factor variables to expose convex sum-of-squares structure.
        model.setParam("presolving/maxrounds", 0)
        diagonal = np.full(n, max(0.0, float(eigvalsh(q, subset_by_index=[0, 0])[0]) * 0.999))
        if formulation == "diagonal":
            binary_start = np.zeros(n)
            binary_start[initial] = 1
            gradient = 2 * q @ binary_start
            signs = 2 * binary_start - 1

            def smallest_eigenvalue(mu):
                shifted = q - np.diag(signs * (gradient - mu))
                return float(eigvalsh(shifted, subset_by_index=[0, 0])[0])

            optimum = minimize_scalar(
                lambda mu: -smallest_eigenvalue(mu),
                bounds=(gradient.min(), gradient.max()), method="bounded",
                options={"xatol": 1e-10},
            )
            # z_i^2 = z_i permits exact diagonal redistribution for binary points.
            # A positive safety shift makes the remaining quadratic PSD.
            correction = max(0.0, -smallest_eigenvalue(optimum.x)) + 1e-8
            diagonal = signs * (gradient - optimum.x) - correction
        residual = q - np.diag(diagonal)
        eigenvalues, vectors = np.linalg.eigh(residual)
        if eigenvalues[0] < -1e-8:
            raise ValueError("Invalid diagonal convexification.")
        factors = np.sqrt(np.maximum(eigenvalues, 0))[:, None] * vectors.T
        transformed = [model.addVar(lb=None, name=f"factor_{i}") for i in range(n)]
        for i in range(n):
            model.addCons(transformed[i] == quicksum(float(factors[i, j]) * z[j] for j in range(n)))
        objective = quicksum(v * v for v in transformed) + quicksum(
            float(diagonal[i]) * z[i] for i in range(n)
        )
    else:
        objective = quicksum(
            float(q[i, i]) * z[i] * z[i]
            + quicksum(float(2 * q[i, j]) * z[i] * z[j] for j in range(i + 1, n))
            for i in range(n)
        )
    model.addCons(objective <= t)
    model.setObjective(t, "minimize")
    initial = sorted(map(int, initial))
    if len(initial) != k or len(set(initial)) != k:
        raise ValueError("Warm-start cardinality mismatch.")
    warm = model.createSol()
    for i, variable in enumerate(z):
        model.setSolVal(warm, variable, float(i in initial))
    if transformed is not None:
        for i, variable in enumerate(transformed):
            model.setSolVal(warm, variable, float(factors[i, initial].sum()))
    model.setSolVal(warm, t, float(q[np.ix_(initial, initial)].sum()) + 1e-7)
    model.addSol(warm)
    model.optimize()
    selected = initial
    if model.getNSols():
        candidate = [i for i, variable in enumerate(z) if model.getVal(variable) > 0.5]
        if len(candidate) == k:
            selected = candidate
    exact = float(q[np.ix_(selected, selected)].sum())
    if initial_value < exact:
        selected, exact = initial, initial_value
    dual = max(0.0, float(model.getDualbound()))
    if not math.isfinite(dual) or dual > exact + 1e-5:
        raise ValueError("Solver returned an invalid objective bound.")
    relative_gap = max(0.0, exact - dual) / max(exact, 1e-12)
    result = {
        "K": k, "status": str(model.getStatus()), "seconds": model.getSolvingTime(),
        "formulation": formulation,
        "nodes": model.getNNodes(), "relative_variance_gap": relative_gap,
        "annual_vol_lower_bound": math.sqrt(max(0, dual)) / 100,
        "annual_vol_upper_bound": math.sqrt(max(0, exact)) / 100,
        "certified_to_requested_gap": relative_gap <= gap + 1e-10,
        "selected_indices": selected, "requested_gap": gap,
    }
    model.freeProb()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds-per-k", type=float, default=60)
    parser.add_argument("--gap", type=float, default=1e-6)
    parser.add_argument("--sizes", type=int, nargs="+")
    parser.add_argument("--formulation", choices=["direct", "factorized", "diagonal"], default="diagonal")
    parser.add_argument("--output", default="optimizer_report.json")
    args = parser.parse_args()
    if args.seconds_per_k <= 0 or not 0 < args.gap < 1:
        parser.error("Time must be positive and relative variance gap must be in (0,1).")
    if Path(args.output).name != args.output or not args.output.endswith(".json"):
        parser.error("--output must be a JSON filename, without directory components.")
    build = ROOT / "build"
    data = np.load(build / "optimization_input.npz")
    tickers = data["tickers"].tolist()
    covariance = data["covariance"]
    lookup = {name: i for i, name in enumerate(tickers)}
    heuristic = pd.read_csv(build / "heuristic_best_by_k.csv").set_index("K")
    manifest = json.loads((build / "heuristic_manifest.json").read_text())
    requested = args.sizes or [18, 19, 22] + [k for k in range(5, 31) if k not in (18, 19, 22)]
    if any(k not in heuristic.index for k in requested):
        raise ValueError("Requested size outside the calibrated range.")
    results = []
    for k in requested:
        initial = [lookup[name] for name in heuristic.loc[k, "tickers"].split(",")]
        result = solve_cardinality(covariance, k, initial, args.seconds_per_k, args.gap, formulation=args.formulation)
        result["tickers"] = sorted(tickers[i] for i in result.pop("selected_indices"))
        results.append(result)
        evidence = {
            "solver": "SCIP via PySCIPOpt 6.2.1",
            "dataset_sha256": manifest["cleaned_returns_sha256"],
            "input_sha256": hashlib.sha256((build / "optimization_input.npz").read_bytes()).hexdigest(),
            "results": sorted(results, key=lambda row: row["K"]),
            "note": "Bounds are numerical certificates under solver tolerances, not exact arithmetic proofs.",
        }
        temp = build / (args.output + ".tmp")
        temp.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        temp.replace(build / args.output)
        print(
            f"K={k}: {result['status']}, "
            f"vol in [{result['annual_vol_lower_bound']:.8%}, "
            f"{result['annual_vol_upper_bound']:.8%}], "
            f"variance gap={result['relative_variance_gap']:.4%}", flush=True
        )


if __name__ == "__main__":
    main()
