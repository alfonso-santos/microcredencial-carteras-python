"""Reproduce the instructor notebook without changing its saved outputs."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT.parent.parent / "src-sol" / "portfolio-diversification-efficiency-challenge-English-SOL.ipynb"


def main() -> None:
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    raw = NOTEBOOK.read_bytes()
    notebook = json.loads(raw)
    namespace = {"__name__": "__calibration__"}
    previous = Path.cwd()
    log = io.StringIO()
    try:
        os.chdir(NOTEBOOK.parent)
        with contextlib.redirect_stdout(log):
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] != "code":
                    continue
                print(f"Calibration cell {index}")
                source = "".join(cell["source"])
                # Do not create a graphical display during CLI calibration.
                if "plt.rc_context" in source:
                    continue
                exec(compile(source, str(NOTEBOOK) + f":cell-{index}", "exec"), namespace)
                if "display" in namespace:
                    namespace["display"] = lambda *args, **kwargs: None
    finally:
        os.chdir(previous)
        (build / "calibration.log").write_text(log.getvalue(), encoding="utf-8")
    if namespace["THRESHOLD_MARGIN"] != 0.0025:
        raise ValueError("Unexpected margin; this package was specified for 0.25%.")
    if namespace["EXPORT_RESULTS"]:
        raise ValueError("Disable notebook EXPORT_RESULTS before packaging.")
    (build / "official_returns.csv").write_bytes(namespace["returns_csv"].encode("utf-8"))
    namespace["results"].to_csv(build / "heuristic_results.csv", index=False, float_format="%.17g")
    namespace["best_by_k"].to_csv(build / "heuristic_best_by_k.csv", index=False, float_format="%.17g")
    namespace["search_runs"].to_csv(build / "search_runs.csv", index=False, float_format="%.17g")
    namespace["cleaning_audit"].to_csv(build / "cleaning_audit.csv", index_label="asset_id")
    np.savez(
        build / "optimization_input.npz",
        covariance=namespace["cov"], tickers=np.array(namespace["tickers"]),
    )
    manifest = dict(namespace["challenge_config"])
    manifest["instructor_notebook_sha256"] = hashlib.sha256(raw).hexdigest()
    (build / "heuristic_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "assets": len(manifest["asset_order"]), "observations": manifest["n_returns"],
        "reference_vol": manifest["reference_vol"], "vol_limit": manifest["vol_limit"],
        "best_qualifying_k": manifest["best_qualifying_k"],
    }, indent=2))


if __name__ == "__main__":
    main()
