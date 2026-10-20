"""Deterministic evaluation, persistent submissions and a public HTML leaderboard."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd


def parse_tickers(value) -> list[str]:
    if isinstance(value, str):
        tokens = re.split(r"[,;\s]+", value.strip()) if value.strip() else []
    elif isinstance(value, (list, tuple)):
        if not all(isinstance(item, str) for item in value):
            raise ValueError("Every ticker must be text.")
        tokens = value
    else:
        raise ValueError("Use a list of tickers or text separated by commas/spaces.")
    cleaned = [token.strip().upper() for token in tokens]
    if not cleaned or any(not token for token in cleaned):
        raise ValueError("The ticker list is empty or contains empty values.")
    if len(cleaned) != len(set(cleaned)):
        raise ValueError("Duplicate tickers are not allowed.")
    return sorted(cleaned)


def load_official_data(directory):
    directory = Path(directory)
    config = json.loads((directory / "competition.json").read_text(encoding="utf-8"))
    identity_payload = {key: value for key, value in config.items() if key != "competition_id"}
    expected_id = hashlib.sha256(json.dumps(identity_payload, sort_keys=True).encode()).hexdigest()[:20]
    if config["competition_id"] != expected_id:
        raise ValueError("Competition rules changed without a new version.")
    raw = (directory / "official_returns.csv").read_bytes()
    actual_hash = hashlib.sha256(raw).hexdigest()
    if actual_hash != config["returns_sha256"]:
        raise ValueError("Official data hash mismatch. Do not evaluate modified data.")
    returns = pd.read_csv(directory / "official_returns.csv", index_col="Date", float_precision="round_trip")
    if list(returns.columns) != config["asset_order"]:
        raise ValueError("Official asset universe/order mismatch.")
    if returns.shape != (config["n_returns"], config["n_assets"]):
        raise ValueError("Official data dimensions mismatch.")
    values = returns.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Official returns must be finite.")
    threshold = config["vol_limit"]
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("Invalid volatility threshold.")
    return returns, config


def evaluate(tickers, returns, config):
    selected = parse_tickers(tickers)
    k = len(selected)
    if not config["min_k"] <= k <= config["max_k"]:
        raise ValueError(f"Choose {config['min_k']} to {config['max_k']} different stocks.")
    unknown = sorted(set(selected) - set(returns.columns))
    if unknown:
        raise ValueError("Unknown tickers: " + ", ".join(unknown))
    portfolio = returns[selected].mean(axis=1)
    annual_vol = float(portfolio.std(ddof=1) * np.sqrt(config["annualization"]))
    if not math.isfinite(annual_vol):
        raise ValueError("Portfolio volatility is not finite.")
    qualifies = annual_vol <= config["vol_limit"] + config["vol_tolerance"]
    anchor = config["score_anchor_k"]
    if not config["min_k"] <= anchor <= config["max_k"]:
        raise ValueError("Invalid score reference.")
    if not qualifies:
        points = 0.0
    elif anchor == config["max_k"]:
        points = 100.0
    else:
        points = float(np.clip(
            100 * (config["max_k"] - k) / (config["max_k"] - anchor), 0, 100
        ))
    return {
        "tickers": selected, "K": k, "annual_vol": annual_vol,
        "qualifies": bool(qualifies), "points": points, "technical_grade": points / 10,
        "improves_reference": bool(qualifies and k < anchor),
        "reference_status": config["reference_status"],
    }


def _write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_register(path, config):
    path = Path(path)
    if not path.exists():
        return {"competition_id": config["competition_id"], "entries": []}
    register = json.loads(path.read_text(encoding="utf-8"))
    if register["competition_id"] != config["competition_id"]:
        raise ValueError("Submission register belongs to another competition version.")
    return register


def record_submission(student_id, display_name, tickers, returns, config, register_path):
    student_id, display_name = str(student_id).strip(), str(display_name).strip()
    if not student_id or not display_name or len(student_id) > 100 or len(display_name) > 100:
        raise ValueError("Student ID and public alias must contain 1 to 100 characters.")
    result = evaluate(tickers, returns, config)
    register = read_register(register_path, config)
    identity = {
        "competition_id": config["competition_id"], "student_id": student_id,
        "tickers": result["tickers"],
    }
    submission_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    for entry in register["entries"]:
        if entry["submission_id"] == submission_id:
            return {**entry, **result, "duplicate": True}
    entry = {
        **result, "student_id": student_id, "display_name": display_name,
        "submission_id": submission_id, "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    register["entries"].append(entry)
    _write_json_atomic(register_path, register)
    return {**entry, "duplicate": False}


PUBLIC_COLUMNS = ["rank", "participant", "K", "annual_vol_pct", "points", "technical_grade", "status"]


def leaderboard(register_path, returns, config):
    entries = read_register(register_path, config)["entries"]
    best = {}
    for entry in entries:
        # Re-evaluate from the authoritative data, not editable stored scores.
        result = evaluate(entry["tickers"], returns, config)
        row = {**entry, **result}
        key = (
            0 if row["qualifies"] else 1,
            row["K"] if row["qualifies"] else config["max_k"] + 1,
            row["annual_vol"], row["submitted_at"],
        )
        old = best.get(row["student_id"])
        if old is None or key < old[0]:
            best[row["student_id"]] = (key, row)
    ordered = sorted(best.values(), key=lambda item: (item[0], item[1]["student_id"]))
    public = []
    previous_key, previous_rank, valid_count = None, None, 0
    for _, row in ordered:
        if row["qualifies"]:
            valid_count += 1
            tie_key = (row["K"], row["annual_vol"])
            rank = previous_rank if tie_key == previous_key else valid_count
            previous_key, previous_rank = tie_key, rank
            status = "Mejora la referencia" if row["improves_reference"] else "Clasificada"
        else:
            rank, status = None, "No cumple el umbral"
        public.append({
            "rank": rank, "participant": row["display_name"], "K": row["K"],
            "annual_vol_pct": 100 * row["annual_vol"], "points": row["points"],
            "technical_grade": row["technical_grade"], "status": status,
        })
    frame = pd.DataFrame(public, columns=PUBLIC_COLUMNS)
    frame["rank"] = pd.array(frame["rank"], dtype="Int64")
    return frame


def publish_leaderboard(frame, config, output_path):
    """Write a self-contained sortable/filterable HTML table, without solutions or IDs."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = {
        "rank": "Posici\u00f3n", "participant": "Participante", "K": "Acciones",
        "annual_vol_pct": "Volatilidad (%)", "points": "Puntos",
        "technical_grade": "Nota t\u00e9cnica / 10", "status": "Estado",
    }
    public = frame[PUBLIC_COLUMNS].rename(columns=labels)
    public["Posici\u00f3n"] = public["Posici\u00f3n"].map(
        lambda value: "" if pd.isna(value) else str(int(value))
    )
    table = public.to_html(
        index=False, escape=True, border=0, table_id="ranking", na_rep="",
        formatters={
            "Volatilidad (%)": lambda value: f"{value:.6f}",
            "Puntos": lambda value: f"{value:.2f}",
            "Nota t\u00e9cnica / 10": lambda value: f"{value:.2f}",
        },
    )
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    status = (
        "Referencia certificada con tolerancias num\u00e9ricas."
        if config["reference_status"] == "certified"
        else "Referencia provisional: mejor soluci\u00f3n encontrada, no \u00f3ptimo demostrado."
    )
    page = """<!doctype html><html lang="es"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Competici\u00f3n de diversificaci\u00f3n</title>
<style>
body{font-family:Georgia,serif;max-width:1100px;margin:36px auto;padding:0 18px;color:#203028;background:#f7f8f3}
h1{font-size:2rem}p,label,input,button,table{font-family:Verdana,sans-serif;font-size:.9rem}
.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white}
th,td{padding:12px;text-align:left;border-bottom:1px solid #dfe5dd}th{background:#dce9de}
th button{border:0;background:none;font-weight:bold;cursor:pointer;text-align:left;padding:0}
input{padding:10px;max-width:90%;margin:12px 0}button{padding:9px}
tr[hidden]{display:none}small{color:#536256}
</style><h1>Competici\u00f3n de diversificaci\u00f3n</h1>
<p>Clasificaci\u00f3n: menos acciones, despu\u00e9s menor volatilidad. Solo clasifican las carteras admisibles.</p>
<p>__STATUS__ Umbral: <strong>__THRESHOLD__ %</strong>.</p>
<label for="filter">Buscar participante o estado</label><br>
<input id="filter" placeholder="Escribe para filtrar">
<button id="refresh">Recargar resultados</button>
<div class="scroll">__TABLE__</div>
<p><small>Actualizada: __STAMP__. Ordenar columnas solo cambia la vista, no la posici\u00f3n oficial.
El profesor regenera este archivo tras registrar entregas. No contiene tickers, correos ni identificadores privados.</small></p>
<script>
const table=document.getElementById('ranking'),body=table.tBodies[0];
document.getElementById('filter').addEventListener('input',e=>{
 const q=e.target.value.toLocaleLowerCase();
 Array.from(body.rows).forEach(r=>r.hidden=!r.textContent.toLocaleLowerCase().includes(q));
});
document.getElementById('refresh').addEventListener('click',()=>location.reload());
Array.from(table.tHead.rows[0].cells).forEach((th,i)=>{
 const label=th.textContent,button=document.createElement('button');let ascending=true;
 button.textContent=label;button.title='Ordenar por '+label;th.textContent='';th.appendChild(button);
 button.addEventListener('click',()=>{
  const numeric=[0,2,3,4,5].includes(i),rows=Array.from(body.rows);
  rows.sort((a,b)=>{
   let x=a.cells[i].textContent.trim(),y=b.cells[i].textContent.trim();
   if(numeric){if(x===''||x==='<NA>')return 1;if(y===''||y==='<NA>')return -1;
    x=Number(x);y=Number(y);return ascending?x-y:y-x;}
   return ascending?x.localeCompare(y):y.localeCompare(x);
  });
  rows.forEach(r=>body.appendChild(r));ascending=!ascending;
 });
});
</script></html>"""
    for placeholder, value in {
        "__TABLE__": table, "__STAMP__": stamp, "__STATUS__": status,
        "__THRESHOLD__": f"{100 * config['vol_limit']:.8f}",
    }.items():
        page = page.replace(placeholder, value if placeholder == "__TABLE__" else html.escape(value))
    temporary = path.with_suffix(".tmp")
    temporary.write_text(page, encoding="utf-8")
    temporary.replace(path)
    return path
