"""Build separate student and instructor notebooks from frozen calibration evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import textwrap
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"


def make_notebook(cells):
    result = []
    for index, (kind, source) in enumerate(cells):
        cell = {
            "cell_type": kind, "id": f"competition-{index:02d}", "metadata": {},
            "source": textwrap.dedent(source).strip().splitlines(keepends=True),
        }
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        result.append(cell)
    return {
        "cells": result, "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    }


def freeze_config():
    manifest = json.loads((BUILD / "heuristic_manifest.json").read_text())
    heuristic = pd.read_csv(BUILD / "heuristic_best_by_k.csv")
    frontier = {
        int(row.K): {
            "K": int(row.K), "upper": float(row.annual_vol), "lower": 0.0,
            "tickers": row.tickers.split(","), "source": "heuristic",
        }
        for row in heuristic.itertuples()
    }
    reports = []
    for path in sorted(BUILD.glob("optimizer*.json")):
        report = json.loads(path.read_text())
        if report["dataset_sha256"] != manifest["cleaned_returns_sha256"]:
            raise ValueError("Optimizer evidence belongs to a different dataset.")
        reports.append(path.name)
        for result in report["results"]:
            target = frontier[result["K"]]
            target["lower"] = max(target["lower"], result["annual_vol_lower_bound"])
            if result["annual_vol_upper_bound"] < target["upper"]:
                target.update(
                    upper=result["annual_vol_upper_bound"], tickers=result["tickers"], source=path.name
                )
    reference = min(frontier.values(), key=lambda row: (row["upper"], row["K"]))
    threshold = reference["upper"] * 1.0025
    anchor = min(k for k, row in frontier.items() if row["upper"] <= threshold + 1e-12)
    lower = min(row["lower"] for row in frontier.values())
    gap = max(0.0, reference["upper"]**2 - lower**2) / reference["upper"]**2
    risk_certified = gap <= 1e-6
    cardinality_certified = all(frontier[k]["lower"] > threshold + 1e-10 for k in range(5, anchor))
    status = "certified" if risk_certified and cardinality_certified else "provisional"
    config = {
        "title": "Diversificacion eficiente: menos acciones, riesgo controlado",
        "min_k": 5, "max_k": 30, "annualization": 252, "ddof": 1,
        "max_missing_fraction": 0.20, "threshold_relative_margin": 0.0025,
        "vol_limit": threshold, "vol_tolerance": 1e-12,
        "returns_sha256": manifest["cleaned_returns_sha256"],
        "asset_order": manifest["asset_order"], "n_assets": len(manifest["asset_order"]),
        "n_returns": manifest["n_returns"],
        "first_return_date": manifest["first_return_date"], "last_return_date": manifest["last_return_date"],
        "score_anchor_k": anchor, "reference_status": status,
        "scoring_policy": "100 * (max_k - K) / (max_k - score_anchor_k), clipped to [0,100], qualifying only",
        "ranking_policy": "qualifies, K ascending, annual_vol ascending; exact ties share rank",
        "technical_grade_policy": "points / 10; not a complete assessment of understanding",
    }
    config["competition_id"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:20]
    evidence = {
        "reference_vol_upper_bound": reference["upper"], "reference_k": reference["K"],
        "global_vol_lower_bound": lower, "relative_variance_gap": gap,
        "risk_certified_to_1e_6": risk_certified, "minimum_k_certified": cardinality_certified,
        "score_anchor_k": anchor, "reference_status": status, "reports": reports,
        "frontier": list(frontier.values()),
        "note": "No certified optimum is claimed if a bound remains open. Scores use a frozen best-known reference.",
    }
    (BUILD / "calibration_evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    return config, evidence


def student_cells(config):
    return [
        ("markdown", f"""
        # Competicion: diversificacion eficiente

        **Objetivo:** encontrar una cartera equiponderada con el menor numero de acciones
        cuya volatilidad anualizada no supere **{100 * config['vol_limit']:.8f} %**.

        ## Reglas
        - Entre **5 y 30 acciones distintas**, del universo oficial de **{config['n_assets']} activos**.
        - Cada accion tiene peso `1/K`. Sin efectivo, renta fija, cortos ni apalancamiento.
        - Utiliza exclusivamente `data/official_returns.csv`, sin cambiar activos, fechas ni datos.
        - Los retornos son logaritmicos. La volatilidad usa `std(ddof=1) * sqrt(252)`.
        - Clasifican las carteras con volatilidad menor o igual al umbral, con tolerancia `1e-12`
          en unidades decimales. No se compara el porcentaje redondeado.
        - Gana la cartera clasificada con **menos acciones**. A igual numero, **menor volatilidad**.
        - Se conserva la mejor entrega de cada participante. Repetir los mismos tickers no crea otra entrega.

        ## Datos y alcance
        La preparacion elimina activos con mas del 20 % de precios ausentes y despues elimina
        fechas incompletas del universo restante. Todos utilizan las mismas {config['n_returns']} observaciones,
        de {config['first_return_date']} a {config['last_return_date']}.
        Se evalua riesgo historico dentro de muestra, no capacidad de predecir el futuro.

        ## Puntuacion tecnica
        Si cumples el umbral: `points = clip(100 * (30 - K) / (30 - {config['score_anchor_k']}), 0, 100)`.
        La nota tecnica sobre 10 es `points / 10`.
        Una cartera de {config['score_anchor_k']} acciones obtiene 100 puntos; una de 30, cero.
        Si no cumples el umbral: cero puntos y estado **No clasificada**.
        La posicion oficial utiliza K y volatilidad, no la nota redondeada.

        **Referencia {config['reference_status']}:** el objetivo se ha calibrado con la mejor solucion
        encontrada mas un margen relativo del 0,25 %. No se anuncia un minimo global sin certificado.
        Si encuentras una cartera admisible con menos de {config['score_anchor_k']} acciones, obtienes
        100 puntos y una mejor posicion. El profesor revisara la referencia; las reglas congeladas
        no cambian durante esta edicion.

        ## Entrega
        Entrega tu identificador, alias publico, lista de tickers y una explicacion breve:
        1. Como elegiste las acciones.
        2. Comparacion con las acciones de menor volatilidad individual y con carteras aleatorias del mismo K.
        3. Efecto de retirar una accion y redistribuir los pesos.
        4. Si utilizaste IA, que tareas delegaste y como verificaste los resultados.

        No necesitas conocer optimizacion formal. El codigo siguiente verifica una propuesta,
        pero no busca una solucion. Conserva tus pruebas para una breve defensa individual.
        """),
        ("code", """
        from pathlib import Path
        import json
        import hashlib
        import re
        import numpy as np
        import pandas as pd
        from IPython.display import display

        DATA_DIR = Path("data")
        config = json.loads((DATA_DIR / "competition.json").read_text(encoding="utf-8"))
        raw = (DATA_DIR / "official_returns.csv").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == config["returns_sha256"], "Official data changed."
        returns = pd.read_csv(DATA_DIR / "official_returns.csv", index_col="Date", float_precision="round_trip")
        assert list(returns.columns) == config["asset_order"]
        print(f"Universe: {returns.shape[1]} assets; observations: {len(returns)}")
        print(f"Volatility limit: {config['vol_limit']:.8%}")
        display(returns.head())
        """),
        ("code", '''
        def check_portfolio(tickers):
            """Verify a proposed equal-weight portfolio without searching for one."""
            values = re.split(r"[,;\\s]+", tickers.strip()) if isinstance(tickers, str) else list(tickers)
            values = [str(value).strip().upper() for value in values]
            if len(values) != len(set(values)):
                raise ValueError("Duplicate tickers.")
            if not config["min_k"] <= len(values) <= config["max_k"]:
                raise ValueError("Choose between 5 and 30 stocks.")
            unknown = sorted(set(values) - set(returns.columns))
            if unknown:
                raise ValueError(f"Unknown tickers: {unknown}")
            portfolio = returns[values].mean(axis=1)
            vol = float(portfolio.std(ddof=1) * np.sqrt(config["annualization"]))
            qualifies = vol <= config["vol_limit"] + config["vol_tolerance"]
            denominator = config["max_k"] - config["score_anchor_k"]
            points = (100.0 if denominator == 0 else float(np.clip(
                100 * (config["max_k"] - len(values)) / denominator, 0, 100
            ))) if qualifies else 0.0
            return {
                "K": len(values), "annual_vol_pct": 100 * vol,
                "qualifies": bool(qualifies), "points": points, "technical_grade": points / 10,
            }

        selected_tickers = []  # Enter your own selection here.
        if selected_tickers:
            display(pd.DataFrame([check_portfolio(selected_tickers)]))
        else:
            print("Enter your tickers in selected_tickers and run this cell.")
        '''),
        ("markdown", """
        ## Tu investigacion

        Incluye aqui tu metodo, los experimentos y su interpretacion.
        Distingue una buena solucion encontrada de una prueba de optimalidad.
        """),
        ("code", "# Add your own analysis here.\n"),
        ("markdown", """
        ## Explicacion de la entrega

        **Metodo utilizado:** ...

        **Comparaciones al mismo numero de acciones:** ...

        **Una sustitucion o retirada y su efecto:** ...

        **Limitaciones y uso de IA:** ...
        """),
    ]


def teacher_cells(config):
    return [
        ("markdown", f"""
        # Profesor: evaluar entregas y publicar la clasificacion

        Este notebook recibe identificador, alias publico y tickers. Calcula los pesos `1/K`,
        volatilidad, elegibilidad, puntos y nota tecnica. Guarda todas las entregas y conserva
        la mejor de cada alumno para la clasificacion.

        **No modifica ninguna aplicacion ni conecta con Firebase o GCS.**
        El registro privado se guarda en `private/submissions.json`.
        La tabla publica se exporta a `public/clasificacion.html`: permite ordenar columnas
        y buscar participantes sin conexion a internet. Cada nueva entrega regenera el archivo.
        Comparte SOLO ese HTML o su tabla; no publiques este notebook ni la carpeta privada.
        Generar el archivo no lo sube automaticamente a ningun servidor. Puedes adjuntarlo al campus
        o alojarlo en el espacio autorizado y reemplazarlo tras nuevas entregas.

        ## Calibracion
        Umbral: **{100 * config['vol_limit']:.8f} %**. Referencia de puntuacion: **{config['score_anchor_k']} acciones**.
        Estado: **{config['reference_status']}**. Consulta la evidencia de optimizacion mas abajo.
        Si quedan brechas abiertas, la referencia es la mejor encontrada, NO un optimo demostrado.
        La escala se congela para esta edicion. Una mejora del alumno genera una alerta, nunca
        una penalizacion ni una actualizacion silenciosa de las notas anteriores.

        **Nota tecnica:** 0 a 10 a partir del numero de acciones si cumple el umbral.
        No sustituye la valoracion de la explicacion o la defensa oral.
        Trabaja con un solo kernel escritor del registro para evitar escrituras simultaneas.
        """),
        ("code", """
        from pathlib import Path
        import json
        import pandas as pd
        from IPython.display import display, FileLink
        from challenge import (
            load_official_data, evaluate, record_submission,
            leaderboard, publish_leaderboard, read_register,
        )

        ROOT = Path.cwd()
        returns, config = load_official_data(ROOT / "data")
        REGISTER = ROOT / "private" / "submissions.json"
        PUBLIC_HTML = ROOT / "public" / "clasificacion.html"
        print(f"Competition ID: {config['competition_id']}")
        print(f"Reference status: {config['reference_status']}")
        print(f"Volatility threshold: {config['vol_limit']:.8%}")
        print(f"Scoring anchor: {config['score_anchor_k']} stocks")
        """),
        ("markdown", """
        ## 1. Consultar los calculos y el estado de los optimos

        `upper` es una volatilidad alcanzada por una cartera concreta; `lower` es una cota inferior
        obtenida por el optimizador. Una diferencia abierta NO demuestra que exista una cartera mejor,
        pero impide asegurar que no exista. La brecha del solver se expresa en varianza.
        La certificacion de riesgo y la del minimo numero de acciones son distintas.
        """),
        ("code", """
        evidence = json.loads((ROOT / "private" / "calibration_evidence.json").read_text(encoding="utf-8"))
        print("Risk optimum certified:", evidence["risk_certified_to_1e_6"])
        print("Minimum number of stocks certified:", evidence["minimum_k_certified"])
        print(f"Global relative variance gap: {evidence['relative_variance_gap']:.4%}")
        frontier = pd.DataFrame(evidence["frontier"])
        display(frontier[["K", "lower", "upper", "source"]].rename(columns={
            "lower": "vol_lower_bound", "upper": "best_found_vol",
        }))
        """),
        ("markdown", """
        ## 2. Verificar sin registrar

        Introduce los tickers separados por comas, espacios o punto y coma.
        No hace falta pedir al alumno sus pesos ni confiar en su volatilidad declarada.
        Una lista invalida produce un error y no se registra. Una cartera valida que excede
        el umbral se puede registrar, pero queda no clasificada y con nota tecnica cero.
        """),
        ("code", """
        candidate_tickers = ""  # Example format: "TICKER1, TICKER2, ..."
        if candidate_tickers.strip():
            preview = evaluate(candidate_tickers, returns, config)
            display(pd.DataFrame([{key: value for key, value in preview.items() if key != "tickers"}]))
        else:
            print("Enter a ticker list to preview a submission.")
        """),
        ("code", '''
        def refresh_ranking():
            """Re-evaluate saved entries and regenerate the public ranking."""
            table = leaderboard(REGISTER, returns, config)
            publish_leaderboard(table, config, PUBLIC_HTML)
            display(table)
            display(FileLink("public/clasificacion.html", result_html_prefix="Tabla interactiva: "))
            return table


        def register_student(student_id, public_alias, tickers):
            """Persist one submission, then refresh the public HTML table."""
            result = record_submission(student_id, public_alias, tickers, returns, config, REGISTER)
            summary = {key: result[key] for key in [
                "K", "annual_vol", "qualifies", "points", "technical_grade", "duplicate",
            ]}
            display(pd.DataFrame([summary]))
            if result["improves_reference"]:
                print("REFERENCE IMPROVED: review calibration; keep the published scale frozen.")
            refresh_ranking()
            return result
        '''),
        ("markdown", """
        ## 3. Registrar una entrega

        Utiliza un identificador estable por alumno (privado) y un alias para la tabla publica.
        No introduzcas correos ni nombres reales como alias si no deben publicarse.
        Reejecutar la misma lista para el mismo identificador no duplica la entrega.
        """),
        ("code", """
        student_id = ""
        public_alias = ""
        submitted_tickers = ""
        if student_id and public_alias and submitted_tickers:
            receipt = register_student(student_id, public_alias, submitted_tickers)
        else:
            print("Fill student_id, public_alias and submitted_tickers, then run this cell.")
        """),
        ("markdown", """
        ## 4. Formulario opcional y tabla

        Si tienes `ipywidgets`, puedes registrar entregas con el formulario. Si no,
        la celda anterior funciona sin dependencias adicionales.
        La tabla HTML admite busqueda y ordenacion; la posicion oficial no cambia al ordenar la vista.
        """),
        ("code", """
        try:
            import ipywidgets as widgets
        except ImportError:
            print("Optional form unavailable. Use the registration cell above.")
        else:
            id_input = widgets.Text(description="ID privado:")
            alias_input = widgets.Text(description="Alias publico:")
            tickers_input = widgets.Textarea(description="Tickers:", layout=widgets.Layout(width="90%", height="100px"))
            save_button = widgets.Button(description="Registrar y actualizar", button_style="primary")
            output = widgets.Output()

            def save_clicked(_):
                with output:
                    output.clear_output()
                    try:
                        register_student(id_input.value, alias_input.value, tickers_input.value)
                    except ValueError as error:
                        print(f"Invalid submission: {error}")

            save_button.on_click(save_clicked)
            display(widgets.VBox([id_input, alias_input, tickers_input, save_button, output]))

        current_ranking = refresh_ranking()
        """),
        ("markdown", """
        ## 5. Historial privado y defensa

        El historial conserva todas las entregas y sus tickers. Nunca se incluye en el HTML publico.
        Para revisar una entrega, filtra por identificador en la celda siguiente.

        Preguntas de defensa:
        1. Por que una accion de volatilidad elevada podria ayudar a diversificar.
        2. Que sucede al quitar una accion y redistribuir los pesos.
        3. Si una menor correlacion implica siempre menor riesgo.
        4. Que diferencia hay entre optimizar en esta muestra y predecir riesgo futuro.

        Antes de comenzar, fija el plazo y las normas de asistencia con IA en tu campus.
        No cambies el dataset, el umbral o la escala de puntuacion durante esta edicion.
        """),
        ("code", """
        inspect_student_id = ""
        if inspect_student_id:
            entries = read_register(REGISTER, config)["entries"]
            display(pd.DataFrame([row for row in entries if row["student_id"] == inspect_student_id]))
        else:
            print("Enter an ID to inspect its private submission history.")
        """),
    ]


def write_notebook(path, cells):
    path.write_text(json.dumps(make_notebook(cells), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    config, evidence = freeze_config()
    version = config["competition_id"]
    destination = ROOT / "dist" / version
    student = destination / "alumnos"
    teacher = destination / "profesor"
    active_register = teacher / "private" / "submissions.json"
    if active_register.exists() and json.loads(active_register.read_text(encoding="utf-8"))["entries"]:
        raise ValueError("This edition already has submissions. Do not rebuild an active class.")
    for folder in (student, teacher):
        (folder / "data").mkdir(parents=True, exist_ok=True)
        shutil.copy2(BUILD / "official_returns.csv", folder / "data" / "official_returns.csv")
        (folder / "data" / "competition.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        shutil.copy2(ROOT / "requirements.txt", folder / "requirements.txt")
    (teacher / "private").mkdir(exist_ok=True)
    (teacher / "private" / "calibration_evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    shutil.copy2(ROOT / "challenge.py", teacher / "challenge.py")
    write_notebook(student / "competicion_alumnos.ipynb", student_cells(config))
    write_notebook(teacher / "evaluacion_profesor.ipynb", teacher_cells(config))
    for path in BUILD.glob("optimizer*.json"):
        shutil.copy2(path, teacher / "private" / path.name)
    shutil.copy2(BUILD / "heuristic_results.csv", teacher / "private" / "heuristic_results.csv")
    (ROOT / "dist" / "latest.json").write_text(json.dumps({
        "competition_id": version, "student_notebook": str(student / "competicion_alumnos.ipynb"),
        "instructor_notebook": str(teacher / "evaluacion_profesor.ipynb"),
        "reference_status": config["reference_status"], "vol_limit": config["vol_limit"],
    }, indent=2), encoding="utf-8")
    for name, folder in [("alumnos", student), ("profesor", teacher)]:
        archive = destination / f"{name}.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
            for path in sorted(folder.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    bundle.write(path, str(path.relative_to(folder)))
    print(json.dumps({
        "directory": str(destination), "reference_status": config["reference_status"],
        "risk_certified": evidence["risk_certified_to_1e_6"],
        "minimum_k_certified": evidence["minimum_k_certified"],
        "vol_limit": config["vol_limit"], "score_anchor_k": config["score_anchor_k"],
    }, indent=2))


if __name__ == "__main__":
    main()
