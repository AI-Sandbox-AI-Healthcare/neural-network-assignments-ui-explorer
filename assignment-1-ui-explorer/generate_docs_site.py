#!/usr/bin/env python3
"""
Generate a minimal static site in ./docs/assignment-1/ suitable for GitHub Pages.

This script exports the patient data and the precomputed oracle table into
JSON files under `docs/assignment-1/` and writes a lightweight `index.html`
that explains how to use the static explorer. The static site is
intentionally limited and does NOT include hidden tests or solutions.

Each assignment gets its own subfolder under `docs/` so multiple assignments
can coexist on the same GitHub Pages site without overwriting each other's
files. See `docs/index.html` for the top-level landing page that links to
each assignment.

Run from repository root:
  python assignment-1-ui-explorer/generate_docs_site.py

Publish the generated `docs/` directory with GitHub Pages (branch or /docs folder).
"""
from pathlib import Path
import json

ROOT = Path(__file__).parent
REPO_ROOT = ROOT.parent
DATA_FILE = ROOT / "data" / "patient_features.csv"
ORACLE_FILE = ROOT / "oracle_table.json"
OUT_DIR = REPO_ROOT / "docs" / "assignment-1"

OUT_DIR.mkdir(parents=True, exist_ok=True)

def export_patients():
    try:
        import pandas as pd
    except Exception:
        print("pandas required to export patient data. Install requirements.txt")
        return
    df = pd.read_csv(DATA_FILE)
    patients = df.to_dict(orient="records")
    (OUT_DIR / "patients.json").write_text(json.dumps(patients, indent=2), encoding="utf-8")
    print(f"Exported {len(patients)} patients -> {OUT_DIR/'patients.json'}")

def export_oracle():
    if ORACLE_FILE.exists():
        txt = ORACLE_FILE.read_text(encoding="utf-8")
        (OUT_DIR / "oracle_table.json").write_text(txt, encoding="utf-8")
        print(f"Copied oracle table -> {OUT_DIR/'oracle_table.json'}")
    else:
        print("Warning: oracle_table.json missing; generate it locally with assignment-1-ui-explorer/generate_oracle.py")

    # Copy precomputed grid if present (see precompute_grids.py)
    precomp = ROOT / "precomputed_grid.json"
    if precomp.exists():
        (OUT_DIR / "precomputed_grid.json").write_text(precomp.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Copied precomputed grid -> {OUT_DIR/'precomputed_grid.json'}")

def write_index():
    index = """<!doctype html>
<html><head><meta charset="utf-8"><title>Assignment 1 Explorer (Static)</title></head>
<body>
<p><a href="../">&larr; All assignments</a></p>
<h1>Assignment 1 — Chronic Pain Classifier (Static)</h1>
<p>This static site contains non-sensitive explorer assets suitable for publishing
on GitHub Pages. It provides dataset browsing and the oracle targets per seed.
It does NOT include hidden tests or solutions.</p>

<h2>Usage</h2>
<ul>
  <li>Open <code>patients.json</code> to browse the dataset.</li>
  <li>Open <code>oracle_table.json</code> to view precomputed oracle metrics by seed.</li>
  <li>To run the interactive trainer (live evaluation), run the local server:
    <pre>python assignment-1-ui-explorer/run_sandbox.py</pre>
  </li>
</ul>

<h2>Security</h2>
<p>Hidden tests and instructor solutions must remain in a private grader repo.
This static explorer exposes only dataset and target metrics — not answers.</p>

</body></html>"""
    (OUT_DIR / "index.html").write_text(index, encoding="utf-8")
    print(f"Wrote static index -> {OUT_DIR/'index.html'}")

def main():
    export_patients()
    export_oracle()
    write_index()

if __name__ == '__main__':
    main()
