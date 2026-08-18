#!/usr/bin/env python3
"""
Generate the full interactive static explorer in ./docs/assignment-1/ for
GitHub Pages.

GitHub Pages can't run the Flask backend, so this script builds a
self-contained page that reproduces run_sandbox.py's UI exactly (same
HTML/CSS/JS, extracted straight from its HTML_TEMPLATE) but computes
everything client-side instead of calling /api/*:

  - static_client_pipeline.js -- a from-scratch JS port of reference.py's
    oracle pipeline (stratified split, standardization, gradient descent,
    metrics). The split is a faithful port of numpy's
    default_rng(seed).permutation() (SeedSequence + PCG64 + numpy's exact
    bounded-rejection Fisher-Yates), verified bit-for-bit against real
    numpy across all 900 seeds x both classes of this dataset -- this is
    what keeps a given student's seed producing the *same* split/targets
    whether they run the local server or the static site.
  - static_api_shim.js -- mirrors the shape of each Flask JSON endpoint
    (/api/data, /api/assign, /api/evaluate, /api/seed_compare) using the
    pipeline above plus the exported patients.json / oracle_table.json.

Both .js files are copied into docs/assignment-1/ as-is, and a handful of
one-line surgical replacements swap run_sandbox.py's `fetch('/api/...')`
calls for calls into the shim -- everything else in the page (CSS, concept
cards, charts, tables, modal) is copied verbatim, so the static site always
matches the local UI exactly whenever this script is regenerated.

If run_sandbox.py's JS changes shape (e.g. one of the replaced functions is
rewritten), this script will raise a clear error rather than silently
producing a broken page -- update the surgical replacements below to match.

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
RUN_SANDBOX_FILE = ROOT / "run_sandbox.py"
CLIENT_PIPELINE_FILE = ROOT / "static_client_pipeline.js"
API_SHIM_FILE = ROOT / "static_api_shim.js"
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


def export_js():
    for src in (CLIENT_PIPELINE_FILE, API_SHIM_FILE):
        (OUT_DIR / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Copied {src.name} -> {OUT_DIR/src.name}")


def _extract_html_template(src: str) -> str:
    start_marker = 'HTML_TEMPLATE = r"""'
    start = src.index(start_marker) + len(start_marker)
    end_marker = '\nif __name__ == "__main__":'
    end_section_start = src.index(end_marker)
    template_end = src.rindex('"""', start, end_section_start)
    return src[start:template_end]


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"generate_docs_site.py: expected exactly 1 occurrence of the "
            f"{label!r} snippet in run_sandbox.py's HTML_TEMPLATE, found "
            f"{count}. run_sandbox.py's JS has likely changed shape -- "
            f"update this script's surgical replacements to match."
        )
    return text.replace(old, new, 1)


def build_interactive_page():
    src = RUN_SANDBOX_FILE.read_text(encoding="utf-8")
    html = _extract_html_template(src)
    html = html.replace("{{ add_timer }}", "0")

    # Swap each fetch('/api/...') call for a call into the local JS shim.
    # The rest of every function (UI updates, chart drawing, etc.) is left
    # completely untouched.
    html = _replace_once(
        html,
        "  const r = await fetch('/api/data');\n  const d = await r.json();",
        "  const d = await apiData();",
        "loadData",
    )
    html = _replace_once(
        html,
        "  const r = await fetch('/api/assign',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({student_id:id})});\n  const d = await r.json();",
        "  const d = await apiAssign(id);",
        "setStudentId",
    )
    html = _replace_once(
        html,
        "  const r = await fetch('/api/evaluate',{method:'POST',\n    headers:{'Content-Type':'application/json'},\n    body:JSON.stringify({seed:state.seed,lr,steps,val_fraction:vf})});\n  const d = await r.json();",
        "  const d = await apiEvaluate(state.seed, lr, steps, vf);",
        "runEval",
    )
    html = _replace_once(
        html,
        "    const r = await fetch('/api/seed_compare',{\n      method:'POST',\n      headers:{'Content-Type':'application/json'},\n      body: JSON.stringify({seed:state.seed, lr, steps, val_fraction}),\n    });\n    const d = await r.json();",
        "    const d = await apiSeedCompare(state.seed, lr, steps, val_fraction);",
        "loadSeedComparison",
    )
    html = _replace_once(
        html,
        "  const r=await fetch('/api/submit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});\n  const d=await r.json();",
        "  const d = await apiSubmit(payload);",
        "submitResult",
    )

    # Load the two shim scripts before the main inline <script>, so
    # apiData/apiAssign/etc. exist as soon as the main script's init
    # code (at the very bottom) calls them.
    html = _replace_once(
        html,
        '<script>\n// ============================================================\n// State',
        '<script src="static_client_pipeline.js"></script>\n'
        '<script src="static_api_shim.js"></script>\n'
        '<script>\n// ============================================================\n// State',
        "main <script> tag",
    )

    (OUT_DIR / "index.html").write_text(html, encoding="utf-8")
    print(f"Wrote interactive static explorer -> {OUT_DIR/'index.html'}")


def main():
    export_patients()
    export_oracle()
    export_js()
    build_interactive_page()


if __name__ == '__main__':
    main()
