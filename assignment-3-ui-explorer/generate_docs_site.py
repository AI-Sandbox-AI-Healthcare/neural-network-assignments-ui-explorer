#!/usr/bin/env python3
"""
Build the full interactive static explorer in ./docs/assignment-3/ for GitHub Pages.

GitHub Pages has no Flask backend, so this writes a self-contained page that uses
the SAME HTML/CSS/JS as run_sandbox.py (extracted verbatim from its HTML_TEMPLATE)
but answers /api/* in the browser via static_api_shim.js:

  data.json          the /api/data payload (320 patients + labels + visit rows)
  oracle_table.json  copied verbatim -> exact per-seed sequence-model oracle target
  static_api_shim.js overrides window.fetch; /api/data and /api/assign are exact,
                     /api/evaluate is a modelled surface anchored to each seed's
                     precomputed oracle (a live PyTorch fit can't run in a browser).

Run from anywhere:  python assignment-3-ui-explorer/generate_docs_site.py
Then publish the ./docs/ folder with GitHub Pages.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
OUT = REPO_ROOT / "docs" / "assignment-3"
RUN_SANDBOX = HERE / "run_sandbox.py"
ORACLE = HERE / "oracle_table.json"
FEATURES_CSV = HERE / "data" / "patient_features.csv"
VISITS_CSV = HERE / "data" / "patient_visits.csv"
SHIM = HERE / "static_api_shim.js"


SHIM_TAG = '<script src="static_api_shim.js"></script>'


def _extract_template():
    src = RUN_SANDBOX.read_text(encoding="utf-8")  # normalises newlines to \n
    start = src.index('HTML_TEMPLATE = r"""') + len('HTML_TEMPLATE = r"""')
    end = src.index('\nif __name__ == "__main__":')
    return src[start:src.rindex('"""', start, end)]


def build_index_html():
    html = _extract_template()
    if "{{" in html or "{%" in html:
        sys.exit("generate_docs_site.py: unexpected Jinja placeholder in HTML_TEMPLATE")
    marker = "\n<script>\nconst state="
    if html.count(marker) != 1:
        sys.exit("generate_docs_site.py: main <script> tag not found / changed shape")
    html = html.replace(marker, "\n" + SHIM_TAG + marker, 1)
    # write LF exactly -- no platform newline translation
    (OUT / "index.html").write_bytes(html.encode("utf-8"))
    print("  index.html")


def build_data_json():
    import pandas as pd
    sys.path.insert(0, str(HERE))
    from reference import MAX_VISITS, VISIT_FEATURE_COLS, _labels

    fdf = pd.read_csv(FEATURES_CSV)
    vdf = pd.read_csv(VISITS_CSV)
    vdf["patient_id"] = vdf["patient_id"].astype(str)

    labels = _labels(fdf["condition_text"].tolist())
    n_pos = int(labels.sum())
    counts = vdf.groupby("patient_id", sort=False).size()
    counts = counts.reindex(fdf["id"].astype(str)).fillna(0).astype(int)
    hist = [int((counts == k).sum()) for k in range(1, MAX_VISITS + 1)]
    visits = [
        [str(r.patient_id), int(r.visit_number), int(r.days_since_first_visit),
         float(r.pain_score_at_visit), int(r.medications_at_visit), int(r.visit_type_code)]
        for r in vdf.itertuples(index=False)
    ]
    payload = {
        "patients": fdf.to_dict(orient="records"),
        "columns": list(fdf.columns),
        "labels": labels.tolist(),
        "visits": visits,
        "seq_len_hist": hist,
        "n_total": len(labels),
        "n_positive": n_pos,
        "n_negative": len(labels) - n_pos,
        "positive_rate": round(n_pos / len(labels), 3),
        "max_visits": MAX_VISITS,
        "visit_feature_cols": list(VISIT_FEATURE_COLS),
    }
    (OUT / "data.json").write_bytes(
        json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    print(f"  data.json ({len(labels)} patients, {len(visits)} visit rows)")


def verify():
    """The docs page must be run_sandbox.py's HTML_TEMPLATE verbatim, plus one
    injected <script src> line, and the copied assets must be byte-identical."""
    tpl = _extract_template().encode("utf-8")
    inject = ("\n" + SHIM_TAG).encode("utf-8")
    got = (OUT / "index.html").read_bytes()
    assert got == tpl.replace(b"\n<script>\nconst state=",
                              inject + b"\n<script>\nconst state=", 1), \
        "docs/index.html is not HTML_TEMPLATE + the single shim <script> line"
    assert got.replace(inject, b"", 1) == tpl, \
        "docs/index.html differs from the local page by more than the shim line"
    assert (OUT / "static_api_shim.js").read_bytes() == SHIM.read_bytes(), \
        "docs static_api_shim.js is not a verbatim copy of the local one"
    assert (OUT / "oracle_table.json").read_bytes() == ORACLE.read_bytes(), \
        "docs oracle_table.json is not a verbatim copy of the local one"
    print("  verified: docs/assignment-3/ == run_sandbox.py's page + one line ("
          + SHIM_TAG + ")")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not ORACLE.exists():
        sys.exit("oracle_table.json missing -- run generate_oracle.py first")
    (OUT / "oracle_table.json").write_bytes(ORACLE.read_bytes())
    print("  oracle_table.json")
    (OUT / "static_api_shim.js").write_bytes(SHIM.read_bytes())
    print("  static_api_shim.js")
    build_data_json()
    build_index_html()
    verify()
    print("\ndocs/assignment-3/ ready -- publish ./docs/ with GitHub Pages.")


if __name__ == "__main__":
    main()
