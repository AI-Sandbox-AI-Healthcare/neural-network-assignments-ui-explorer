#!/usr/bin/env python3
"""
Build the full interactive static explorer in ./docs/assignment-2/ for GitHub Pages.

GitHub Pages has no Flask backend, so this writes a self-contained page that uses
the SAME HTML/CSS/JS as run_sandbox.py (extracted verbatim from its HTML_TEMPLATE)
but answers /api/* in the browser via static_api_shim.js:

  data.json          the /api/data payload (320 patients + labels)
  oracle_table.json  copied verbatim -> exact per-seed Random-Forest oracle target
  arena.json         every Architecture-Arena config precomputed (exact simulation)
  static_api_shim.js overrides window.fetch; /api/data, /api/assign and /api/arena
                     are exact, /api/evaluate is a modelled surface anchored to the
                     real oracle (a live sklearn fit can't run in a browser).

Run from the repo root:  python assignment-2-ui-explorer/generate_docs_site.py
Then publish the ./docs/ folder with GitHub Pages.
"""
import itertools
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
OUT = REPO_ROOT / "docs" / "assignment-2"
RUN_SANDBOX = HERE / "run_sandbox.py"
ORACLE = HERE / "oracle_table.json"
DATA_CSV = HERE / "data" / "patient_features.csv"
SHIM = HERE / "static_api_shim.js"

PAIN_KEYWORDS = [
    "chronic", "pain", "arthritis", "osteoarthritis", "rheumatoid",
    "fibromyalgia", "migraine", "neuropathy", "neuralgia",
    "sciatica", "back pain", "neck pain", "spinal", "fracture",
    "injury", "burn", "wound", "trauma", "sprain", "strain",
    "tendon", "ligament", "joint", "osteoporosis", "gout",
    "lupus", "paralysis", "amputation", "surgery", "postoperative", "whiplash",
]


def build_index_html():
    src = RUN_SANDBOX.read_text(encoding="utf-8")
    start = src.index('HTML_TEMPLATE = r"""') + len('HTML_TEMPLATE = r"""')
    end = src.index('\nif __name__ == "__main__":')
    html = src[start:src.rindex('"""', start, end)]
    if "{{" in html or "{%" in html:
        sys.exit("generate_docs_site.py: unexpected Jinja placeholder in HTML_TEMPLATE")
    marker = "\n<script>\nconst state="
    if html.count(marker) != 1:
        sys.exit("generate_docs_site.py: main <script> tag not found / changed shape")
    html = html.replace(marker, '\n<script src="static_api_shim.js"></script>' + marker, 1)
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print("  index.html")


def build_data_json():
    import pandas as pd
    df = pd.read_csv(DATA_CSV)
    rx = re.compile(r"\b(?:" + "|".join(re.escape(k) for k in PAIN_KEYWORDS) + r")\b")
    labels = [1 if rx.search(str(t).lower()) else 0 for t in df["condition_text"]]
    n_pos = sum(labels)
    payload = {
        "patients": df.to_dict(orient="records"),
        "columns": list(df.columns),
        "labels": labels,
        "n_total": len(labels),
        "n_positive": n_pos,
        "n_negative": len(labels) - n_pos,
        "positive_rate": round(n_pos / len(labels), 3),
    }
    (OUT / "data.json").write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"  data.json ({len(labels)} patients)")


def build_arena_json():
    sys.path.insert(0, str(HERE))
    from reference import simulate_arena_panel
    cfgs = [
        {"preset": p, "activation": a, "dropout": d, "early_stopping": e}
        for p, a, d, e in itertools.product(
            ["deep", "wide"], ["relu", "tanh"], [False, True], [False, True])
    ]
    key = lambda c: (c["preset"][0] + c["activation"][0]
                     + ("D" if c["dropout"] else "d")
                     + ("E" if c["early_stopping"] else "e"))
    table = {}
    for seed in range(100, 1000):
        row = {}
        for c in cfgs:
            for panel in ("A", "B"):
                m = simulate_arena_panel(seed, c, panel)
                row[key(c) + panel] = [m["train_loss"], m["val_loss"], m["val_f1"],
                                       m["stopped_epoch"], m["best_epoch"]]
        table[str(seed)] = row
    txt = json.dumps(table, separators=(",", ":"))
    (OUT / "arena.json").write_text(txt, encoding="utf-8")
    print(f"  arena.json ({len(table)} seeds, {len(txt) / 1e6:.1f} MB)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not ORACLE.exists():
        sys.exit("oracle_table.json missing -- run generate_oracle.py first")
    (OUT / "oracle_table.json").write_text(ORACLE.read_text(encoding="utf-8"), encoding="utf-8")
    print("  oracle_table.json")
    (OUT / "static_api_shim.js").write_text(SHIM.read_text(encoding="utf-8"), encoding="utf-8")
    print("  static_api_shim.js")
    build_data_json()
    build_arena_json()
    build_index_html()
    print(f"\ndocs/assignment-2/ ready -- publish ./docs/ with GitHub Pages.")


if __name__ == "__main__":
    main()
