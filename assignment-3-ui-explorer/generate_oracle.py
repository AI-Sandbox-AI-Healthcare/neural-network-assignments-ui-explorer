"""
assignment-3-ui-explorer/generate_oracle.py   (INSTRUCTOR-ONLY)

Pre-compute the per-seed sequence-model oracle target for seeds 100-999 and
write oracle_table.json. Run once from anywhere:

    python assignment-3-ui-explorer/generate_oracle.py

run_sandbox.py also computes any missing seed lazily on first request, so a
partial table still works. The oracle is the ANALYTIC surrogate from
reference.py (a real per-seed PyTorch search would take hours); the hidden
grader trains the real reference LSTM live and only requires the student's model
to land within tolerance of it.
"""
import json
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from reference import (  # noqa: E402
    AUC_MAX, AUC_MIN, VISIT_FEATURE_COLS, oracle_metrics,
)

SEEDS = range(100, 1000)
OUT_FILE = _HERE / "oracle_table.json"
FEATURES_FILE = _HERE / "data" / "patient_features.csv"
VISITS_FILE = _HERE / "data" / "patient_visits.csv"


def main():
    print(f"Loading {FEATURES_FILE.name} / {VISITS_FILE.name} ...")
    fdf = pd.read_csv(FEATURES_FILE)
    vdf = pd.read_csv(VISITS_FILE)
    vdf["patient_id"] = vdf["patient_id"].astype(str)
    for c in VISIT_FEATURE_COLS:
        vdf[c] = vdf[c].astype(float)
    print(f"  {len(fdf)} patients, {len(vdf)} visit rows.")

    table, skipped = {}, []
    for seed in SEEDS:
        m = oracle_metrics(fdf, vdf, seed)
        table[str(seed)] = m
        if not (AUC_MIN <= m["auc"] <= AUC_MAX):
            skipped.append((seed, m["auc"]))
        if seed % 100 == 99:
            print(f"  seeds {seed - 99}-{seed} done")

    OUT_FILE.write_text(json.dumps(table, indent=1), encoding="utf-8")
    print(f"\noracle_table.json written ({len(table)} seeds).")
    if skipped:
        print(f"  {len(skipped)} seeds outside AUC [{AUC_MIN}, {AUC_MAX}] "
              f"(the UI walks to the next in-band seed):")
        for s, a in skipped[:10]:
            print(f"    seed {s}: AUC={a:.3f}")


if __name__ == "__main__":
    main()
