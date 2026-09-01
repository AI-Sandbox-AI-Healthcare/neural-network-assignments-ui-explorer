"""
assignment-2-ui-explorer/generate_oracle.py

Pre-compute the Random Forest oracle target for seeds 100-999 and write
oracle_table.json. Run once (instructor-only) from the repo root:

    python assignment-2-ui-explorer/generate_oracle.py

run_sandbox.py also computes any missing seed lazily on first request, so a
partial table still works.
"""
import json
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from reference import best_config_for_seed  # noqa: E402

SEEDS = range(100, 1000)
OUT_FILE = _HERE / "oracle_table.json"
DATA_FILE = _HERE / "data" / "patient_features.csv"

AUC_MIN, AUC_MAX = 0.80, 0.97


def main():
    print(f"Loading {DATA_FILE} ...")
    df = pd.read_csv(DATA_FILE)
    print(f"  {len(df)} patients loaded.")

    table, skipped = {}, []
    for seed in SEEDS:
        m = best_config_for_seed(df, seed)
        if not (AUC_MIN <= m["auc"] <= AUC_MAX):
            skipped.append((seed, m["auc"]))
        table[str(seed)] = m
        if seed % 100 == 99:
            print(f"  seeds {seed - 99}-{seed} done")

    OUT_FILE.write_text(json.dumps(table, indent=1), encoding="utf-8")
    print(f"\noracle_table.json written ({len(table)} seeds).")
    if skipped:
        print(f"  {len(skipped)} seeds outside AUC [{AUC_MIN}, {AUC_MAX}]:")
        for s, a in skipped[:10]:
            print(f"    seed {s}: AUC={a:.3f}")


if __name__ == "__main__":
    main()
