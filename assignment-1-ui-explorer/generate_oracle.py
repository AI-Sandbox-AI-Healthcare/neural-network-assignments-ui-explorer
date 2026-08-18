"""
assignment-1-ui-explorer/generate_oracle.py

Pre-compute oracle metrics for seeds 100-999 and write oracle_table.json.
Run once (instructor-only) from the repo root:
    python assignment-1-ui-explorer/generate_oracle.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).parent         # assignment-1-ui-explorer/
sys.path.insert(0, str(_HERE))        # so 'from reference import' works

from reference import evaluate_seed, OPTIMAL_LR, OPTIMAL_STEPS, OPTIMAL_VAL_FRACTION

SEEDS    = range(100, 1000)
OUT_FILE = _HERE / "oracle_table.json"
DATA_FILE = _HERE / "data" / "patient_features.csv"

# Seeds where the oracle AUC is a degenerate outlier (skip and use +1)
AUC_MIN = 0.74
AUC_MAX = 0.93


def main():
    print(f"Loading {DATA_FILE} ...")
    df = pd.read_csv(DATA_FILE)
    print(f"  {len(df)} patients loaded.")

    try:
        import torch  # noqa: F401
        use_torch = True
        print("  Using PyTorch oracle.")
    except ImportError:
        use_torch = False
        print("  torch not found -- using numpy oracle (same math).")

    table = {}
    skipped = []
    for seed in SEEDS:
        m = evaluate_seed(df, seed, OPTIMAL_LR, OPTIMAL_STEPS, OPTIMAL_VAL_FRACTION,
                          use_torch=use_torch)
        auc = m["auc"]
        if auc < AUC_MIN or auc > AUC_MAX:
            skipped.append((seed, auc))
        # Store without the large loss_history to keep file small
        table[str(seed)] = {k: v for k, v in m.items() if k != "loss_history"}
        if seed % 100 == 99:
            print(f"  seeds {seed - 99}-{seed} done")

    OUT_FILE.write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(f"\noracle_table.json written ({len(table)} seeds).")
    if skipped:
        print(f"  WARNING: {len(skipped)} seeds with AUC outside [{AUC_MIN},{AUC_MAX}]:")
        for s, a in skipped[:10]:
            print(f"    seed {s}: AUC={a:.3f}")


if __name__ == "__main__":
    main()
