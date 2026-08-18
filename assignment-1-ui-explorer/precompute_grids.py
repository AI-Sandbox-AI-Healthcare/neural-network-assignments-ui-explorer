#!/usr/bin/env python3
"""
Precompute parameter-grid evaluations for each seed in the oracle table
and export them to `docs/assignment-1/precomputed_grid.json` for a static UI.

This produces a mapping: seed -> list of evaluations, where each evaluation
is {"lr":..., "steps":..., "val_fraction":..., "auc":..., "accuracy":..., "f1":..., "loss":...}

Run from repository root:
  python assignment-1-ui-explorer/precompute_grids.py --out docs/assignment-1/precomputed_grid.json

Be mindful: this may take time depending on the number of seeds and grid size.
Use --limit to restrict number of seeds processed for a quick run.
"""
from pathlib import Path
import json
import argparse
import sys

ROOT = Path(__file__).parent
REPO_ROOT = ROOT.parent
DATA_FILE = ROOT / "data" / "patient_features.csv"
ORACLE_FILE = ROOT / "oracle_table.json"
OUT_DEFAULT = REPO_ROOT / "docs" / "assignment-1" / "precomputed_grid.json"

sys.path.insert(0, str(ROOT))  # so 'from reference import' works

def load_oracle():
    if not ORACLE_FILE.exists():
        raise SystemExit("oracle_table.json not found; run assignment-1-ui-explorer/generate_oracle.py first")
    return json.loads(ORACLE_FILE.read_text(encoding="utf-8"))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(OUT_DEFAULT))
    p.add_argument("--use-torch", action="store_true", help="Use PyTorch when available (slower)")
    p.add_argument("--limit", type=int, default=0, help="Limit number of seeds (0 = all)")
    args = p.parse_args()

    oracle = load_oracle()
    seeds = sorted(int(k) for k in oracle.keys())
    if args.limit and args.limit > 0:
        seeds = seeds[: args.limit]

    # Parameter grid (coarse but sufficient for UI simulation)
    lrs = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
    steps = [50, 100, 200, 500]
    vfs = [0.10, 0.20, 0.30]

    from reference import evaluate_seed
    import pandas as pd

    df = pd.read_csv(DATA_FILE)
    out = {}
    for s in seeds:
        print(f"Processing seed {s}...")
        evals = []
        for lr in lrs:
            for st in steps:
                for vf in vfs:
                    m = evaluate_seed(df, int(s), lr=lr, steps=st, val_fraction=vf, use_torch=args.use_torch)
                    evals.append({
                        "lr": lr, "steps": st, "val_fraction": vf,
                        "auc": m["auc"], "accuracy": m["accuracy"], "f1": m["f1"], "loss": m["final_loss"],
                    })
        out[str(s)] = evals

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote precomputed grid -> {out_path}")

if __name__ == '__main__':
    main()
