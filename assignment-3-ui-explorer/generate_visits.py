"""
assignment-3-ui-explorer/generate_visits.py  (INSTRUCTOR-ONLY)

Turns the static 320-patient table (data/patient_features.csv) into a
long-format clinical-visit table (data/patient_visits.csv) for Assignment 3.

Every patient gets 1-6 visits.  Each visit carries four numeric per-visit
measurements:

    days_since_first_visit   int   0 at visit 1, monotonically increasing
    pain_score_at_visit      float 0.0 - 10.0, a NOISY reading of the patient's
                                   underlying pain level (single visit is weak;
                                   the average over a few visits is strong)
    medications_at_visit     int   0 - 9, prescriptions written at that visit
    visit_type_code          int   1 - 5  (1 routine, 2 urgent, 3 specialist,
                                   4 procedure, 5 telehealth) -- never 0, so a
                                   real visit is never an all-zero row and a
                                   Keras Masking(mask_value=0.0) layer only ever
                                   masks true padding.

The chronic-pain label is the SAME keyword-derived label as Assignments 1-2
(rebuilt from condition_text); it is not written to this file -- the pipeline
and the grader rebuild it from patient_features.csv, exactly as before.

Generation is fully deterministic: each patient's visits come from
np.random.default_rng(sha256("a3-visits|<id>")).  Re-running this script
reproduces data/patient_visits.csv byte-for-byte.

Run once from anywhere:

    python assignment-3-ui-explorer/generate_visits.py
"""
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

VISIT_SALT = "a3-visits"
MAX_VISITS = 6

PAIN_KEYWORDS = [
    "chronic", "pain", "arthritis", "osteoarthritis", "rheumatoid",
    "fibromyalgia", "migraine", "neuropathy", "neuralgia",
    "sciatica", "back pain", "neck pain", "spinal", "fracture",
    "injury", "burn", "wound", "trauma", "sprain", "strain",
    "tendon", "ligament", "joint", "osteoporosis", "gout",
    "lupus", "paralysis", "amputation", "surgery", "postoperative", "whiplash",
]

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent

# All the data/ folders that must carry an identical copy of the CSV.
FEATURES_SOURCES = [
    _REPO_ROOT / "neural-network-assignment-2" / "data" / "patient_features.csv",
    _HERE / "data" / "patient_features.csv",
]
VISIT_TARGETS = [
    _HERE / "data" / "patient_visits.csv",
    _REPO_ROOT / "neural-network-assignment-3" / "data" / "patient_visits.csv",
    _REPO_ROOT / "private-instructor-graders" / "INFO-557" / "hw03" / "v1" / "patient_visits.csv",
]
FEATURE_COPY_TARGETS = [
    _HERE / "data" / "patient_features.csv",
    _REPO_ROOT / "neural-network-assignment-3" / "data" / "patient_features.csv",
    _REPO_ROOT / "private-instructor-graders" / "INFO-557" / "hw03" / "v1" / "patient_features.csv",
]


def _label_from_text(text: str) -> int:
    pattern = r"\b(?:" + "|".join(re.escape(k) for k in PAIN_KEYWORDS) + r")\b"
    return 1 if re.compile(pattern).search(str(text).lower()) else 0


def _patient_rng(pid: str) -> np.random.Generator:
    h = int(hashlib.sha256(f"{VISIT_SALT}|{pid}".encode()).hexdigest(), 16)
    return np.random.default_rng(h % (2 ** 32))


def _find_features_csv() -> Path:
    for p in FEATURES_SOURCES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "patient_features.csv not found in any known location: "
        + ", ".join(str(p) for p in FEATURES_SOURCES)
    )


def _visit_type(rng: np.random.Generator, pain: float, label: int) -> int:
    """1 routine, 2 urgent, 3 specialist, 4 procedure, 5 telehealth."""
    w = np.array([0.40, 0.14, 0.16, 0.06, 0.24], dtype=float)      # baseline mix
    if pain >= 7.0:
        w += np.array([-0.18, 0.16, 0.16, 0.10, -0.14])
    elif pain >= 4.0:
        w += np.array([-0.06, 0.06, 0.08, 0.02, -0.06])
    if label:
        w += np.array([-0.05, 0.03, 0.06, 0.03, -0.05])
    w = np.clip(w, 0.02, None)
    w /= w.sum()
    return int(rng.choice([1, 2, 3, 4, 5], p=w))


def build_visits(features_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, patient in features_df.iterrows():
        pid = str(patient["id"])
        rng = _patient_rng(pid)
        label = _label_from_text(patient["condition_text"])
        severity = float(patient["pain_severity"])                 # 0 - 10
        encounters = float(patient["number_of_encounters"])

        # --- how many visits (1-6): pain patients come back more often --------
        lam = 2.7 + 0.05 * encounters + (1.5 if label else 0.0)
        n_visits = int(np.clip(1 + rng.poisson(max(lam - 1.0, 0.05)), 1, MAX_VISITS))

        # --- the patient's underlying pain level + a worsening trend ----------
        # The chronic-pain label lifts the whole visit history: an explicit
        # offset plus a fraction of the static pain_severity feature, then per
        # visit Gaussian noise on top.  Single visits stay noisy; the mean over
        # a few visits separates the classes cleanly (~AUC 0.9).
        base_pain = float(np.clip(
            0.35 * severity + (2.6 if label else 1.0) + 1.0 + rng.normal(0.0, 0.7),
            0.0, 10.0,
        ))
        trend = float(rng.normal(0.55 if label else 0.02, 0.35))   # per-visit drift

        day = 0
        for k in range(n_visits):
            if k > 0:
                day += int(rng.integers(14, 121))                  # 2 weeks - 4 months
            # single-visit reading is deliberately NOISY (std ~2.5 on a 0-10
            # scale) so one visit is a weak signal (~AUC 0.73) but the mean
            # over a few is strong (~AUC 0.87) -- this is what makes pooling /
            # recurrence worthwhile, and keeps the personal oracle target
            # inside the 0.80-0.97 "good seed" band with room to tune.
            pain = base_pain + trend * k + float(rng.normal(0.0, 2.5))
            pain = round(float(np.clip(pain, 0.0, 10.0)), 1)
            meds = int(np.clip(
                rng.poisson(0.5 + 0.25 * pain + (0.6 if label else 0.0)), 0, 9
            ))
            vtype = _visit_type(rng, pain, label)
            rows.append({
                "patient_id": pid,
                "visit_number": k + 1,
                "days_since_first_visit": day,
                "pain_score_at_visit": pain,
                "medications_at_visit": meds,
                "visit_type_code": vtype,
            })

    return pd.DataFrame(rows, columns=[
        "patient_id", "visit_number", "days_since_first_visit",
        "pain_score_at_visit", "medications_at_visit", "visit_type_code",
    ])


def _report(features_df: pd.DataFrame, visits_df: pd.DataFrame) -> None:
    from collections import Counter

    labels = features_df["condition_text"].map(_label_from_text).to_numpy()
    per_patient = visits_df.groupby("patient_id", sort=False)
    counts = per_patient.size()
    hist = Counter(counts.tolist())
    total_slots = len(counts) * MAX_VISITS
    real_slots = int(counts.sum())

    # crude separability check: mean pain per patient vs. label
    mean_pain = per_patient["pain_score_at_visit"].mean()
    mean_pain = mean_pain.reindex(features_df["id"].astype(str)).to_numpy()
    pos, neg = mean_pain[labels == 1], mean_pain[labels == 0]
    auc = np.mean([1.0 * (a > b) + 0.5 * (a == b)
                   for a in pos for b in neg])

    print(f"  patients          : {len(counts)}")
    print(f"  visit rows        : {len(visits_df)}")
    print(f"  visits/patient    : "
          + ", ".join(f"{k}:{hist.get(k, 0)}" for k in range(1, MAX_VISITS + 1)))
    print(f"  padding @ max=6   : {100 * (1 - real_slots / total_slots):.1f}% of slots")
    print(f"  label balance     : {int(labels.sum())} pos / {int((1 - labels).sum())} neg")
    print(f"  mean-pain vs label AUC (linear-probe ceiling): {auc:.3f}")


def main() -> None:
    src = _find_features_csv()
    print(f"Loading {src} ...")
    features_df = pd.read_csv(src)
    print(f"  {len(features_df)} patients.")

    visits_df = build_visits(features_df)
    _report(features_df, visits_df)

    for target in VISIT_TARGETS:
        target.parent.mkdir(parents=True, exist_ok=True)
        visits_df.to_csv(target, index=False)
        print(f"  wrote {target}")
    for target in FEATURE_COPY_TARGETS:
        if target.resolve() == src.resolve():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(src.read_bytes())
        print(f"  copied features -> {target}")


if __name__ == "__main__":
    main()
