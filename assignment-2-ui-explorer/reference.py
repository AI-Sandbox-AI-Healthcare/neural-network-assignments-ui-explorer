"""
assignment-2-ui-explorer/reference.py

Random Forest oracle + full imbalance-handling pipeline, plus a deterministic
feed-forward "Architecture Arena" training simulation.

Independent of the student's pipeline.py -- used to pre-compute and serve
optimal results for each student seed. The oversampling algorithm here is a
byte-for-byte match of pipeline.random_oversample() (with a tunable ratio),
which is why oversample_ratio in get_sandbox_params() must match the UI.
"""
import hashlib
import math
import re

import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------------------------
# Shared constants (same dataset / label as Assignment 1)
# ---------------------------------------------------------------------------
PAIN_KEYWORDS = [
    "chronic", "pain", "arthritis", "osteoarthritis", "rheumatoid",
    "fibromyalgia", "migraine", "neuropathy", "neuralgia",
    "sciatica", "back pain", "neck pain", "spinal", "fracture",
    "injury", "burn", "wound", "trauma", "sprain", "strain",
    "tendon", "ligament", "joint", "osteoporosis", "gout",
    "lupus", "paralysis", "amputation", "surgery", "postoperative", "whiplash",
]

FEATURE_COLS = [
    "age", "is_female", "number_of_unique_meds", "number_of_encounters",
    "number_of_procedures", "unique_procedures", "pain_severity",
    "body_height", "body_weight", "body_mass_index",
    "systolic_blood_pressure", "diastolic_blood_pressure",
    "heart_rate", "respiratory_rate",
    "qaly", "daly", "qols", "healthcare_expenses", "healthcare_coverage",
]

# Oracle search space -- generate_oracle.py picks the best cell per seed.
GRID_N_ESTIMATORS = [80, 140, 200]
GRID_MAX_DEPTH = [6, 10]
OPTIMAL_OVERSAMPLE_RATIO = 1.0
OPTIMAL_VAL_FRACTION = 0.20
THRESHOLD = 0.5


def student_to_seed(student_id: str) -> int:
    """Deterministic seed from student ID, independent of PYTHONHASHSEED."""
    h = int(hashlib.sha256(student_id.lower().strip().encode()).hexdigest(), 16)
    return h % 900 + 100


# ---------------------------------------------------------------------------
# Pipeline pieces
# ---------------------------------------------------------------------------
def _flag_keywords(descriptions):
    pattern = r"\b(?:" + "|".join(re.escape(k) for k in PAIN_KEYWORDS) + r")\b"
    compiled = re.compile(pattern)
    return np.array([bool(compiled.search(d.lower())) for d in descriptions])


def _stratified_split(labels, val_fraction, seed):
    rng = np.random.default_rng(seed)
    is_val = np.zeros(len(labels), dtype=bool)
    for cls in sorted(np.unique(labels)):
        idx = np.where(labels == cls)[0]
        shuffled = rng.permutation(idx)
        n_val = round(val_fraction * len(idx))
        is_val[shuffled[:n_val]] = True
    return is_val


def class_weights(labels):
    labels = np.asarray(labels)
    counts = np.bincount(labels)
    n_samples = len(labels)
    present = np.unique(labels)
    return {int(c): float(n_samples / (len(present) * counts[c])) for c in present}


def random_oversample(features, labels, seed, ratio=1.0):
    """Duplicate minority rows until minority_count ~= ratio * majority_count.

    ratio == 1.0 fully balances (identical to pipeline.random_oversample).
    ratio == 0.0 leaves the data untouched.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    majority = int(np.bincount(labels).max())
    target = int(round(max(0.0, min(1.0, ratio)) * majority))
    feats_parts = [features]
    label_parts = [labels]
    for cls in sorted(np.unique(labels)):
        idx = np.where(labels == cls)[0]
        deficit = target - len(idx)
        if deficit > 0:
            extra = rng.choice(idx, size=deficit, replace=True)
            feats_parts.append(features[extra])
            label_parts.append(labels[extra])
    return np.concatenate(feats_parts), np.concatenate(label_parts)


def _auc(y_true, scores):
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    total = 0.0
    for sp in pos:
        for sn in neg:
            if sp > sn:
                total += 1.0
            elif sp == sn:
                total += 0.5
    return total / (len(pos) * len(neg))


def _roc_curve_points(y_true, probs, n=30):
    pts = []
    for t in np.linspace(1.0, 0.0, n):
        yp = (probs >= t).astype(int)
        tp = int(np.sum((yp == 1) & (y_true == 1)))
        fp = int(np.sum((yp == 1) & (y_true == 0)))
        fn = int(np.sum((yp == 0) & (y_true == 1)))
        tn = int(np.sum((yp == 0) & (y_true == 0)))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        pts.append([round(fpr, 3), round(tpr, 3)])
    return pts


def evaluate_seed(df, seed, n_estimators, max_depth, oversample_ratio,
                  val_fraction=OPTIMAL_VAL_FRACTION):
    """Full imbalance + Random Forest pipeline for one seed / config."""
    labels = _flag_keywords(df["condition_text"].tolist()).astype(int)
    features = df[FEATURE_COLS].values.astype(float)

    is_val = _stratified_split(labels, val_fraction, seed)
    x_train, y_train = features[~is_val], labels[~is_val]
    x_val, y_val = features[is_val], labels[is_val]

    x_bal, y_bal = random_oversample(x_train, y_train, seed, oversample_ratio)
    model = RandomForestClassifier(
        n_estimators=int(n_estimators), max_depth=int(max_depth), random_state=seed
    )
    model.fit(x_bal, y_bal)

    probs = model.predict_proba(x_val)[:, 1]
    y_pred = (probs >= THRESHOLD).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_val == 1)))
    fp = int(np.sum((y_pred == 1) & (y_val == 0)))
    tn = int(np.sum((y_pred == 0) & (y_val == 0)))
    fn = int(np.sum((y_pred == 0) & (y_val == 1)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / len(y_val)
    auc = _auc(y_val, probs)

    train_pred = model.predict(x_bal)
    train_acc = float(np.mean(train_pred == y_bal))

    ranking = sorted(
        zip(FEATURE_COLS, [float(v) for v in model.feature_importances_]),
        key=lambda t: t[1], reverse=True,
    )

    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    return {
        "n_estimators": int(n_estimators),
        "max_depth": int(max_depth),
        "oversample_ratio": round(float(oversample_ratio), 2),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(prec, 4), "recall": round(rec, 4),
        "f1": round(f1, 4), "accuracy": round(acc, 4), "auc": round(auc, 4),
        "train_accuracy": round(train_acc, 4),
        "train_n": int(np.sum(~is_val)), "val_n": int(np.sum(is_val)),
        "train_pos": n_pos, "train_neg": n_neg,
        "balanced_n": int(len(y_bal)),
        "feature_importances": [[n, round(v, 4)] for n, v in ranking],
        "roc_curve": _roc_curve_points(y_val, probs),
        "class_weights": {str(k): round(v, 4)
                          for k, v in class_weights(y_train).items()},
    }


def best_config_for_seed(df, seed):
    """Grid-search the oracle target for one seed (simpler models win ties)."""
    best = None
    for n_estimators in GRID_N_ESTIMATORS:
        for max_depth in GRID_MAX_DEPTH:
            m = evaluate_seed(df, seed, n_estimators, max_depth,
                              OPTIMAL_OVERSAMPLE_RATIO)
            key = (-m["auc"], max_depth, n_estimators)
            if best is None or key < best[0]:
                best = (key, m)
    return best[1]


# ---------------------------------------------------------------------------
# Architecture Arena -- deterministic feed-forward training simulation
# ---------------------------------------------------------------------------
ARENA_PRESETS = {
    "deep": [32, 32, 32, 16],
    "wide": [96],
}
ARENA_EPOCHS = 40
EARLY_STOP_PATIENCE = 5


def _arena_seed(seed, cfg, panel):
    key = f"{seed}|{panel}|{cfg['preset']}|{cfg['activation']}|{int(cfg['dropout'])}|{int(cfg['early_stopping'])}"
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2 ** 32)


def simulate_arena_panel(seed, cfg, panel):
    """Return plausible, fully deterministic train/val loss curves + val F1.

    Not a real PyTorch run -- a closed-form model of how depth/width,
    activation, dropout and early stopping change the loss curves, matching
    what students see when they later train build_feedforward_model() for real.
    """
    rng = np.random.default_rng(_arena_seed(seed, cfg, panel))
    preset = cfg["preset"]
    wide = preset == "wide"
    tanh = cfg["activation"] == "tanh"
    dropout = bool(cfg["dropout"])
    early = bool(cfg["early_stopping"])

    # Capacity / optimisation character of the configuration.
    start = 0.70 + 0.03 * rng.standard_normal()
    floor = 0.30 + (0.015 if wide else 0.0) + (0.02 if tanh else 0.0)
    floor -= 0.01 if not dropout else 0.0            # dropout raises train loss slightly
    train_rate = 0.16 + (0.03 if wide else 0.0) - (0.02 if tanh else 0.0)
    # Overfitting pressure: wide + tanh + no-dropout overfit hardest.
    overfit = 0.11
    overfit += 0.05 if wide else 0.0
    overfit += 0.03 if tanh else 0.0
    overfit -= 0.09 if dropout else 0.0
    overfit = max(0.0, overfit)
    overfit_onset = 8 + (4 if dropout else 0) + rng.integers(0, 3)

    train_loss, val_loss = [], []
    best_val, best_epoch, wait = math.inf, 0, 0
    stopped_epoch = ARENA_EPOCHS
    for epoch in range(1, ARENA_EPOCHS + 1):
        decay = math.exp(-train_rate * epoch)
        t = floor + (start - floor) * decay + 0.006 * rng.standard_normal()
        gap = overfit * max(0.0, 1 - math.exp(-(max(0, epoch - overfit_onset)) / 6.0))
        v = t + gap + 0.03 + 0.008 * rng.standard_normal()
        t = max(0.05, t)
        v = max(0.06, v)
        train_loss.append(round(t, 4))
        val_loss.append(round(v, 4))
        if v < best_val - 1e-4:
            best_val, best_epoch, wait = v, epoch, 0
        else:
            wait += 1
        if early and wait >= EARLY_STOP_PATIENCE and stopped_epoch == ARENA_EPOCHS:
            stopped_epoch = epoch
            break

    final_val_loss = val_loss[best_epoch - 1] if early else val_loss[-1]
    # Map the achieved val loss onto a validation F1 in a believable band. The
    # best configuration for any seed clears the "Try your own pair" goal of
    # F1 >= 0.787; the worst sits around 0.45-0.55.
    val_f1 = 0.90 - 1.5 * (final_val_loss - 0.30)
    val_f1 = float(np.clip(val_f1 + 0.01 * rng.standard_normal(), 0.35, 0.90))

    return {
        "preset": preset,
        "hidden_units": ARENA_PRESETS[preset],
        "activation": cfg["activation"],
        "dropout": dropout,
        "early_stopping": early,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "stopped_epoch": int(stopped_epoch),
        "best_epoch": int(best_epoch),
        "val_f1": round(val_f1, 4),
        "final_val_loss": round(final_val_loss, 4),
    }


def _describe(panel):
    bits = [
        "wide" if panel["preset"] == "wide" else "deep",
        panel["activation"],
        "dropout" if panel["dropout"] else "no dropout",
        "early stopping" if panel["early_stopping"] else "no early stopping",
    ]
    return ", ".join(bits)


def arena_caption(a, b):
    """One-sentence 'what changed?' narration comparing the two panels."""
    def verdict(p):
        overfit_gap = p["val_loss"][-1] - p["train_loss"][-1]
        if p["early_stopping"] and p["stopped_epoch"] < ARENA_EPOCHS:
            return f"stopped early at epoch {p['stopped_epoch']} with val F1 {p['val_f1']:.3f}"
        if overfit_gap > 0.12:
            return f"overfit after ~{p['best_epoch']} epochs (val F1 {p['val_f1']:.3f})"
        return f"kept validation loss flat, ending at val F1 {p['val_f1']:.3f}"

    winner = "A" if a["val_f1"] >= b["val_f1"] else "B"
    return (
        f"Model A ({_describe(a)}) {verdict(a)}; "
        f"Model B ({_describe(b)}) {verdict(b)}. "
        f"Higher validation F1: Model {winner}."
    )


def run_arena(seed, cfg_a, cfg_b):
    a = simulate_arena_panel(seed, cfg_a, "A")
    b = simulate_arena_panel(seed, cfg_b, "B")
    return {"a": a, "b": b, "caption": arena_caption(a, b), "epochs": ARENA_EPOCHS}
