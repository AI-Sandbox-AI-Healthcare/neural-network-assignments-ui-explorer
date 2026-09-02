"""
assignment-3-ui-explorer/reference.py

Sequence-model oracle + a deterministic "Recurrent vs. Convolutional" arena
simulation for Assignment 3.

Independent of the student's pipeline.py. A real PyTorch LSTM cannot train in a
browser (or 900x in a pre-compute pass), so the metrics here are an ANALYTIC
surrogate: a masked-mean-pool + logistic probe gives each seed / padding length
its achievable ceiling, and closed-form quality factors model how cell type,
hidden width and bidirectionality move a trained recurrent model around that
ceiling. The always-on Conv1D baseline sits a little below the recurrent model
because part of this dataset's signal is in the ORDER and RECENCY of visits.

The oracle architecture (oracle_config) is byte-for-byte the same function the
hidden grader's solutions.py uses, so "optimal in the UI" == "the reference the
autograder trains".
"""
import hashlib
import math
import re

import numpy as np

PAIN_KEYWORDS = [
    "chronic", "pain", "arthritis", "osteoarthritis", "rheumatoid",
    "fibromyalgia", "migraine", "neuropathy", "neuralgia",
    "sciatica", "back pain", "neck pain", "spinal", "fracture",
    "injury", "burn", "wound", "trauma", "sprain", "strain",
    "tendon", "ligament", "joint", "osteoporosis", "gout",
    "lupus", "paralysis", "amputation", "surgery", "postoperative", "whiplash",
]

VISIT_FEATURE_COLS = [
    "days_since_first_visit",
    "pain_score_at_visit",
    "medications_at_visit",
    "visit_type_code",
]

MAX_VISITS = 6
VAL_FRACTION = 0.20
AUC_MIN, AUC_MAX = 0.80, 0.97
GREEN = 0.995
ARENA_EPOCHS = 30


def student_to_seed(student_id: str) -> int:
    h = int(hashlib.sha256(student_id.lower().strip().encode()).hexdigest(), 16)
    return h % 900 + 100


def oracle_config(seed: int) -> dict:
    """Per-seed target architecture. MUST match solutions.oracle_config()."""
    rng = np.random.default_rng((int(seed) * 2654435761) % (2 ** 32))
    max_seq_len = int(rng.choice([5, 6], p=[0.25, 0.75]))
    hidden_units = int(rng.choice([16, 24, 32], p=[0.34, 0.41, 0.25]))
    return {
        "max_seq_len": max_seq_len,
        "hidden_units": hidden_units,
        "cell_type": "lstm",
        "bidirectional": False,
        "val_fraction": VAL_FRACTION,
    }


# --------------------------------------------------------------------------- #
# Data prep (mirrors pipeline pad/build/mask/split)                           #
# --------------------------------------------------------------------------- #

def _labels(condition_text):
    pattern = r"\b(?:" + "|".join(re.escape(k) for k in PAIN_KEYWORDS) + r")\b"
    compiled = re.compile(pattern)
    return np.array([1 if compiled.search(t.lower()) else 0 for t in condition_text],
                    dtype=int)


def _pad(visits, max_len):
    t, f = visits.shape
    if t > max_len:
        return visits[t - max_len:]
    out = np.zeros((max_len, f), dtype=float)
    out[:t] = visits
    return out


def build_sequences(features_df, visits_df, max_len):
    patient_ids = features_df["id"].astype(str).to_numpy()
    grouped = {pid: g for pid, g in visits_df.groupby("patient_id", sort=False)}
    seqs, lengths = [], []
    for pid in patient_ids:
        g = grouped[str(pid)].sort_values("visit_number")
        arr = g[VISIT_FEATURE_COLS].to_numpy(dtype=float)
        lengths.append(len(arr))
        seqs.append(_pad(arr, max_len))
    return np.stack(seqs), np.array(lengths, dtype=int), patient_ids


def patient_level_split(labels, val_fraction, seed):
    rng = np.random.default_rng(seed)
    is_val = np.zeros(len(labels), dtype=bool)
    for cls in sorted(np.unique(labels)):
        idx = np.where(labels == cls)[0]
        shuffled = rng.permutation(idx)
        is_val[shuffled[:round(val_fraction * len(idx))]] = True
    return is_val


def _mask(lengths, max_len):
    return np.arange(max_len)[None, :] < lengths[:, None]


def _masked_mean(seqs, mask):
    m = mask.astype(float)[:, :, None]
    return (seqs * m).sum(axis=1) / np.clip(mask.sum(axis=1, keepdims=True), 1, None)


def _logistic_probe(x_tr, y_tr, x_va, seed):
    """Tiny deterministic full-batch logistic regression -> val scores."""
    mu, sd = x_tr.mean(0), x_tr.std(0)
    sd[sd == 0] = 1.0
    xt = (x_tr - mu) / sd
    xv = (x_va - mu) / sd
    xt = np.c_[np.ones(len(xt)), xt]
    xv = np.c_[np.ones(len(xv)), xv]
    rng = np.random.default_rng(seed ^ 0x5EED)
    w = 0.01 * rng.standard_normal(xt.shape[1])
    for _ in range(400):
        p = 1.0 / (1.0 + np.exp(-xt @ w))
        w -= 0.15 * (xt.T @ (p - y_tr)) / len(y_tr)
    return 1.0 / (1.0 + np.exp(-xv @ w))


def _auc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))


def _f1_acc(y, s, thr):
    yp = (s >= thr).astype(int)
    tp = int(((yp == 1) & (y == 1)).sum()); fp = int(((yp == 1) & (y == 0)).sum())
    tn = int(((yp == 0) & (y == 0)).sum()); fn = int(((yp == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return f1, (tp + tn) / len(y), (tp, fp, tn, fn)


def _roc_points(y, s, n=26):
    pts = []
    for thr in np.linspace(1.0, 0.0, n):
        yp = (s >= thr).astype(int)
        tp = ((yp == 1) & (y == 1)).sum(); fp = ((yp == 1) & (y == 0)).sum()
        fn = ((yp == 0) & (y == 1)).sum(); tn = ((yp == 0) & (y == 0)).sum()
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        pts.append([round(float(fpr), 3), round(float(tpr), 3)])
    return pts


def _loss_curve(seed, key, floor, rate, overfit, epochs=ARENA_EPOCHS):
    rng = np.random.default_rng(int(hashlib.sha256(f"{seed}|{key}".encode()).hexdigest(), 16) % (2 ** 32))
    start = 0.69 + 0.02 * rng.standard_normal()
    onset = 7 + rng.integers(0, 4)
    tr, va = [], []
    for e in range(1, epochs + 1):
        t = floor + (start - floor) * math.exp(-rate * e) + 0.005 * rng.standard_normal()
        gap = overfit * max(0.0, 1 - math.exp(-max(0, e - onset) / 6.0))
        v = t + gap + 0.028 + 0.006 * rng.standard_normal()
        tr.append(round(max(0.05, t), 4))
        va.append(round(max(0.06, v), 4))
    return tr, va


# --------------------------------------------------------------------------- #
# The surrogate evaluator                                                     #
# --------------------------------------------------------------------------- #

def _base_ceiling(features_df, visits_df, seed, max_len):
    """Masked-mean-pool + logistic probe: the achievable metric ceiling for this
    seed at this padding length (more visits kept -> stronger)."""
    labels = _labels(features_df["condition_text"].tolist())
    seqs, lengths, _ = build_sequences(features_df, visits_df, max_len)
    mask = _mask(lengths, max_len)
    pooled = _masked_mean(seqs, mask)
    is_val = patient_level_split(labels, VAL_FRACTION, seed)
    s_va = _logistic_probe(pooled[~is_val], labels[~is_val].astype(float),
                           pooled[is_val], seed)
    y_va = labels[is_val]
    thr = float(np.median(_logistic_probe(pooled[~is_val], labels[~is_val].astype(float),
                                          pooled[~is_val], seed)))
    auc = _auc(y_va, s_va)
    f1, acc, _ = _f1_acc(y_va, s_va, thr)
    return dict(auc=auc, f1=f1, accuracy=acc, y_va=y_va, s_va=s_va, thr=thr,
                train_n=int((~is_val).sum()), val_n=int(is_val.sum()),
                n_pos=int((y_va == 1).sum()), n_neg=int((y_va == 0).sum()))


def _recurrent_metrics(features_df, visits_df, seed, max_seq_len, hidden_units,
                       cell_type, bidirectional):
    """Analytic surrogate for a trained recurrent model's validation metrics."""
    cfg = oracle_config(seed)
    base = _base_ceiling(features_df, visits_df, seed, int(max_seq_len))
    rng = np.random.default_rng(int(hashlib.sha256(
        f"{seed}|{max_seq_len}|{hidden_units}|{cell_type}|{int(bool(bidirectional))}"
        .encode()).hexdigest(), 16) % (2 ** 32))
    jit = lambda: rng.standard_normal()

    hu = int(hidden_units)
    q_hidden = 1.0 - 0.010 * abs(hu - cfg["hidden_units"]) - 0.0015 * max(0, 12 - hu)
    q_hidden = float(np.clip(q_hidden, 0.72, 1.0))
    q_cell = 1.0 if cell_type == "lstm" else 0.992
    q_bidir = 1.004 if bidirectional else 1.0
    q_len = 1.0 - 0.02 * max(0, cfg["max_seq_len"] - int(max_seq_len))
    q = q_hidden * q_cell * q_bidir * float(np.clip(q_len, 0.8, 1.0))
    nz = 1.06 - q

    rec_bonus = 1.015
    rec_auc = float(np.clip(base["auc"] * q * rec_bonus + 0.006 * jit() * nz,
                            0.5, min(0.995, base["auc"] * 1.03)))
    rec_f1 = float(np.clip(base["f1"] * q * rec_bonus + 0.012 * jit() * nz,
                           0.1, min(0.99, base["f1"] * 1.05)))
    rec_acc = float(np.clip(base["accuracy"] * (0.94 + 0.06 * q) + 0.010 * jit() * nz,
                            0.45, min(0.99, base["accuracy"] * 1.04)))
    return dict(auc=rec_auc, f1=rec_f1, accuracy=rec_acc, q=q, base=base, jit=jit)


def evaluate_seed(features_df, visits_df, seed, max_seq_len, hidden_units,
                  cell_type, bidirectional):
    if cell_type not in ("lstm", "gru"):
        raise ValueError("cell_type must be 'lstm' or 'gru'")
    cfg = oracle_config(seed)
    rm = _recurrent_metrics(features_df, visits_df, seed, max_seq_len,
                            hidden_units, cell_type, bidirectional)
    base, jit = rm["base"], rm["jit"]
    hu = int(hidden_units)
    rec_auc, rec_f1, rec_acc = rm["auc"], rm["f1"], rm["accuracy"]

    # --- fixed Conv1D baseline: a touch below, because order/recency matter ---
    conv_auc = float(np.clip(base["auc"] * 0.955 + 0.004 * jit(), 0.5, rec_auc))
    conv_f1 = float(np.clip(base["f1"] * 0.93 + 0.008 * jit(), 0.1, rec_f1 + 0.01))
    conv_acc = float(np.clip(base["accuracy"] * 0.965 + 0.006 * jit(), 0.45, rec_acc + 0.01))

    y = base["y_va"]
    shrink = np.clip((rec_auc - 0.5) / max(base["auc"] - 0.5, 1e-6), 0, 1)
    rec_scores = 0.5 + (base["s_va"] - 0.5) * shrink
    shrink_c = np.clip((conv_auc - 0.5) / max(base["auc"] - 0.5, 1e-6), 0, 1)
    conv_scores = 0.5 + (base["s_va"] - 0.5) * shrink_c
    _, _, rec_cm = _f1_acc(y, rec_scores, base["thr"])
    _, _, conv_cm = _f1_acc(y, conv_scores, base["thr"])

    q = rm["q"]
    rec_tr, rec_va = _loss_curve(seed, f"rec|{hu}|{cell_type}|{int(bool(bidirectional))}",
                                 floor=0.30 + 0.05 * (1 - q), rate=0.14 + 0.03 * q,
                                 overfit=0.10 + 0.06 * (1 - q))
    conv_tr, conv_va = _loss_curve(seed, "conv", floor=0.34, rate=0.19, overfit=0.13)

    oracle = oracle_metrics(features_df, visits_df, seed)
    is_optimal = bool(
        rec_f1 >= GREEN * oracle["f1"]
        and rec_acc >= GREEN * oracle["accuracy"]
        and rec_auc >= GREEN * oracle["auc"]
    )
    return {
        "max_seq_len": int(max_seq_len), "hidden_units": hu,
        "cell_type": cell_type, "bidirectional": bool(bidirectional),
        "recurrent": {
            "auc": round(rec_auc, 4), "f1": round(rec_f1, 4), "accuracy": round(rec_acc, 4),
            "tp": rec_cm[0], "fp": rec_cm[1], "tn": rec_cm[2], "fn": rec_cm[3],
            "roc_curve": _roc_points(y, rec_scores), "train_loss": rec_tr, "val_loss": rec_va,
        },
        "conv1d": {
            "auc": round(conv_auc, 4), "f1": round(conv_f1, 4), "accuracy": round(conv_acc, 4),
            "tp": conv_cm[0], "fp": conv_cm[1], "tn": conv_cm[2], "fn": conv_cm[3],
            "roc_curve": _roc_points(y, conv_scores), "train_loss": conv_tr, "val_loss": conv_va,
        },
        "epochs": ARENA_EPOCHS,
        "train_n": base["train_n"], "val_n": base["val_n"],
        "val_pos": base["n_pos"], "val_neg": base["n_neg"],
        "is_optimal": is_optimal,
        "hint": _hint(rec_auc, rec_f1, oracle, max_seq_len, hidden_units, cell_type,
                      bidirectional, cfg),
        "caption": _caption(cell_type, hu, bidirectional, rec_auc, conv_auc, rec_f1, conv_f1, cfg),
        "oracle": oracle,
    }


_ORACLE_MARGIN = 0.97   # target sits 3% under the reference model's own metrics,
                        # so evaluating AT oracle_config always clears the green line.


def oracle_metrics(features_df, visits_df, seed):
    """The per-seed personal target: the reference LSTM at oracle_config, held a
    few percent below its own achieved metrics so a well-tuned student clears
    the 'optimal' line."""
    cfg = oracle_config(seed)
    rm = _recurrent_metrics(features_df, visits_df, seed, cfg["max_seq_len"],
                            cfg["hidden_units"], cfg["cell_type"], cfg["bidirectional"])
    return {
        "max_seq_len": cfg["max_seq_len"], "hidden_units": cfg["hidden_units"],
        "cell_type": cfg["cell_type"],
        "auc": round(min(AUC_MAX, rm["auc"] * _ORACLE_MARGIN), 4),
        "f1": round(rm["f1"] * _ORACLE_MARGIN, 4),
        "accuracy": round(rm["accuracy"] * _ORACLE_MARGIN, 4),
        "train_n": rm["base"]["train_n"], "val_n": rm["base"]["val_n"],
    }


def _hint(auc, f1, oracle, max_seq_len, hidden_units, cell_type, bidirectional, cfg):
    """A short, actionable nudge. Only shown while the model is NOT optimal, so it
    always points at a knob to turn: hidden_units / cell_type / bidirectional, or
    max_seq_len when it is below 5."""
    hu = int(hidden_units)
    msl = int(max_seq_len)
    behind = max(oracle["auc"] - auc, oracle["f1"] - f1)

    if msl < 5:
        return (f"You are keeping only {msl} of up to 6 visits. Raise max_seq_len in the "
                f"Sequence Explorer tab so the model sees more recent history.")
    if cell_type != cfg["cell_type"] and behind > 0.01:
        return (f"Try the {cfg['cell_type'].upper()} cell -- for this seed it separates the "
                f"classes a little better than {cell_type.upper()} at the same width.")
    if hu <= 8 and behind > 0.005:
        return (f"The recurrent layer is very narrow ({hu} units). Increase hidden_units so it "
                f"has room to learn the pattern.")
    if hu >= 52 and behind > 0.01:
        return (f"{hu} hidden units is wide for ~{oracle['val_n']} validation patients -- it "
                f"is likely overfitting. Lower hidden_units, turn bidirectional off, or try GRU.")
    if behind <= 0.02:
        return ("So close -- you're right on the oracle line. A small change to hidden_units, "
                "cell_type, or the bidirectional toggle should tip all three bars green.")
    return ("Adjust hidden_units, cell_type or bidirectional to lift F1 and AUC above the oracle "
            "target.")


def _caption(cell_type, hidden_units, bidirectional, rec_auc, conv_auc, rec_f1, conv_f1, cfg):
    who = ("Bidirectional " if bidirectional else "") + cell_type.upper()
    delta = rec_auc - conv_auc
    if delta >= 0.03:
        tail = ("the Conv1D baseline trained faster but missed patients whose pain "
                "pattern only emerges over their last 2-3 visits.")
    elif delta >= 0.005:
        tail = ("the Conv1D baseline is close behind — this dataset rewards reading "
                "visits in order, but not by a huge margin.")
    else:
        tail = ("the Conv1D baseline essentially matched it — with these settings the "
                "recurrent model isn't using visit order well.")
    return (f"{who} with {hidden_units} hidden units reached val AUC {rec_auc:.3f} "
            f"(F1 {rec_f1:.3f}); the Conv1D baseline reached AUC {conv_auc:.3f} "
            f"(F1 {conv_f1:.3f}). {tail}")


def good_seed(base_seed, oracle_table):
    for offset in range(900):
        cand = (base_seed - 100 + offset) % 900 + 100
        e = oracle_table.get(str(cand))
        if e and AUC_MIN <= e["auc"] <= AUC_MAX:
            return cand
    return base_seed
