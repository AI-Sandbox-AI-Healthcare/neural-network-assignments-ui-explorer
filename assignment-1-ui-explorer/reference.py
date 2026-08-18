"""
assignment-1-ui-explorer/reference.py

PyTorch oracle model and full evaluation pipeline.
Independent of student's pipeline.py — used to pre-compute and serve
optimal results for each student seed.
"""
import hashlib
import re

import numpy as np

OPTIMAL_LR = 0.5
OPTIMAL_STEPS = 300
OPTIMAL_VAL_FRACTION = 0.20
THRESHOLD = 0.5

PAIN_KEYWORDS = [
    "chronic", "pain", "arthritis", "osteoarthritis", "rheumatoid",
    "fibromyalgia", "migraine", "neuropathy", "neuralgia",
    "sciatica", "back pain", "neck pain", "spinal", "fracture",
    "injury", "burn", "wound", "trauma", "sprain", "strain",
    "tendon", "ligament", "joint", "osteoporosis", "gout",
    "lupus", "paralysis", "amputation", "surgery", "postoperative", "whiplash",
]

FEATURE_COLS = [
    "age", "is_female", "number_of_unique_meds", "number_of_encounters", "number_of_procedures",
    "unique_procedures", "pain_severity", "body_height", "body_weight",
    "body_mass_index", "systolic_blood_pressure", "diastolic_blood_pressure",
    "heart_rate", "respiratory_rate", "qaly", "daly", "qols",
    "healthcare_expenses", "healthcare_coverage",
]


def student_to_seed(student_id: str) -> int:
    """Deterministic seed from student ID, independent of PYTHONHASHSEED."""
    h = int(hashlib.sha256(student_id.lower().strip().encode()).hexdigest(), 16)
    return h % 900 + 100


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


def _standardize(X_train, X_val):
    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0)
    stds = np.where(stds == 0, 1.0, stds)
    return (X_train - means) / stds, (X_val - means) / stds


def _auc(y_true, scores):
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    total = 0.0
    for sp in pos:
        for sn in neg:
            if sp > sn:
                total += 1.0
            elif sp == sn:
                total += 0.5
    return total / (len(pos) * len(neg))


def _metrics(y_val, probs, threshold=THRESHOLD):
    y_pred = (probs >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_val == 1)))
    fp = int(np.sum((y_pred == 1) & (y_val == 0)))
    tn = int(np.sum((y_pred == 0) & (y_val == 0)))
    fn = int(np.sum((y_pred == 0) & (y_val == 1)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / len(y_val)
    auc = _auc(y_val, probs)
    return dict(tp=tp, fp=fp, tn=tn, fn=fn,
                precision=round(prec, 4), recall=round(rec, 4),
                f1=round(f1, 4), accuracy=round(acc, 4), auc=round(auc, 4))


def _bce_loss(probs, y):
    eps = 1e-9
    return float(-np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps)))


def _roc_curve_points(y_true, probs, n=30):
    """Return list of [fpr, tpr] pairs swept across thresholds for a live ROC plot."""
    thresholds = np.linspace(1.0, 0.0, n)
    pts = []
    for t in thresholds:
        yp = (probs >= t).astype(int)
        tp = int(np.sum((yp == 1) & (y_true == 1)))
        fp = int(np.sum((yp == 1) & (y_true == 0)))
        fn = int(np.sum((yp == 0) & (y_true == 1)))
        tn = int(np.sum((yp == 0) & (y_true == 0)))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        pts.append([round(fpr, 3), round(tpr, 3)])
    return pts


def run_torch_oracle(X_train, y_train, X_val, y_val, lr, steps):
    """Train with PyTorch SGD logistic regression, return (probs_val, loss_history)."""
    import torch
    import torch.nn as nn

    class _Model(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.fc = nn.Linear(n, 1, bias=True)
            nn.init.zeros_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)

        def forward(self, x):
            return torch.sigmoid(self.fc(x))

    model = _Model(X_train.shape[1])
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    Xt = torch.FloatTensor(X_train)
    yt = torch.FloatTensor(y_train)

    loss_hist = []
    for _ in range(steps):
        model.train()
        out = model(Xt).squeeze()
        loss = criterion(out, yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_hist.append(round(loss.item(), 6))

    model.eval()
    with torch.no_grad():
        probs = model(torch.FloatTensor(X_val)).squeeze().numpy()

    return probs, loss_hist


def run_numpy_oracle(X_train, y_train, X_val, y_val, lr, steps):
    """Numpy fallback when torch is unavailable (same math, same result)."""
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    w = np.zeros(X_train.shape[1])
    b = 0.0
    loss_hist = []
    n = X_train.shape[0]
    for _ in range(steps):
        pred = _sigmoid(X_train.dot(w) + b)
        eps = 1e-9
        loss_hist.append(round(float(-np.mean(
            y_train * np.log(pred + eps) + (1 - y_train) * np.log(1 - pred + eps)
        )), 6))
        err = pred - y_train
        w -= lr * (X_train.T.dot(err) / n)
        b -= lr * float(np.mean(err))

    probs = _sigmoid(X_val.dot(w) + b)
    return probs, loss_hist


def evaluate_seed(df, seed, lr=OPTIMAL_LR, steps=OPTIMAL_STEPS,
                  val_fraction=OPTIMAL_VAL_FRACTION, use_torch=True):
    """Full pipeline for one seed. Returns metrics dict with loss_history."""
    labels = _flag_keywords(df["condition_text"].tolist()).astype(float)
    X = df[FEATURE_COLS].values.astype(float)

    is_val = _stratified_split(labels.astype(int), val_fraction, seed)
    X_train, y_train = X[~is_val], labels[~is_val]
    X_val, y_val = X[is_val], labels[is_val]
    X_train_s, X_val_s = _standardize(X_train, X_val)

    if use_torch:
        try:
            probs, loss_hist = run_torch_oracle(X_train_s, y_train, X_val_s, y_val, lr, steps)
        except ImportError:
            probs, loss_hist = run_numpy_oracle(X_train_s, y_train, X_val_s, y_val, lr, steps)
    else:
        probs, loss_hist = run_numpy_oracle(X_train_s, y_train, X_val_s, y_val, lr, steps)

    m = _metrics(y_val.astype(int), probs)
    m["final_loss"] = round(loss_hist[-1], 4)
    m["train_n"] = int(np.sum(~is_val))
    m["val_n"] = int(np.sum(is_val))
    m["loss_history"] = loss_hist
    m["roc_curve"] = _roc_curve_points(y_val.astype(int), probs)
    return m
