# Neural Network Assignments — UI Explorer

Interactive, in-browser explorers that accompany the neural network
assignments. Each assignment gets its own self-contained folder
(`assignment-N-ui-explorer/`) with its own server, dataset, and README
notes below.

---

## GitHub Pages

Every explorer also runs entirely in the browser with no backend —
`static_api_shim.js` overrides `fetch` and answers the `/api/*` routes
client-side.

- Landing page: https://ai-sandbox-ai-healthcare.github.io/neural-network-assignments-ui-explorer/

- Assignment 1: https://ai-sandbox-ai-healthcare.github.io/neural-network-assignments-ui-explorer/assignment-1/

- Assignment 2: https://ai-sandbox-ai-healthcare.github.io/neural-network-assignments-ui-explorer/assignment-2/

- Assignment 3: https://ai-sandbox-ai-healthcare.github.io/neural-network-assignments-ui-explorer/assignment-3/

Each `docs/assignment-N/` page is generated from that assignment's
`run_sandbox.py` by `assignment-N-ui-explorer/generate_docs_site.py` — run it after changing a UI, then publish the `docs/` folder. 

---

## Assignment 1 — Chronic Pain Classifier (`assignment-1-ui-explorer/`)

A logistic-regression-from-scratch exercise on a synthetic EHR dataset
(inspired by Synthea, 320 patients). Students implement the pipeline
(keyword labeling, stratified split, standardization, gradient descent,
AUC/F1) in their own `pipeline.py`; this explorer is the companion UI that
lets them:

- **Overview & Concepts** — 11 interactive concept cards (gradient descent,
  standardization, stratified split, AUC-ROC, precision/recall, log loss,
  confusion matrix, label leakage, decision threshold, preprocessing
  leakage, etc.).
- **Dataset Explorer** — browse all 320 patients/23 columns, test the
  keyword-labeling regex live, and see class balance.
- **Training Explorer** — drag learning-rate / steps / validation-fraction
  sliders and watch the loss curve, ROC curve, and confusion matrix update
  live against their personal oracle target.

### Running locally

Requirements: Python 3.10 – 3.13 (`requirements.txt` includes PyTorch, whose
wheels are not yet published for 3.14).

**Windows (PowerShell):**

```powershell
python -m venv myenv
.\myenv\Scripts\Activate.ps1
pip install -r assignment-1-ui-explorer/requirements.txt
.\assignment-1-ui-explorer\start.bat
```

**Linux / macOS:**

```bash
python3 -m venv myenv
source myenv/bin/activate
pip install -r assignment-1-ui-explorer/requirements.txt
python assignment-1-ui-explorer/run_sandbox.py
```

Open `http://localhost:3001` in a browser (it should also open
automatically). Enter any student ID in the sidebar to see a seeded target
and start training.

---

## Assignment 2 — Tame the Tabular Baselines (`assignment-2-ui-explorer/`)

Same 320-patient dataset, same chronic-pain label, same personal seed — the
question shifts from "build from scratch" to "make it reliable under class
imbalance". Students:

- **Overview & Concepts** — 10 concept cards: class imbalance, class
  weighting, oversampling vs. SMOTE, bagging & Random Forests, feature
  importance, depth-vs-width, ReLU vs. tanh, dropout, early stopping,
  accuracy vs. AUC under imbalance.
- **Dataset Explorer** — patient table + keyword tester, plus a *Rebalance
  Preview* slider that redraws the class-balance chart after simulated
  oversampling.
- **Baseline Trainer** — drag `n_estimators` / `max_depth` /
  `oversample_ratio` and watch a live feature-importance chart, ROC curve
  and confusion matrix against a personal **Random Forest oracle**.
- **Architecture Arena** — run four preset feedforward comparisons
  (deep vs. wide, ReLU vs. tanh, dropout vs. none, early stopping vs. none),
  each with auto-generated "What changed?" narration; a free-mix mode then
  unlocks so students can try any combination and re-run.

### Running locally

Requirements: Python 3.10+ (no deep-learning framework needed — the Arena is a
deterministic simulation).

**Windows (PowerShell):**

```powershell
python -m venv myenv
.\myenv\Scripts\Activate.ps1
pip install -r assignment-2-ui-explorer/requirements.txt
.\assignment-2-ui-explorer\start.bat
```

**Linux / macOS:**

```bash
python3 -m venv myenv
source myenv/bin/activate
pip install -r assignment-2-ui-explorer/requirements.txt
python assignment-2-ui-explorer/run_sandbox.py
```

Open `http://localhost:3002` (it should open automatically). Enter any
student ID for a seeded Random Forest target.

---

## Assignment 3 — Model the Patient Journey (`assignment-3-ui-explorer/`)

Same 320 patients, same chronic-pain label, same personal seed — but each
patient is now a **sequence of 1–6 clinical visits** (`patient_visits.csv`).
Students:

- **Overview & Concepts** — three intro cards (the problem / the dataset / the
  approach), a **Pipeline Stages** strip, and **8 concept cards**:
  sequential clinical data, padding, masking, LSTM gates, GRU gates,
  bidirectional processing, 1D convolution as a visit-pattern detector,
  overfitting in small sequence datasets.
- **Patient Timeline Explorer** — a bold "click a patient" instruction, a
  scrollable patient list, a per-patient visit timeline (markers on top,
  pain-score chart below, split 50/50), and a sequence-length histogram with a
  computed read-out (mode, average, how many have the full 6 visits).
- **Sequence Explorer** — `max_seq_len` **buttons (1–6)** with *no default*:
  the padding-fraction stat, truncation note and per-patient preview stay blank
  until a value is picked, and picking one is part of completing the activity.
  The **Per-Patient Padding View** (first 20 patients) uses fixed-width visit
  cells and a reserved "⚠ N oldest dropped" column. Plus the worked
  2-patient / 3-visit mini-example that matches `pipeline/test_pipeline.py`.
- **Recurrent Arena** — tune `cell_type` (LSTM/GRU), `hidden_units` (4–64) and
  `bidirectional` against a personal LSTM oracle; `max_seq_len` is carried over
  from the Sequence Explorer and **nothing runs until it is set**. An always-on
  **Conv1D baseline** on the same sequences, live Current-vs-Target bars,
  overlapping loss curves, an ROC curve, a confusion matrix, an auto-generated
  "What's happening?" caption, and an **LSTM/GRU vs. Conv1D architecture
  comparison table** (memory-based vs. pattern-finding: strengths / weaknesses /
  runtime).

### Running locally

Requirements: Python 3.10+ (no deep-learning framework needed — the Arena is a
deterministic analytic simulation; the real LSTM/GRU is built in
`neural-network-assignment-3/pipeline/pipeline.py`).

**Windows (PowerShell):**

```powershell
python -m venv myenv
.\myenv\Scripts\Activate.ps1
pip install -r assignment-3-ui-explorer/requirements.txt
.\assignment-3-ui-explorer\start.bat
```

**Linux / macOS:**

```bash
python3 -m venv myenv
source myenv/bin/activate
pip install -r assignment-3-ui-explorer/requirements.txt
python assignment-3-ui-explorer/run_sandbox.py
```

Open `http://localhost:3003` (it should open automatically). Enter any student
ID for a seeded sequence-model target.

