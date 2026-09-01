# Neural Network Assignments — UI Explorer

Interactive, in-browser explorers that accompany the neural network
assignments. Each assignment gets its own self-contained folder
(`assignment-N-ui-explorer/`) with its own server, dataset, and README
notes below.

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

Requirements: Python 3.10+.

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

### GitHub Pages

The full interactive explorer, running entirely in-browser (no server) —

https://ai-sandbox-ai-healthcare.github.io/neural-network-assignments-ui-explorer/

https://ai-sandbox-ai-healthcare.github.io/neural-network-assignments-ui-explorer/assignment-1/

https://ai-sandbox-ai-healthcare.github.io/neural-network-assignments-ui-explorer/assignment-2/

---

## Future assignments

Additional assignments should follow the same pattern.