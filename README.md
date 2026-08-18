# Neural Network Assignments — UI Explorer

Interactive, in-browser explorers that accompany the neural network
assignments. Each assignment gets its own self-contained folder
(`assignment-N-ui-explorer/`) with its own server, dataset, and README
notes below.

Every student gets a personal, deterministic seed derived from their student
ID (`student_to_seed()` in `reference.py`). That seed fixes their train/val
split and their target ("oracle") metrics, so results are reproducible per
student but differ across the class. This behavior is load-bearing for how
the assignments are graded and is not touched by any of the cleanup in this
repo.

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

### Publishing to GitHub Pages

GitHub Pages can only serve static files, so it cannot run the Flask
backend — Pages hosts a read-only snapshot (dataset + each seed's oracle
target), not the live, slider-driven Training Explorer. That still requires
running the server locally as described above.

**One-time setup**, after pushing this repo to GitHub:

1. On GitHub, go to **Settings → Pages**.
2. Under **Build and deployment**, set **Source** to **Deploy from a branch**.
3. Set **Branch** to `main` and the folder to `/docs`, then **Save**.
4. GitHub publishes the site at `https://<org-or-user>.github.io/<repo-name>/`
   within a minute or two (check the Pages settings page for the exact URL
   and build status).

After that, every future `git push` to `main` that changes `docs/`
automatically updates the live site — no rebuild step on GitHub's side.

If you edit anything under `assignment-1-ui-explorer/` that affects the
static export (the dataset, the oracle table), regenerate it before pushing:

```powershell
python assignment-1-ui-explorer/generate_docs_site.py
```

This writes into `docs/assignment-1/` (its own subfolder, not the shared
`docs/` root) so it won't collide with other assignments' static exports.
`docs/index.html` is the hand-maintained landing page that links to each
assignment's subfolder — update it whenever a new assignment's static site
is added. Commit and push both the source change and the regenerated
`docs/` output.

### Regenerating data (instructor-only)

- `generate_oracle.py` — recomputes `oracle_table.json` (optimal metrics for
  every seed 100–999). Only needed if the dataset or optimal hyperparameters
  change.
- `precompute_grids.py` — optionally exports a coarse parameter grid to
  `docs/assignment-1/precomputed_grid.json` for future static-site
  enhancements.

---

## Future assignments

Additional assignments should follow the same pattern:

1. A new `assignment-N-ui-explorer/` folder with its own server, data,
   `requirements.txt`, and `start.bat` (copy `assignment-1-ui-explorer/` as
   a starting point).
2. Its own `generate_docs_site.py`-style script that writes its static export
   into `docs/assignment-N/` — **its own subfolder**, so it doesn't overwrite
   another assignment's `index.html`/`patients.json`/etc.
3. A new link added to `docs/index.html` (the root landing page) pointing at
   `assignment-N/`.
4. A new section in this README describing what it teaches, how to run it,
   and how to access its pages.
