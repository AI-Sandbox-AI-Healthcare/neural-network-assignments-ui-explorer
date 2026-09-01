// ============================================================
// docs/assignment-2 static API shim.
//
// GitHub Pages has no Flask backend, so this file overrides
// window.fetch and answers run_sandbox.py's /api/* routes
// entirely in the browser. Keep in sync with run_sandbox.py.
//
//   /api/data     exact  -> data.json
//   /api/assign   exact  -> student_to_seed + _good_seed over oracle_table.json
//   /api/arena    exact  -> arena.json (the same deterministic simulation,
//                           precomputed for all 16 configs x 900 seeds x A/B)
//   /api/evaluate MODELLED -> a smooth response surface anchored to each seed's
//                           REAL Random-Forest oracle (oracle_table.json). A live
//                           scikit-learn fit is impossible in-browser, so the
//                           intermediate metrics while you drag the sliders are
//                           modelled -- but the oracle TARGETS are the real ones,
//                           and the surface peaks exactly at the oracle config, so
//                           the tune -> "Optimal Performance Reached!" -> copy-params
//                           loop behaves the same as the local server.
// ============================================================
(function () {
  "use strict";
  const realFetch = window.fetch.bind(window);

  const GOOD_MIN = 0.80, GOOD_MAX = 0.97;   // run_sandbox.py AUC_OK_MIN / AUC_OK_MAX
  const GREEN = 0.995;                        // run_sandbox.py GREEN

  let _data = null, _oracle = null, _arena = null;
  const getData = async () => _data || (_data = await (await realFetch("data.json")).json());
  const getOracle = async () => _oracle || (_oracle = await (await realFetch("oracle_table.json")).json());
  const getArena = async () => _arena || (_arena = await (await realFetch("arena.json")).json());

  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const mulberry32 = (a) => () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const hashStr = (s) => {
    let h = 2166136261 >>> 0;
    for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
    return h >>> 0;
  };

  // student_to_seed: int(sha256(id.lower().strip()), 16) % 900 + 100
  async function studentToSeed(id) {
    const bytes = new TextEncoder().encode(String(id || "").toLowerCase().trim());
    const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
    let h = 0n;
    for (const b of digest) h = (h << 8n) | BigInt(b);
    return Number(h % 900n) + 100;
  }
  async function goodSeed(base) {
    const t = await getOracle();
    for (let o = 0; o < 900; o++) {
      const c = ((base - 100 + o) % 900) + 100;
      const e = t[String(c)];
      if (e && e.auc >= GOOD_MIN && e.auc <= GOOD_MAX) return c;
    }
    return base;
  }

  // ---- /api/data ----
  const apiData = async () => await getData();

  // ---- /api/assign ----
  async function apiAssign(studentId) {
    const id = String(studentId || "").trim();
    if (!id) return { error: "student_id required" };
    const seed = await goodSeed(await studentToSeed(id));
    return { seed, oracle: (await getOracle())[String(seed)] };
  }

  // ---- /api/evaluate (modelled surface, anchored to the real oracle) ----
  async function apiEvaluate(b) {
    const seed = +b.seed, ne = +b.n_estimators, md = +b.max_depth, r = +b.oversample_ratio;
    const o = (await getOracle())[String(seed)];
    const rnd = mulberry32(hashStr(seed + "|" + ne + "|" + md + "|" + r.toFixed(2)));
    const jit = () => rnd() - 0.5;

    // quality factors: exactly 1.0 at this seed's oracle config, degrading away.
    const qmd = clamp(1 - 0.018 * Math.max(0, o.max_depth - md) - 0.010 * Math.max(0, md - o.max_depth), 0.55, 1);
    const qne = clamp(1 - 0.0018 * Math.max(0, 60 - ne) - 0.00030 * Math.abs(ne - o.n_estimators), 0.75, 1);
    const base = qmd * qne;
    const rc = clamp(r, 0, 1);
    const nz = 1.05 - base;   // noise fades out as you approach the oracle

    let auc = o.auc * base * (0.945 + 0.055 * Math.min(rc / 0.9, 1)) + 0.010 * jit() * nz;
    let f1 = o.f1 * base * (0.68 + 0.32 * Math.min(rc / 0.95, 1)) + 0.020 * jit() * nz;
    let acc = o.accuracy * (0.90 + 0.10 * base) * (0.965 + 0.035 * Math.min(rc, 1)) + 0.012 * jit() * nz;
    auc = clamp(auc, 0.5, o.auc * 1.003);
    f1 = clamp(f1, 0.15, o.f1 * 1.005);
    acc = clamp(acc, 0.45, o.accuracy * 1.005);

    const P = o.tp + o.fn, N = o.fp + o.tn;
    let tp, fp, fn, tn;
    if (f1 <= 0.01) { tp = 0; fn = P; fp = 0; tn = N; }
    else { tp = Math.round(f1 * P); fn = P - tp; fp = Math.min(N, Math.round(tp * (1 - f1) / f1)); tn = N - fp; }

    const train_accuracy = clamp(acc + 0.04 + 0.02 * Math.max(0, md - 5) + 0.05 * (1 - base), 0, 0.999);

    const flat = 1 / o.feature_importances.length;
    let fi = o.feature_importances.map(([n, v]) => [n, flat + base * (v - flat) + 0.004 * jit()]);
    const s = fi.reduce((a, x) => a + Math.max(x[1], 0), 0) || 1;
    fi = fi.map(([n, v]) => [n, +(Math.max(v, 0) / s).toFixed(4)]).sort((x, y) => y[1] - x[1]);

    const shrink = clamp((auc - 0.5) / ((o.auc - 0.5) || 1), 0, 1);
    const roc = o.roc_curve.map(([fpr, tpr]) => [fpr, +(fpr + (tpr - fpr) * shrink).toFixed(3)]);

    const m = {
      n_estimators: ne, max_depth: md, oversample_ratio: +r.toFixed(2),
      tp, fp, tn, fn,
      precision: +((tp + fp) > 0 ? tp / (tp + fp) : 0).toFixed(4),
      recall: +((tp + fn) > 0 ? tp / (tp + fn) : 0).toFixed(4),
      f1: +f1.toFixed(4), accuracy: +acc.toFixed(4), auc: +auc.toFixed(4),
      train_accuracy: +train_accuracy.toFixed(4),
      train_n: o.train_n, val_n: o.val_n,
      feature_importances: fi, roc_curve: roc,
    };
    const is_optimal = m.f1 >= GREEN * o.f1 && m.accuracy >= GREEN * o.accuracy && m.auc >= GREEN * o.auc;
    return Object.assign({}, m, { is_optimal, hint: makeHint(m, o, rc), oracle: o });
  }

  function makeHint(m, o, r) {   // port of run_sandbox.py _make_hint
    const gap = o.auc - m.auc, overfit = m.train_accuracy - m.accuracy;
    if (r < 0.5) return "Your training set is still imbalanced -- raise oversample_ratio toward 1.0 so the chronic-pain class isn't drowned out.";
    if (m.fn > m.tp && gap > 0.02) return "The model is over-predicting the majority class (more misses than catches). Raise oversample_ratio, or add trees with n_estimators.";
    if (overfit > 0.30 && m.max_depth >= 12) return "The forest is memorising training data (train acc " + m.train_accuracy.toFixed(2) + " vs val " + m.accuracy.toFixed(2) + "). Lower max_depth.";
    if (gap > 0.05) return "Underfitting -- try a deeper max_depth or more trees (higher n_estimators).";
    if (gap > 0.0 && r < 1.0) return "So close -- every metric is a hair under target and you are at oversample_ratio " + r.toFixed(2) + ". Push it to 1.00 (full balance) to close the gap.";
    if (gap > 0.015) return "Close! AUC " + m.auc.toFixed(3) + " vs target " + o.auc.toFixed(3) + ". Nudge n_estimators or max_depth one step.";
    if (gap > 0.0) return "Almost there -- a small tweak should push you over the oracle line.";
    return "You're at or above the oracle target. Lock in these parameters.";
  }

  // ---- /api/arena (exact: precomputed) ----
  const PRESET_UNITS = { deep: [32, 32, 32, 16], wide: [96] };
  const cfgKey = (c) => c.preset[0] + c.activation[0] + (c.dropout ? "D" : "d") + (c.early_stopping ? "E" : "e");
  function panelFrom(rec, c) {
    return {
      preset: c.preset, hidden_units: PRESET_UNITS[c.preset], activation: c.activation,
      dropout: !!c.dropout, early_stopping: !!c.early_stopping,
      train_loss: rec[0], val_loss: rec[1], val_f1: rec[2], stopped_epoch: rec[3], best_epoch: rec[4],
    };
  }
  const describe = (p) => [
    p.preset === "wide" ? "wide" : "deep", p.activation,
    p.dropout ? "dropout" : "no dropout",
    p.early_stopping ? "early stopping" : "no early stopping",
  ].join(", ");
  function verdict(p, E) {
    const g = p.val_loss[p.val_loss.length - 1] - p.train_loss[p.train_loss.length - 1];
    if (p.early_stopping && p.stopped_epoch < E) return "stopped early at epoch " + p.stopped_epoch + " with val F1 " + p.val_f1.toFixed(3);
    if (g > 0.12) return "overfit after ~" + p.best_epoch + " epochs (val F1 " + p.val_f1.toFixed(3) + ")";
    return "kept validation loss flat, ending at val F1 " + p.val_f1.toFixed(3);
  }
  async function apiArena(b) {
    const A = await getArena();
    const E = 40, se = A[String(b.seed || 100)] || A["100"];
    const a = panelFrom(se[cfgKey(b.a) + "A"], b.a);
    const p = panelFrom(se[cfgKey(b.b) + "B"], b.b);
    const winner = a.val_f1 >= p.val_f1 ? "A" : "B";
    const caption = "Model A (" + describe(a) + ") " + verdict(a, E) + "; Model B (" +
      describe(p) + ") " + verdict(p, E) + ". Higher validation F1: Model " + winner + ".";
    return { a, b: p, caption, epochs: E };
  }

  // ---- install ----
  window.fetch = async function (url, init) {
    const u = typeof url === "string" ? url : (url && url.url) || "";
    if (!/(^|\/)api\/(data|assign|evaluate|arena)\b/.test(u)) return realFetch(url, init);
    const body = init && init.body ? JSON.parse(init.body) : {};
    let out;
    if (/\/api\/data\b/.test(u)) out = await apiData();
    else if (/\/api\/assign\b/.test(u)) out = await apiAssign(body.student_id);
    else if (/\/api\/evaluate\b/.test(u)) out = await apiEvaluate(body);
    else out = await apiArena(body);
    return new Response(JSON.stringify(out), { status: 200, headers: { "Content-Type": "application/json" } });
  };
})();
