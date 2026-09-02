// ============================================================
// docs/assignment-3 static API shim.
//
// GitHub Pages has no Flask backend, so this file overrides
// window.fetch and answers run_sandbox.py's /api/* routes
// entirely in the browser. Keep in sync with run_sandbox.py + reference.py.
//
//   /api/data     exact    -> data.json
//   /api/assign   exact    -> student_to_seed + good_seed over oracle_table.json
//   /api/evaluate MODELLED  -> a smooth response surface anchored to each seed's
//                            precomputed oracle target (oracle_table.json). The
//                            real surrogate is a NumPy logistic probe that can't
//                            run in-browser, so the live metrics while you drag
//                            the sliders are modelled -- but the oracle TARGETS
//                            are the real precomputed ones and the surface peaks
//                            exactly at each seed's oracle_config, so the
//                            tune -> "Optimal Performance Reached!" -> copy-params
//                            loop behaves the same as the local server.
// ============================================================
(function () {
  "use strict";
  const realFetch = window.fetch.bind(window);

  const AUC_MIN = 0.80, AUC_MAX = 0.97;   // reference.AUC_MIN / AUC_MAX
  const GREEN = 0.995;                     // reference.GREEN
  const EPOCHS = 30;                       // reference.ARENA_EPOCHS
  const ORACLE_MARGIN = 0.97;             // reference._ORACLE_MARGIN

  let _data = null, _oracle = null;
  const getData = async () => _data || (_data = await (await realFetch("data.json")).json());
  const getOracle = async () => _oracle || (_oracle = await (await realFetch("oracle_table.json")).json());

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
      if (e && e.auc >= AUC_MIN && e.auc <= AUC_MAX) return c;
    }
    return base;
  }

  // reference.oracle_config -- must match reference.py / solutions.py exactly.
  function oracleConfig(seed) {
    // np.random.default_rng((seed * 2654435761) % 2**32).choice with given p.
    // The precomputed oracle already carries max_seq_len / hidden_units, so we
    // just read them back from the table; this helper is only a fallback.
    const rnd = mulberry32((Math.imul(seed, 2654435761) >>> 0));
    const msl = rnd() < 0.25 ? 5 : 6;
    const r = rnd();
    const hu = r < 0.34 ? 16 : (r < 0.75 ? 24 : 32);
    return { max_seq_len: msl, hidden_units: hu, cell_type: "lstm" };
  }

  const apiData = async () => await getData();

  async function apiAssign(studentId) {
    const id = String(studentId || "").trim();
    if (!id) return { error: "student_id required" };
    const seed = await goodSeed(await studentToSeed(id));
    return { seed, oracle: (await getOracle())[String(seed)] };
  }

  function lossCurve(seed, key, floor, rate, overfit) {
    const rnd = mulberry32(hashStr(seed + "|" + key));
    const gauss = () => (rnd() + rnd() + rnd() + rnd() - 2) * 0.9;
    const start = 0.69 + 0.02 * gauss();
    const onset = 7 + Math.floor(rnd() * 4);
    const tr = [], va = [];
    for (let e = 1; e <= EPOCHS; e++) {
      const t = floor + (start - floor) * Math.exp(-rate * e) + 0.005 * gauss();
      const gap = overfit * Math.max(0, 1 - Math.exp(-Math.max(0, e - onset) / 6));
      const v = t + gap + 0.028 + 0.006 * gauss();
      tr.push(+Math.max(0.05, t).toFixed(4));
      va.push(+Math.max(0.06, v).toFixed(4));
    }
    return [tr, va];
  }

  function rocPoints(auc, n) {
    // A smooth concave ROC with the given area (power-curve tpr = fpr^k).
    const a = clamp(auc, 0.5, 0.999);
    const k = Math.max(0.05, (1 - a) / a);
    const pts = [];
    for (let i = 0; i < n; i++) {
      const f = i / (n - 1);
      pts.push([+f.toFixed(3), +Math.pow(f, k).toFixed(3)]);
    }
    return pts;
  }

  async function apiEvaluate(b) {
    const seed = +b.seed, msl = +b.max_seq_len, hu = +b.hidden_units;
    const cell = String(b.cell_type), bidir = !!b.bidirectional;
    const oracle = (await getOracle())[String(seed)];
    const cfg = { max_seq_len: oracle.max_seq_len, hidden_units: oracle.hidden_units,
                  cell_type: oracle.cell_type };

    const rnd = mulberry32(hashStr(seed + "|" + msl + "|" + hu + "|" + cell + "|" + (bidir ? 1 : 0)));
    const jit = () => rnd() - 0.5;

    // quality vs. the per-seed oracle architecture (mirror reference._recurrent_metrics)
    let qh = 1 - 0.010 * Math.abs(hu - cfg.hidden_units) - 0.0015 * Math.max(0, 12 - hu);
    qh = clamp(qh, 0.72, 1.0);
    const qc = cell === "lstm" ? 1.0 : 0.992;
    const qb = bidir ? 1.004 : 1.0;
    const ql = clamp(1 - 0.02 * Math.max(0, cfg.max_seq_len - msl), 0.8, 1.0);
    const q = qh * qc * qb * ql;
    const nz = 1.06 - q;

    // oracle target = reference model metrics * ORACLE_MARGIN, so recover the
    // reference model's own metrics, then scale by this config's quality.
    const refAuc = oracle.auc / ORACLE_MARGIN, refF1 = oracle.f1 / ORACLE_MARGIN,
          refAcc = oracle.accuracy / ORACLE_MARGIN;
    let recAuc = clamp(refAuc * q + 0.010 * jit() * nz, 0.5, 0.995);
    let recF1 = clamp(refF1 * q + 0.020 * jit() * nz, 0.1, 0.99);
    let recAcc = clamp(refAcc * (0.94 + 0.06 * q) / 1.0 + 0.014 * jit() * nz, 0.45, 0.99);

    const convAuc = clamp(refAuc * 0.94 + 0.006 * jit(), 0.5, recAuc);
    const convF1 = clamp(refF1 * 0.92 + 0.010 * jit(), 0.1, recF1 + 0.01);
    const convAcc = clamp(refAcc * 0.95 + 0.008 * jit(), 0.45, recAcc + 0.01);

    const P = oracle.val_n ? Math.round((oracle.val_n) * 0.42) : 26;
    const N = (oracle.val_n || 64) - P;
    const cm = (f1) => {
      if (f1 <= 0.01) return [0, 0, N, P];
      const tp = Math.round(f1 * P), fn = P - tp;
      const fp = Math.min(N, Math.round(tp * (1 - f1) / f1)), tn = N - fp;
      return [tp, fp, tn, fn];
    };
    const [rtp, rfp, rtn, rfn] = cm(recF1);
    const [ctp, cfp, ctn, cfn] = cm(convF1);

    const [rtr, rva] = lossCurve(seed, "rec|" + hu + "|" + cell + "|" + (bidir ? 1 : 0),
      0.30 + 0.05 * (1 - q), 0.14 + 0.03 * q, 0.10 + 0.06 * (1 - q));
    const [ctr, cva] = lossCurve(seed, "conv", 0.34, 0.19, 0.13);

    const isOptimal = recF1 >= GREEN * oracle.f1 && recAcc >= GREEN * oracle.accuracy
      && recAuc >= GREEN * oracle.auc;

    return {
      max_seq_len: msl, hidden_units: hu, cell_type: cell, bidirectional: bidir,
      recurrent: {
        auc: +recAuc.toFixed(4), f1: +recF1.toFixed(4), accuracy: +recAcc.toFixed(4),
        tp: rtp, fp: rfp, tn: rtn, fn: rfn,
        roc_curve: rocPoints(recAuc, 26), train_loss: rtr, val_loss: rva,
      },
      conv1d: {
        auc: +convAuc.toFixed(4), f1: +convF1.toFixed(4), accuracy: +convAcc.toFixed(4),
        tp: ctp, fp: cfp, tn: ctn, fn: cfn,
        roc_curve: rocPoints(convAuc, 26), train_loss: ctr, val_loss: cva,
      },
      epochs: EPOCHS,
      is_optimal: isOptimal,
      hint: makeHint(recAuc, recF1, oracle, msl, hu, cell, cfg, bidir),
      caption: makeCaption(cell, hu, bidir, recAuc, convAuc, recF1, convF1),
      oracle,
    };
  }

  // Only shown while NOT optimal -- keep in sync with reference._hint().
  function makeHint(auc, f1, oracle, msl, hu, cell, cfg, bidir) {
    const behind = Math.max(oracle.auc - auc, oracle.f1 - f1);
    if (msl < 5)
      return "You are keeping only " + msl + " of up to 6 visits. Raise max_seq_len in the "
        + "Sequence Explorer tab so the model sees more recent history.";
    if (cell !== cfg.cell_type && behind > 0.01)
      return "Try the " + cfg.cell_type.toUpperCase() + " cell -- for this seed it separates the "
        + "classes a little better than " + cell.toUpperCase() + " at the same width.";
    if (hu <= 8 && behind > 0.005)
      return "The recurrent layer is very narrow (" + hu + " units). Increase hidden_units so it "
        + "has room to learn the pattern.";
    if (hu >= 52 && behind > 0.01)
      return hu + " hidden units is wide for ~" + (oracle.val_n || 64) + " validation patients -- "
        + "it is likely overfitting. Lower hidden_units, turn bidirectional off, or try GRU.";
    if (behind <= 0.02)
      return "So close -- you're right on the oracle line. A small change to hidden_units, "
        + "cell_type, or the bidirectional toggle should tip all three bars green.";
    return "Adjust hidden_units, cell_type or bidirectional to lift F1 and AUC above the oracle target.";
  }

  function makeCaption(cell, hu, bidir, recAuc, convAuc, recF1, convF1) {
    const who = (bidir ? "Bidirectional " : "") + cell.toUpperCase();
    const d = recAuc - convAuc;
    let tail;
    if (d >= 0.03) tail = "the Conv1D baseline trained faster but missed patients whose pain pattern only emerges over their last 2-3 visits.";
    else if (d >= 0.005) tail = "the Conv1D baseline is close behind -- this dataset rewards reading visits in order, but not by a huge margin.";
    else tail = "the Conv1D baseline essentially matched it -- with these settings the recurrent model isn't using visit order well.";
    return who + " with " + hu + " hidden units reached val AUC " + recAuc.toFixed(3)
      + " (F1 " + recF1.toFixed(3) + "); the Conv1D baseline reached AUC " + convAuc.toFixed(3)
      + " (F1 " + convF1.toFixed(3) + "). " + tail;
  }

  // ---- install ----
  window.fetch = async function (url, init) {
    const u = typeof url === "string" ? url : (url && url.url) || "";
    if (!/(^|\/)api\/(data|assign|evaluate)\b/.test(u)) return realFetch(url, init);
    const body = init && init.body ? JSON.parse(init.body) : {};
    let out;
    if (/\/api\/data\b/.test(u)) out = await apiData();
    else if (/\/api\/assign\b/.test(u)) out = await apiAssign(body.student_id);
    else out = await apiEvaluate(body);
    return new Response(JSON.stringify(out), { status: 200, headers: { "Content-Type": "application/json" } });
  };
})();
