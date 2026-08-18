// ============================================================
// Static-site API shims: mirror run_sandbox.py's Flask JSON
// endpoints, computing everything in-browser via
// NNExplorerPipeline (static_client_pipeline.js), since GitHub
// Pages has no backend to call. Keep these in sync with
// run_sandbox.py if its /api/* routes change.
// ============================================================
const AUC_OK_MIN = 0.75, AUC_OK_MAX = 0.92;
const OPTIMAL_LR = 0.5, OPTIMAL_STEPS = 300, OPTIMAL_VAL_FRACTION = 0.20;

let _patientsCache = null;
async function _getPatients(){
  if (_patientsCache) return _patientsCache;
  const r = await fetch('patients.json');
  _patientsCache = await r.json();
  return _patientsCache;
}

let _oracleTableCache = null;
async function _getOracleTable(){
  if (_oracleTableCache) return _oracleTableCache;
  const r = await fetch('oracle_table.json');
  _oracleTableCache = await r.json();
  return _oracleTableCache;
}

let _painRe = null;
function _flagKeywords(text){
  // Built lazily: PAIN_KEYWORDS is declared later, in the main inline script.
  if (!_painRe) _painRe = new RegExp('\\b(?:' + PAIN_KEYWORDS.map(k=>k.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')).join('|') + ')\\b');
  return _painRe.test((text||'').toLowerCase()) ? 1 : 0;
}

function _goodSeed(baseSeed, oracleTable){
  for (let offset = 0; offset < 900; offset++){
    const candidate = ((baseSeed - 100 + offset) % 900) + 100;
    const entry = oracleTable[String(candidate)];
    if (entry && entry.auc >= AUC_OK_MIN && entry.auc <= AUC_OK_MAX) return candidate;
  }
  return baseSeed;
}

function _makeHint(m, oracle, lr, steps, valFraction){
  if (!oracle || !Object.keys(oracle).length)
    return "Explore different parameter combinations to find the best performance.";
  const aucDiff = oracle.auc - m.auc;
  const hist = m.loss_history || [];
  if (Math.abs(valFraction - 0.20) > 0.01)
    return "Val fraction should be 0.20. The oracle uses exactly 20% validation -- adjust that slider first.";
  if (hist.length >= 20){
    const tailDelta = Math.abs(hist[hist.length-1] - hist[hist.length-10]);
    if (tailDelta < 0.001 && m.final_loss > oracle.final_loss + 0.05)
      return "Loss has plateaued too high -- learning rate is probably too small. Try 0.3 to 0.7.";
    if (hist[hist.length-1] > hist[hist.length-10] && m.final_loss > oracle.final_loss + 0.02)
      return "Loss is oscillating -- learning rate might be too large. Try reducing it.";
  }
  if (steps < 80 && aucDiff > 0.05)
    return "Model hasn't converged yet. Increase the number of steps (try 200+).";
  if (lr < 0.15 && aucDiff > 0.05)
    return "Learning rate is very small. Try 0.3-0.6.";
  if (aucDiff > 0.06)
    return `AUC is ${m.auc.toFixed(3)} -- target is ${oracle.auc.toFixed(3)}. Try a higher learning rate.`;
  if (aucDiff > 0.025)
    return `Getting closer! AUC ${m.auc.toFixed(3)} vs target ${oracle.auc.toFixed(3)}. Fine-tune lr or add steps.`;
  if (aucDiff > 0.005)
    return `Almost there -- AUC within ${aucDiff.toFixed(3)}. Small tweaks should get you over the line.`;
  return "Keep experimenting -- you're in the right range. Try small adjustments.";
}

async function apiData(){
  const patients = await _getPatients();
  const labels = patients.map(p => _flagKeywords(p.condition_text));
  const nPos = labels.reduce((a,b)=>a+b,0);
  const nNeg = labels.length - nPos;
  return {
    patients,
    n_total: labels.length, n_positive: nPos, n_negative: nNeg,
    positive_rate: Math.round((nPos/labels.length)*1000)/1000,
    labels,
    columns: Object.keys(patients[0] || {}),
  };
}

async function apiAssign(studentId){
  const id = (studentId||'').trim();
  if (!id) return {error: 'student_id required'};
  const [patients, oracleTable] = await Promise.all([_getPatients(), _getOracleTable()]);
  const rawSeed = await NNExplorerPipeline.studentToSeed(id);
  const seed = _goodSeed(rawSeed, oracleTable);
  let oracle = oracleTable[String(seed)];
  if (!oracle){
    const labels = patients.map(p => _flagKeywords(p.condition_text));
    const m = NNExplorerPipeline.evaluateSeed(patients, labels, seed, OPTIMAL_LR, OPTIMAL_STEPS, OPTIMAL_VAL_FRACTION);
    const { loss_history, roc_curve, ...rest } = m;
    oracle = rest;
  }
  return {seed, oracle};
}

async function apiEvaluate(seed, lr, steps, valFraction){
  const [patients, oracleTable] = await Promise.all([_getPatients(), _getOracleTable()]);
  const labels = patients.map(p => _flagKeywords(p.condition_text));
  const m = NNExplorerPipeline.evaluateSeed(patients, labels, seed, lr, steps, valFraction);
  const oracle = oracleTable[String(seed)] || {};
  const isOptimal = !!(Object.keys(oracle).length &&
    Math.abs(m.auc - oracle.auc) <= 0.005 &&
    Math.abs(m.accuracy - oracle.accuracy) <= 0.010 &&
    Math.abs(m.f1 - oracle.f1) <= 0.010 &&
    m.final_loss <= oracle.final_loss + 0.010);
  const hint = _makeHint(m, oracle, lr, steps, valFraction);
  return Object.assign({}, m, {is_optimal: isOptimal, hint, oracle});
}

async function apiSeedCompare(seed, lr, steps, valFraction){
  const patients = await _getPatients();
  const labels = patients.map(p => _flagKeywords(p.condition_text));
  const results = [];
  for (let s of [seed-1, seed, seed+1]){
    s = Math.max(100, Math.min(999, s));
    const m = NNExplorerPipeline.evaluateSeed(patients, labels, s, lr, steps, valFraction);
    results.push({seed: s, auc: m.auc, accuracy: m.accuracy, f1: m.f1, loss: m.final_loss});
  }
  return {results, student_seed: seed};
}

async function apiSubmit(_payload){
  return {error: 'submission disabled in public explorer'};
}
