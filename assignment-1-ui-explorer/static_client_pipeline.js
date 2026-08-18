// ============================================================
// Client-side port of reference.py's oracle pipeline, for the
// static GitHub Pages build (no backend available).
//
// The stratified split below is a faithful port of numpy's
// default_rng(seed).permutation() (SeedSequence + PCG64 + the
// exact bounded-rejection Fisher-Yates numpy uses), verified
// bit-for-bit against real numpy across all 900 seeds x both
// classes of this dataset before being ported here. This must
// stay in sync with reference.py if that file's math changes.
// ============================================================
(function(global){
"use strict";

const MASK32 = 0xffffffffn;
const MASK64 = 0xffffffffffffffffn;
const MASK128 = (1n << 128n) - 1n;

const XSHIFT = 16n;
const MULT_A = 0x931e8875n;
const MULT_B = 0x58f38dedn;
const MIX_MULT_L = 0xca01f9ddn;
const MIX_MULT_R = 0x4973f715n;
const INIT_A = 0x43b0d7e5n;
const INIT_B = 0x8b51f9ddn;
const PCG_MULT_128 = 0x2360ed051fc65da44385df649fccf645n;

function intToUint32Words(nBig){
  // little-endian uint32 words (lowest bits first), matching numpy's
  // _int_to_uint32_array
  const words = [];
  if (nBig === 0n) { words.push(0n); return words; }
  let n = nBig;
  while (n > 0n) { words.push(n & MASK32); n >>= 32n; }
  return words;
}

class SeedSequence {
  constructor(entropyInt){
    this.entropy = BigInt(entropyInt);
    this.poolSize = 4;
    this.pool = this._makePool();
  }
  _makePool(){
    let hashConst = INIT_A;
    const hashmix = (value) => {
      value &= MASK32;
      value ^= hashConst;
      hashConst = (hashConst * MULT_A) & MASK32;
      value = (value * hashConst) & MASK32;
      value ^= value >> XSHIFT;
      return value & MASK32;
    };
    const mix = (x, y) => {
      let result = (MIX_MULT_L * x - MIX_MULT_R * y) & MASK32;
      result ^= result >> XSHIFT;
      return result & MASK32;
    };
    const entropyWords = intToUint32Words(this.entropy); // spawn_key is always empty here
    const pool = new Array(this.poolSize).fill(0n);
    for (let i = 0; i < this.poolSize; i++){
      pool[i] = i < entropyWords.length ? hashmix(entropyWords[i]) : hashmix(0n);
    }
    for (let iSrc = 0; iSrc < this.poolSize; iSrc++){
      for (let iDst = 0; iDst < this.poolSize; iDst++){
        if (iSrc !== iDst) pool[iDst] = mix(pool[iDst], hashmix(pool[iSrc]));
      }
    }
    for (let iSrc = this.poolSize; iSrc < entropyWords.length; iSrc++){
      for (let iDst = 0; iDst < this.poolSize; iDst++){
        pool[iDst] = mix(pool[iDst], hashmix(entropyWords[iSrc]));
      }
    }
    return pool;
  }
  generateState(nWords){
    let hashConst = INIT_B;
    const state = [];
    let srcIdx = 0;
    for (let iDst = 0; iDst < nWords; iDst++){
      let dataVal = this.pool[srcIdx];
      dataVal ^= hashConst;
      hashConst = (hashConst * MULT_B) & MASK32;
      dataVal = (dataVal * hashConst) & MASK32;
      dataVal ^= dataVal >> XSHIFT;
      state.push(dataVal & MASK32);
      srcIdx = (srcIdx + 1) % this.poolSize;
    }
    return state;
  }
}

function wordsToUint64(words32){
  const out = [];
  for (let i = 0; i < words32.length; i += 2){
    const lo = words32[i];
    const hi = words32[i+1];
    out.push((hi << 32n) | lo);
  }
  return out;
}

class PCG64 {
  constructor(seedSeq){
    const words32 = seedSeq.generateState(8); // 8 uint32 -> 4 uint64
    const u64 = wordsToUint64(words32);
    // numpy's pcg64_set_seed: s=(seed[0]<<64)|seed[1]; i=(inc[0]<<64)|inc[1]
    // where seed=[val0,val1], inc=[val2,val3]
    const initstate = (u64[0] << 64n) | u64[1];
    const initseq   = (u64[2] << 64n) | u64[3];
    this.state = 0n;
    this.inc = ((initseq << 1n) | 1n) & MASK128;
    this._step();
    this.state = (this.state + initstate) & MASK128;
    this._step();
  }
  _step(){
    this.state = (this.state * PCG_MULT_128 + this.inc) & MASK128;
  }
  nextUint64(){
    this._step();
    const s = this.state;
    const xored = ((s >> 64n) ^ (s & MASK64)) & MASK64;
    const rot = (s >> 122n) & 0x3fn;
    if (rot === 0n) return xored;
    return ((xored >> rot) | (xored << (64n - rot))) & MASK64;
  }
}

class Rng {
  constructor(seedInt){
    this.bitgen = new PCG64(new SeedSequence(seedInt));
    this._bufferedUint32 = null;
  }
  _nextUint32(){
    if (this._bufferedUint32 !== null){
      const v = this._bufferedUint32;
      this._bufferedUint32 = null;
      return v;
    }
    const v64 = this.bitgen.nextUint64();
    const low = v64 & MASK32;
    const high = (v64 >> 32n) & MASK32;
    this._bufferedUint32 = high;
    return low;
  }
  _randomInterval(mxNum){
    // returns a Number (safe: mx is always small here, < 320)
    let mx = BigInt(mxNum);
    if (mx === 0n) return 0;
    let mask = mx;
    mask |= mask >> 1n; mask |= mask >> 2n; mask |= mask >> 4n;
    mask |= mask >> 8n; mask |= mask >> 16n; mask |= mask >> 32n;
    if (mx <= MASK32){
      for(;;){ const value = this._nextUint32() & mask; if (value <= mx) return Number(value); }
    } else {
      for(;;){ const value = this.bitgen.nextUint64() & mask; if (value <= mx) return Number(value); }
    }
  }
  permutation(arr){
    const x = arr.slice();
    for (let i = x.length - 1; i > 0; i--){
      const j = this._randomInterval(i);
      const tmp = x[i]; x[i] = x[j]; x[j] = tmp;
    }
    return x;
  }
}

// ============================================================
// student_to_seed: sha256(lower(strip(id))) as big-endian int % 900 + 100
// ============================================================
async function studentToSeed(studentId){
  const s = studentId.toLowerCase().trim();
  const bytes = new TextEncoder().encode(s);
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  let h = 0n;
  for (const b of digest) h = (h << 8n) | BigInt(b);
  return Number(h % 900n) + 100;
}

// ============================================================
// Pipeline: stratified split, standardize, train, metrics
// ============================================================
const FEATURE_COLS = [
  "age", "is_female", "number_of_unique_meds", "number_of_encounters", "number_of_procedures",
  "unique_procedures", "pain_severity", "body_height", "body_weight",
  "body_mass_index", "systolic_blood_pressure", "diastolic_blood_pressure",
  "heart_rate", "respiratory_rate", "qaly", "daly", "qols",
  "healthcare_expenses", "healthcare_coverage",
];

function stratifiedSplit(labels, valFraction, seed){
  const n = labels.length;
  const isVal = new Array(n).fill(false);
  const rng = new Rng(seed);
  for (const cls of [0, 1]){
    const idx = [];
    for (let i = 0; i < n; i++) if (labels[i] === cls) idx.push(i);
    const shuffled = rng.permutation(idx);
    const nVal = Math.round(valFraction * idx.length);
    for (let k = 0; k < nVal; k++) isVal[shuffled[k]] = true;
  }
  return isVal;
}

function standardize(Xtrain, Xval){
  const nFeat = Xtrain[0].length;
  const nTrain = Xtrain.length;
  const means = new Array(nFeat).fill(0);
  const stds = new Array(nFeat).fill(0);
  for (let j = 0; j < nFeat; j++){
    let s = 0;
    for (let i = 0; i < nTrain; i++) s += Xtrain[i][j];
    means[j] = s / nTrain;
  }
  for (let j = 0; j < nFeat; j++){
    let s = 0;
    for (let i = 0; i < nTrain; i++){ const d = Xtrain[i][j] - means[j]; s += d*d; }
    const variance = s / nTrain;
    let std = Math.sqrt(variance);
    if (std === 0) std = 1.0;
    stds[j] = std;
  }
  const scale = (X) => X.map(row => row.map((v,j) => (v - means[j]) / stds[j]));
  return { Xtrain: scale(Xtrain), Xval: scale(Xval), means, stds };
}

function sigmoid(z){ return 1.0 / (1.0 + Math.exp(-z)); }

function trainNumpyStyle(Xtrain, ytrain, lr, steps){
  const nFeat = Xtrain[0].length;
  const n = Xtrain.length;
  let w = new Array(nFeat).fill(0);
  let b = 0;
  const lossHist = [];
  const eps = 1e-9;
  for (let step = 0; step < steps; step++){
    const preds = new Array(n);
    for (let i = 0; i < n; i++){
      let z = b;
      const row = Xtrain[i];
      for (let j = 0; j < nFeat; j++) z += row[j] * w[j];
      preds[i] = sigmoid(z);
    }
    let lossSum = 0;
    for (let i = 0; i < n; i++){
      const y = ytrain[i], p = preds[i];
      lossSum += y * Math.log(p + eps) + (1 - y) * Math.log(1 - p + eps);
    }
    lossHist.push(round(-lossSum / n, 6));
    const errs = new Array(n);
    for (let i = 0; i < n; i++) errs[i] = preds[i] - ytrain[i];
    const gradW = new Array(nFeat).fill(0);
    for (let j = 0; j < nFeat; j++){
      let s = 0;
      for (let i = 0; i < n; i++) s += Xtrain[i][j] * errs[i];
      gradW[j] = s / n;
    }
    let gradB = 0;
    for (let i = 0; i < n; i++) gradB += errs[i];
    gradB /= n;
    for (let j = 0; j < nFeat; j++) w[j] -= lr * gradW[j];
    b -= lr * gradB;
  }
  return { w, b, lossHist };
}

function round(v, digits){
  const f = Math.pow(10, digits);
  return Math.round(v * f) / f;
}

function aucPairwise(yVal, probs){
  const pos = [], neg = [];
  for (let i = 0; i < yVal.length; i++) (yVal[i] === 1 ? pos : neg).push(probs[i]);
  let total = 0;
  for (const sp of pos) for (const sn of neg) { if (sp > sn) total += 1; else if (sp === sn) total += 0.5; }
  return total / (pos.length * neg.length);
}

function rocCurvePoints(yVal, probs, n=30){
  const pts = [];
  for (let k = 0; k < n; k++){
    const t = 1.0 - (k/(n-1));
    let tp=0, fp=0, fn=0, tn=0;
    for (let i = 0; i < probs.length; i++){
      const yp = probs[i] >= t ? 1 : 0;
      if (yp===1 && yVal[i]===1) tp++;
      else if (yp===1 && yVal[i]===0) fp++;
      else if (yp===0 && yVal[i]===1) fn++;
      else tn++;
    }
    const tpr = (tp+fn) > 0 ? tp/(tp+fn) : 0;
    const fpr = (fp+tn) > 0 ? fp/(fp+tn) : 0;
    pts.push([round(fpr,3), round(tpr,3)]);
  }
  return pts;
}

function evaluateSeed(patients, labels, seed, lr, steps, valFraction){
  const X = patients.map(p => FEATURE_COLS.map(c => Number(p[c])));
  const isVal = stratifiedSplit(labels, valFraction, seed);

  const Xtrain=[], ytrain=[], Xval=[], yval=[];
  for (let i = 0; i < X.length; i++){
    if (isVal[i]) { Xval.push(X[i]); yval.push(labels[i]); }
    else { Xtrain.push(X[i]); ytrain.push(labels[i]); }
  }

  const { Xtrain: XtrainS, Xval: XvalS } = standardize(Xtrain, Xval);
  const { w, b, lossHist } = trainNumpyStyle(XtrainS, ytrain, lr, steps);

  const probs = XvalS.map(row => {
    let z = b;
    for (let j = 0; j < row.length; j++) z += row[j]*w[j];
    return sigmoid(z);
  });

  const threshold = 0.5;
  let tp=0, fp=0, tn=0, fn=0;
  for (let i = 0; i < probs.length; i++){
    const pred = probs[i] >= threshold ? 1 : 0;
    const y = yval[i];
    if (pred===1 && y===1) tp++;
    else if (pred===1 && y===0) fp++;
    else if (pred===0 && y===0) tn++;
    else fn++;
  }
  const prec = (tp+fp)>0 ? tp/(tp+fp) : 0;
  const rec = (tp+fn)>0 ? tp/(tp+fn) : 0;
  const f1 = (prec+rec)>0 ? 2*prec*rec/(prec+rec) : 0;
  const acc = (tp+tn)/yval.length;
  const auc = aucPairwise(yval, probs);

  return {
    tp, fp, tn, fn,
    precision: round(prec,4), recall: round(rec,4), f1: round(f1,4),
    accuracy: round(acc,4), auc: round(auc,4),
    final_loss: round(lossHist[lossHist.length-1],4),
    train_n: Xtrain.length, val_n: Xval.length,
    loss_history: lossHist,
    roc_curve: rocCurvePoints(yval, probs),
  };
}

global.NNExplorerPipeline = {
  studentToSeed, evaluateSeed, FEATURE_COLS,
  _internal: { Rng, SeedSequence, PCG64 },
};

})(window);
