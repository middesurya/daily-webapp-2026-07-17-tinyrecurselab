/* TinyRecurseLab core — from-scratch reverse-mode tensor autograd + TRM-style
   recursive reasoner trained on 4x4 Sudoku. Pure JS, no deps.
   Shared by every module in index.html. Tested standalone in Node. */
(function (global) {
'use strict';

// ---------- tiny RNG (seedable, reproducible) ----------
function RNG(seed) { this.s = (seed >>> 0) || 1; }
RNG.prototype.next = function () { // xorshift32 -> [0,1)
  let x = this.s;
  x ^= x << 13; x ^= x >>> 17; x ^= x << 5; x >>>= 0;
  this.s = x; return x / 4294967296;
};
RNG.prototype.int = function (n) { return Math.floor(this.next() * n); };
RNG.prototype.normal = function () { // Box-Muller
  let u = 1 - this.next(), v = this.next();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};

// ======================================================================
// AUTOGRAD ENGINE  — Tensor = [rows, cols] flat Float64Array, row-major.
// Each op records a backward closure; T.backward() runs reverse-mode.
// ======================================================================
let TAPE = null;            // current recording tape (array of nodes) or null
function tapeOn() { TAPE = []; }
function tapeReset() { TAPE = null; }

function Tensor(rows, cols, data) {
  this.r = rows; this.c = cols;
  this.data = data || new Float64Array(rows * cols);
  this.grad = null;         // lazily allocated on backward
  this._bw = null;          // backward closure
  if (TAPE) TAPE.push(this);
}
Tensor.prototype.zeroGrad = function () {
  if (!this.grad) this.grad = new Float64Array(this.r * this.c);
  else this.grad.fill(0);
};
function T(rows, cols, data) { return new Tensor(rows, cols, data); }

// matmul: A[m,k] @ B[k,n] -> [m,n]
function matmul(A, B) {
  const m = A.r, k = A.c, n = B.c;
  const out = T(m, n);
  const a = A.data, b = B.data, o = out.data;
  for (let i = 0; i < m; i++) {
    for (let p = 0; p < k; p++) {
      const av = a[i * k + p];
      if (av === 0) continue;
      const bo = p * n, io = i * n;
      for (let j = 0; j < n; j++) o[io + j] += av * b[bo + j];
    }
  }
  out._bw = function () {
    const go = out.grad;
    A.grad = A.grad || new Float64Array(m * k);
    B.grad = B.grad || new Float64Array(k * n);
    const ga = A.grad, gb = B.grad;
    // dA = dO @ B^T ; dB = A^T @ dO
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        const g = go[i * n + j];
        if (g === 0) continue;
        for (let p = 0; p < k; p++) {
          ga[i * k + p] += g * b[p * n + j];
          gb[p * n + j] += g * a[i * k + p];
        }
      }
    }
  };
  return out;
}

// addBias: X[m,n] + b[1,n] (broadcast over rows)
function addBias(X, b) {
  const m = X.r, n = X.c, out = T(m, n);
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++)
    out.data[i * n + j] = X.data[i * n + j] + b.data[j];
  out._bw = function () {
    const go = out.grad;
    X.grad = X.grad || new Float64Array(m * n);
    b.grad = b.grad || new Float64Array(n);
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) {
      X.grad[i * n + j] += go[i * n + j];
      b.grad[j] += go[i * n + j];
    }
  };
  return out;
}

// elementwise add of two equal-shape tensors (residual connections)
function add(A, B) {
  const m = A.r, n = A.c, out = T(m, n);
  for (let i = 0; i < m * n; i++) out.data[i] = A.data[i] + B.data[i];
  out._bw = function () {
    const go = out.grad;
    A.grad = A.grad || new Float64Array(m * n);
    B.grad = B.grad || new Float64Array(m * n);
    for (let i = 0; i < m * n; i++) { A.grad[i] += go[i]; B.grad[i] += go[i]; }
  };
  return out;
}

function relu(X) {
  const n = X.r * X.c, out = T(X.r, X.c);
  for (let i = 0; i < n; i++) out.data[i] = X.data[i] > 0 ? X.data[i] : 0;
  out._bw = function () {
    const go = out.grad;
    X.grad = X.grad || new Float64Array(n);
    for (let i = 0; i < n; i++) if (X.data[i] > 0) X.grad[i] += go[i];
  };
  return out;
}

function tanh(X) {
  const n = X.r * X.c, out = T(X.r, X.c);
  for (let i = 0; i < n; i++) out.data[i] = Math.tanh(X.data[i]);
  out._bw = function () {
    const go = out.grad, od = out.data;
    X.grad = X.grad || new Float64Array(n);
    for (let i = 0; i < n; i++) X.grad[i] += go[i] * (1 - od[i] * od[i]);
  };
  return out;
}

// concat two tensors along columns: A[m,a] , B[m,b] -> [m,a+b]
function concat(A, B) {
  const m = A.r, a = A.c, b = B.c, out = T(m, a + b);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < a; j++) out.data[i * (a + b) + j] = A.data[i * a + j];
    for (let j = 0; j < b; j++) out.data[i * (a + b) + a + j] = B.data[i * b + j];
  }
  out._bw = function () {
    const go = out.grad;
    A.grad = A.grad || new Float64Array(m * a);
    B.grad = B.grad || new Float64Array(m * b);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < a; j++) A.grad[i * a + j] += go[i * (a + b) + j];
      for (let j = 0; j < b; j++) B.grad[i * b + j] += go[i * (a + b) + a + j];
    }
  };
  return out;
}

// softmax cross-entropy over groups of `K` logits.
// logits X[m, cells*K], target int array length m*cells. Returns scalar loss.
// Only cells with target>=0 count (allow ignoring). Returns {loss, node}.
function softmaxCE(X, targets, cells, K) {
  const m = X.r; // batch
  const out = T(1, 1);
  let loss = 0, count = 0;
  const probs = new Float64Array(m * cells * K);
  for (let i = 0; i < m; i++) {
    for (let c = 0; c < cells; c++) {
      const base = i * cells * K + c * K;
      let mx = -Infinity;
      for (let k = 0; k < K; k++) mx = Math.max(mx, X.data[base + k]);
      let sum = 0;
      for (let k = 0; k < K; k++) { const e = Math.exp(X.data[base + k] - mx); probs[base + k] = e; sum += e; }
      for (let k = 0; k < K; k++) probs[base + k] /= sum;
      const tgt = targets[i * cells + c];
      if (tgt >= 0) { loss += -Math.log(probs[base + tgt] + 1e-12); count++; }
    }
  }
  out.data[0] = count ? loss / count : 0;
  out._bw = function () {
    const scale = (out.grad ? out.grad[0] : 1) / (count || 1);
    X.grad = X.grad || new Float64Array(m * cells * K);
    for (let i = 0; i < m; i++) for (let c = 0; c < cells; c++) {
      const tgt = targets[i * cells + c];
      if (tgt < 0) continue;
      const base = i * cells * K + c * K;
      for (let k = 0; k < K; k++)
        X.grad[base + k] += scale * (probs[base + k] - (k === tgt ? 1 : 0));
    }
  };
  return out;
}

// backward from a scalar tensor
function backward(root) {
  // seed
  root.grad = root.grad || new Float64Array(root.r * root.c);
  root.grad[0] = 1;
  const tape = TAPE;
  for (let i = tape.length - 1; i >= 0; i--) {
    const node = tape[i];
    if (node._bw && node.grad) node._bw();
  }
}

// ======================================================================
// 4x4 SUDOKU  (digits 1..4, 2x2 boxes). 0 = blank.
// ======================================================================
const N = 4, CELLS = 16;
function boxIndex(r, c) { return (Math.floor(r / 2) * 2 + Math.floor(c / 2)); }
function okToPlace(g, r, c, v) {
  for (let k = 0; k < N; k++) {
    if (g[r * N + k] === v) return false;       // row
    if (g[k * N + c] === v) return false;        // col
  }
  const br = Math.floor(r / 2) * 2, bc = Math.floor(c / 2) * 2;
  for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++)
    if (g[(br + i) * N + (bc + j)] === v) return false;
  return true;
}
// count solutions up to `cap`, filling given grid (0=blank). returns count; if solveInto given, stores first solution.
function countSolutions(grid, cap, rng, solveInto) {
  const g = grid.slice();
  let count = 0;
  function rec() {
    if (count >= cap) return;
    let pos = -1;
    for (let i = 0; i < CELLS; i++) if (g[i] === 0) { pos = i; break; }
    if (pos === -1) { count++; if (solveInto && count === 1) for (let i = 0; i < CELLS; i++) solveInto[i] = g[i]; return; }
    const r = Math.floor(pos / N), c = pos % N;
    const order = [1, 2, 3, 4];
    if (rng) for (let i = 3; i > 0; i--) { const j = rng.int(i + 1); const t = order[i]; order[i] = order[j]; order[j] = t; }
    for (const v of order) {
      if (okToPlace(g, r, c, v)) { g[pos] = v; rec(); g[pos] = 0; if (count >= cap) return; }
    }
  }
  rec();
  return count;
}
function fullSolution(rng) {
  const empty = new Int8Array(CELLS);
  const sol = new Int8Array(CELLS);
  countSolutions(empty, 1, rng, sol);
  return sol;
}
// make a puzzle with a unique solution by removing cells greedily.
function makePuzzle(rng, targetBlanks) {
  const sol = fullSolution(rng);
  const puz = sol.slice();
  const idx = [...Array(CELLS).keys()];
  for (let i = CELLS - 1; i > 0; i--) { const j = rng.int(i + 1); const t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
  let blanks = 0;
  for (const p of idx) {
    if (blanks >= targetBlanks) break;
    const saved = puz[p]; puz[p] = 0;
    // keep removal only if still unique
    if (countSolutions(puz, 2, null, null) === 1) blanks++;
    else puz[p] = saved;
  }
  return { puzzle: puz, solution: sol, blanks };
}

// encode a batch of puzzles -> X one-hot [B, 16*5], targets int [B*16] (0..3)
// givens channel: index 0 = blank, 1..4 = digit.
function encodeBatch(items) {
  const B = items.length;
  const X = new Float64Array(B * CELLS * 5);
  const targets = new Int32Array(B * CELLS);
  for (let b = 0; b < B; b++) {
    const it = items[b];
    for (let c = 0; c < CELLS; c++) {
      const g = it.puzzle[c];
      X[b * CELLS * 5 + c * 5 + g] = 1;          // 0=blank else digit
      targets[b * CELLS + c] = it.solution[c] - 1; // 0..3
    }
  }
  return { X, targets, B };
}

// ======================================================================
// TINY RECURSIVE MODEL
//  state: Y logits [B,16*4], z latent [B,Hz].
//  core f (2-layer MLP over concat(Xflat, Y, z)) produces dz and dy.
//  supervision steps T; within each, recurse z n times then update Y once.
//  mode 'trm'  -> backprop through the whole supervision step (full unroll)
//  mode 'hrm'  -> 1-step gradient: detach z,Y before final update (approx)
// ======================================================================
const XIN = CELLS * 5;   // 80
const YIN = CELLS * 4;   // 64

function heInit(rng, r, c, scale) {
  const t = T(r, c);
  const s = (scale || Math.sqrt(2 / r));
  for (let i = 0; i < r * c; i++) t.data[i] = rng.normal() * s;
  return t;
}

function makeModel(rng, H, Hz) {
  H = H || 48; Hz = Hz || 48;
  const inDim = XIN + YIN + Hz;
  tapeReset();
  const m = {
    H, Hz,
    W1: heInit(rng, inDim, H), b1: T(1, H),
    W2: heInit(rng, H, H), b2: T(1, H),
    Wz: heInit(rng, H, Hz, 0.1), bz: T(1, Hz),
    Wy: heInit(rng, H, YIN, 0.1), by: T(1, YIN),
    // halt head (ACT): predicts P(halt) from z -> scalar
    Wh: heInit(rng, Hz, 1, 0.1), bh: T(1, 1),
  };
  m.params = [m.W1, m.b1, m.W2, m.b2, m.Wz, m.bz, m.Wy, m.by, m.Wh, m.bh];
  m.nParams = m.params.reduce((s, p) => s + p.r * p.c, 0);
  // Adam state
  m.adam = m.params.map(p => ({ mt: new Float64Array(p.r * p.c), vt: new Float64Array(p.r * p.c) }));
  m.t = 0;
  return m;
}

// one core pass: given leaf tensors Xt,Yt,zt -> {dz, dy, h2}
function corePass(m, Xt, Yt, zt) {
  const inp = concat(concat(Xt, Yt), zt);      // [B, inDim]
  const h1 = relu(addBias(matmul(inp, m.W1), m.b1));
  const h2 = relu(addBias(matmul(h1, m.W2), m.b2));
  const dz = addBias(matmul(h2, m.Wz), m.bz);
  const dy = addBias(matmul(h2, m.Wy), m.by);
  return { dz, dy, h2 };
}

// Forward+loss for one batch. Returns {loss, YdataFinal, zFinal, lossVal, haltInfo}
// If train, builds tape & accumulates grads into params (does NOT step).
function trainBatch(m, enc, cfg) {
  const B = enc.B, n = cfg.n, Tsup = cfg.T, mode = cfg.mode || 'trm';
  // persistent numeric state carried across supervision steps (detached)
  let Ydata = new Float64Array(B * YIN);     // start logits 0
  let zdata = new Float64Array(B * m.Hz);
  let totalLoss = 0;
  for (const p of m.params) { p.grad = p.grad || new Float64Array(p.r * p.c); p.grad.fill(0); }

  for (let s = 0; s < Tsup; s++) {
    tapeOn();
    const Xt = T(B, XIN, Float64Array.from(enc.X)); // constant input leaf
    let Yt = T(B, YIN, Float64Array.from(Ydata));
    let zt = T(B, m.Hz, Float64Array.from(zdata));

    // recurse z n times (updating latent scratchpad), Y held
    for (let i = 0; i < n; i++) {
      if (mode === 'hrm' && i < n - 1) {
        // 1-step gradient: run without tape, update numeric z only
        tapeReset();
        const cp = corePass(m, Xt, Yt, zt);
        const nz = new Float64Array(B * m.Hz);
        for (let q = 0; q < nz.length; q++) nz[q] = Math.tanh(zt.data[q] + cp.dz.data[q]);
        tapeOn();
        zt = T(B, m.Hz, nz);
      } else {
        const cp = corePass(m, Xt, Yt, zt);
        zt = tanh(add(zt, cp.dz));
      }
    }
    // one Y update using final z
    const cpY = corePass(m, Xt, Yt, zt);
    const Ynew = add(Yt, cpY.dy);
    // deep supervision loss at this step
    const loss = softmaxCE(Ynew, enc.targets, CELLS, 4);
    backward(loss);
    totalLoss += loss.data[0];

    // carry state forward (detach): copy numeric values
    Ydata = Float64Array.from(Ynew.data);
    zdata = Float64Array.from(zt.data);
    tapeReset();
  }
  // average grad over supervision steps
  const invT = 1 / Tsup;
  for (const p of m.params) for (let i = 0; i < p.grad.length; i++) p.grad[i] *= invT;
  return { lossVal: totalLoss / Tsup, Ydata };
}

// Adam step
function adamStep(m, lr, clip) {
  m.t++;
  const b1 = 0.9, b2 = 0.999, eps = 1e-8;
  const bc1 = 1 - Math.pow(b1, m.t), bc2 = 1 - Math.pow(b2, m.t);
  for (let pi = 0; pi < m.params.length; pi++) {
    const p = m.params[pi], st = m.adam[pi], g = p.grad;
    for (let i = 0; i < g.length; i++) {
      let gi = g[i];
      if (clip) { if (gi > clip) gi = clip; else if (gi < -clip) gi = -clip; }
      st.mt[i] = b1 * st.mt[i] + (1 - b1) * gi;
      st.vt[i] = b2 * st.vt[i] + (1 - b2) * gi * gi;
      const mh = st.mt[i] / bc1, vh = st.vt[i] / bc2;
      p.data[i] -= lr * mh / (Math.sqrt(vh) + eps);
    }
  }
}

// Inference: run recursion for a chosen number of improvement steps, return
// predicted grid + per-step exact-match info + halt probs. No tape.
function infer(m, enc, steps, n) {
  const B = enc.B;
  let Ydata = new Float64Array(B * YIN);
  let zdata = new Float64Array(B * m.Hz);
  const stepsHistory = [];
  tapeReset();
  for (let s = 0; s < steps; s++) {
    const Xt = T(B, XIN, Float64Array.from(enc.X));
    let Yt = T(B, YIN, Ydata);
    let zt = T(B, m.Hz, zdata);
    for (let i = 0; i < n; i++) {
      const cp = corePass(m, Xt, Yt, zt);
      const nz = new Float64Array(B * m.Hz);
      for (let q = 0; q < nz.length; q++) nz[q] = Math.tanh(zt.data[q] + cp.dz.data[q]);
      zt = T(B, m.Hz, nz);
    }
    const cpY = corePass(m, Xt, Yt, zt);
    const nY = new Float64Array(B * YIN);
    for (let q = 0; q < nY.length; q++) nY[q] = Yt.data[q] + cpY.dy.data[q];
    Ydata = nY; zdata = zt.data;
    // halt prob from z: sigmoid(z@Wh+bh)
    const halt = new Float64Array(B);
    for (let b = 0; b < B; b++) {
      let acc = m.bh.data[0];
      for (let j = 0; j < m.Hz; j++) acc += zdata[b * m.Hz + j] * m.Wh.data[j];
      halt[b] = 1 / (1 + Math.exp(-acc));
    }
    stepsHistory.push({ Ydata: Float64Array.from(Ydata), z: Float64Array.from(zdata), halt });
  }
  return { Ydata, stepsHistory, zdata };
}

// decode Y logits -> predicted digits [B*16] (1..4); compute exact-match & cell acc vs solution
function evaluate(Ydata, items) {
  const B = items.length;
  let exact = 0, cellCorrect = 0, cellTotal = 0;
  const preds = new Int8Array(B * CELLS);
  for (let b = 0; b < B; b++) {
    let allRight = true;
    for (let c = 0; c < CELLS; c++) {
      const base = b * YIN + c * 4;
      let best = 0, bv = -Infinity;
      for (let k = 0; k < 4; k++) if (Ydata[base + k] > bv) { bv = Ydata[base + k]; best = k; }
      const pred = best + 1;
      preds[b * CELLS + c] = pred;
      const sol = items[b].solution[c];
      if (pred === sol) cellCorrect++; else allRight = false;
      cellTotal++;
    }
    if (allRight) exact++;
  }
  return { exactAcc: exact / B, cellAcc: cellCorrect / cellTotal, preds };
}

function genDataset(rng, count, blanks) {
  const items = [];
  for (let i = 0; i < count; i++) items.push(makePuzzle(rng, blanks));
  return items;
}

const API = {
  RNG, Tensor, T, tapeOn, tapeReset, matmul, addBias, add, relu, tanh, concat,
  softmaxCE, backward,
  N, CELLS, XIN, YIN, okToPlace, countSolutions, fullSolution, makePuzzle,
  encodeBatch, makeModel, corePass, trainBatch, adamStep, infer, evaluate, genDataset,
};
if (typeof module !== 'undefined' && module.exports) module.exports = API;
global.TRL = API;
})(typeof window !== 'undefined' ? window : globalThis);
