const A = require('./core.js');

// ---------- 1. Sudoku validity ----------
function validSolution(g) {
  const N = 4;
  for (let r = 0; r < N; r++) { const s = new Set(); for (let c = 0; c < N; c++) s.add(g[r*N+c]); if (s.size !== 4) return false; }
  for (let c = 0; c < N; c++) { const s = new Set(); for (let r = 0; r < N; r++) s.add(g[r*N+c]); if (s.size !== 4) return false; }
  for (let br = 0; br < 2; br++) for (let bc = 0; bc < 2; bc++) {
    const s = new Set(); for (let i=0;i<2;i++) for (let j=0;j<2;j++) s.add(g[(br*2+i)*N+(bc*2+j)]);
    if (s.size !== 4) return false;
  }
  return true;
}
const rng = new A.RNG(12345);
let badSol = 0, badUniq = 0, blanksSum = 0;
for (let i = 0; i < 200; i++) {
  const { puzzle, solution, blanks } = A.makePuzzle(rng, 8);
  if (!validSolution(solution)) badSol++;
  if (A.countSolutions(puzzle, 2, null, null) !== 1) badUniq++;
  blanksSum += blanks;
}
console.log(`[Sudoku] invalid solutions: ${badSol}/200, non-unique puzzles: ${badUniq}/200, avg blanks: ${(blanksSum/200).toFixed(1)}`);

// ---------- 2. Gradient check (finite differences) ----------
// Build a small differentiable scalar from the same ops and compare analytic grad to numeric.
function gradCheck() {
  const r = new A.RNG(7);
  const m = A.makeModel(r, 8, 8);
  // one puzzle batch
  const items = A.genDataset(r, 4, 8);
  const enc = A.encodeBatch(items);
  // T=1 isolates the autograd graph (no deliberate cross-step truncation)
  const cfg = { n: 3, T: 1, mode: 'trm' };

  // analytic grads
  A.trainBatch(m, enc, cfg);
  const analytic = m.params.map(p => Float64Array.from(p.grad));

  // numeric grads for a few random entries of W1 and Wy
  const eps = 1e-5;
  function lossOnly(model) {
    // recompute loss with tape but ignore grads; reuse trainBatch loss (it also sets grads but we only read loss)
    return A.trainBatch(model, enc, cfg).lossVal;
  }
  let maxRel = 0, checks = 0;
  const targets = [0, 4, 6]; // param indices W1,Wz? indices: 0=W1,1=b1,2=W2,3=b2,4=Wz,5=bz,6=Wy...
  for (const pi of targets) {
    const p = m.params[pi];
    const nCheck = Math.min(6, p.data.length);
    for (let t = 0; t < nCheck; t++) {
      const idx = Math.floor(r.next() * p.data.length);
      const orig = p.data[idx];
      p.data[idx] = orig + eps; const lp = lossOnly(m);
      p.data[idx] = orig - eps; const lm = lossOnly(m);
      p.data[idx] = orig;
      const num = (lp - lm) / (2 * eps);
      const ana = analytic[pi][idx];
      const rel = Math.abs(num - ana) / (Math.max(Math.abs(num), Math.abs(ana)) + 1e-8);
      maxRel = Math.max(maxRel, rel); checks++;
    }
  }
  console.log(`[GradCheck] ${checks} params, max relative error vs finite-diff: ${maxRel.toExponential(3)}`);
  return maxRel;
}
const gcErr = gradCheck();

// ---------- 3. Real training run ----------
function trainRun() {
  const r = new A.RNG(2026);
  const H = 48, Hz = 48;
  const m = A.makeModel(r, H, Hz);
  console.log(`[Model] params = ${m.nParams}`);
  const blanks = 6;
  const train = A.genDataset(r, 400, blanks);
  const test = A.genDataset(r, 100, blanks);
  const testEnc = A.encodeBatch(test);
  const cfg = { n: 3, T: 3, mode: 'trm' };
  const lr = 0.01, B = 64;
  let firstLoss = null;
  for (let step = 0; step < 400; step++) {
    const batch = [];
    for (let b = 0; b < B; b++) batch.push(train[r.int(train.length)]);
    const enc = A.encodeBatch(batch);
    const { lossVal } = A.trainBatch(m, enc, cfg);
    A.adamStep(m, lr, 1.0);
    if (firstLoss === null) firstLoss = lossVal;
    if (step % 100 === 99) {
      const inf = A.infer(m, testEnc, cfg.T, cfg.n);
      const ev = A.evaluate(inf.Ydata, test);
      console.log(`  step ${step+1}: loss=${lossVal.toFixed(3)} testExact=${(ev.exactAcc*100).toFixed(1)}% testCell=${(ev.cellAcc*100).toFixed(1)}%`);
    }
  }
  // test-time compute: more steps than trained
  console.log('[Test-time compute] exact-match vs inference improvement steps:');
  for (const steps of [1, 2, 3, 4, 6, 8]) {
    const inf = A.infer(m, testEnc, steps, cfg.n);
    const ev = A.evaluate(inf.Ydata, test);
    console.log(`   steps=${steps}: exact=${(ev.exactAcc*100).toFixed(1)}% cell=${(ev.cellAcc*100).toFixed(1)}%`);
  }
}
trainRun();

// ---------- 4. TRM vs HRM quick convergence ----------
function ablation() {
  console.log('[TRM vs HRM] loss after 150 steps (same seed/data):');
  for (const mode of ['trm', 'hrm']) {
    const r = new A.RNG(99);
    const m = A.makeModel(r, 48, 48);
    const train = A.genDataset(r, 300, 6);
    const cfg = { n: 3, T: 3, mode };
    let last = 0;
    for (let step = 0; step < 150; step++) {
      const batch = []; for (let b = 0; b < 64; b++) batch.push(train[r.int(train.length)]);
      const enc = A.encodeBatch(batch);
      last = A.trainBatch(m, enc, cfg).lossVal;
      A.adamStep(m, 0.01, 1.0);
    }
    const testEnc = A.encodeBatch(A.genDataset(r, 100, 6));
    const inf = A.infer(m, testEnc, 3, 3);
    const ev = A.evaluate(inf.Ydata, A.genDataset(new A.RNG(99), 100, 6)); // note: separate; just for loss trend we use last
    console.log(`   mode=${mode}: finalLoss=${last.toFixed(3)}`);
  }
}
abl