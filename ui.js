/* TinyRecurseLab — UI layer. Depends on global TRL (core.js). */
(function () {
'use strict';
const A = window.TRL;
const $ = id => document.getElementById(id);
const CSS = getComputedStyle(document.documentElement);
const COL = {
  a1: '#6ee7ff', a2: '#b57bff', a3: '#5df2a0', a4: '#ffcf6e', a5: '#ff7ab0',
  grid: '#233150', line: '#1e2940', mut: '#8b98b0', dim: '#5b6982', bad: '#ff7086', ink: '#e8edf6'
};

// ---------- shared state ----------
const S = {
  model: null, cfg: { n: 3, T: 3, mode: 'trm' }, blanks: 6,
  test: null, ready: false, job: null
};

// ---------- canvas helpers ----------
function setup(cv) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  if (cv._dpr !== dpr) {
    const w = cv.width, h = cv.height;           // attribute size (CSS px) — valid before scaling
    cv.style.width = w + 'px'; cv.style.height = h + 'px';
    cv._w = w; cv._h = h; cv.width = w * dpr; cv.height = h * dpr; cv._dpr = dpr;
  }
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cv._w, cv._h);
  return { ctx, w: cv._w, h: cv._h };
}
function axes(ctx, w, h, pad, opts) {
  ctx.strokeStyle = COL.line; ctx.lineWidth = 1;
  ctx.strokeRect(pad.l, pad.t, w - pad.l - pad.r, h - pad.t - pad.b);
  ctx.fillStyle = COL.dim; ctx.font = '10px ui-monospace,monospace';
  if (opts && opts.xlabel) { ctx.textAlign = 'center'; ctx.fillText(opts.xlabel, pad.l + (w - pad.l - pad.r) / 2, h - 4); }
  if (opts && opts.ylabel) { ctx.save(); ctx.translate(11, pad.t + (h - pad.t - pad.b) / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center'; ctx.fillText(opts.ylabel, 0, 0); ctx.restore(); }
}
function plotLine(ctx, pts, xr, yr, box, color, lw) {
  if (pts.length < 1) return;
  const sx = x => box.l + (x - xr[0]) / (xr[1] - xr[0] + 1e-9) * (box.w);
  const sy = y => box.t + box.h - (y - yr[0]) / (yr[1] - yr[0] + 1e-9) * (box.h);
  ctx.strokeStyle = color; ctx.lineWidth = lw || 2; ctx.lineJoin = 'round';
  ctx.beginPath();
  pts.forEach((p, i) => { const X = sx(p.x), Y = sy(p.y); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); });
  ctx.stroke();
  return { sx, sy };
}
function gridlines(ctx, box, yr, ticks, fmt) {
  ctx.fillStyle = COL.dim; ctx.font = '10px ui-monospace,monospace'; ctx.textAlign = 'right';
  for (let i = 0; i <= ticks; i++) {
    const v = yr[0] + (yr[1] - yr[0]) * i / ticks;
    const y = box.t + box.h - (v - yr[0]) / (yr[1] - yr[0] + 1e-9) * box.h;
    ctx.strokeStyle = COL.line; ctx.globalAlpha = .5; ctx.beginPath(); ctx.moveTo(box.l, y); ctx.lineTo(box.l + box.w, y); ctx.stroke(); ctx.globalAlpha = 1;
    ctx.fillText(fmt ? fmt(v) : v.toFixed(2), box.l - 5, y + 3);
  }
}

// softmax of 4 logits -> {digit(1..4), conf, probs[4]}
function cellPred(Y, b, c) {
  const base = b * A.YIN + c * 4; let mx = -1e9;
  for (let k = 0; k < 4; k++) mx = Math.max(mx, Y[base + k]);
  let s = 0; const p = [0, 0, 0, 0];
  for (let k = 0; k < 4; k++) { p[k] = Math.exp(Y[base + k] - mx); s += p[k]; }
  let best = 0; for (let k = 0; k < 4; k++) { p[k] /= s; if (p[k] > p[best]) best = k; }
  return { digit: best + 1, conf: p[best], probs: p };
}
function hexA(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

// ---------- Sudoku DOM grid ----------
function buildGrid(host) {
  host.innerHTML = '';
  const cells = [];
  for (let i = 0; i < 16; i++) {
    const r = Math.floor(i / 4), c = i % 4;
    const d = document.createElement('div');
    d.className = 'cell box';
    if ((Math.floor(r / 2) + Math.floor(c / 2)) % 2 === 0) d.style.background = '#0b1120';
    else d.style.background = '#0e1526';
    host.appendChild(d); cells.push(d);
  }
  return cells;
}
function renderGrid(cells, item, Y) {
  for (let c = 0; c < 16; c++) {
    const el = cells[c], given = item.puzzle[c];
    el.className = 'cell box';
    if (given !== 0) { el.classList.add('given'); el.textContent = given; el.style.color = ''; continue; }
    if (!Y) { el.textContent = ''; continue; }
    const pr = cellPred(Y, 0, c);
    el.textContent = pr.digit;
    const correct = pr.digit === item.solution[c];
    const base = correct ? COL.a3 : COL.bad;
    el.style.background = hexA(base, 0.12 + 0.6 * (pr.conf - 0.25) / 0.75);
    el.style.color = correct ? '#08101c' : COL.bad;
    el.style.fontWeight = 800;
  }
}

// ======================================================================
//  Async trainer (single active job)
// ======================================================================
function startJob(fn) { const tok = {}; S.job = tok; fn(tok); return tok; }
function alive(tok) { return S.job === tok; }

// ======================================================================
//  Tabs
// ======================================================================
const tabs = [...document.querySelectorAll('.tab')];
const mods = [...document.querySelectorAll('.module')];
tabs.forEach(t => t.addEventListener('click', () => {
  tabs.forEach(x => x.classList.remove('on')); t.classList.add('on');
  mods.forEach(m => m.classList.remove('on'));
  const i = +t.dataset.m; mods[i].classList.add('on');
  if (i === 0) refreshLoop();
}));

// ======================================================================
//  MODULE 0 : The Loop
// ======================================================================
const loopGrid = buildGrid($('loopGrid'));
let loopItem = null, loopHist = null, loopMax = 8;
function newLoopPuzzle() {
  const rng = new A.RNG((Math.random() * 1e9) | 0);
  loopItem = A.makePuzzle(rng, S.blanks);
  loopHist = null;
  $('loopStep').value = 0; $('loopStepV').textContent = '0';
  renderGrid(loopGrid, loopItem, null);
  drawLoopMaps(null);
  $('loopExact').textContent = '—'; $('loopCell').textContent = '—'; $('loopDz').textContent = '—';
}
function computeLoop() {
  if (!S.model || !loopItem) return;
  const enc = A.encodeBatch([loopItem]);
  const inf = A.infer(S.model, enc, loopMax, S.model.cfg.n);
  loopHist = inf.stepsHistory;
  $('loopStep').max = loopMax;
}
function showLoopStep(step) {
  if (!loopHist) return;
  const h = loopHist[Math.min(step, loopHist.length - 1)];
  renderGrid(loopGrid, loopItem, h.Ydata);
  const ev = A.evaluate(h.Ydata, [loopItem]);
  $('loopExact').textContent = ev.exactAcc >= 1 ? 'SOLVED ✓' : (ev.exactAcc * 100).toFixed(0) + '%';
  $('loopCell').textContent = (ev.cellAcc * 100).toFixed(0) + '% (' + Math.round(ev.cellAcc * 16) + '/16)';
  let dz = '—';
  if (step > 0) { const p = loopHist[step - 1].z, q = h.z; let s = 0; for (let i = 0; i < q.length; i++) s += (q[i] - p[i]) ** 2; dz = Math.sqrt(s / q.length).toFixed(3); }
  $('loopDz').textContent = dz;
  drawLoopMaps(h);
}
function drawLoopMaps(h) {
  // z heatmap
  const z = setup($('loopZ'));
  if (h) {
    const Hz = h.z.length, cols = Hz, cw = z.w / cols;
    let mx = 1e-6; for (const v of h.z) mx = Math.max(mx, Math.abs(v));
    for (let i = 0; i < Hz; i++) {
      const v = h.z[i] / mx;
      z.ctx.fillStyle = v >= 0 ? hexA(COL.a3, Math.abs(v)) : hexA(COL.a5, Math.abs(v));
      z.ctx.fillRect(i * cw, 0, cw - 0.5, z.h);
    }
  } else { z.ctx.fillStyle = COL.dim; z.ctx.font = '12px ui-monospace'; z.ctx.fillText('press Run the loop', 12, 24); }
  // y logits heatmap (16 cells x 4 digits)
  const y = setup($('loopY'));
  if (h) {
    const cw = y.w / 16, ch = y.h / 4;
    for (let c = 0; c < 16; c++) {
      const pr = cellPred(h.Ydata, 0, c);
      for (let k = 0; k < 4; k++) {
        y.ctx.fillStyle = hexA(COL.a1, pr.probs[k]);
        y.ctx.fillRect(c * cw, k * ch, cw - 1, ch - 1);
      }
      // outline chosen
      y.ctx.strokeStyle = pr.digit - 1 === (loopItem.solution[c] - 1) ? COL.a3 : COL.bad;
      y.ctx.lineWidth = 1.5; y.ctx.strokeRect(c * cw + .5, (pr.digit - 1) * ch + .5, cw - 2, ch - 2);
    }
    y.ctx.fillStyle = COL.dim; y.ctx.font = '9px ui-monospace'; y.ctx.textAlign = 'left';
    y.ctx.fillText('cells 1..16 →', 2, y.h - 2);
  }
}
$('loopNew').addEventListener('click', () => { newLoopPuzzle(); });
$('loopPlay').addEventListener('click', () => {
  if (!S.model) return;
  computeLoop();
  let s = 0; const iv = setInterval(() => {
    $('loopStep').value = s; $('loopStepV').textContent = s;
    showLoopStep(s); s++;
    if (s > loopMax) clearInterval(iv);
  }, 320);
});
$('loopStep').addEventListener('input', e => {
  const s = +e.target.value; $('loopStepV').textContent = s;
  if (!loopHist) computeLoop(); showLoopStep(s);
});
function refreshLoop() {
  if (!loopItem) newLoopPuzzle();
  if (S.ready) $('loopNote').innerHTML = 'A trained model is loaded. Drag the <b>improvement step</b> slider or press <b>Run the loop</b> — early steps are a rough draft; the scratchpad <b style="color:var(--a3)">z</b> keeps changing while the answer <b style="color:var(--a2)">y</b> sharpens and locks in.';
}

// ======================================================================
//  MODULE 1 : Train
// ======================================================================
const trainSample = buildGrid($('trainSample'));
const T1 = { model: null, data: null, test: null, sampleItem: null, hist: [], acc: [], step: 0, running: false, lastT: 0, lastStep: 0 };
function readCfg() {
  return {
    n: +$('tn').value, T: +$('tT').value, H: +$('th').value,
    lr: (+$('tlr').value) / 1000, blanks: +$('tb').value, mode: $('tmode').value
  };
}
['tn', 'tT', 'th', 'tlr', 'tb'].forEach(id => $(id).addEventListener('input', () => {
  $('tnV').textContent = $('tn').value; $('tTV').textContent = $('tT').value;
  $('thV').textContent = $('th').value; $('tlrV').textContent = ((+$('tlr').value) / 1000).toFixed(3);
  $('tbV').textContent = $('tb').value;
}));
function buildTrainModel() {
  const c = readCfg();
  const rng = new A.RNG(1234 + (Math.random() * 1000 | 0));
  T1.model = A.makeModel(rng, c.H, c.H);
  T1.data = A.genDataset(rng, 400, c.blanks);
  T1.test = A.genDataset(rng, 120, c.blanks);
  T1.testEnc = A.encodeBatch(T1.test);
  T1.sampleItem = T1.test[0];
  T1.hist = []; T1.acc = []; T1.step = 0; T1.rng = rng;
  $('trainParams').textContent = 'params: ' + T1.model.nParams.toLocaleString();
  T1.model.cfg = { n: c.n, T: c.T, mode: c.mode };
  drawLoss(); drawAcc();
  renderGrid(trainSample, T1.sampleItem, null);
}
function trainTick(tok) {
  if (!alive(tok) || !T1.running) return;
  const c = readCfg();
  T1.model.cfg = { n: c.n, T: c.T, mode: c.mode };
  const B = 64, chunk = 6;
  for (let i = 0; i < chunk; i++) {
    const batch = []; for (let b = 0; b < B; b++) batch.push(T1.data[T1.rng.int(T1.data.length)]);
    const enc = A.encodeBatch(batch);
    const r = A.trainBatch(T1.model, enc, { n: c.n, T: c.T, mode: c.mode });
    A.adamStep(T1.model, c.lr, 1.0);
    T1.step++; T1.hist.push({ x: T1.step, y: Math.max(r.lossVal, 1e-4) });
  }
  if (T1.step - T1.lastStep >= 18) {
    const inf = A.infer(T1.model, T1.testEnc, c.T, c.n);
    const ev = A.evaluate(inf.Ydata, T1.test);
    T1.acc.push({ x: T1.step, exact: ev.exactAcc, cell: ev.cellAcc });
    $('tExact').textContent = (ev.exactAcc * 100).toFixed(1) + '%';
    $('tCell').textContent = (ev.cellAcc * 100).toFixed(1) + '%';
    // sample render
    const encS = A.encodeBatch([T1.sampleItem]);
    const infS = A.infer(T1.model, encS, c.T, c.n);
    renderGrid(trainSample, T1.sampleItem, infS.Ydata);
    drawAcc();
    T1.lastStep = T1.step;
    // publish to shared state
    S.model = T1.model; S.blanks = c.blanks; S.ready = true;
  }
  const now = performance.now();
  if (now - T1.lastT > 400) { $('trainSpeed').textContent = (chunk * 1000 / (now - T1.lastT) * (T1.lastT ? 1 : 0) || 0).toFixed(0) + ' steps/s'; }
  T1.lastT2 = T1.lastT2 || now;
  $('tStep').textContent = T1.step;
  $('tLoss').textContent = T1.hist.length ? T1.hist[T1.hist.length - 1].y.toFixed(4) : '—';
  drawLoss();
  requestAnimationFrame(() => trainTick(tok));
}
function drawLoss() {
  const { ctx, w, h } = setup($('lossCanvas'));
  const box = { l: 44, t: 12, r: 12, b: 26, w: w - 56, h: h - 38 };
  const pts = T1.hist.map(p => ({ x: p.x, y: Math.log10(p.y) }));
  const yr = [Math.log10(1e-3), Math.log10(2)];
  axes(ctx, w, h, { l: 44, t: 12, r: 12, b: 26 }, { xlabel: 'training step' });
  gridlines(ctx, box, yr, 4, v => Math.pow(10, v).toExponential(0));
  if (pts.length) {
    const xr = [0, Math.max(pts[pts.length - 1].x, 10)];
    plotLine(ctx, pts, xr, yr, box, COL.a4, 2);
  } else { ctx.fillStyle = COL.dim; ctx.font = '12px ui-monospace'; ctx.fillText('press Train', box.l + 10, box.t + 24); }
}
function drawAcc() {
  const { ctx, w, h } = setup($('accCanvas'));
  const box = { l: 40, t: 12, r: 12, b: 26, w: w - 52, h: h - 38 };
  const yr = [0, 100];
  axes(ctx, w, h, { l: 40, t: 12, r: 12, b: 26 }, { xlabel: 'training step' });
  gridlines(ctx, box, yr, 4, v => v.toFixed(0));
  if (T1.acc.length) {
    const xr = [0, Math.max(T1.acc[T1.acc.length - 1].x, 10)];
    plotLine(ctx, T1.acc.map(p => ({ x: p.x, y: p.cell * 100 })), xr, yr, box, COL.a1, 2);
    plotLine(ctx, T1.acc.map(p => ({ x: p.x, y: p.exact * 100 })), xr, yr, box, COL.a3, 2.5);
  } else { ctx.fillStyle = COL.dim; ctx.font = '12px ui-monospace'; ctx.fillText('held-out accuracy appears here', box.l + 10, box.t + 24); }
}
$('trainBtn').addEventListener('click', () => {
  if (!T1.model) buildTrainModel();
  T1.running = true; T1.lastT = 0;
  $('trainBtn').disabled = true; $('pauseBtn').disabled = false;
  $('trainNote').innerHTML = 'Training… loss should fall fast and exact-match should climb. Try nudging <b>recursion depth n</b> mid-run — more loops = more effective compute per example.';
  startJob(trainTick);
});
$('pauseBtn').addEventListener('click', () => {
  T1.running = false; S.job = null;
  $('trainBtn').disabled = false; $('pauseBtn').disabled = true;
});
$('resetBtn').addEventListener('click', () => {
  T1.running = false; S.job = null;
  $('trainBtn').disabled = false; $('pauseBtn').disabled = true;
  buildTrainModel();
  $('tStep').textContent = '0'; $('tLoss').textContent = '—'; $('tExact').textContent = '—'; $('tCell').textContent = '—';
});
$('th').addEventListener('change', () => { if (!T1.running) buildTrainModel(); });
$('tb').addEventListener('change', () => { if (!T1.running) buildTrainModel(); });

// ======================================================================
//  MODULE 2 : Test-Time Compute
// ======================================================================
$('sweepBtn').addEventListener('click', () => {
  if (!S.model) { $('sweepStatus').textContent = 'train a model first (tab 02) or wait for warm-up'; return; }
  $('sweepStatus').textContent = 'running…';
  $('sweepTrained').textContent = 'trained budget: T=' + S.model.cfg.T;
  setTimeout(() => {
    const rng = new A.RNG(555);
    const test = A.genDataset(rng, 160, S.blanks);
    const enc = A.encodeBatch(test);
    const res = [];
    for (let steps = 1; steps <= 12; steps++) {
      const inf = A.infer(S.model, enc, steps, S.model.cfg.n);
      const ev = A.evaluate(inf.Ydata, test);
      res.push({ steps, exact: ev.exactAcc, cell: ev.cellAcc });
    }
    drawSweep(res); $('sweepStatus').textContent = 'done';
    const best = res.reduce((a, b) => b.exact > a.exact ? b : a);
    $('sweepNote').innerHTML = `Peak exact-match <b>${(best.exact * 100).toFixed(1)}%</b> at <b>${best.steps}</b> steps (trained for T=${S.model.cfg.T}). Beyond the peak the tiny model tends to <b style="color:var(--a4)">over-think</b> — extra loops drift the answer. Bigger models & higher training T push the peak rightward.`;
  }, 30);
});
function drawSweep(res) {
  const { ctx, w, h } = setup($('sweepCanvas'));
  const box = { l: 46, t: 14, r: 14, b: 30, w: w - 60, h: h - 46 };
  axes(ctx, w, h, { l: 46, t: 14, r: 14, b: 30 }, { xlabel: 'inference improvement steps', ylabel: 'accuracy %' });
  gridlines(ctx, box, [0, 100], 5, v => v.toFixed(0));
  const xr = [1, 12];
  const sx = x => box.l + (x - xr[0]) / (xr[1] - xr[0]) * box.w;
  // trained budget marker
  if (S.model) { const bx = sx(S.model.cfg.T); ctx.strokeStyle = hexA(COL.a4, .8); ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(bx, box.t); ctx.lineTo(bx, box.t + box.h); ctx.stroke(); ctx.setLineDash([]); }
  const cell = plotLine(ctx, res.map(r => ({ x: r.steps, y: r.cell * 100 })), xr, [0, 100], box, COL.a1, 2.5);
  const exact = plotLine(ctx, res.map(r => ({ x: r.steps, y: r.exact * 100 })), xr, [0, 100], box, COL.a3, 3);
  // dots + labels
  res.forEach(r => {
    ctx.fillStyle = COL.a3; ctx.beginPath(); ctx.arc(exact.sx(r.steps), exact.sy(r.exact * 100), 3.2, 0, 7); ctx.fill();
  });
  ctx.fillStyle = COL.dim; ctx.font = '10px ui-monospace'; ctx.textAlign = 'center';
  res.forEach(r => ctx.fillText(r.steps, sx(r.steps), box.t + box.h + 14));
}

// ======================================================================
//  MODULE 3 : TRM vs HRM
// ======================================================================
$('ablBtn').addEventListener('click', () => {
  $('ablStatus').textContent = 'racing…'; $('ablBtn').disabled = true;
  const runs = { trm: [], hrm: [] };
  const models = {};
  const rngData = new A.RNG(99);
  const data = A.genDataset(rngData, 300, 6);
  const testItems = A.genDataset(rngData, 120, 6);
  const testEnc = A.encodeBatch(testItems);
  ['trm', 'hrm'].forEach(mode => { const r = new A.RNG(77); models[mode] = A.makeModel(r, 48, 48); models[mode].rng = r; });
  let step = 0; const MAX = 160;
  startJob(function race(tok) {
    if (!alive(tok)) { $('ablBtn').disabled = false; return; }
    for (let c = 0; c < 4; c++) {
      ['trm', 'hrm'].forEach(mode => {
        const m = models[mode];
        const batch = []; for (let b = 0; b < 64; b++) batch.push(data[m.rng.int(data.length)]);
        const enc = A.encodeBatch(batch);
        const r = A.trainBatch(m, enc, { n: 3, T: 3, mode });
        A.adamStep(m, 0.01, 1.0);
        runs[mode].push({ x: step, y: Math.max(r.lossVal, 1e-4) });
      });
      step++;
    }
    drawAbl(runs);
    if (step < MAX) { requestAnimationFrame(() => race(tok)); }
    else {
      $('ablBtn').disabled = false; $('ablStatus').textContent = 'done';
      ['trm', 'hrm'].forEach(mode => {
        const inf = A.infer(models[mode], testEnc, 3, 3);
        const ev = A.evaluate(inf.Ydata, testItems);
        const fl = runs[mode][runs[mode].length - 1].y;
        $(mode === 'trm' ? 'ablTrm' : 'ablHrm').textContent = `loss ${fl.toFixed(3)} · exact ${(ev.exactAcc * 100).toFixed(1)}%`;
      });
    }
  });
});
function drawAbl(runs) {
  const { ctx, w, h } = setup($('ablCanvas'));
  const box = { l: 46, t: 12, r: 12, b: 26, w: w - 58, h: h - 38 };
  const yr = [Math.log10(3e-3), Math.log10(2)];
  axes(ctx, w, h, { l: 46, t: 12, r: 12, b: 26 }, { xlabel: 'training step' });
  gridlines(ctx, box, yr, 4, v => Math.pow(10, v).toExponential(0));
  const xr = [0, 160];
  plotLine(ctx, runs.trm.map(p => ({ x: p.x, y: Math.log10(p.y) })), xr, yr, box, COL.a1, 2.2);
  plotLine(ctx, runs.hrm.map(p => ({ x: p.x, y: Math.log10(p.y) })), xr, yr, box, COL.a5, 2.2);
}
// parameter calculator
function updatePC() {
  const d = +$('pcWidth').value, L = +$('pcLayers').value;
  $('pcW').textContent = d; $('pcL').textContent = L;
  const io = A.XIN + A.YIN + 64; // rough embedding/io constant
  const perLayer = d * d;
  const trm = L * perLayer + io * d + d * (A.YIN + 64);
  const hrm = 2 * L * perLayer + io * d + d * (A.YIN + 64) + d * d; // 2 nets + coupling
  $('pcTrm').textContent = trm.toLocaleString();
  $('pcHrm').textContent = hrm.toLocaleString();
  $('pcRatio').textContent = (hrm / trm).toFixed(2) + '×';
  const { ctx, w, h } = setup($('pcBar'));
  const max = Math.max(trm, hrm), bw = w - 120;
  [['TRM', trm, COL.a1], ['HRM', hrm, COL.a5]].forEach((r, i) => {
    const y = 14 + i * 40;
    ctx.fillStyle = COL.mut; ctx.font = '11px ui-monospace'; ctx.textAlign = 'left'; ctx.fillText(r[0], 4, y + 14);
    ctx.fillStyle = hexA(r[2], .85); ctx.fillRect(46, y, bw * r[1] / max, 20);
    ctx.fillStyle = COL.ink; ctx.fillText(r[1].toLocaleString(), 52 + bw * r[1] / max, y + 14);
  });
}
$('pcWidth').addEventListener('input', updatePC);
$('pcLayers').addEventListener('input', updatePC);

// ======================================================================
//  MODULE 4 : Adaptive Halting
// ======================================================================
$('actTh').addEventListener('input', () => $('actThV').textContent = ((+$('actTh').value) / 100).toFixed(2));
$('actBtn').addEventListener('click', () => {
  if (!S.model) { $('actStatus').textContent = 'train a model first (tab 02) or wait for warm-up'; return; }
  $('actStatus').textContent = 'running…';
  setTimeout(() => {
    const th = (+$('actTh').value) / 100, maxS = 12, rng = new A.RNG(303);
    const byDiff = {}; const dots = [];
    for (const blanks of [3, 4, 5, 6, 7, 8, 9, 10]) {
      byDiff[blanks] = { steps: [], correct: 0, n: 0 };
      for (let q = 0; q < 14; q++) {
        const item = A.makePuzzle(rng, blanks);
        const enc = A.encodeBatch([item]);
        const inf = A.infer(S.model, enc, maxS, S.model.cfg.n);
        // find halt step: first step where softmax-prob change < th
        let used = maxS;
        for (let s = 1; s < inf.stepsHistory.length; s++) {
          const a = inf.stepsHistory[s - 1].Ydata, b = inf.stepsHistory[s].Ydata;
          let acc = 0;
          for (let c = 0; c < 16; c++) { const pa = cellPred(a, 0, c).probs, pb = cellPred(b, 0, c).probs; for (let k = 0; k < 4; k++) acc += (pa[k] - pb[k]) ** 2; }
          const rms = Math.sqrt(acc / 64);
          if (rms < th) { used = s + 1; break; }
        }
        const ev = A.evaluate(inf.stepsHistory[Math.min(used, inf.stepsHistory.length) - 1].Ydata, [item]);
        byDiff[blanks].steps.push(used); byDiff[blanks].n++; if (ev.exactAcc >= 1) byDiff[blanks].correct++;
        dots.push({ blanks, used, correct: ev.exactAcc >= 1 });
      }
    }
    drawAct(byDiff, dots, maxS);
    $('actStatus').textContent = 'done';
  }, 30);
});
function drawAct(byDiff, dots, maxS) {
  // scatter
  const sc = setup($('actScatter'));
  const b1 = { l: 40, t: 14, r: 12, b: 28, w: sc.w - 52, h: sc.h - 42 };
  axes(sc.ctx, sc.w, sc.h, { l: 40, t: 14, r: 12, b: 28 }, { xlabel: 'blanks (difficulty)', ylabel: 'steps to halt' });
  gridlines(sc.ctx, b1, [0, maxS], maxS <= 12 ? 6 : 6, v => v.toFixed(0));
  const xr = [2.5, 10.5];
  dots.forEach(d => {
    const x = b1.l + (d.blanks - xr[0]) / (xr[1] - xr[0]) * b1.w + (Math.random() - .5) * 14;
    const y = b1.t + b1.h - d.used / maxS * b1.h + (Math.random() - .5) * 6;
    sc.ctx.fillStyle = d.correct ? hexA(COL.a3, .8) : hexA(COL.bad, .7);
    sc.ctx.beginPath(); sc.ctx.arc(x, y, 3, 0, 7); sc.ctx.fill();
  });
  sc.ctx.fillStyle = COL.dim; sc.ctx.font = '10px ui-monospace'; sc.ctx.textAlign = 'left';
  sc.ctx.fillText('green = solved · red = unsolved', b1.l + 4, b1.t + 12);
  // bars
  const br = setup($('actBars'));
  const b2 = { l: 40, t: 14, r: 30, b: 28, w: br.w - 70, h: br.h - 42 };
  axes(br.ctx, br.w, br.h, { l: 40, t: 14, r: 30, b: 28 }, { xlabel: 'blanks (difficulty)' });
  const keys = Object.keys(byDiff).map(Number).sort((a, b) => a - b);
  const bw = b2.w / keys.length;
  let totalUsed = 0, totalN = 0;
  keys.forEach((k, i) => {
    const d = byDiff[k], avg = d.steps.reduce((s, v) => s + v, 0) / d.n; totalUsed += avg; totalN++;
    const acc = d.correct / d.n;
    const x = b2.l + i * bw;
    br.ctx.fillStyle = hexA(COL.a2, .85); br.ctx.fillRect(x + bw * .18, b2.t + b2.h - avg / maxS * b2.h, bw * .3, avg / maxS * b2.h);
    br.ctx.fillStyle = hexA(COL.a3, .85); br.ctx.fillRect(x + bw * .52, b2.t + b2.h - acc * b2.h, bw * .3, acc * b2.h);
    br.ctx.fillStyle = COL.dim; br.ctx.font = '10px ui-monospace'; br.ctx.textAlign = 'center';
    br.ctx.fillText(k, x + bw / 2, b2.t + b2.h + 14);
  });
  const avgAll = totalUsed / totalN;
  $('actSaved').textContent = (maxS - avgAll).toFixed(1) + ' steps (' + ((1 - avgAll / maxS) * 100).toFixed(0) + '% less compute)';
}

// ======================================================================
//  MODULE 5 : Under the Hood — gradient check
// ======================================================================
$('gcBtn').addEventListener('click', () => {
  $('gcStatus').textContent = 'checking…'; $('gcBtn').disabled = true;
  setTimeout(() => {
    const r = new A.RNG(7);
    const m = A.makeModel(r, 10, 10);
    const enc = A.encodeBatch(A.genDataset(r, 4, 8));
    const cfg = { n: 3, T: 1, mode: 'trm' };
    A.trainBatch(m, enc, cfg);
    const analytic = m.params.map(p => Float64Array.from(p.grad));
    const eps = 1e-5; const pairs = []; let maxRel = 0;
    const idxs = [0, 2, 4, 6];
    for (const pi of idxs) {
      const p = m.params[pi];
      for (let t = 0; t < 12; t++) {
        const idx = r.int(p.data.length); const o = p.data[idx];
        p.data[idx] = o + eps; const lp = A.trainBatch(m, enc, cfg).lossVal;
        p.data[idx] = o - eps; const lm = A.trainBatch(m, enc, cfg).lossVal;
        p.data[idx] = o;
        const num = (lp - lm) / (2 * eps), ana = analytic[pi][idx];
        const rel = Math.abs(num - ana) / (Math.max(Math.abs(num), Math.abs(ana)) + 1e-8);
        maxRel = Math.max(maxRel, rel);
        pairs.push({ num, ana });
      }
    }
    drawGC(pairs);
    $('gcErr').textContent = maxRel.toExponential(2);
    const pass = maxRel < 1e-4;
    $('gcVerdict').innerHTML = pass ? '<span style="color:var(--a3)">PASS ✓ gradients are exact</span>' : '<span style="color:var(--bad)">FAIL</span>';
    $('gcStatus').textContent = pass ? 'verified' : 'mismatch';
    $('gcBtn').disabled = false;
  }, 30);
});
function drawGC(pairs) {
  const { ctx, w, h } = setup($('gcCanvas'));
  const box = { l: 44, t: 12, r: 12, b: 28, w: w - 56, h: h - 40 };
  let mx = 1e-6; pairs.forEach(p => { mx = Math.max(mx, Math.abs(p.num), Math.abs(p.ana)); });
  axes(ctx, w, h, { l: 44, t: 12, r: 12, b: 28 }, { xlabel: 'analytic gradient', ylabel: 'finite-diff' });
  // y=x line
  ctx.strokeStyle = hexA(COL.dim, .8); ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(box.l, box.t + box.h); ctx.lineTo(box.l + box.w, box.t); ctx.stroke(); ctx.setLineDash([]);
  const sx = v => box.l + (v + mx) / (2 * mx) * box.w;
  const sy = v => box.t + box.h - (v + mx) / (2 * mx) * box.h;
  pairs.forEach(p => { ctx.fillStyle = hexA(COL.a1, .85); ctx.beginPath(); ctx.arc(sx(p.ana), sy(p.num), 3.5, 0, 7); ctx.fill(); });
  ctx.fillStyle = COL.dim; ctx.font = '10px ui-monospace'; ctx.textAlign = 'left';
  ctx.fillText('points on the dashed line ⇒ autodiff == calculus', box.l + 6, box.t + 14);
}

// ======================================================================
//  Warm-up: train a default model in the background
// ======================================================================
function warmUp() {
  const rng = new A.RNG(2026);
  const m = A.makeModel(rng, 48, 48);
  m.cfg = { n: 3, T: 3, mode: 'trm' };
  const data = A.genDataset(rng, 350, S.blanks);
  const testEnc = A.encodeBatch(A.genDataset(rng, 100, S.blanks));
  const testItems = A.genDataset(new A.RNG(2027), 100, S.blanks);
  const TARGET = 260; let step = 0;
  $('chipTxt').textContent = 'warming up model… 0%';
  startJob(function loop(tok) {
    if (!alive(tok)) return;
    for (let c = 0; c < 6; c++) {
      const batch = []; for (let b = 0; b < 64; b++) batch.push(data[rng.int(data.length)]);
      const enc = A.encodeBatch(batch);
      A.trainBatch(m, enc, m.cfg); A.adamStep(m, 0.012, 1.0); step++;
    }
    const pct = Math.min(100, (step / TARGET * 100) | 0);
    $('chipTxt').textContent = 'warming up model… ' + pct + '%';
    if (step < TARGET) { requestAnimationFrame(() => loop(tok)); }
    else {
      const inf = A.infer(m, testEnc, 3, 3); const ev = A.evaluate(inf.Ydata, testItems);
      S.model = m; S.ready = true; S.job = null;
      $('chip').innerHTML = '<span class="dotp" style="background:var(--a3);animation:none"></span><span>model ready · ' + (ev.exactAcc * 100).toFixed(0) + '% exact-match · ' + m.nParams.toLocaleString() + ' params</span>';
      setTimeout(() => { $('chip').style.opacity = '.55'; }, 4000);
      refreshLoop();
    }
  });
}

// init
updatePC();
newLoopPuzzle();
warmUp();
})();
