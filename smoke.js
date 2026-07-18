// Headless smoke test: stub DOM+canvas, run core.js + ui.js, exercise handlers.
const fs = require('fs');
const path = require('path');

// ---- stub 2d context (no-op everything) ----
const ctxProto = new Proxy({}, { get: () => () => {} });
function makeCtx() {
  return {
    setTransform(){}, clearRect(){}, strokeRect(){}, fillRect(){}, fillText(){},
    beginPath(){}, moveTo(){}, lineTo(){}, arc(){}, stroke(){}, fill(){},
    save(){}, restore(){}, translate(){}, rotate(){}, setLineDash(){},
    set fillStyle(v){}, get fillStyle(){return '#000';},
    set strokeStyle(v){}, get strokeStyle(){return '#000';},
    set lineWidth(v){}, set lineJoin(v){}, set font(v){}, set textAlign(v){},
    set globalAlpha(v){}, get globalAlpha(){return 1;}
  };
}
// ---- stub element ----
const store = {};
function El(id, tag) {
  return {
    id, tagName: tag || 'div', _val: (tag==='input'||tag==='select')?'3':'',
    width: 520, height: 220, style: {}, dataset: {}, classList: { _s:new Set(), add(c){this._s.add(c);}, remove(c){this._s.delete(c);}, contains(c){return this._s.has(c);} },
    children: [], innerHTML: '', textContent: '',
    _listeners: {},
    getContext(){ return makeCtx(); },
    addEventListener(ev, fn){ (this._listeners[ev]=this._listeners[ev]||[]).push(fn); },
    appendChild(c){ this.children.push(c); },
    querySelectorAll(){ return []; },
    get value(){ return this._val; }, set value(v){ this._val=String(v); },
    fire(ev){ (this._listeners[ev]||[]).forEach(f=>f({target:this})); },
  };
}
const ids = ['loopGrid','loopPlay','loopNew','loopStep','loopStepV','loopZ','loopY','loopExact','loopCell','loopDz','loopNote',
 'tn','tT','th','tlr','tb','tmode','tnV','tTV','thV','tlrV','tbV','trainBtn','pauseBtn','resetBtn','trainParams','trainSpeed',
 'lossCanvas','accCanvas','trainSample','tStep','tLoss','tExact','tCell','trainNote',
 'sweepBtn','sweepTrained','sweepStatus','sweepCanvas','sweepNote',
 'ablBtn','ablStatus','ablCanvas','ablTrm','ablHrm','pcWidth','pcLayers','pcW','pcL','pcTrm','pcHrm','pcRatio','pcBar',
 'actBtn','actTh','actThV','actStatus','actScatter','actBars','actSaved',
 'gcBtn','gcStatus','gcCanvas','gcErr','gcVerdict','chip','chipTxt'];
ids.forEach(id => {
  const tag = /Canvas|loopZ|loopY|pcBar|actScatter|actBars|gcCanvas/.test(id) ? 'canvas'
    : /^(tn|tT|th|tlr|tb|loopStep|pcWidth|pcLayers|actTh)$/.test(id) ? 'input'
    : /mode/.test(id) ? 'select' : 'div';
  store[id] = El(id, tag);
});
// tabs
const tabEls = [];
for (let i=0;i<6;i++){ const t=El('tab'+i,'div'); t.dataset.m=String(i); t._listeners={}; tabEls.push(t); }
const modEls = []; for (let i=0;i<6;i++) modEls.push(El('m'+i,'section'));

global.window = {
  devicePixelRatio: 1,
  addEventListener(){}, TRL: null,
};
global.requestAnimationFrame = (fn) => { global.__raf.push(fn); };
global.__raf = [];
global.performance = { now: () => Date.now() };
global.setTimeout = (fn) => { global.__timeouts.push(fn); return 0; };
global.__timeouts = [];
global.setInterval = () => 0; global.clearInterval = () => {};
global.Math.random = (()=>{let s=42;return ()=>{s=(s*1103515245+12345)&0x7fffffff;return s/0x7fffffff;};})();
global.getComputedStyle = () => ({ getPropertyValue: () => '#000' });
global.document = {
  getElementById: (id) => store[id] || El(id),
  querySelectorAll: (sel) => sel === '.tab' ? tabEls : sel === '.module' ? modEls : [],
  createElement: (tag) => El('new', tag),
  documentElement: {},
};

// load core.js (as browser script -> sets window.TRL & global.TRL)
const coreSrc = fs.readFileSync(path.join(__dirname,'core.js'),'utf8');
eval(coreSrc);
if (!window.TRL && global.TRL) window.TRL = global.TRL;
console.log('core loaded, TRL fns:', Object.keys(window.TRL).length);

// load ui.js
const uiSrc = fs.readFileSync(path.join(__dirname,'ui.js'),'utf8');
let errors = 0;
try { eval(uiSrc); console.log('ui.js init OK'); } catch(e){ console.log('UI INIT ERROR:', e.message, '\n', e.stack.split('\n').slice(0,4).join('\n')); errors++; }

// drain warm-up rAF loop (bounded)
function drainRaf(max){ let c=0; while(global.__raf.length && c<max){ const f=global.__raf.shift(); try{ f(); }catch(e){ console.log('rAF ERROR:', e.message); errors++; break; } c++; } return c; }
console.log('warm-up rAF frames run:', drainRaf(500));

// exercise handlers
function fire(id, ev){ try{ store[id].fire(ev||'click'); }catch(e){ console.log('HANDLER ERROR ['+id+']:', e.message); errors++; } }
// module 3 param calc slider
store.pcWidth._val='256'; fire('pcWidth','input');
store.pcLayers._val='4'; fire('pcLayers','input');
console.log('pc TRM/HRM:', store.pcTrm.textContent, '/', store.pcHrm.textContent, 'ratio', store.pcRatio.textContent);
// train handlers
fire('tn','input'); fire('trainBtn'); console.log('train frames:', drainRaf(60)); fire('pauseBtn');
console.log('train step after run:', store.tStep.textContent, 'loss:', store.tLoss.textContent);
// gradient check (uses setTimeout)
fire('gcBtn'); global.__timeouts.forEach(f=>{try{f();}catch(e){console.log('GC ERROR:',e.message);errors++;}}); global.__timeouts=[];
console.log('gradcheck err:', store.gcErr.textContent, 'verdict:', store.gcVerdict.textContent.replace(/<[^>]+>/g,''));
// sweep (needs S.model — set via train above publishing to shared state after >=18 steps)
fire('sweepBtn'); global.__timeouts.forEach(f=>{try{f();}catch(e){console.log('SWEEP ERROR:',e.message);errors++;}}); global.__timeouts=[];
console.log('sweep status:', store.sweepStatus.textContent);
// act
fire('actBtn'); global.__timeouts.forEach(f=>{try{f();}catch(e){console.log('ACT ERROR:',e.message);errors++;}}); global.__timeouts=[];
console.log('act status:', store.actStatus.textContent, 'saved:', store.actSaved.textContent);
// ablation race
fire('ablBtn'); console.log('ablation frames:', drainRaf(300));
console.log('abl TRM:', store.ablTrm.textContent, '| HRM:', store.ablHrm.textContent);
// tab switch
tabEls[0].fire('click');

console.log(errors===0 ? '\nSMOKE TEST PASS ✓' : '\nSMOKE TEST: '+errors+' error(s)');