/* =====================================================================
 *  Tape Stock Count — runs entirely on the phone (AI counting + label reading + calculation)
 *  Setup: put the Apps Script Web app URL (ending in /exec) on the line below
 * ===================================================================== */
const API_URL = 'https://script.google.com/macros/s/AKfycbxMBuBhTnEJITKQe_7zt05faNLFZOZGs3klvY61N5INKpnZ4YXE0nEko8ITzY-vS-Fo/exec';                 // ← empty = DEMO mode (sample data, nothing is saved)
const DAY_CHANGE_HOUR = 12;         // saved before noon = counts for yesterday (night shift 19:00–07:00)
const MODEL_URL = 'model.onnx';
const CACHE = 'stocktape-v1';

const $ = id => document.getElementById(id);
const fmt = (v, d = 1) => v == null ? '–' : Number(v).toLocaleString('en-US', {minimumFractionDigits: d, maximumFractionDigits: d});
const store = {get(k){try{return localStorage.getItem(k)}catch(e){return null}}, set(k,v){try{localStorage.setItem(k,v)}catch(e){}}};
const DEMO_MODE = !API_URL;

let CFG = null;   // {settings, specs:{key:row}, machines, plan_updated, done, date}
let R = null;     // report being edited

/* ------------------------- report date ------------------------- */
function reportDate(now = new Date()){
  const d = new Date(now); if(d.getHours() < DAY_CHANGE_HOUR) d.setDate(d.getDate() - 1);
  return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`;
}
const thDate = iso => iso ? iso.split('-').reverse().join('/') : '-';

/* ------------------------- formulas ------------------------- */
const specKey = (den, tw) => `${Math.round(+den)}/${(+tw).toFixed(1)}`;
function parseSku(text){
  const s = String(text || '').replace(/\s/g, '').toUpperCase().replace(/O/g, '0');
  const m = s.match(/W1\d{13}/); if(!m) return null;
  const sku = m[0], n = +sku.slice(7, 10);
  const denier = n < 640 ? n * 10 : n, tw = +sku.slice(10, 12) / 10;   // lowest Denier in the plant is 640
  return {sku, denier, tw, spec: specKey(denier, tw)};
}
const Calc = {
  fullKg(spec, kind){ const r = CFG.specs[spec]; return r ? r.denier * (kind === 'warp' ? r.warp_len : r.weft_len) / 9e6 : null; },
  creel(m){  // ① Width×10×2 ÷ (25.4 ÷ warp density) × full warp bobbin weight
    if(!m.width || !m.dan) return [null, null];
    const q = Math.round(m.width * 10 * 2 / (25.4 / m.dan)), fk = this.fullKg(m.spec, 'warp');
    return [q, fk == null ? null : q * fk];
  },
  shuttle(m){ const fk = this.fullKg(m.spec, 'weft'); return fk == null ? null : fk * CFG.settings.shuttles; },   // ③
  floor(m, f){ const fk = this.fullKg(m.spec, 'weft'), s = CFG.settings;                                          // ⑥
    return fk == null ? null : fk * ((f.full||0) * s.floor_full + (f.half||0) * s.floor_half + (f.low||0) * s.floor_low); },
  photo(circles, spec, kind){  // bobbins from photo: % tape left = (D² − core²) ÷ (Dfull² − core²)
    const s = CFG.settings, core = s.core_cm, full = kind === 'warp' ? s.warp_full_cm : s.weft_full_cm;
    const ratios = circles.filter(c => c.core).map(c => c.r / c.core).sort((a, b) => a - b);
    const med = ratios.length ? ratios[ratios.length >> 1] : full / core;
    const bins = [0,0,0,0,0,0]; let eq = 0;
    for(const c of circles){
      let ratio = c.core ? c.r / c.core : med;
      if(ratio < 0.6 * med || ratio > 1.6 * med) ratio = med;
      const d = ratio * core, f = Math.max(0, Math.min(1, (d*d - core*core) / (full*full - core*core)));
      eq += f; const p = f * 100;
      bins[p >= 95 ? 0 : p >= 85 ? 1 : p >= 75 ? 2 : p >= 65 ? 3 : p >= 55 ? 4 : 5]++;
    }
    const fk = this.fullKg(spec, kind);
    return {count: circles.length, equiv_full: eq, kg: fk == null ? null : eq * fk, bins};
  }
};

/* ------------------------- AI bobbin detection ------------------------- */
const AI = {
  session: null, loading: null,
  async load(onProgress){
    if(this.session) return this.session;
    if(this.loading) return this.loading;
    this.loading = (async () => {
      ort.env.wasm.wasmPaths = new URL('lib/', location.href).href;
      ort.env.wasm.numThreads = 1;
      let buf;
      try {
        const c = await caches.open(CACHE); let r = await c.match(MODEL_URL);
        if(!r){ r = await fetchProgress(MODEL_URL, onProgress); await c.put(MODEL_URL, r.clone()); }
        buf = await r.arrayBuffer();
      } catch(e){ buf = await (await fetch(MODEL_URL)).arrayBuffer(); }
      this.session = await ort.InferenceSession.create(buf, {executionProviders: ['wasm']});
      return this.session;
    })();
    return this.loading;
  },
  async detect(bmp){
    const s = await this.load(), S = 1024, conf = CFG.settings.ai_conf || 0.65;
    const sc = S / Math.max(bmp.width, bmp.height), nw = Math.round(bmp.width * sc), nh = Math.round(bmp.height * sc);
    const px = (S - nw) >> 1, py = (S - nh) >> 1;
    const cv = document.createElement('canvas'); cv.width = cv.height = S;
    const x = cv.getContext('2d'); x.fillStyle = 'rgb(114,114,114)'; x.fillRect(0, 0, S, S); x.drawImage(bmp, px, py, nw, nh);
    const d = x.getImageData(0, 0, S, S).data, f = new Float32Array(3 * S * S);
    for(let i = 0, p = 0; i < S * S; i++, p += 4){ f[i] = d[p] / 255; f[S*S + i] = d[p+1] / 255; f[2*S*S + i] = d[p+2] / 255; }
    const out = (await s.run({[s.inputNames[0]]: new ort.Tensor('float32', f, [1, 3, S, S])}))[s.outputNames[0]];
    const N = out.dims[2], o = out.data, cand = [];
    for(let j = 0; j < N; j++){
      const score = o[4*N + j]; if(score < conf) continue;
      cand.push({cx: o[j], cy: o[N+j], w: o[2*N+j], h: o[3*N+j], score});
    }
    cand.sort((a, b) => b.score - a.score);
    const keep = [];
    const iou = (a, b) => { const ix = Math.max(0, Math.min(a.cx+a.w/2, b.cx+b.w/2) - Math.max(a.cx-a.w/2, b.cx-b.w/2));
      const iy = Math.max(0, Math.min(a.cy+a.h/2, b.cy+b.h/2) - Math.max(a.cy-a.h/2, b.cy-b.h/2));
      const I = ix * iy; return I / (a.w*a.h + b.w*b.h - I); };
    for(const c of cand) if(keep.every(k => iou(k, c) < 0.5)) keep.push(c);
    return keep.map(c => ({x: (c.cx - px) / sc, y: (c.cy - py) / sc, r: (c.w + c.h) / 4 / sc, score: c.score}));
  }
};
async function fetchProgress(url, cb){
  const r = await fetch(url); if(!r.ok) throw new Error('Could not load the AI model');
  const total = +r.headers.get('content-length') || 0; if(!total || !r.body || !cb) return r;
  const rd = r.body.getReader(), chunks = []; let got = 0;
  for(;;){ const {done, value} = await rd.read(); if(done) break; chunks.push(value); got += value.length; cb(got / total); }
  return new Response(new Blob(chunks), {headers: {'Content-Type': 'application/octet-stream'}});
}

/* core radius: walk out from the centre until brightness reaches tape level */
function measureCores(bmp, dets){
  const W = bmp.width, H = bmp.height, cv = document.createElement('canvas'); cv.width = W; cv.height = H;
  const x = cv.getContext('2d', {willReadFrequently: true}); x.drawImage(bmp, 0, 0);
  const d = x.getImageData(0, 0, W, H).data;
  const g = (px, py) => { let s = 0, n = 0;
    for(let dy = -1; dy <= 1; dy++) for(let dx = -1; dx <= 1; dx++){ const X = px+dx, Y = py+dy;
      if(X >= 0 && Y >= 0 && X < W && Y < H){ const i = (Y*W + X) * 4; s += 0.299*d[i] + 0.587*d[i+1] + 0.114*d[i+2]; n++; } }
    return n ? s / n : -1; };
  return dets.map(({x: cx, y: cy, r: R}) => {
    const ring = [];
    for(let a = 0; a < 16; a++) for(const t of [0.65, 0.75, 0.85]){
      const px = Math.round(cx + Math.cos(a*Math.PI/8) * R * t), py = Math.round(cy + Math.sin(a*Math.PI/8) * R * t);
      const v = g(px, py); if(v >= 0) ring.push(v); }
    if(ring.length < 10) return null;
    ring.sort((a, b) => a - b); const tape = ring[ring.length >> 1], rs = [];
    for(let a = 0; a < 24; a++){ const ca = Math.cos(a*Math.PI/12), sa = Math.sin(a*Math.PI/12);
      for(let t = 0.12; t < 0.8; t += 0.01){ const px = Math.round(cx + ca*R*t), py = Math.round(cy + sa*R*t);
        const v = g(px, py); if(v < 0) break; if(v >= 0.85 * tape){ rs.push(t * R); break; } } }
    if(rs.length < 8) return null;
    rs.sort((a, b) => a - b); return rs[rs.length >> 1];
  });
}

/* ------------------------- label SKU reader ------------------------- */
const OCR = {
  w: null,
  async get(){ if(!this.w) this.w = Tesseract.createWorker('eng', 1, {workerPath: 'lib/tess/worker.min.js',
    corePath: 'lib/tess/core/', langPath: 'lang', gzip: true}); return this.w; },
  async read(blob){
    const w = await this.get(), {data} = await w.recognize(blob);
    let pallet = '';
    try { if('BarcodeDetector' in window){ const b = await new BarcodeDetector({formats: ['qr_code']}).detect(await createImageBitmap(blob));
      if(b[0]) pallet = b[0].rawValue; } } catch(e){}
    return {p: parseSku(data.text), pallet};
  }
};

/* ------------------------- API (Google Apps Script) ------------------------- */
async function apiGet(params){
  const u = API_URL + '?' + new URLSearchParams({...params, token: $('pin').value});
  let r; try { r = await fetch(u); } catch(e){ throw new Error('Cannot reach Google — check internet connection'); }
  const j = await r.json(); if(!j.ok) throw new Error(j.error); return j;
}
async function apiPost(body){
  // no Content-Type header (avoids CORS preflight) — Apps Script reads e.postData.contents
  const r = await fetch(API_URL, {method: 'POST', body: JSON.stringify({...body, token: body.token || $('pin').value})});
  const j = await r.json(); if(!j.ok) throw new Error(j.error); return j;
}
/* send queue (IndexedDB) — nothing is lost when offline */
const Outbox = {
  db: null,
  open(){ return this.db ||= new Promise((res, rej) => { const q = indexedDB.open('stocktape', 1);
    q.onupgradeneeded = () => q.result.createObjectStore('outbox', {keyPath: 'k'}); q.onsuccess = () => res(q.result); q.onerror = () => rej(q.error); }); },
  async tx(mode, fn){ const db = await this.open(); return new Promise((res, rej) => { const t = db.transaction('outbox', mode);
    const r = fn(t.objectStore('outbox')); t.oncomplete = () => res(r && r.result); t.onerror = () => rej(t.error); }); },
  put(item){ return this.tx('readwrite', s => s.put(item)); },
  del(k){ return this.tx('readwrite', s => s.delete(k)); },
  all(){ return this.tx('readonly', s => s.getAll()); },
  async flush(){
    if(DEMO_MODE) return 0;
    let items = []; try { items = await this.all(); } catch(e){ return 0; }
    for(const it of items){ try { await apiPost(it.body); await this.del(it.k); } catch(e){ break; } }
    const left = (await this.all()).length; updateHeader(left); return left;
  }
};

/* ------------------------- load data ------------------------- */
async function loadConfig(){
  const date = reportDate();
  let j;
  if(DEMO_MODE) j = {...DEMO, done: JSON.parse(store.get('demo_done_' + date) || '{}')};
  else j = await apiGet({action: 'config', date});
  CFG = {settings: j.settings, specs: Object.fromEntries(j.specs.map(r => [specKey(r.denier, r.tw), r])),
         machines: j.machines, plan_updated: j.plan_updated, done: j.done || {}, date};
  store.set('cfg_cache', JSON.stringify(j));
  render();
}
let PENDING = 0;
function updateHeader(pending = PENDING){
  PENDING = pending;
  $('hdrDate').textContent = 'Report date ' + thDate(CFG?.date || reportDate());
  $('hdrPlan').textContent = 'Plan: ' + (CFG?.plan_updated ? thDate(CFG.plan_updated) : 'none');
  $('hdrSync').textContent = DEMO_MODE ? '⚠ DEMO mode (not connected to Google)' : (pending ? `${pending} waiting to send` : 'All data sent');
}
function render(){
  updateHeader();
  const b = [];
  if(!CFG.machines.length) b.push(['bad', 'No plan yet — waiting for the plan file from the LINE group']);
  else if(CFG.plan_updated && CFG.plan_updated < CFG.date) b.push(['warn', `Latest plan is dated ${thDate(CFG.plan_updated)} — you can continue, the latest plan will be used`]);
  $('banners').innerHTML = b.map(([c, t]) => `<div class="banner ${c}">${t}</div>`).join('');
  const sel = $('machine'), cur = sel.value;
  sel.innerHTML = '<option value="">— Select loom —</option>' + ['SL', 'BL'].map(g =>
    `<optgroup label="${g === 'SL' ? 'Circular looms (SL)' : 'Flat looms (BL)'}">` + CFG.machines.filter(m => m.group === g)
      .map(m => `<option value="${m.id}">${CFG.done[m.id] ? '✔ ' : ''}${m.id}</option>`).join('') + '</optgroup>').join('');
  sel.value = cur;
  renderChips();
}

/* loom chips: grouped SL / BL · tap to select */
function renderChips(){
  const cur = $('machine').value;
  $('doneList').innerHTML = [['SL', 'Circular looms'], ['BL', 'Flat looms']].map(([g, label]) => {
    const ms = CFG.machines.filter(m => m.group === g); if(!ms.length) return '';
    const d = ms.filter(m => CFG.done[m.id]).length;
    return `<div class="lgrp"><div class="lgrp-h"><b>${label} (${g})</b><span class="cnt2">${d} / ${ms.length} done</span></div>
      <div class="bar2"><i style="width:${Math.round(d / ms.length * 100)}%"></i></div>
      <div class="chips">${ms.map(m => `<div class="chip${CFG.done[m.id] ? ' done' : ''}${m.id === cur ? ' cur' : ''}${CFG.specs[m.spec] ? '' : ' bad'}"
        onclick="pickLoom('${m.id}')">${m.id}</div>`).join('')}</div></div>`;
  }).join('') + '<div class="legend"><span>Not counted</span><span class="d">Done</span><span class="b">Spec not in table</span></div>';
}
function pickLoom(id){ const s = $('machine'); s.value = id; s.dispatchEvent(new Event('change')); $('mCard').scrollIntoView({behavior: 'smooth', block: 'start'}); }

/* ------------------------- form ------------------------- */
$('machine').onchange = e => {
  const id = e.target.value;
  renderChips();
  if(!id){ $('form').style.display = 'none'; $('sumBar').style.display = 'none'; return; }
  const m = CFG.machines.find(x => x.id === id);
  R = {machine: id, photos: []};
  ['flFull', 'flHalf', 'flLow', 'note'].forEach(k => $(k).value = '');
  renderMachine(m); renderPhotos(); $('resCard').style.display = 'none';
  $('form').style.display = ''; $('sumBar').style.display = '';
  if(CFG.done[id]) showResult({totals: CFG.done[id], warnings: []}, true);
};
function renderMachine(m){
  const ok = !!CFG.specs[m.spec], [q, ck] = Calc.creel(m);
  $('mCard').innerHTML = `<h2>${m.id} <span class="tag ${m.group === 'SL' ? 'ok' : 'warn'}">${m.type}</span>
    ${ok ? '' : '<span class="tag bad">Spec not in table</span>'}</h2>
    <div class="kv"><span>Spec (Denier/T.W.)</span><b>${m.spec || '-'}</b>
    <span>Width / Dàn</span><b>${m.width ?? '-'} cm / ${m.dan ?? '-'}</b>
    <span>① Warp on creel</span><b>${q ?? '-'} bobbins · ${fmt(ck)} kg</b>
    <span>③ Weft in shuttles</span><b>${fmt(Calc.shuttle(m), 2)} kg</b></div>
    <div class="muted" style="margin-top:6px">①③ calculated automatically from the plan — no input needed</div>`;
}

let pendingPick = null;
$('fileIn').onchange = async e => {
  const f = e.target.files[0]; e.target.value = '';
  if(!f || !pendingPick) return;
  const fn = pendingPick; pendingPick = null;
  try { await fn(f); } catch(err){ console.error(err); alert(err.message || err); } finally { busy(false); }
};
function pick(fn){ pendingPick = fn; $('fileIn').click(); }
function busy(on, t){ $('busy').classList.toggle('on', on); if(t) $('busyTxt').textContent = t; }

/* resize photo (EXIF rotation applied automatically) */
async function shrink(file, max){
  const src = await createImageBitmap(file, {imageOrientation: 'from-image'});
  const s = Math.min(1, max / Math.max(src.width, src.height));
  const c = document.createElement('canvas'); c.width = Math.round(src.width * s); c.height = Math.round(src.height * s);
  c.getContext('2d').drawImage(src, 0, 0, c.width, c.height);
  const blob = await new Promise(r => c.toBlob(r, 'image/jpeg', 0.85));
  return {blob, bmp: await createImageBitmap(blob), url: URL.createObjectURL(blob), w: c.width, h: c.height};
}
async function runDetect(file){
  busy(true, AI.session ? 'AI is counting bobbins…' : 'Loading AI (first time only)…');
  const im = await shrink(file, 1600);
  await AI.load(p => $('busyTxt').textContent = `Loading AI ${Math.round(p * 100)}% (first time only)`);
  busy(true, 'AI is counting bobbins…');
  const dets = await AI.detect(im.bmp), cores = measureCores(im.bmp, dets);
  const circles = dets.map((d, i) => ({x: d.x, y: d.y, r: d.r, core: cores[i]}));
  return {blob: im.blob, url: im.url, w: im.w, h: im.h, circles, ai_count: circles.length};
}
function addPhoto(place){ pick(async f => { R.photos.push({place, ...(await runDetect(f))}); renderPhotos(); openEditor(R.photos.length - 1); }); }
function addCart(place){ R.photos.push({place, circles: [], sku: '', spec: ''}); renderPhotos(); }
function cartLabel(i){
  pick(async f => {
    busy(true, 'Reading label…');
    const im = await shrink(f, 2000), {p, pallet} = await OCR.read(im.blob);
    if(!p){ alert('Could not read the SKU — take a closer, sharper photo or type the SKU'); return; }
    setSku(i, p, pallet);
  });
}
function cartSkuText(i, v){ const p = parseSku(v); if(!p){ alert('Invalid SKU format (W + 14 digits)'); return; } setSku(i, p); }
function setSku(i, p, pallet){ Object.assign(R.photos[i], {sku: p.sku, spec: p.spec, pallet: pallet || R.photos[i].pallet || ''}); renderPhotos(); }
function cartPile(i){
  if(!R.photos[i].spec){ alert('Scan the label / enter the SKU first'); return; }
  pick(async f => { Object.assign(R.photos[i], await runDetect(f)); renderPhotos(); openEditor(i); });
}
function delPhoto(i){ if(confirm('Delete this item?')){ R.photos.splice(i, 1); renderPhotos(); } }

const imgs = {};
function getImg(url){ return imgs[url] ||= new Promise(res => { const im = new Image(); im.onload = () => res(im); im.src = url; }); }
async function drawThumb(i){
  const p = R.photos[i], c = $('th' + i); if(!c || !p.url) return;
  const im = await getImg(p.url), x = c.getContext('2d'), s = Math.max(192 / im.width, 192 / im.height);
  const ox = (192 - im.width * s) / 2, oy = (192 - im.height * s) / 2;
  x.drawImage(im, ox, oy, im.width * s, im.height * s); x.strokeStyle = '#ff2d2d'; x.lineWidth = 2;
  for(const q of p.circles){ x.beginPath(); x.arc(ox + q.x * s, oy + q.y * s, Math.max(2, q.r * s), 0, 7); x.stroke(); }
}
function renderPhotos(){
  for(const place of ['loom', 'cart_warp', 'cart_weft']){
    $(place + 'List').innerHTML = R.photos.map((p, i) => [p, i]).filter(([p]) => p.place === place).map(([p, i], k) => {
      const th = p.url ? `<canvas id="th${i}" width="192" height="192" onclick="openEditor(${i})"></canvas>` : '';
      const cnt = p.url ? `<div class="cnt">${p.circles.length} bobbins</div><div class="muted">AI counted ${p.ai_count} · tap photo to edit</div>` : '';
      if(place === 'loom') return `<div class="photo">${th}<div class="info">${cnt}</div><button class="sm danger" onclick="delPhoto(${i})">Delete</button></div>`;
      const ok = p.spec && CFG.specs[p.spec];
      return `<div class="cart"><div style="display:flex;justify-content:space-between;align-items:center">
          <b>Cart ${k + 1}</b><button class="sm danger" onclick="delPhoto(${i})">Delete</button></div>
        <label>SKU on label</label>
        <div style="display:flex;gap:8px"><input value="${p.sku || ''}" placeholder="W10000108525501" inputmode="text" onchange="cartSkuText(${i}, this.value)">
          <button class="sm" onclick="cartLabel(${i})">📷 Label</button></div>
        ${p.spec ? `<div class="specbig">${p.spec.replace('/', 'D / ')} mm ${ok ? '<span class="tag ok">✔</span>' : '<span class="tag bad">Not in table</span>'}</div>
          <div class="muted">Check it matches the label${p.pallet ? ' · Pallet ' + p.pallet : ''}</div>` : ''}
        ${p.url ? `<div class="photo">${th}<div class="info">${cnt}</div></div>` : ''}
        <button class="big" style="margin-top:8px" onclick="cartPile(${i})">📷 ${p.url ? 'Retake bobbin photo' : 'Photo of bobbins'}</button></div>`;
    }).join('');
  }
  R.photos.forEach((p, i) => p.url && drawThumb(i));
  const c = pl => R.photos.filter(p => p.place === pl).reduce((a, p) => a + p.circles.length, 0);
  $('sumTxt').textContent = `${R.machine} · rack ${c('loom')} · warp cart ${c('cart_warp')} · weft cart ${c('cart_weft')} bobbins`;
}

/* ------------------------- circle editor ------------------------- */
let E = null;
async function openEditor(i){
  const p = R.photos[i], im = await getImg(p.url);
  E = {i, im, undo: [], z: Math.min(1, window.innerWidth / im.width)}; $('ed').classList.add('on'); edDraw();
}
function edDraw(){
  const {im, z} = E, p = R.photos[E.i], c = $('edCv');
  c.width = Math.round(im.width * z); c.height = Math.round(im.height * z);
  const x = c.getContext('2d'); x.drawImage(im, 0, 0, c.width, c.height); x.lineWidth = 2.5;
  for(const q of p.circles){ x.strokeStyle = x.fillStyle = q.added ? '#00e0ff' : '#ff2d2d';
    x.beginPath(); x.arc(q.x * z, q.y * z, q.r * z, 0, 7); x.stroke(); x.beginPath(); x.arc(q.x * z, q.y * z, 3, 0, 7); x.fill(); }
  $('edCount').textContent = p.circles.length + ' bobbins';
}
$('edCv').addEventListener('click', ev => {
  const p = R.photos[E.i], rc = ev.target.getBoundingClientRect(), x = (ev.clientX - rc.left) / E.z, y = (ev.clientY - rc.top) / E.z;
  E.undo.push(JSON.stringify(p.circles));
  let best = -1, bd = 1e9;
  p.circles.forEach((q, k) => { const d = Math.hypot(q.x - x, q.y - y); if(d < q.r * 0.85 && d < bd){ bd = d; best = k; } });
  if(best >= 0) p.circles.splice(best, 1);
  else { const rs = p.circles.map(q => q.r).sort((a, b) => a - b); p.circles.push({x, y, r: rs.length ? rs[rs.length >> 1] : 40, core: null, added: true}); }
  edDraw();
});
function edZoom(k){ E.z = Math.max(0.1, Math.min(4, E.z * k)); edDraw(); }
function edUndo(){ if(E.undo.length){ R.photos[E.i].circles = JSON.parse(E.undo.pop()); edDraw(); } }
function edClose(){ $('ed').classList.remove('on'); E = null; renderPhotos(); }

/* ------------------------- save ------------------------- */
const b64 = blob => new Promise(r => { const fr = new FileReader(); fr.onload = () => r(fr.result.split(',')[1]); fr.readAsDataURL(blob); });
function compute(m, floor){
  const warn = [], [creel_qty, creel] = Calc.creel(m), parts = {creel, shuttle: Calc.shuttle(m), cart_warp: 0, loom: 0, cart_weft: 0, floor: 0};
  if(!CFG.specs[m.spec]) warn.push(`Loom spec ${m.spec} is not in the spec table — ①③④⑥ not calculated`);
  const photos = R.photos.map(p => {
    const kind = p.place === 'cart_warp' ? 'warp' : 'weft', spec = p.place === 'loom' ? m.spec : p.spec;
    const r = Calc.photo(p.circles, spec, kind);
    if(r.kg == null) warn.push(`Photo ${p.place}: spec ${spec} is not in the spec table — no weight calculated`); else parts[p.place] += r.kg;
    return {place: p.place, sku: p.sku || '', pallet: p.pallet || '', spec, ai_count: p.ai_count, count: r.count,
            equiv_full: +r.equiv_full.toFixed(2), kg: r.kg == null ? null : +r.kg.toFixed(2), bins: r.bins};
  });
  parts.floor = Calc.floor(m, floor) || 0;
  const warp = (parts.creel || 0) + parts.cart_warp, weft = (parts.shuttle || 0) + parts.loom + parts.cart_weft + parts.floor;
  const totals = Object.fromEntries(Object.entries(parts).map(([k, v]) => [k, v == null ? null : +v.toFixed(2)]));
  Object.assign(totals, {creel_qty, warp: +warp.toFixed(2), weft: +weft.toFixed(2), total: +(warp + weft).toFixed(2)});
  return {photos, totals, warnings: warn};
}
async function save(){
  if(!$('reporter').value.trim()){ alert('Enter your name first'); $('reporter').focus(); return; }
  if(!DEMO_MODE && !$('pin').value){ alert('Enter PIN first'); $('pin').focus(); return; }
  if(R.photos.some(p => p.place !== 'loom' && (!p.spec || !p.url))){ alert('Some carts have no SKU or no bobbin photo yet'); return; }
  const m = CFG.machines.find(x => x.id === R.machine);
  const floor = {full: +$('flFull').value || 0, half: +$('flHalf').value || 0, low: +$('flLow').value || 0};
  const c = compute(m, floor);
  const report = {date: CFG.date, machine: m.id, type: m.type, spec: m.spec, width: m.width, dan: m.dan,
    plan_updated: CFG.plan_updated, reporter: $('reporter').value.trim(), note: $('note').value, floor,
    totals: c.totals, warnings: c.warnings, saved_at: new Date().toISOString()};
  busy(true, 'Saving…');
  try {
    if(DEMO_MODE){
      CFG.done[m.id] = c.totals; store.set('demo_done_' + CFG.date, JSON.stringify(CFG.done));
      showResult({...report, warnings: [...c.warnings, 'DEMO mode — not sent to Google Sheet']});
    } else {
      const photos = await Promise.all(R.photos.map(async (p, i) => ({...c.photos[i],
        image: p.blob ? await b64(p.blob) : null,
        circles: p.circles.map(q => [Math.round(q.x), Math.round(q.y), Math.round(q.r * 10) / 10, q.core ? Math.round(q.core * 10) / 10 : 0])})));
      const body = {action: 'saveReport', token: $('pin').value, ...report, photos};
      const k = `${report.date}|${report.machine}`;
      await Outbox.put({k, body});
      const left = await Outbox.flush();
      CFG.done[m.id] = c.totals;
      showResult(left ? {...report, warnings: [...c.warnings, 'Not sent yet (no signal) — kept on this phone, will send automatically when online']} : report);
    }
    render(); $('machine').value = m.id;
  } catch(err){ alert(err.message); } finally { busy(false); }
}
function showResult(r, old){
  const t = r.totals; $('resCard').style.display = '';
  $('resCard').innerHTML = `<h2>${old ? 'Already saved (saving again overwrites)' : '✔ Saved'}</h2>
    ${(r.warnings || []).map(w => `<div class="banner warn">${w}</div>`).join('')}
    <table class="res">
    <tr><td>① Warp on creel (${t.creel_qty ?? '-'} bobbins)</td><td>${fmt(t.creel)} kg</td></tr>
    <tr><td>② Warp carts</td><td>${fmt(t.cart_warp)} kg</td></tr>
    <tr class="t"><td>Total warp</td><td>${fmt(t.warp)} kg</td></tr>
    <tr><td>③ Shuttles</td><td>${fmt(t.shuttle)} kg</td></tr>
    <tr><td>④ Loom rack</td><td>${fmt(t.loom)} kg</td></tr>
    <tr><td>⑤ Weft carts</td><td>${fmt(t.cart_weft)} kg</td></tr>
    <tr><td>⑥ Floor / basket</td><td>${fmt(t.floor)} kg</td></tr>
    <tr class="t"><td>Total weft</td><td>${fmt(t.weft)} kg</td></tr>
    <tr class="t"><td>Total for loom</td><td>${fmt(t.total)} kg</td></tr></table>`;
  if(!old) $('resCard').scrollIntoView({behavior: 'smooth'});
}

/* ------------------------- start ------------------------- */
$('reporter').value = store.get('reporter') || ''; $('pin').value = store.get('pin') || '';
$('reporter').onchange = e => store.set('reporter', e.target.value);
$('pin').onchange = e => { store.set('pin', e.target.value); loadConfig().catch(err => banner(err.message)); };
if(DEMO_MODE) $('pinWrap').style.display = 'none';
function banner(t){ $('banners').innerHTML = `<div class="banner bad">${t}</div>`; }
async function boot(){
  if(location.protocol === 'file:'){
    banner('This page was opened as a local file (file://). The AI cannot load this way — open it from the web link (GitHub Pages / https://…) instead.');
  }
  try { await loadConfig(); }
  catch(err){
    const c = store.get('cfg_cache');
    if(c){ const j = JSON.parse(c); CFG = {settings: j.settings, specs: Object.fromEntries(j.specs.map(r => [specKey(r.denier, r.tw), r])),
      machines: j.machines, plan_updated: j.plan_updated, done: {}, date: reportDate()}; render(); banner('Offline — using last saved data: ' + err.message); }
    else banner(err.message + (DEMO_MODE ? '' : ' (check your PIN)'));
  }
  Outbox.flush().catch(() => {});
  setTimeout(() => AI.load().catch(() => {}), 1500);   // preload AI
}
window.addEventListener('online', () => Outbox.flush());
setInterval(() => Outbox.flush().catch(() => {}), 60000);
boot();
