/* =====================================================================
 *  Label text reader — PaddleOCR (PP-OCRv4 det + rec) running in the phone with onnxruntime-web
 *  Much better than Tesseract on tilted / blurry / dim photos. Models are cached after the first load.
 * ===================================================================== */
const POCR = {
  det: null, rec: null, cls: null, loading: null,
  DET_URL: 'ocr_det.onnx', REC_URL: 'ocr_rec.onnx', KEYS_URL: 'ocr_keys.json',

  async load(onProgress){
    if(this.rec) return;
    if(this.loading) return this.loading;
    this.loading = (async () => {
      ort.env.wasm.wasmPaths = new URL('lib/', location.href).href;
      ort.env.wasm.numThreads = 1;
      const files = [this.DET_URL, this.REC_URL], bufs = [];
      for(let i = 0; i < files.length; i++){
        const u = files[i], prog = p => onProgress && onProgress((i + p) / files.length);
        let buf;
        try {
          const c = await caches.open(CACHE); let r = await c.match(u);
          if(!r){ r = await fetchProgress(u, prog); await c.put(u, r.clone()); }
          buf = await r.arrayBuffer();
        } catch(e){ buf = await (await fetch(u)).arrayBuffer(); }
        bufs.push(buf);
      }
      const keys = await (await fetch(this.KEYS_URL)).json();      // [[classIndex, char], …] — only the characters a label can contain
      this.allowed = Int32Array.from(keys.map(k => k[0])); this.chars = {}; keys.forEach(([i, c]) => this.chars[i] = c);
      this.det = await ort.InferenceSession.create(bufs[0], {executionProviders: ['wasm']});
      this.rec = await ort.InferenceSession.create(bufs[1], {executionProviders: ['wasm']});
    })();
    try { await this.loading; } catch(e){ this.loading = null; throw e; }
  },

  /* ---------- text detection (DB) ---------- */
  async detect(src, sx, sy, sw, sh, maxSide){
    let r = Math.min(1, maxSide / Math.max(sw, sh));
    if(Math.min(sw, sh) * r < 640) r = Math.min(maxSide * 1.25 / Math.max(sw, sh), 640 / Math.min(sw, sh));
    const w = Math.max(32, Math.round(sw * r / 32) * 32), h = Math.max(32, Math.round(sh * r / 32) * 32);
    const cv = document.createElement('canvas'); cv.width = w; cv.height = h;
    const g = cv.getContext('2d', {willReadFrequently: true}); g.drawImage(src, sx, sy, sw, sh, 0, 0, w, h);
    const d = g.getImageData(0, 0, w, h).data, n = w * h, f = new Float32Array(3 * n);
    for(let i = 0, p = 0; i < n; i++, p += 4){      // BGR, (v/255 − 0.5)/0.5
      f[i] = d[p+2] / 127.5 - 1; f[n + i] = d[p+1] / 127.5 - 1; f[2*n + i] = d[p] / 127.5 - 1; }
    const out = (await this.det.run({[this.det.inputNames[0]]: new ort.Tensor('float32', f, [1, 3, h, w])}))[this.det.outputNames[0]];
    const prob = out.data, bm = new Uint8Array(n);
    for(let y = 0; y < h; y++) for(let x = 0; x < w; x++){           // threshold 0.3 + 2×2 dilation
      const i = y * w + x; if(prob[i] > 0.3){ bm[i] = 1; if(x + 1 < w) bm[i+1] = 1; if(y + 1 < h){ bm[i+w] = 1; if(x + 1 < w) bm[i+w+1] = 1; } } }
    const seen = new Uint8Array(n), boxes = [], stack = [];
    const kx = sw / w, ky = sh / h;
    for(let s = 0; s < n; s++){
      if(!bm[s] || seen[s]) continue;
      const pts = []; let sum = 0, cnt = 0; stack.push(s); seen[s] = 1;
      while(stack.length){
        const i = stack.pop(), x = i % w, y = (i / w) | 0; sum += prob[i]; cnt++;
        // only border pixels matter for the hull
        if(x === 0 || y === 0 || x === w - 1 || y === h - 1 || !bm[i-1] || !bm[i+1] || !bm[i-w] || !bm[i+w]) pts.push([x, y], [x + 1, y + 1]);
        for(let dy = -1; dy <= 1; dy++) for(let dx = -1; dx <= 1; dx++){
          const xx = x + dx, yy = y + dy; if(xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
          const j = yy * w + xx; if(bm[j] && !seen[j]){ seen[j] = 1; stack.push(j); } }
      }
      if(cnt < 10 || sum / cnt < 0.5) continue;                       // box score threshold
      const rc = minRect(hull(pts)); if(!rc || Math.min(rc.w, rc.h) < 3) continue;
      const dist = rc.w * rc.h * 1.6 / (2 * (rc.w + rc.h));            // unclip ratio 1.6
      rc.w += 2 * dist; rc.h += 2 * dist; if(Math.min(rc.w, rc.h) < 5) continue;
      // back to source coordinates (scale x and y separately — fine for near-uniform scale)
      boxes.push({cx: sx + rc.cx * kx, cy: sy + rc.cy * ky, w: rc.w * kx, h: rc.h * ky, a: rc.a});
    }
    return boxes;
  },

  /* ---------- text recognition (CTC), vocabulary limited to label characters ---------- */
  async recognize(src, b){
    let L = b.w, S = b.h, a = b.a;                                   // make the long side the reading direction
    if(S > L){ [L, S] = [S, L]; a += Math.PI / 2; }
    if(Math.cos(a) < 0) a += Math.PI;                                  // keep text upright-ish (left → right)
    const H = 48, Wt = Math.min(1600, Math.max(8, Math.ceil(H * L / S))), Wp = Math.max(320, Wt);
    const cv = document.createElement('canvas'); cv.width = Wt; cv.height = H;
    const g = cv.getContext('2d', {willReadFrequently: true}); g.imageSmoothingQuality = 'high';
    g.translate(Wt / 2, H / 2); g.scale(Wt / L, H / S); g.rotate(-a); g.translate(-b.cx, -b.cy); g.drawImage(src, 0, 0);
    const d = g.getImageData(0, 0, Wt, H).data, n = H * Wp, f = new Float32Array(3 * n);   // zero padding on the right
    for(let y = 0; y < H; y++) for(let x = 0; x < Wt; x++){
      const p = (y * Wt + x) * 4, i = y * Wp + x;
      f[i] = d[p+2] / 127.5 - 1; f[n + i] = d[p+1] / 127.5 - 1; f[2*n + i] = d[p] / 127.5 - 1; }
    const out = (await this.rec.run({[this.rec.inputNames[0]]: new ort.Tensor('float32', f, [1, 3, H, Wp])}))[this.rec.outputNames[0]];
    const T = out.dims[1], C = out.dims[2], o = out.data, al = this.allowed;
    let txt = '', prev = -1, conf = 0, nc = 0;
    for(let t = 0; t < T; t++){
      const base = t * C; let bi = 0, bv = o[base];                    // class 0 = CTC blank
      for(let k = 0; k < al.length; k++){ const v = o[base + al[k]]; if(v > bv){ bv = v; bi = al[k]; } }
      if(bi !== 0 && bi !== prev){ txt += this.chars[bi]; conf += bv; nc++; }
      prev = bi;
    }
    return {text: txt, conf: nc ? conf / nc : 0};
  },

  async lines(src, sx, sy, sw, sh, maxSide){
    const boxes = await this.detect(src, sx, sy, sw, sh, maxSide), out = [];
    boxes.sort((p, q) => (p.cy - q.cy) || (p.cx - q.cx));
    for(const b of boxes){ const r = await this.recognize(src, b); if(r.text.trim() && r.conf > 0.4) out.push({...r, box: b}); }
    return out;
  },

  /* read a label photo → {text, boxes}; second pass on the label area at higher resolution if something is missing */
  async read(src, need){
    const W = src.width, H = src.height;
    let res = await this.lines(src, 0, 0, W, H, 1280), text = res.map(r => r.text).join('\n');
    if(res.length && need && !need(text)){
      const xs = [], ys = [];
      for(const {box: b} of res){ const e = Math.max(b.w, b.h) / 2; xs.push(b.cx - e, b.cx + e); ys.push(b.cy - e, b.cy + e); }
      let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
      const mx = (x1 - x0) * 0.06 + 10, my = (y1 - y0) * 0.1 + 10;
      x0 = Math.max(0, x0 - mx); y0 = Math.max(0, y0 - my); x1 = Math.min(W, x1 + mx); y1 = Math.min(H, y1 + my);
      if((x1 - x0) * (y1 - y0) < W * H * 0.8){
        const res2 = await this.lines(src, x0, y0, x1 - x0, y1 - y0, 1280);
        text += '\n' + res2.map(r => r.text).join('\n');
      }
    }
    return text;
  }
};

/* convex hull (monotone chain) */
function hull(P){
  P.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cr = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]), lo = [], up = [];
  for(const p of P){ while(lo.length >= 2 && cr(lo[lo.length-2], lo[lo.length-1], p) <= 0) lo.pop(); lo.push(p); }
  for(let i = P.length - 1; i >= 0; i--){ const p = P[i]; while(up.length >= 2 && cr(up[up.length-2], up[up.length-1], p) <= 0) up.pop(); up.push(p); }
  up.pop(); lo.pop(); return lo.concat(up);
}
/* minimum-area rectangle of a convex polygon (rotating edges) → {cx, cy, w, h, a} with w along angle a */
function minRect(hp){
  if(hp.length < 3) return null;
  let best = null;
  for(let i = 0; i < hp.length; i++){
    const p = hp[i], q = hp[(i + 1) % hp.length], a = Math.atan2(q[1] - p[1], q[0] - p[0]), c = Math.cos(a), s = Math.sin(a);
    let u0 = Infinity, u1 = -Infinity, v0 = Infinity, v1 = -Infinity;
    for(const [x, y] of hp){ const u = x * c + y * s, v = -x * s + y * c; if(u < u0) u0 = u; if(u > u1) u1 = u; if(v < v0) v0 = v; if(v > v1) v1 = v; }
    const area = (u1 - u0) * (v1 - v0);
    if(!best || area < best.area){ const uc = (u0 + u1) / 2, vc = (v0 + v1) / 2;
      best = {area, cx: uc * c - vc * s, cy: uc * s + vc * c, w: u1 - u0, h: v1 - v0, a}; }
  }
  return best;
}
