/* DL-Edu 3장부터 쓰는 공통 그림·계산 도구. dl.js 다음, chNN.js 앞에 읽는다
   Shared figure and maths helpers for chapter 3 onwards. Load after dl.js and before chNN.js
   - 짧은 도구 (숫자 쓰기, 글자 배경, 화살표, timer)
     small helpers (number formatting, text with background, arrows, timers)
   - 등고선 그림, 2D 분류 경계 그림, 신경망 그림, pixel 그림
     contour maps, 2D decision fields, network diagrams, pixel images
   - TinyNet: browser 안에서 학습하는 작은 신경망 (backprop, SGD·momentum·RMSprop·Adam, L1·L2, dropout)
     TinyNet: a small neural network that trains in the browser (backprop, SGD, momentum, RMSprop, Adam, L1, L2, dropout) */
"use strict";

const HAS_DOM = typeof document !== "undefined";
const $ = id => document.getElementById(id);
const reduceMotion = HAS_DOM && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
function fx(v, d = 2) { let s = (+v).toFixed(d); if (/^-0(\.0*)?$/.test(s)) s = s.slice(1); return s.replace("-", "−"); }
const paren = (v, d = 2) => (v < 0 ? "(" + fx(v, d) + ")" : fx(v, d));
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const SUP = n => String(n).replace(/./g, ch => "⁰¹²³⁴⁵⁶⁷⁸⁹"["0123456789".indexOf(ch)] || (ch === "-" ? "⁻" : ch));
const SUB = n => String(n).replace(/./g, ch => "₀₁₂₃₄₅₆₇₈₉"["0123456789".indexOf(ch)] || ch);
function sci(v) {
  const a = Math.abs(v); if (a === 0) return "0";
  if (a >= 1e4 || a < 1e-3) { let e = Math.floor(Math.log10(a)); if (Math.abs(+(v / 10 ** e).toFixed(1)) >= 10) e++; return fx(v / 10 ** e, 1) + "×10" + SUP(e); }
  return a >= 100 ? fx(v, 0) : a >= 1 ? fx(v, 2) : fx(v, 4);
}
const commas = n => String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
function txtBg(ctx, c, s, x, y, o = {}) {
  ctx.save(); ctx.font = (o.w || "400") + " " + (o.size || 11.5) + "px " + (o.mono ? FM : FS);
  const wd = ctx.measureText(s).width, x0 = o.align === "right" ? x - wd : o.align === "center" ? x - wd / 2 : x;
  ctx.fillStyle = c.bg; ctx.globalAlpha = 0.9; ctx.fillRect(x0 - 3, y - (o.size || 11.5) + 1, wd + 6, (o.size || 11.5) + 5); ctx.restore();
  txt(ctx, s, x, y, o);
}
function textW(ctx, s, size = 11.5, w = "400", mono = false) { ctx.save(); ctx.font = w + " " + size + "px " + (mono ? FM : FS); const v = ctx.measureText(s).width; ctx.restore(); return v; }
function panelTitle(ctx, c, s, x, y) { txt(ctx, s, x, y, { color: c.ink, w: "600", size: 12.5 }); }
function curve(A, f, xr, n = 240) { const p = []; for (let i = 0; i <= n; i++) { const x = xr[0] + (xr[1] - xr[0]) * i / n; p.push([A.X(x), A.Y(f(x))]); } return p; }
function arrowS(ctx, x0, y0, x1, y1, color, width = 1.4, head = 5) {
  line(ctx, [[x0, y0], [x1, y1]], color, width);
  const a = Math.atan2(y1 - y0, x1 - x0);
  ctx.save(); ctx.fillStyle = color; ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x1 - head * Math.cos(a - 0.45), y1 - head * Math.sin(a - 0.45)); ctx.lineTo(x1 - head * Math.cos(a + 0.45), y1 - head * Math.sin(a + 0.45)); ctx.closePath(); ctx.fill(); ctx.restore();
}
function rrect(ctx, x, y, w, h, r, fill, stroke, lw = 1.2) {
  ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
  if (fill) { ctx.fillStyle = fill; ctx.fill(); } if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.stroke(); }
}
// 걸음을 차례로 보여 주는 timer (움직임 줄이기 설정이면 한 번에)
// Timer that plays steps one after another (all at once with reduced motion)
function stopRun(S) { clearInterval(S.timer); S.timer = 0; }
function runSteps(S, n, step, ms = 140, done) {
  stopRun(S); let k = 0;
  if (reduceMotion) { while (k++ < n && step() !== false); if (done) done(); return; }
  S.timer = setInterval(() => { if (k++ >= n || step() === false) { stopRun(S); if (done) done(); } }, ms);
}
function setRange(id, v, d = 2) { const el = $(id); el.value = v; const o = $(id.replace(/^i/, "o")); if (o) o.textContent = fx(v, d); }
function niceStepC(raw) { const p = 10 ** Math.floor(Math.log10(raw)), m = raw / p; return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p; }
function ticksR(lo, hi, n = 5) { const st = niceStepC((hi - lo) / n), out = []; for (let v = Math.ceil(lo / st) * st; v <= hi + 1e-9; v += st) out.push(+v.toFixed(10)); return out; }
function intTicks(r) { const out = []; for (let v = Math.ceil(r[0]); v <= Math.floor(r[1]); v++) out.push(v); return out; }
function equalRange(B, yHalf, cx = 0, cy = 0) { const xh = yHalf * B.w / B.h; return { xr: [cx - xh, cx + xh], yr: [cy - yHalf, cy + yHalf] }; }

// 색 섞기
// Colour mixing
function rgb(hex) { let h = String(hex).trim().replace("#", ""); if (h.length === 3) h = h.split("").map(ch => ch + ch).join(""); const n = parseInt(h, 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; }
const mixRGB = (a, b, t) => [0, 1, 2].map(i => Math.round(a[i] + (b[i] - a[i]) * t));
const cssRGB = v => `rgb(${v[0]},${v[1]},${v[2]})`;

// 2D 함수의 등고선 그림: 낮을수록 진한 파랑, 띠의 경계 = 등고선. 같은 조건이면 다시 계산하지 않는다
// Contour picture of a 2D function: lower = deeper blue, band edges = contour lines. Cached for identical inputs
const contourCache = new Map();
function contour(ctx, c, B, f, xr, yr, key, levels = 13) {
  const cell = 2, nx = Math.max(2, Math.round(B.w / cell)), ny = Math.max(2, Math.round(B.h / cell));
  const k = [key, nx, ny, xr, yr, c.bg, c.sig].join("|");
  let off = contourCache.get(k);
  if (!off) {
    const v = new Float32Array(nx * ny); let lo = Infinity, hi = -Infinity;
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const z = f(xr[0] + (i + 0.5) / nx * (xr[1] - xr[0]), yr[1] - (j + 0.5) / ny * (yr[1] - yr[0]));
      v[j * nx + i] = z; if (z < lo) lo = z; if (z > hi) hi = z;
    }
    const top = Math.sqrt(hi - lo) || 1, band = new Int16Array(nx * ny);
    for (let i = 0; i < v.length; i++) band[i] = Math.min(levels - 1, Math.floor(Math.sqrt(v[i] - lo) / top * levels));
    off = document.createElement("canvas"); off.width = nx; off.height = ny;
    const ox = off.getContext("2d"), img = ox.createImageData(nx, ny), b0 = rgb(c.bg), b1 = rgb(c.sig), lc = rgb(c.muted);
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const q = j * nx + i, L = band[q], edge = (i < nx - 1 && band[q + 1] !== L) || (j < ny - 1 && band[q + nx] !== L);
      const a = edge ? 0.5 : 0.07 + 0.5 * (1 - L / (levels - 1)), to = edge ? lc : b1;
      for (let ch = 0; ch < 3; ch++) img.data[q * 4 + ch] = Math.round(b0[ch] + (to[ch] - b0[ch]) * a);
      img.data[q * 4 + 3] = 255;
    }
    ox.putImageData(img, 0, 0);
    if (contourCache.size > 60) contourCache.clear();
    contourCache.set(k, off);
  }
  ctx.save(); ctx.imageSmoothingEnabled = false; ctx.drawImage(off, B.l, B.t, B.w, B.h); ctx.restore();
}

// 2D 분류 경계 그림: 칸마다 class 확률로 색을 섞는다. fn(x, y) → 확률 배열 (2 class면 [p0, p1])
// 2D decision field: each cell mixes class colours by probability. fn(x, y) → array of probabilities
function classColors(c) { return [c.sig, c.miss, c.ok, c.vio]; }
function drawField(ctx, c, B, xr, yr, fn, o = {}) {
  const cell = o.cell || 4, nx = Math.max(2, Math.round(B.w / cell)), ny = Math.max(2, Math.round(B.h / cell));
  const off = document.createElement("canvas"); off.width = nx; off.height = ny;
  const ox = off.getContext("2d"), img = ox.createImageData(nx, ny), bg = rgb(c.bg), cols = classColors(c).map(rgb), amax = o.alpha || 0.42;
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const p = fn(xr[0] + (i + 0.5) / nx * (xr[1] - xr[0]), yr[1] - (j + 0.5) / ny * (yr[1] - yr[0]));
    let best = 0; for (let k = 1; k < p.length; k++) if (p[k] > p[best]) best = k;
    const conf = p.length > 1 ? (p[best] - 1 / p.length) / (1 - 1 / p.length) : 0;
    const v = mixRGB(bg, cols[best], 0.08 + amax * conf), q = (j * nx + i) * 4;
    img.data[q] = v[0]; img.data[q + 1] = v[1]; img.data[q + 2] = v[2]; img.data[q + 3] = 255;
  }
  ox.putImageData(img, 0, 0);
  ctx.save(); ctx.imageSmoothingEnabled = true; ctx.drawImage(off, B.l, B.t, B.w, B.h); ctx.restore();
}
// 점 찍기: 모양으로도 class를 구분한다 (0 = 원, 1 = 사각형, 2 = 삼각형)
// Scatter points: classes differ by shape too (0 = circle, 1 = square, 2 = triangle)
function drawPoints(ctx, c, A, X, Y, o = {}) {
  const cols = classColors(c), r = o.r || 3.6;
  X.forEach((p, i) => {
    const k = Y[i], x = A.X(p[0]), y = A.Y(p[1]); ctx.save(); ctx.globalAlpha = o.alpha || 0.95;
    ctx.fillStyle = cols[k]; ctx.strokeStyle = c.bg; ctx.lineWidth = 1;
    ctx.beginPath();
    if (k === 0) ctx.arc(x, y, r, 0, 2 * Math.PI);
    else if (k === 1) ctx.rect(x - r * 0.9, y - r * 0.9, r * 1.8, r * 1.8);
    else { ctx.moveTo(x, y - r * 1.1); ctx.lineTo(x + r, y + r * 0.8); ctx.lineTo(x - r, y + r * 0.8); ctx.closePath(); }
    ctx.fill(); ctx.stroke(); ctx.restore();
  });
}

// 신경망 그림: layers = [2, 3, 1]. o.vals[l][i] = node 값, o.W[l][j][i] = weight (선 색 = 부호, 굵기 = 크기)
// Network diagram: layers = [2, 3, 1]. o.vals[l][i] = node values, o.W[l][j][i] = weights (colour = sign, width = size)
function drawNet(ctx, c, layers, B, o = {}) {
  const L = layers.length, r = o.r || Math.min(16, B.h / (Math.max(...layers) * 2.6));
  const pos = layers.map((n, l) => Array.from({ length: n }, (_, i) => [B.l + (L === 1 ? B.w / 2 : l * B.w / (L - 1)), B.t + (i + 0.5) * B.h / n]));
  for (let l = 1; l < L; l++) for (let j = 0; j < layers[l]; j++) for (let i = 0; i < layers[l - 1]; i++) {
    const w = o.W && o.W[l - 1] ? o.W[l - 1][j][i] : null, hl = o.hl && o.hl(l, j, i);
    const col = w === null ? c.line : w >= 0 ? c.sig : c.fa, lw = w === null ? 1.2 : 0.8 + Math.min(4, Math.abs(w) * (o.wScale || 1.6));
    ctx.save(); ctx.globalAlpha = hl === false ? 0.15 : 0.85;
    line(ctx, [pos[l - 1][i], pos[l][j]], hl ? c.miss : col, hl ? lw + 1 : lw); ctx.restore();
  }
  pos.forEach((col, l) => col.forEach(([x, y], i) => {
    const v = o.vals && o.vals[l] ? o.vals[l][i] : null, on = o.nodeOn ? o.nodeOn(l, i) : true;
    ctx.save(); ctx.globalAlpha = on ? 1 : 0.25;
    dot(ctx, x, y, r, o.fill ? o.fill(l, i, v) : c.bg, o.stroke ? o.stroke(l, i) : c.ink);
    if (v !== null && v !== undefined && o.showVals !== false) txt(ctx, typeof v === "string" ? v : fx(v, o.dp === undefined ? 2 : o.dp), x, y + 4, { color: c.ink, size: o.valSize || 10.5, mono: true, align: "center", w: "600" });
    ctx.restore();
  }));
  return pos;
}

// pixel 그림: data = 길이 w·h 배열 (0~1). 음수가 있으면 o.div = true로 파랑/빨강
// Pixel image: data = array of length w·h (0–1). With negatives use o.div = true for blue/red
function drawImg(ctx, c, data, w, h, x, y, cell, o = {}) {
  const ink = rgb(c.ink), bg = rgb(o.bg || c.bg), pos = rgb(c.sig), neg = rgb(c.fa), mx = o.max || (o.div ? Math.max(1e-9, ...data.map(Math.abs)) : 1);
  for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
    const v = data[j * w + i];
    ctx.fillStyle = o.div ? cssRGB(mixRGB(bg, v >= 0 ? pos : neg, Math.min(1, Math.abs(v) / mx))) : cssRGB(mixRGB(bg, ink, clamp(v / mx, 0, 1)));
    ctx.fillRect(x + i * cell, y + j * cell, Math.ceil(cell), Math.ceil(cell));
  }
  if (o.grid) { ctx.save(); ctx.strokeStyle = c.line; ctx.lineWidth = 0.6; for (let i = 0; i <= w; i++) line(ctx, [[x + i * cell, y], [x + i * cell, y + h * cell]], c.line, 0.6); for (let j = 0; j <= h; j++) line(ctx, [[x, y + j * cell], [x + w * cell, y + j * cell]], c.line, 0.6); ctx.restore(); }
  if (o.frame !== false) { ctx.strokeStyle = c.line; ctx.lineWidth = 1; ctx.strokeRect(x - 0.5, y - 0.5, w * cell + 1, h * cell + 1); }
}

// 행렬 그림: 칸마다 숫자. o.cw, o.ch = 칸 크기, o.col = 테두리 색, o.label = 아래 이름, o.hl(i, j) = 강조할 칸
// Matrix picture: a number in each cell. o.cw, o.ch = cell size, o.col = border colour, o.label = name below, o.hl(i, j) = cells to highlight
function drawMat(ctx, c, M, x, y, o = {}) {
  const rows = M.length, cols = M[0].length, cw = o.cw || 46, ch = o.ch || 24, col = o.col || c.ink, W = cols * cw, H = rows * ch;
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
    const hl = o.hl && o.hl(i, j);
    if (hl) { ctx.fillStyle = hl === true ? c.sigwash : hl; ctx.fillRect(x + j * cw + 1, y + i * ch + 1, cw - 2, ch - 2); }
    const v = M[i][j];
    txt(ctx, typeof v === "string" ? v : fx(v, o.dp === undefined ? 2 : o.dp), x + j * cw + cw / 2, y + i * ch + ch / 2 + 4, { color: o.ink || c.ink, size: o.size || 11.5, mono: true, align: "center" });
  }
  ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = 1.4; ctx.beginPath();
  ctx.moveTo(x + 5, y); ctx.lineTo(x, y); ctx.lineTo(x, y + H); ctx.lineTo(x + 5, y + H);
  ctx.moveTo(x + W - 5, y); ctx.lineTo(x + W, y); ctx.lineTo(x + W, y + H); ctx.lineTo(x + W - 5, y + H); ctx.stroke(); ctx.restore();
  if (o.label) txt(ctx, o.label, x + W / 2, y + H + 15, { color: o.labelColor || col, size: 11, align: "center", w: "600" });
  return { w: W, h: H };
}

/* ================= 수학 도구 / Maths helpers ================= */
const ACT = {
  linear: { f: z => z, d: () => 1 },
  sigmoid: { f: z => 1 / (1 + Math.exp(-z)), d: (z, a) => a * (1 - a) },
  tanh: { f: z => Math.tanh(z), d: (z, a) => 1 - a * a },
  relu: { f: z => (z > 0 ? z : 0), d: z => (z > 0 ? 1 : 0) },
  leaky: { f: z => (z > 0 ? z : 0.1 * z), d: z => (z > 0 ? 1 : 0.1) },
  elu: { f: z => (z > 0 ? z : Math.exp(z) - 1), d: (z, a) => (z > 0 ? 1 : a + 1) },
  gelu: { f: z => 0.5 * z * (1 + Math.tanh(0.7978845608 * (z + 0.044715 * z ** 3))), d: z => { const u = 0.7978845608 * (z + 0.044715 * z ** 3), t = Math.tanh(u); return 0.5 * (1 + t) + 0.5 * z * (1 - t * t) * 0.7978845608 * (1 + 3 * 0.044715 * z * z); } }
};
function softmaxV(z) { const m = Math.max(...z), e = z.map(v => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0); return e.map(v => v / s); }
const matVec = (W, x) => W.map(row => row.reduce((s, w, i) => s + w * x[i], 0));
const matMul = (A, B) => A.map(row => B[0].map((_, j) => row.reduce((s, v, k) => s + v * B[k][j], 0)));
const transpose = A => A[0].map((_, j) => A.map(r => r[j]));
const meanOf = a => a.reduce((s, v) => s + v, 0) / a.length;
const sdOf = a => { const m = meanOf(a); return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length); };
function shuffled(n, rng) { const a = Array.from({ length: n }, (_, i) => i); for (let i = n - 1; i > 0; i--) { const j = Math.floor(rng.u() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; }

// 2D 연습 data (점 X와 class Y)
// 2D practice data (points X and classes Y)
function makeData(kind, n, rng, noise = 0.1) {
  const X = [], Y = [], g = () => rng.n() * noise;
  for (let i = 0; i < n; i++) {
    let x, y, k;
    if (kind === "blobs") { k = i % 2; x = (k ? 0.6 : -0.6) + g() * 3; y = (k ? 0.35 : -0.35) + g() * 3; }
    else if (kind === "xor") { k = i % 2; const sx = rng.u() < 0.5 ? 1 : -1, sy = k ? -sx : sx; x = sx * (0.25 + 0.7 * rng.u()) + g(); y = sy * (0.25 + 0.7 * rng.u()) + g(); }
    else if (kind === "circles") { k = i % 2; const a = rng.u() * 2 * Math.PI, rr = (k ? 1.0 : 0.42) + g(); x = rr * Math.cos(a); y = rr * Math.sin(a); }
    else if (kind === "moons") { k = i % 2; const a = rng.u() * Math.PI; x = k ? 1 - Math.cos(a) - 0.5 : Math.cos(a) - 0.5; y = k ? 0.25 - Math.sin(a) + 0.25 : Math.sin(a) - 0.25; x *= 0.85; y *= 0.85; x += g(); y += g(); }
    else if (kind === "spiral") { k = i % 2; const t = 0.25 + rng.u() * 0.75, a = t * 3.6 * Math.PI + k * Math.PI; x = t * Math.cos(a) * 1.1 + g() * 0.6; y = t * Math.sin(a) * 1.1 + g() * 0.6; }
    else if (kind === "three") { k = i % 3; const cx = [-0.7, 0.7, 0][k], cy = [-0.45, -0.45, 0.65][k]; x = cx + g() * 3; y = cy + g() * 3; }
    X.push([x, y]); Y.push(k);
  }
  return { X, Y };
}

/* ================= TinyNet: 작은 신경망 / a small neural network ================= */
// sizes = [입력, 숨은 층..., 출력]. hidden = 숨은 층 activation. out = "sigmoid" | "softmax" | "linear". loss = "bce" | "ce" | "mse"
// sizes = [input, hidden..., output]. hidden = hidden activation. out = "sigmoid" | "softmax" | "linear". loss = "bce" | "ce" | "mse"
class TinyNet {
  constructor(sizes, o = {}) {
    this.sizes = sizes; this.hidden = o.hidden || "tanh"; this.out = o.out || (sizes[sizes.length - 1] > 1 ? "softmax" : "sigmoid");
    this.lossName = o.loss || (this.out === "softmax" ? "ce" : this.out === "sigmoid" ? "bce" : "mse");
    this.rng = o.rng || makeRng(o.seed || 1);
    this.W = []; this.b = [];
    for (let l = 1; l < sizes.length; l++) {
      const nin = sizes[l - 1], nout = sizes[l], sd = TinyNet.initSD(o.init || "xavier", nin, nout, o.sd);
      this.W.push(Array.from({ length: nout }, () => Array.from({ length: nin }, () => this.rng.n() * sd)));
      this.b.push(new Array(nout).fill(0));
    }
    this.opt = { name: "sgd", lr: 0.1, beta: 0.9, beta2: 0.999, eps: 1e-8, l2: 0, l1: 0, drop: 0, ...o.opt };
    this.state = null; this.t = 0;
  }
  static initSD(kind, nin, nout, sd) {
    if (kind === "xavier") return Math.sqrt(2 / (nin + nout));
    if (kind === "kaiming") return Math.sqrt(2 / nin);
    return sd === undefined ? 1 : sd;
  }
  nParams() { return this.W.reduce((s, W, l) => s + W.length * W[0].length + this.b[l].length, 0); }
  act(l) { return l === this.W.length - 1 ? this.out : this.hidden; }
  // 한 sample의 forward. train이면 dropout을 쓴다. 층마다 z(곱한 합)와 a(activation 뒤)를 돌려준다
  // Forward pass for one sample. Uses dropout when train is true. Returns z (weighted sum) and a (after activation) per layer
  forward(x, train = false) {
    const zs = [], as = [x.slice()], masks = [];
    let a = x;
    for (let l = 0; l < this.W.length; l++) {
      const z = this.W[l].map((row, j) => row.reduce((s, w, i) => s + w * a[i], this.b[l][j]));
      let an;
      if (this.act(l) === "softmax") an = softmaxV(z);
      else { const F = ACT[this.act(l)]; an = z.map(F.f); }
      if (train && this.opt.drop > 0 && l < this.W.length - 1) {
        const keep = 1 - this.opt.drop, m = an.map(() => (this.rng.u() < keep ? 1 / keep : 0));
        an = an.map((v, j) => v * m[j]); masks.push(m);
      } else masks.push(null);
      zs.push(z); as.push(an); a = an;
    }
    return { zs, as, masks };
  }
  predict(x) { const f = this.forward(x); return f.as[f.as.length - 1]; }
  // 출력 → class 확률 배열 (그림용)
  // Output → array of class probabilities (for drawing)
  probs(x) { const p = this.predict(x); return this.out === "sigmoid" && p.length === 1 ? [1 - p[0], p[0]] : p; }
  target(y) { const k = this.sizes[this.sizes.length - 1]; if (Array.isArray(y)) return y; return k === 1 ? [y] : Array.from({ length: k }, (_, i) => (i === y ? 1 : 0)); }
  lossOne(p, t) {
    if (this.lossName === "mse") return p.reduce((s, v, i) => s + (v - t[i]) ** 2, 0) / p.length;
    if (this.lossName === "bce") return -p.reduce((s, v, i) => s + t[i] * Math.log(Math.max(v, 1e-12)) + (1 - t[i]) * Math.log(Math.max(1 - v, 1e-12)), 0) / p.length;
    return -p.reduce((s, v, i) => s + t[i] * Math.log(Math.max(v, 1e-12)), 0);
  }
  loss(X, Y) { return X.reduce((s, x, i) => s + this.lossOne(this.predict(x), this.target(Y[i])), 0) / X.length; }
  accuracy(X, Y) {
    let ok = 0;
    X.forEach((x, i) => { const p = this.predict(x), k = p.length === 1 ? (p[0] > 0.5 ? 1 : 0) : p.indexOf(Math.max(...p)); if (k === Y[i]) ok++; });
    return ok / X.length;
  }
  // batch의 평균 gradient (backprop). 반환: {gW, gb, loss}
  // Mean gradient over a batch (backprop). Returns {gW, gb, loss}
  grads(X, Y, train = true) {
    const gW = this.W.map(W => W.map(r => r.map(() => 0))), gb = this.b.map(b => b.map(() => 0));
    let loss = 0; const L = this.W.length, n = X.length;
    X.forEach((x, s) => {
      const t = this.target(Y[s]), { zs, as, masks } = this.forward(x, train), p = as[L];
      loss += this.lossOne(p, t);
      // 출력층의 delta = dLoss/dz
      // Output delta = dLoss/dz
      let delta;
      if ((this.out === "softmax" && this.lossName === "ce") || (this.out === "sigmoid" && this.lossName === "bce")) delta = p.map((v, i) => (v - t[i]) / (this.lossName === "bce" ? p.length : 1));
      else {
        const dA = p.map((v, i) => (this.lossName === "mse" ? 2 * (v - t[i]) / p.length : this.lossName === "bce" ? (v - t[i]) / Math.max(v * (1 - v), 1e-12) / p.length : -t[i] / Math.max(v, 1e-12)));
        if (this.out === "softmax") delta = p.map((pi, i) => p.reduce((s2, pj, j) => s2 + dA[j] * pj * ((i === j ? 1 : 0) - pi), 0));
        else { const F = ACT[this.out]; delta = dA.map((g, i) => g * F.d(zs[L - 1][i], p[i])); }
      }
      for (let l = L - 1; l >= 0; l--) {
        const aPrev = as[l];
        for (let j = 0; j < delta.length; j++) { gb[l][j] += delta[j]; for (let i = 0; i < aPrev.length; i++) gW[l][j][i] += delta[j] * aPrev[i]; }
        if (l > 0) {
          const F = ACT[this.hidden], m = masks[l - 1];
          const back = aPrev.map((_, i) => this.W[l].reduce((s2, row, j) => s2 + row[i] * delta[j], 0));
          // dropout으로 꺼진 unit은 gradient도 0. 켜진 unit은 1/keep 배
          // Units dropped out get zero gradient; kept units are scaled by 1/keep
          delta = back.map((g, i) => { const aRaw = m ? (m[i] ? aPrev[i] / m[i] : 0) : aPrev[i]; return g * (m ? m[i] : 1) * F.d(zs[l - 1][i], aRaw); });
        }
      }
    });
    gW.forEach(W => W.forEach(r => r.forEach((_, i) => { r[i] /= n; })));
    gb.forEach(b => b.forEach((_, i) => { b[i] /= n; }));
    return { gW, gb, loss: loss / n };
  }
  // optimizer로 한 걸음
  // One optimizer step
  step(X, Y) {
    const o = this.opt, { gW, gb, loss } = this.grads(X, Y, true);
    if (!this.state) this.state = { mW: gW.map(W => W.map(r => r.map(() => 0))), vW: gW.map(W => W.map(r => r.map(() => 0))), mb: gb.map(b => b.map(() => 0)), vb: gb.map(b => b.map(() => 0)) };
    this.t++;
    const S = this.state, upd = (P, G, M, V, isW) => {
      for (let l = 0; l < P.length; l++) for (let j = 0; j < P[l].length; j++) {
        const row = isW ? P[l][j] : null, n = isW ? row.length : 1;
        for (let i = 0; i < n; i++) {
          const get = () => (isW ? row[i] : P[l][j]), set = v => { if (isW) row[i] = v; else P[l][j] = v; };
          let g = isW ? G[l][j][i] : G[l][j];
          if (isW) g += o.l2 * get() + o.l1 * Math.sign(get());
          const mRow = isW ? M[l][j] : M[l], vRow = isW ? V[l][j] : V[l], ix = isW ? i : j;
          let d;
          if (o.name === "sgd") d = g;
          else if (o.name === "momentum") { mRow[ix] = o.beta * mRow[ix] + g; d = mRow[ix]; }
          else if (o.name === "rmsprop") { vRow[ix] = o.beta * vRow[ix] + (1 - o.beta) * g * g; d = g / (Math.sqrt(vRow[ix]) + o.eps); }
          else { mRow[ix] = o.beta * mRow[ix] + (1 - o.beta) * g; vRow[ix] = o.beta2 * vRow[ix] + (1 - o.beta2) * g * g; const mh = mRow[ix] / (1 - o.beta ** this.t), vh = vRow[ix] / (1 - o.beta2 ** this.t); d = mh / (Math.sqrt(vh) + o.eps); }
          set(get() - o.lr * d);
        }
      }
    };
    upd(this.W, gW, S.mW, S.vW, true); upd(this.b, gb, S.mb, S.vb, false);
    return loss;
  }
  // 한 epoch: data를 섞어 batch 크기로 나눠 걸음을 걷는다
  // One epoch: shuffle the data and take one step per batch
  epoch(X, Y, batch = 16) {
    const idx = shuffled(X.length, this.rng); let tot = 0, nb = 0;
    for (let s = 0; s < idx.length; s += batch) { const ii = idx.slice(s, s + batch); tot += this.step(ii.map(i => X[i]), ii.map(i => Y[i])); nb++; }
    return tot / nb;
  }
}

if (typeof module !== "undefined") module.exports = { TinyNet, makeData, ACT, softmaxV, matMul, matVec, transpose, shuffled };
