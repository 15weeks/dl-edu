/* 11장 그림과 조작. 한글판(ch11-cnn-KR.html)과 영어판(ch11-cnn.html)이 같이 쓴다
   Chapter 11 figures and interactions, shared by the Korean and English pages
   학습한 CNN weight와 결과는 assets/data/ch11_data.js (tools/ch11_numbers.py가 만든다)
   The trained CNN weights and results live in assets/data/ch11_data.js (made by tools/ch11_numbers.py) */
"use strict";

const D11 = CH11_DATA;
const b64bytes = s => Uint8Array.from(atob(s), ch => ch.charCodeAt(0));
const deqF = d => { const q = new Int8Array(b64bytes(d.q).buffer); return Float32Array.from(q, v => v * d.s); };
const NET = D11.cnn.layers.map(l => ({ W: deqF(l.W), shape: l.W.shape, b: l.b }));
const MN = b64bytes(D11.mnist.x), MY = D11.mnist.y, NM = MY.length, FMX = b64bytes(D11.fmnist.x);
const mnist = i => Array.from(MN.subarray(i * 784, (i + 1) * 784), v => v / 255);
const fmn = i => Array.from(FMX.subarray(i * 784, (i + 1) * 784), v => v / 255);
const FCLS = { ko: ["티셔츠", "바지", "풀오버", "원피스", "코트", "샌들", "셔츠", "운동화", "가방", "발목 부츠"], en: ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"] };
const fname = k => tr(FCLS.ko[k], FCLS.en[k]);

// convolution (stride 1): x = C장의 H×H 지도, 출력 = O장의 (H + 2pad − k + 1)² 지도
// Convolution (stride 1): x = C maps of H×H; output = O maps of (H + 2pad − k + 1)²
function conv2d(x, H, L, pad) {
  const [O, C, k] = L.shape, Ho = H + 2 * pad - k + 1, out = [];
  for (let o = 0; o < O; o++) {
    const m = new Float32Array(Ho * Ho).fill(L.b[o]);
    for (let ci = 0; ci < C; ci++) {
      const xm = x[ci];
      for (let u = 0; u < k; u++) for (let v = 0; v < k; v++) {
        const w = L.W[((o * C + ci) * k + u) * k + v]; if (!w) continue;
        for (let i = 0; i < Ho; i++) {
          const r = i + u - pad; if (r < 0 || r >= H) continue;
          for (let j = 0; j < Ho; j++) { const cc = j + v - pad; if (cc >= 0 && cc < H) m[i * Ho + j] += w * xm[r * H + cc]; }
        }
      }
    }
    out.push(m);
  }
  return out;
}
const reluM = ms => ms.map(m => m.map(v => (v > 0 ? v : 0)));
function pool2(m, H, mode = "max") {
  const h = H >> 1, o = new Float32Array(h * h);
  for (let i = 0; i < h; i++) for (let j = 0; j < h; j++) {
    const a = m[2 * i * H + 2 * j], b = m[2 * i * H + 2 * j + 1], c = m[(2 * i + 1) * H + 2 * j], d = m[(2 * i + 1) * H + 2 * j + 1];
    o[i * h + j] = mode === "mean" ? (a + b + c + d) / 4 : Math.max(a, b, c, d);
  }
  return o;
}
// 이 장의 MNIST CNN: conv 5×5 (8장) → pool → conv 5×5 (16장) → pool → Linear(784, 10)
// This chapter's MNIST CNN: conv 5×5 (8 maps) → pool → conv 5×5 (16 maps) → pool → Linear(784, 10)
function cnnForward(x) {
  const c1 = reluM(conv2d([x], 28, NET[0], 2)), p1 = c1.map(m => pool2(m, 28)), c2 = reluM(conv2d(p1, 14, NET[1], 2)), p2 = c2.map(m => pool2(m, 14));
  const flat = []; p2.forEach(m => flat.push(...m));
  const L = NET[2], logits = L.b.map((b, j) => { let s = b; for (let i = 0; i < 784; i++) s += L.W[j * 784 + i] * flat[i]; return s; });
  return { c1, p1, c2, p2, probs: softmaxV(logits) };
}
// 숫자가 적힌 칸 그림. o.div = 양수 파랑·음수 빨강, o.hl(i, j) = 테두리 색
// Grid of numbered cells. o.div = positive blue, negative red; o.hl(i, j) = border colour
function gridNums(ctx, c, M, n, m, x, y, cell, o = {}) {
  const bg = rgb(c.bg), mx = o.max || Math.max(1e-9, ...M.map(v => Math.abs(v === null ? 0 : v)));
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) {
    const v = M[i * m + j], X = x + j * cell, Y = y + i * cell;
    if (v !== null) { ctx.fillStyle = o.div ? cssRGB(mixRGB(bg, rgb(v >= 0 ? c.sig : c.fa), Math.min(1, Math.abs(v) / mx) * 0.75)) : cssRGB(mixRGB(bg, rgb(c.ink), clamp(v / mx, 0, 1) * 0.3)); ctx.fillRect(X, Y, cell, cell); }
    ctx.strokeStyle = c.line; ctx.lineWidth = 1; ctx.strokeRect(X + 0.5, Y + 0.5, cell - 1, cell - 1);
    if (v !== null && cell >= 18) txt(ctx, Math.abs(v - Math.round(v)) < 1e-9 ? String(Math.round(v)).replace("-", "−") : fx(v, 2), X + cell / 2, Y + cell / 2 + 4, { color: c.ink, size: Math.min(13, cell * 0.42), mono: true, align: "center" });
  }
  if (o.hl) for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) { const h = o.hl(i, j); if (h) { ctx.strokeStyle = h; ctx.lineWidth = 2.4; ctx.strokeRect(x + j * cell + 1.2, y + i * cell + 1.2, cell - 2.4, cell - 2.4); } }
}

/* ================= 1절 / Section 1 ================= */
// 7×7 그림 위를 3×3 kernel이 한 칸씩 미끄러진다. 칸마다 곱해서 더한 값이 출력 한 칸
// A 3×3 kernel slides one step at a time over a 7×7 picture. Multiply cell by cell and add: one output cell
const X1 = Array.from({ length: 49 }, (_, k) => { const r = Math.floor(k / 7), q = k % 7; return r >= 2 && r <= 4 && q >= 2 && q <= 4 ? 1 : 0; });
const KER = { v: [-1, 0, 1, -1, 0, 1, -1, 0, 1], h: [-1, -1, -1, 0, 0, 0, 1, 1, 1], blur: Array(9).fill(1 / 9) };
const conv3 = (X, N, K, i, j) => { let s = 0; for (let u = 0; u < 3; u++) for (let v = 0; v < 3; v++) s += X[(i + u) * N + j + v] * K[u * 3 + v]; return s; };
const S1 = { k: "v", p: 10, timer: 0 };
const F1 = fig("cv1", w => (w >= 640 ? 300 : 520), (ctx, w, h, c) => {
  const wide = w >= 640, cell = wide ? Math.min(34, (w - 150) / 15) : Math.min(30, (w - 60) / 10), K = KER[S1.k], pi = Math.floor(S1.p / 5), pj = S1.p % 5;
  const ix = 16, iy = 34, kx = wide ? ix + 7 * cell + 46 : ix + 7 * cell + 30, ky = iy + 2 * cell, ox = wide ? kx + 3 * cell + 46 : ix, oy = wide ? iy + cell : iy + 7 * cell + 50;
  panelTitle(ctx, c, tr("입력 7×7", "input 7×7"), ix, iy - 12);
  gridNums(ctx, c, X1, 7, 7, ix, iy, cell, { max: 1, hl: (i, j) => (i >= pi && i < pi + 3 && j >= pj && j < pj + 3 ? c.miss : null) });
  ctx.strokeStyle = c.miss; ctx.lineWidth = 3; ctx.strokeRect(ix + pj * cell, iy + pi * cell, 3 * cell, 3 * cell);
  panelTitle(ctx, c, "kernel 3×3", kx, ky - 12);
  txt(ctx, "⊙", kx - (wide ? 24 : 16), ky + 1.5 * cell + 6, { color: c.muted, size: 18, align: "center" });
  gridNums(ctx, c, K, 3, 3, kx, ky, cell, { div: true, max: 1 });
  const O = Array.from({ length: 25 }, (_, k) => (k <= S1.p ? conv3(X1, 7, K, Math.floor(k / 5), k % 5) : null));
  panelTitle(ctx, c, tr("출력 (feature map) 5×5", "output (feature map) 5×5"), ox, oy - 12);
  if (wide) txt(ctx, "=", ox - 24, oy + 2.5 * cell + 6, { color: c.muted, size: 18, align: "center" });
  gridNums(ctx, c, O, 5, 5, ox, oy, cell, { div: true, max: 3, hl: (i, j) => (i === pi && j === pj ? c.miss : null) });
});
function upd1() {
  paint(F1);
  const K = KER[S1.k], pi = Math.floor(S1.p / 5), pj = S1.p % 5, terms = [];
  for (let u = 0; u < 3; u++) for (let v = 0; v < 3; v++) { const x = X1[(pi + u) * 7 + pj + v]; if (x && K[u * 3 + v]) terms.push(`${x}·(${S1.k === "blur" ? "1/9" : String(K[u * 3 + v]).replace("-", "−")})`); }
  const y = conv3(X1, 7, K, pi, pj);
  $("r1").textContent = tr(`출력 (${pi}, ${pj}) = 창 안 9칸의 (입력 × kernel) 합 = ${terms.length ? terms.join(" + ") : "0"} = ${fx(y, S1.k === "blur" ? 2 : 0)}  (0인 곱은 뺐다)`,
    `output (${pi}, ${pj}) = sum over the 9 cells in the window of (input × kernel) = ${terms.length ? terms.join(" + ") : "0"} = ${fx(y, S1.k === "blur" ? 2 : 0)}  (zero products left out)`);
}
bindSeg("g1k", v => { S1.k = v; upd1(); });
$("b1step").addEventListener("click", () => { stopRun(S1); S1.p = (S1.p + 1) % 25; upd1(); });
$("b1play").addEventListener("click", () => { S1.p = 0; upd1(); runSteps(S1, 24, () => { S1.p++; upd1(); }, 380); });

/* ================= 2절 / Section 2 ================= */
// 진짜 숫자에 kernel 적용: kernel 칸을 누르면 숫자가 −1 → 0 → 1 → 2로 바뀐다
// A kernel on a real digit: click a kernel cell to cycle its number −1 → 0 → 1 → 2
const PRE2 = { v: KER.v, h: KER.h, blur: KER.blur, sharp: [0, -1, 0, -1, 5, -1, 0, -1, 0], id: [0, 0, 0, 0, 1, 0, 0, 0, 0] };
const S2 = { i: 6, k: [...KER.v], lay: null };
const conv28 = (x, K) => conv2d([x], 28, { W: Float32Array.from(K), shape: [1, 1, 3], b: [0] }, 1)[0];
const F2 = fig("cv2", w => (w >= 640 ? 230 : 470), (ctx, w, h, c) => {
  const wide = w >= 640, cell = wide ? Math.min(6, (w - 260) / 56) : Math.min(5.5, (w - 50) / 56), kc = 30, x = mnist(S2.i), y = conv28(x, S2.k);
  const ix = 16, iy = 34, kx = wide ? ix + 28 * cell + 40 : ix + 28 * cell + 30, ky = iy + 14 * cell - 1.5 * kc, ox = wide ? kx + 3 * kc + 40 : ix, oy = wide ? iy : iy + 28 * cell + 50;
  panelTitle(ctx, c, tr(`입력: 숫자 ${MY[S2.i]}`, `input: digit ${MY[S2.i]}`), ix, iy - 12); drawImg(ctx, c, x, 28, 28, ix, iy, cell);
  panelTitle(ctx, c, tr("kernel (칸을 누른다)", "kernel (click a cell)"), kx, ky - 12);
  gridNums(ctx, c, S2.k, 3, 3, kx, ky, kc, { div: true, max: 2 });
  S2.lay = { kx, ky, kc };
  panelTitle(ctx, c, tr("출력 (파랑 +, 빨강 −)", "output (blue +, red −)"), ox, oy - 12); drawImg(ctx, c, y, 28, 28, ox, oy, cell, { div: true });
});
function upd2() {
  paint(F2); paint(F2b);
  const y = conv28(mnist(S2.i), S2.k), s = S2.k.reduce((a, b) => a + b, 0);
  $("r2").textContent = tr(`kernel 9칸의 합 = ${fx(s, 2)}, 출력 범위 ${fx(Math.min(...y), 2)} ~ ${fx(Math.max(...y), 2)}\n합이 0이면 고른 곳은 0이 되고 바뀌는 곳(경계)만 남는다. 합이 1이면 밝기는 그대로 두고 모양만 바꾼다.`,
    `sum of the 9 kernel cells = ${fx(s, 2)}, output range ${fx(Math.min(...y), 2)} to ${fx(Math.max(...y), 2)}\nA sum of 0 turns flat areas into 0 and keeps only changes (edges). A sum of 1 keeps the brightness and only reshapes.`);
}
F2.cv.addEventListener("click", ev => {
  const L = S2.lay; if (!L) return; const r = F2.cv.getBoundingClientRect(), j = Math.floor((ev.clientX - r.left - L.kx) / L.kc), i = Math.floor((ev.clientY - r.top - L.ky) / L.kc);
  if (i < 0 || i > 2 || j < 0 || j > 2) return;
  const v = S2.k[i * 3 + j], nxt = [-1, 0, 1, 2].find(t => t > v + 1e-9); S2.k[i * 3 + j] = nxt === undefined ? -1 : nxt; upd2();
});
bindSeg("g2k", v => { S2.k = [...PRE2[v]]; upd2(); });
$("b2next").addEventListener("click", () => { S2.i = (S2.i + 3) % NM; upd2(); });
// 학습한 CNN 첫 층의 kernel 8개와 그 feature map
// The 8 kernels of the trained CNN's first layer and their feature maps
const F2b = fig("cv2b", w => (w >= 640 ? 200 : 380), (ctx, w, h, c) => {
  const per = w >= 640 ? 8 : 4, colW = (w - 24) / per, kc = Math.min(7, (colW - 10) / 5), mc = Math.min(3.6, (colW - 8) / 28), f = cnnForward(mnist(S2.i));
  panelTitle(ctx, c, tr("학습한 CNN 1층: kernel 5×5 (위)와 feature map 28×28 (아래, ReLU 뒤)", "Trained CNN, layer 1: 5×5 kernels (top) and 28×28 feature maps (below, after ReLU)"), 12, 16);
  for (let o = 0; o < 8; o++) {
    const col = o % per, row = Math.floor(o / per), x0 = 12 + col * colW, y0 = 30 + row * (5 * kc + 28 * mc + 30);
    drawImg(ctx, c, Array.from(NET[0].W.subarray(o * 25, o * 25 + 25)), 5, 5, x0 + (28 * mc - 5 * kc) / 2, y0, kc, { div: true });
    drawImg(ctx, c, f.c1[o], 28, 28, x0, y0 + 5 * kc + 8, mc, { max: Math.max(1e-6, ...f.c1[o]) });
  }
});

/* ================= 3절 / Section 3 ================= */
// 출력 크기 계산기: 입력 N, kernel K, stride S, padding P
// Output-size calculator: input N, kernel K, stride S, padding P
const S3 = { n: 7, k: 3, s: 2, p: 1 };
const outSize = (n, k, s, p) => Math.floor((n + 2 * p - k) / s) + 1;
// 높이는 지금 N, P, 출력 크기에 맞춘다 (upd3에서 resize)
// The height follows the current N, P and output size (resized in upd3)
const cell3 = (w, NP) => (w >= 640 ? Math.min(26, (w * 0.5) / NP) : Math.min(24, (w - 40) / NP));
const h3 = w => { const NP = S3.n + 2 * S3.p, O = Math.max(1, outSize(S3.n, S3.k, S3.s, S3.p)), cl = cell3(w, NP); return Math.round(w >= 640 ? 34 + NP * cl + 24 : 34 + NP * cl + 46 + O * Math.min(cl, (w - 40) / O) + 20); };
const F3 = fig("cv3", h3, (ctx, w, h, c) => {
  const { n, k, s, p } = S3, NP = n + 2 * p, O = outSize(n, k, s, p), wide = w >= 640, cell = cell3(w, NP), x0 = 16, y0 = 34;
  panelTitle(ctx, c, tr(`입력 ${n}×${n} + padding ${p} (점선 칸 = 0)`, `input ${n}×${n} + padding ${p} (dashed cells = 0)`), x0, y0 - 12);
  for (let i = 0; i < NP; i++) for (let j = 0; j < NP; j++) {
    const pad = i < p || j < p || i >= n + p || j >= n + p, X = x0 + j * cell, Y = y0 + i * cell;
    ctx.fillStyle = pad ? c.bg : c.sigwash; ctx.fillRect(X, Y, cell, cell);
    ctx.save(); ctx.strokeStyle = c.line; ctx.lineWidth = 1; if (pad) ctx.setLineDash([3, 3]); ctx.strokeRect(X + 0.5, Y + 0.5, cell - 1, cell - 1); ctx.restore();
    if (pad && cell >= 16) txt(ctx, "0", X + cell / 2, Y + cell / 2 + 4, { color: c.muted, size: 10, mono: true, align: "center" });
  }
  if (O >= 1) {
    ctx.save(); ctx.lineWidth = 3; ctx.strokeStyle = c.miss; ctx.strokeRect(x0 + 1.5, y0 + 1.5, k * cell - 3, k * cell - 3);
    if (O >= 2) { ctx.strokeStyle = c.ok; ctx.setLineDash([6, 4]); ctx.strokeRect(x0 + s * cell + 1.5, y0 + 1.5, k * cell - 3, k * cell - 3); arrowS(ctx, x0 + k * cell / 2, y0 - 3, x0 + (s + k / 2) * cell, y0 - 3, c.ok, 1.6, 6); }
    ctx.restore();
  }
  const ox = wide ? x0 + NP * cell + 60 : x0, oy = wide ? y0 : y0 + NP * cell + 46, oc = Math.min(cell, wide ? (w - ox - 20) / Math.max(1, O) : (w - 40) / Math.max(1, O));
  if (O >= 1) {
    panelTitle(ctx, c, tr(`출력 ${O}×${O}`, `output ${O}×${O}`), ox, oy - 12);
    for (let i = 0; i < O; i++) for (let j = 0; j < O; j++) {
      ctx.fillStyle = i === 0 && j === 0 ? c.wash : c.bg; ctx.fillRect(ox + j * oc, oy + i * oc, oc, oc);
      ctx.strokeStyle = i === 0 && j === 0 ? c.miss : i === 0 && j === 1 ? c.ok : c.line; ctx.lineWidth = i === 0 && j < 2 ? 2.4 : 1; ctx.strokeRect(ox + j * oc + 1, oy + i * oc + 1, oc - 2, oc - 2);
    }
  } else txt(ctx, tr("kernel이 입력보다 크다: 출력 없음", "kernel bigger than the input: no output"), ox, oy + 20, { color: c.fa, size: 12.5, w: "700" });
});
function upd3() {
  resize(F3); paint(F3);
  const { n, k, s, p } = S3, O = outSize(n, k, s, p), rest = (n + 2 * p - k) % s;
  $("r3").textContent = tr(`출력 = ⌊(N + 2P − K) / S⌋ + 1 = ⌊(${n} + ${2 * p} − ${k}) / ${s}⌋ + 1 = ${O}` + (O >= 1 && rest ? `\n나누어떨어지지 않는다: 오른쪽·아래 ${rest}칸은 kernel이 닿지 않고 버려진다.` : "") + (O >= 1 && s === 1 && 2 * p === k - 1 ? "\nS = 1, P = (K − 1)/2이면 출력 = 입력 크기 (\"same\" padding)." : ""),
    `output = ⌊(N + 2P − K) / S⌋ + 1 = ⌊(${n} + ${2 * p} − ${k}) / ${s}⌋ + 1 = ${O}` + (O >= 1 && rest ? `\nNot an exact division: the last ${rest} column(s) and row(s) are never reached by the kernel and are dropped.` : "") + (O >= 1 && s === 1 && 2 * p === k - 1 ? "\nWith S = 1 and P = (K − 1)/2 the output keeps the input size (\"same\" padding)." : ""));
}
bindRange("i3n", "o3n", v => String(v), v => { S3.n = v; upd3(); });
bindRange("i3k", "o3k", v => String(v), v => { S3.k = v; upd3(); });
bindRange("i3s", "o3s", v => String(v), v => { S3.s = v; upd3(); });
bindRange("i3p", "o3p", v => String(v), v => { S3.p = v; upd3(); });

/* ================= 4절 / Section 4 ================= */
// transpose convolution: 입력 한 칸이 (값 × kernel)을 출력에 찍는다. 겹치는 곳은 더한다
// Transpose convolution: each input cell stamps (value × kernel) onto the output. Overlaps add up
const X4 = [1, 2, 3, 4], S4 = { s: 2, n: 1 };
function tconv(n, s) {
  const O = s + 3, out = new Array(O * O).fill(0);
  for (let q = 0; q < n; q++) { const i = Math.floor(q / 2), j = q % 2; for (let u = 0; u < 3; u++) for (let v = 0; v < 3; v++) out[(i * s + u) * O + j * s + v] += X4[q]; }
  return out;
}
const F4 = fig("cv4", w => (w >= 600 ? 260 : 440), (ctx, w, h, c) => {
  const wide = w >= 600, cell = wide ? Math.min(36, (w - 200) / 12) : Math.min(34, (w - 60) / 8), O = S4.s + 3, q = S4.n - 1, qi = Math.floor(q / 2), qj = q % 2;
  const ix = 16, iy = 40, kx = ix + 2 * cell + 40, ky = iy, ox = wide ? kx + 3 * cell + 50 : ix, oy = wide ? iy : iy + 3 * cell + 50;
  panelTitle(ctx, c, tr("입력 2×2", "input 2×2"), ix, iy - 12);
  gridNums(ctx, c, X4, 2, 2, ix, iy, cell, { max: 4, hl: (i, j) => (S4.n && i === qi && j === qj ? c.miss : null) });
  panelTitle(ctx, c, "kernel 3×3", kx, ky - 12); gridNums(ctx, c, Array(9).fill(1), 3, 3, kx, ky, cell, { div: true, max: 2 });
  panelTitle(ctx, c, tr(`출력 ${O}×${O} (stride ${S4.s})`, `output ${O}×${O} (stride ${S4.s})`), ox, oy - 12);
  gridNums(ctx, c, tconv(S4.n, S4.s), O, O, ox, oy, cell, { max: 10, hl: (i, j) => (S4.n && i >= qi * S4.s && i < qi * S4.s + 3 && j >= qj * S4.s && j < qj * S4.s + 3 ? c.miss : null) });
});
function upd4() {
  paint(F4);
  const q = S4.n - 1, qi = Math.floor(q / 2), qj = q % 2, O = S4.s + 3;
  $("r4").textContent = S4.n ? tr(`입력 (${qi}, ${qj}) = ${X4[q]} → 출력의 (${qi * S4.s}~${qi * S4.s + 2}, ${qj * S4.s}~${qj * S4.s + 2}) 칸에 ${X4[q]} × kernel을 더한다 (${S4.n}/4)\n출력 크기 = (N − 1)·S + K = (2 − 1)·${S4.s} + 3 = ${O}: 2×2가 ${O}×${O}로 커진다.`,
    `input (${qi}, ${qj}) = ${X4[q]} → add ${X4[q]} × kernel to output cells (${qi * S4.s}–${qi * S4.s + 2}, ${qj * S4.s}–${qj * S4.s + 2}) (${S4.n}/4)\noutput size = (N − 1)·S + K = (2 − 1)·${S4.s} + 3 = ${O}: 2×2 grows to ${O}×${O}.`) : tr("아직 아무것도 찍지 않았다.", "Nothing stamped yet.");
}
bindSeg("g4s", v => { S4.s = +v; upd4(); });
$("b4step").addEventListener("click", () => { S4.n = S4.n >= 4 ? 1 : S4.n + 1; upd4(); });
$("b4all").addEventListener("click", () => { S4.n = 4; upd4(); });

/* ================= 5절 / Section 5 ================= */
// pooling: 1층 feature map 28×28 → 14×14 → 7×7, max와 mean을 나란히
// Pooling: a layer-1 feature map 28×28 → 14×14 → 7×7, max and mean side by side
const S5 = { i: 10, ch: 2 };
const F5 = fig("cv5", w => (w >= 600 ? 380 : 340), (ctx, w, h, c) => {
  const wide = w >= 600, side = wide ? Math.min(130, (w - 140) / 4) : Math.min(100, (w - 70) / 3), m28 = cnnForward(mnist(S5.i)).c1[S5.ch];
  const rows = [["max", c.sig], ["mean", c.ok]];
  rows.forEach(([mode, col], r) => {
    const m14 = pool2(m28, 28, mode), m7 = pool2(m14, 14, mode), y0 = wide ? 48 + r * (side + 50) : 48 + r * (side + 60), xs = wide ? [80, 80 + side + 40, 80 + 2 * (side + 40)] : [16, 16 + side + 20, 16 + 2 * (side + 20)];
    txt(ctx, `${mode} pool`, wide ? 70 : 16, wide ? y0 + side / 2 + 4 : y0 - 22, { color: col, size: 12.5, w: "700", align: wide ? "right" : "left" });
    [[m28, 28], [m14, 14], [m7, 7]].forEach(([m, n], k) => {
      if (r === 0 || !wide) txt(ctx, `${n}×${n}`, xs[k] + side / 2, y0 - 8, { color: c.muted, size: 11, mono: true, align: "center" });
      if (k === 0 && r === 1) { txt(ctx, tr("(같은 그림)", "(same picture)"), xs[0] + side / 2, y0 + side / 2, { color: c.muted, size: 11, align: "center" }); ctx.strokeStyle = c.line; ctx.strokeRect(xs[0], y0, side, side); return; }
      drawImg(ctx, c, m, n, n, xs[k], y0, side / n, { max: Math.max(1e-6, ...m28) });
      if (k) arrowS(ctx, xs[k] - (wide ? 34 : 18), y0 + side / 2, xs[k] - 6, y0 + side / 2, c.muted, 1.4, 6);
    });
  });
});
function upd5() { paint(F5); $("r5").textContent = tr(`숫자 ${MY[S5.i]}, 1층 feature map ${S5.ch + 1}번 (ReLU 뒤). 2×2 칸마다 하나로: max는 가장 센 반응을 남겨 획이 굵고 또렷하다, mean은 평균이라 흐려진다.\n크기는 한 번에 가로·세로 반, 칸 수는 1/4. 학습할 parameter는 0개.`, `Digit ${MY[S5.i]}, layer-1 feature map no. ${S5.ch + 1} (after ReLU). One value per 2×2 block: max keeps the strongest response, so strokes stay thick and sharp; mean averages, so they blur.\nEach step halves width and height (a quarter of the cells). No parameters to learn.`); }
$("b5ch").addEventListener("click", () => { S5.ch = (S5.ch + 1) % 8; upd5(); });
$("b5next").addEventListener("click", () => { S5.i = (S5.i + 3) % NM; upd5(); });

/* ================= 6절 / Section 6 ================= */
// CNN shape 따라가기: 층마다 (C, H, W)와 parameter 수
// Following the shapes through a CNN: (C, H, W) and parameter count at every layer
const NETS6 = {
  mnist: { inp: [1, 28], ops: [["conv", 8, 5, 2], ["pool"], ["conv", 16, 5, 2], ["pool"], ["flat"], ["fc", 10]] },
  fmnist: { inp: [1, 28], ops: [["conv", 16, 3, 1], ["pool"], ["conv", 32, 3, 1], ["pool"], ["flat"], ["fc", 64], ["fc", 10]] },
  cifar: { inp: [3, 32], ops: [["conv", 16, 3, 1], ["pool"], ["conv", 32, 3, 1], ["pool"], ["flat"], ["fc", 64], ["fc", 10]] }
};
const S6 = { n: "mnist" };
function shapes6(N) {
  let C = N.inp[0], H = N.inp[1], F = 0; const st = [{ lab: tr("입력", "input"), C, H, P: 0 }];
  N.ops.forEach(op => {
    if (op[0] === "conv") { const P = op[1] * C * op[2] * op[2] + op[1]; H = H + 2 * op[3] - op[2] + 1; C = op[1]; st.push({ lab: `conv ${op[2]}×${op[2]}`, C, H, P }); }
    else if (op[0] === "pool") { H = H >> 1; st.push({ lab: "max pool 2", C, H, P: 0 }); }
    else if (op[0] === "flat") { F = C * H * H; st.push({ lab: "flatten", F, P: 0 }); }
    else { const P = F * op[1] + op[1]; F = op[1]; st.push({ lab: `Linear → ${op[1]}`, F, P }); }
  });
  return st;
}
const F6 = fig("cv6", w => (w >= 700 ? 250 : 470), (ctx, w, h, c) => {
  const st = shapes6(NETS6[S6.n]), per = w >= 700 ? st.length : Math.ceil(st.length / 2), colW = (w - 24) / per, base = w >= 700 ? 160 : 150;
  st.forEach((s, k) => {
    const col = k % per, row = Math.floor(k / per), cx = 12 + col * colW + colW / 2, yb = 30 + row * 220 + base - 40;
    if (s.F === undefined) {
      const side = 8 + s.H * 2.1, sl = Math.min(8, Math.ceil(s.C / 2)), off = 3;
      for (let q = sl - 1; q >= 0; q--) { const x = cx - side / 2 + q * off - (sl - 1) * off / 2, y = yb - side - q * off; ctx.fillStyle = q ? c.bg : (k ? c.sigwash : c.okwash); ctx.fillRect(x, y, side, side); ctx.strokeStyle = k ? c.sig : c.ok; ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, side, side); }
      txt(ctx, `(${s.C}, ${s.H}, ${s.H})`, cx, yb + 18, { color: c.ink, size: 11.5, mono: true, align: "center", w: "600" });
    } else {
      const hh = Math.min(120, 8 + Math.sqrt(s.F) * 2.6);
      ctx.fillStyle = c.wash; ctx.fillRect(cx - 7, yb - hh, 14, hh); ctx.strokeStyle = c.miss; ctx.strokeRect(cx - 6.5, yb - hh + 0.5, 13, hh - 1);
      txt(ctx, `(${s.F})`, cx, yb + 18, { color: c.ink, size: 11.5, mono: true, align: "center", w: "600" });
    }
    txt(ctx, s.lab, cx, yb + 34, { color: c.muted, size: 11, align: "center" });
    if (s.P) txt(ctx, `${commas(s.P)} p`, cx, yb + 50, { color: c.vio, size: 11, mono: true, align: "center" });
  });
});
function upd6() {
  paint(F6);
  const st = shapes6(NETS6[S6.n]), tot = st.reduce((a, s) => a + s.P, 0), fl = st.find(s => s.lab === "flatten");
  $("r6").textContent = tr(`parameter 합계 ${commas(tot)}개 (보라 숫자 = 그 층의 parameter). flatten 길이 ${fl.F} = C × H × W.\nconv는 크기를 지키고(padding) pool이 반씩 줄인다. channel은 늘고 칸은 줄어든다.`, `${commas(tot)} parameters in total (violet = parameters of that layer). Flatten length ${fl.F} = C × H × W.\nThe convs keep the size (padding) and the pools halve it. Channels grow while the grid shrinks.`);
}
bindSeg("g6n", v => { S6.n = v; upd6(); });

/* ================= 7절 / Section 7 ================= */
// 숫자 그리기 → CNN 확률과 2층 feature map 16장
// Draw a digit → CNN probabilities and the 16 layer-2 feature maps
const S7 = { a: mnist(14), lay: null, prev: null, si: 14 };
function prep(a) {
  let x0 = 28, x1 = -1, y0 = 28, y1 = -1, tot = 0;
  for (let r = 0; r < 28; r++) for (let k = 0; k < 28; k++) if (a[r * 28 + k] > 0.05) { x0 = Math.min(x0, k); x1 = Math.max(x1, k); y0 = Math.min(y0, r); y1 = Math.max(y1, r); }
  if (x1 < 0) return new Array(784).fill(0);
  // 8장과 같은 MNIST식 맞춤: 글자 상자를 20칸으로, 무게 중심을 (14, 14)로
  // The same MNIST-style fitting as chapter 8: ink box to 20 cells, centre of mass to (14, 14)
  const bw = x1 - x0 + 1, bh = y1 - y0 + 1, s = 20 / Math.max(bw, bh), out = new Array(784).fill(0), nw = Math.max(1, Math.round(bw * s)), nh = Math.max(1, Math.round(bh * s)), tmp = [];
  for (let r = 0; r < nh; r++) for (let k = 0; k < nw; k++) {
    const sx = x0 + (k + 0.5) / s - 0.5, sy = y0 + (r + 0.5) / s - 0.5, ix = Math.floor(sx), iy = Math.floor(sy), fx_ = sx - ix, fy = sy - iy;
    const g = (yy, xx) => (yy >= 0 && yy < 28 && xx >= 0 && xx < 28 ? a[yy * 28 + xx] : 0);
    tmp.push(g(iy, ix) * (1 - fx_) * (1 - fy) + g(iy, ix + 1) * fx_ * (1 - fy) + g(iy + 1, ix) * (1 - fx_) * fy + g(iy + 1, ix + 1) * fx_ * fy);
  }
  let cx = 0, cy = 0; tmp.forEach((v, i) => { cx += v * (i % nw); cy += v * Math.floor(i / nw); tot += v; });
  cx /= tot || 1; cy /= tot || 1;
  const ox = Math.round(14 - cx - 0.5), oy = Math.round(14 - cy - 0.5);
  tmp.forEach((v, i) => { const xx = (i % nw) + ox, yy = Math.floor(i / nw) + oy; if (xx >= 0 && xx < 28 && yy >= 0 && yy < 28) out[yy * 28 + xx] = Math.min(1, v); });
  return out;
}
function probBars(ctx, c, p, B) {
  const bw = B.w / 10, k = p.indexOf(Math.max(...p));
  for (let d = 0; d < 10; d++) {
    const hgt = p[d] * (B.h - 30), x = B.l + d * bw;
    ctx.fillStyle = d === k ? c.miss : c.sig; ctx.globalAlpha = d === k ? 0.95 : 0.55; ctx.fillRect(x + bw * 0.15, B.t + B.h - 16 - hgt, bw * 0.7, hgt); ctx.globalAlpha = 1;
    txt(ctx, String(d), x + bw / 2, B.t + B.h - 2, { color: c.ink, size: 12, w: d === k ? "700" : "400", align: "center", mono: true });
    if (p[d] > 0.05) txt(ctx, Math.round(p[d] * 100) + "%", x + bw / 2, B.t + B.h - 20 - hgt, { color: c.ink, size: 10, align: "center", mono: true });
  }
}
const F7 = fig("cv7", w => (w >= 700 ? 330 : 720), (ctx, w, h, c) => {
  const wide = w >= 700, cell = wide ? Math.min(9, (w * 0.3) / 28) : Math.min(10, (w - 40) / 28), G = { l: 16, t: 30, s: cell * 28 }, f = cnnForward(prep(S7.a));
  S7.lay = { G, cell };
  panelTitle(ctx, c, tr("여기에 숫자를 그린다 (끌기)", "Draw a digit here (drag)"), G.l, 18);
  drawImg(ctx, c, S7.a, 28, 28, G.l, G.t, cell);
  const mx = wide ? G.l + G.s + 30 : 16, my = wide ? G.t : G.t + G.s + 40, per = wide ? 4 : 8, mc = wide ? Math.min(4, (w - mx - 250 - per * 6) / per / 14) : Math.min(3.6, (w - 32 - per * 6) / per / 14);
  panelTitle(ctx, c, tr("2층 feature map 16장 (14×14)", "16 layer-2 feature maps (14×14)"), mx, my - 12);
  f.c2.forEach((m, o) => drawImg(ctx, c, m, 14, 14, mx + (o % per) * (14 * mc + 6), my + Math.floor(o / per) * (14 * mc + 6), mc, { max: Math.max(1e-6, ...m) }));
  const B = wide ? { l: mx + per * (14 * mc + 6) + 24, t: G.t, w: w - (mx + per * (14 * mc + 6) + 24) - 12, h: G.s } : { l: 16, t: my + (16 / per) * (14 * mc + 6) + 30, w: w - 32, h: 190 };
  panelTitle(ctx, c, tr("CNN의 softmax 확률", "CNN softmax probabilities"), B.l, B.t - 12);
  probBars(ctx, c, f.probs, B);
});
function upd7() {
  paint(F7);
  const p = cnnForward(prep(S7.a)).probs, k = p.indexOf(Math.max(...p));
  $("r7").textContent = tr(`답: ${k} (${Math.round(p[k] * 100)}%). CNN parameter ${commas(D11.cnn.params)}개, MNIST 시험 정확도 ${(D11.cnn.acc * 100).toFixed(1)}% (page의 int8 weight로)\n8장의 FFN 196-64-10은 parameter 13,258개로 95.8%.`, `Answer: ${k} (${Math.round(p[k] * 100)}%). CNN with ${commas(D11.cnn.params)} parameters, MNIST test accuracy ${(D11.cnn.acc * 100).toFixed(1)}% (with the page's int8 weights)\nChapter 8's FFN 196-64-10: 13,258 parameters, 95.8%.`);
}
function paint7(px, py) {
  const { G, cell } = S7.lay, cx = (px - G.l) / cell, cy = (py - G.t) / cell;
  for (let r = Math.floor(cy - 2); r <= Math.ceil(cy + 2); r++) for (let k = Math.floor(cx - 2); k <= Math.ceil(cx + 2); k++) {
    if (r < 0 || r >= 28 || k < 0 || k >= 28) continue;
    const d = Math.hypot(k + 0.5 - cx, r + 0.5 - cy); if (d < 1.7) S7.a[r * 28 + k] = Math.min(1, S7.a[r * 28 + k] + (d < 0.9 ? 1 : 0.45));
  }
}
dragOn(F7, (px, py) => { const L = S7.lay; if (!L) return null; return px >= L.G.l && px <= L.G.l + L.G.s && py >= L.G.t && py <= L.G.t + L.G.s ? 1 : null; }, (k, px, py) => {
  if (S7.prev) { const [qx, qy] = S7.prev, n = Math.ceil(Math.hypot(px - qx, py - qy) / 3); for (let i = 1; i <= n; i++) paint7(qx + (px - qx) * i / n, qy + (py - qy) * i / n); } else paint7(px, py);
  S7.prev = [px, py]; upd7();
});
F7.cv.addEventListener("pointerup", () => { S7.prev = null; });
$("b7clear").addEventListener("click", () => { S7.a = new Array(784).fill(0); upd7(); });
$("b7sample").addEventListener("click", () => { S7.si = (S7.si + 3) % NM; S7.a = mnist(S7.si); upd7(); });
// 옮긴 그림에서의 정확도: CNN vs FFN (PyTorch, 시험 10,000장)
// Accuracy on shifted pictures: CNN vs FFN (PyTorch, 10,000 test images)
const SH = D11.shift;
const F7b = fig("cv7b", 260, (ctx, w, h, c) => {
  const B = { l: 52, t: 30, w: w - 70, h: h - 74 }, A = axes(ctx, c, B, [0, 6], [0, 1], { xt: SH.s, yt: [[0, "0%"], [0.25, "25%"], [0.5, "50%"], [0.75, "75%"], [1, "100%"]], xlabel: tr("왼쪽·오른쪽으로 옮긴 칸 수", "pixels shifted left or right") });
  panelTitle(ctx, c, tr("옮긴 숫자의 시험 정확도", "Test accuracy on shifted digits"), B.l, 16);
  [[SH.cnn, c.ok, `CNN (${commas(SH.params[0])} p)`], [SH.ffn, c.fa, `FFN 784-64-10 (${commas(SH.params[1])} p)`]].forEach(([a, col, lab]) => {
    line(ctx, a.map((v, i) => [A.X(SH.s[i]), A.Y(v)]), col, 2.2); a.forEach((v, i) => dot(ctx, A.X(SH.s[i]), A.Y(v), 3, col));
    txtBg(ctx, c, lab, A.X(2) + 8, A.Y(a[2]) + (col === c.ok ? -10 : 18), { color: col, size: 11.5, w: "700" });
  });
});
// Gaussian 폭 분류 data 보기 (Python과 같은 방법으로 browser에서 만든다)
// A look at the Gaussian-width data (made in the browser the same way as in Python)
const S7c = { seed: 4 };
const F7c = fig("cv7c", w => (w >= 600 ? 150 : 290), (ctx, w, h, c) => {
  const r = makeRng(S7c.seed), n = 8, per = w >= 600 ? 8 : 4, cell = Math.min(3, (w - 24 - per * 12) / per / 32);
  for (let k = 0; k < n; k++) {
    const wide = k % 2 === 1, sig = wide ? 3 + r.u() : 1.5 + r.u(), cx = 8 + 16 * r.u(), cy = 8 + 16 * r.u(), img = [];
    for (let i = 0; i < 32; i++) for (let j = 0; j < 32; j++) img.push(Math.exp(-((j - cx) ** 2 + (i - cy) ** 2) / (2 * sig * sig)) + 0.3 * r.n());
    const x0 = 12 + (k % per) * (32 * cell + 12), y0 = 26 + Math.floor(k / per) * (32 * cell + 40);
    drawImg(ctx, c, img, 32, 32, x0, y0, cell, { max: 1 });
    txt(ctx, `${wide ? tr("넓음", "wide") : tr("좁음", "narrow")} σ ${fx(sig, 1)}`, x0 + 16 * cell, y0 + 32 * cell + 15, { color: wide ? c.miss : c.sig, size: 11, align: "center", w: "600" });
  }
  panelTitle(ctx, c, tr("32×32, 잡음 표준편차 0.3: 좁은 Gaussian인가 넓은 Gaussian인가?", "32×32, noise SD 0.3: a narrow Gaussian or a wide one?"), 12, 14);
});
$("b7g").addEventListener("click", () => { S7c.seed++; paint(F7c); });

/* ================= 8절 / Section 8 ================= */
// image transform 모음: 같은 옷 그림에 transform을 하나씩
// A gallery of image transforms: one transform at a time on the same clothing picture
const S8 = { i: 0, seed: 1 };
function warp(a, ang, sc, dx, dy, flip) {
  const o = new Array(784).fill(0), cs = Math.cos(ang), sn = Math.sin(ang);
  for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
    let u = ((x - 13.5 - dx) * cs + (y - 13.5 - dy) * sn) / sc + 13.5; const v = (-(x - 13.5 - dx) * sn + (y - 13.5 - dy) * cs) / sc + 13.5;
    if (flip) u = 27 - u;
    const iu = Math.floor(u), iv = Math.floor(v), fu = u - iu, fv = v - iv, g = (yy, xx) => (yy >= 0 && yy < 28 && xx >= 0 && xx < 28 ? a[yy * 28 + xx] : 0);
    o[y * 28 + x] = g(iv, iu) * (1 - fu) * (1 - fv) + g(iv, iu + 1) * fu * (1 - fv) + g(iv + 1, iu) * (1 - fu) * fv + g(iv + 1, iu + 1) * fu * fv;
  }
  return o;
}
const F8 = fig("cv8", w => (w >= 640 ? 170 : 330), (ctx, w, h, c) => {
  const a = fmn(S8.i), r = makeRng(S8.seed), per = w >= 640 ? 7 : 4, cell = Math.min(3.4, (w - 24 - per * 14) / per / 28);
  const ang = (r.u() - 0.5) * 40, dx = (r.u() - 0.5) * 5.6, dy = (r.u() - 0.5) * 5.6, sc = 0.8 + 0.4 * r.u(), ex = Math.floor(r.u() * 18), ey = Math.floor(r.u() * 18), ew = 6 + Math.floor(r.u() * 5);
  const items = [
    [tr("원래", "original"), a, null],
    [`Rotation(${Math.round(ang)}°)`, warp(a, ang * Math.PI / 180, 1, 0, 0, false), null],
    [`Affine(${fx(dx, 1)}, ${fx(dy, 1)})`, warp(a, 0, 1, dx, dy, false), null],
    [`scale ×${fx(sc, 2)}`, warp(a, 0, sc, 0, 0, false), null],
    ["HorizontalFlip", warp(a, 0, 1, 0, 0, true), null],
    ["RandomErasing", a.map((v, k) => { const y = Math.floor(k / 28), x = k % 28; return x >= ex && x < ex + ew && y >= ey && y < ey + ew ? 0 : v; }), null],
    ["Normalize", a.map(v => (v - 0.5) / 0.5), { div: true, max: 1 }]
  ];
  items.forEach(([lab, img, o], k) => {
    const x0 = 12 + (k % per) * (28 * cell + 14), y0 = 18 + Math.floor(k / per) * (28 * cell + 46);
    drawImg(ctx, c, img, 28, 28, x0, y0, cell, o || {});
    txt(ctx, lab, x0 + 14 * cell, y0 + 28 * cell + 15, { color: k ? c.ink : c.miss, size: 10.5, mono: true, align: "center", w: k ? "400" : "700" });
  });
});
function upd8() { paint(F8); $("r8").textContent = tr(`${fname(S8.i)} 한 장. 학습 때마다 무작위로 바꿔 보여 주면 data가 늘어난 효과가 난다 (5장 data augmentation). Normalize는 모양은 그대로, 숫자 범위만 0~1에서 −1~1로.\n좌우 뒤집기는 옷에는 괜찮지만 숫자(MNIST)에는 안 된다: 바꾼 뒤에도 정답이 같아야 한다.`, `One ${fname(S8.i)}. Showing a different random version every time it is used for training acts like more data (chapter 5, data augmentation). Normalize keeps the shape and only moves the range from 0–1 to −1–1.\nA horizontal flip is fine for clothes but not for digits (MNIST): the label must still be right after the change.`); }
$("b8roll").addEventListener("click", () => { S8.seed++; upd8(); });
$("b8next").addEventListener("click", () => { S8.i = (S8.i + 1) % 10; upd8(); });

/* ================= 9절 / Section 9 ================= */
// FMNIST CNN의 confusion matrix (시험 10,000장). 칸을 누르면 그 수가 나온다
// Confusion matrix of the FMNIST CNN (10,000 test images). Click a cell to read its count
const CM = D11.fmnist.cm, S9 = { sel: [6, 0], lay: null };
const F9 = fig("cv9", w => (w >= 600 ? 440 : 470), (ctx, w, h, c) => {
  const lw = w >= 600 ? 120 : 92, cell = Math.min(34, (w - lw - 30) / 10), x0 = lw, y0 = 64, bg = rgb(c.bg);
  S9.lay = { x0, y0, cell };
  txt(ctx, tr("예측 →", "predicted →"), x0 + 5 * cell, 14, { color: c.muted, size: 11.5, align: "center" });
  for (let j = 0; j < 10; j++) drawImg(ctx, c, fmn(j), 28, 28, x0 + j * cell + (cell - 28 * Math.min(1, cell / 30)) / 2, 22, Math.min(1, cell / 30), { frame: false });
  for (let i = 0; i < 10; i++) {
    const tot = CM[i].reduce((a, b) => a + b, 0);
    txt(ctx, fname(i), x0 - 8, y0 + i * cell + cell / 2 + 4, { color: c.ink, size: 11, align: "right" });
    for (let j = 0; j < 10; j++) {
      const v = CM[i][j], f = v / tot, X = x0 + j * cell, Y = y0 + i * cell;
      ctx.fillStyle = cssRGB(mixRGB(bg, rgb(i === j ? c.ok : c.fa), Math.min(1, i === j ? f : f * 5) * 0.85)); ctx.fillRect(X, Y, cell, cell);
      ctx.strokeStyle = c.line; ctx.lineWidth = 1; ctx.strokeRect(X + 0.5, Y + 0.5, cell - 1, cell - 1);
      if (v >= 10 && cell >= 26) txt(ctx, String(v), X + cell / 2, Y + cell / 2 + 4, { color: c.ink, size: 10, mono: true, align: "center" });
    }
  }
  const [si, sj] = S9.sel; ctx.strokeStyle = c.ink; ctx.lineWidth = 2.4; ctx.strokeRect(x0 + sj * cell + 1, y0 + si * cell + 1, cell - 2, cell - 2);
  txt(ctx, tr("↑ 정답", "↑ true"), x0 - 8, y0 - 6, { color: c.muted, size: 11, align: "right" });
});
function upd9() {
  paint(F9);
  const [i, j] = S9.sel, tot = CM[i].reduce((a, b) => a + b, 0), rec = CM.map((r, k) => r[k] / r.reduce((a, b) => a + b, 0)), worst = rec.indexOf(Math.min(...rec));
  $("r9").textContent = tr(`정답 ${fname(i)} → 예측 ${fname(j)}: ${CM[i][j]}장 (${fname(i)} ${tot}장 중 ${(CM[i][j] / tot * 100).toFixed(1)}%)\n전체 정확도 ${(D11.fmnist.acc * 100).toFixed(2)}%, 가장 어려운 class: ${fname(worst)} (recall ${(rec[worst] * 100).toFixed(1)}%)`,
    `true ${fname(i)} → predicted ${fname(j)}: ${CM[i][j]} images (${(CM[i][j] / tot * 100).toFixed(1)}% of ${tot} ${fname(i)} images)\noverall accuracy ${(D11.fmnist.acc * 100).toFixed(2)}%, hardest class: ${fname(worst)} (recall ${(rec[worst] * 100).toFixed(1)}%)`);
}
F9.cv.addEventListener("click", ev => {
  const L = S9.lay; if (!L) return; const r = F9.cv.getBoundingClientRect(), j = Math.floor((ev.clientX - r.left - L.x0) / L.cell), i = Math.floor((ev.clientY - r.top - L.y0) / L.cell);
  if (i >= 0 && i < 10 && j >= 0 && j < 10) { S9.sel = [i, j]; upd9(); }
});

// 처음 그리기
// First paint
upd1(); upd2(); upd3(); upd4(); upd5(); upd6(); upd7(); paint(F7b); paint(F7c); upd8(); upd9();
repaintAll();
