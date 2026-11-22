/* 8장 그림과 조작. 한글판(ch08-ffn-data-gpu-KR.html)과 영어판(ch08-ffn-data-gpu.html)이 같이 쓴다
   Chapter 8 figures and interactions, shared by the Korean and English pages
   MNIST로 학습한 weight와 실험 결과는 assets/data/ch08_data.js (tools/ch08_numbers.py가 만든다)
   The MNIST-trained weights and experiment results live in assets/data/ch08_data.js (made by tools/ch08_numbers.py) */
"use strict";

const D8 = CH08_DATA;
const b64bytes = s => Uint8Array.from(atob(s), ch => ch.charCodeAt(0));
function deq(d) { const q = new Int8Array(b64bytes(d.q).buffer), [r, c] = d.shape; return Array.from({ length: r }, (_, i) => Array.from({ length: c }, (_, j) => q[i * c + j] * d.s)); }
const W1 = deq(D8.net.W1), W2 = deq(D8.net.W2), B1 = D8.net.b1, B2 = D8.net.b2;
const SAMP = b64bytes(D8.samples.x), SY = D8.samples.y, NS = SY.length;
const sample28 = i => Array.from(SAMP.subarray(i * 784, (i + 1) * 784), v => v / 255);
// 28×28 → 14×14 (2×2 평균)
// 28×28 → 14×14 (average of each 2×2)
const pool14 = a => Array.from({ length: 196 }, (_, k) => { const r = Math.floor(k / 14) * 2, c = (k % 14) * 2; return (a[r * 28 + c] + a[r * 28 + c + 1] + a[(r + 1) * 28 + c] + a[(r + 1) * 28 + c + 1]) / 4; });
function predict(x196) { const h = W1.map((row, j) => Math.max(0, row.reduce((s, w, i) => s + w * x196[i], B1[j]))); return softmaxV(W2.map((row, k) => row.reduce((s, w, j) => s + w * h[j], B2[k]))); }
function probBars(ctx, c, p, B, hl) {
  const bw = B.w / 10, k = p.indexOf(Math.max(...p));
  for (let d = 0; d < 10; d++) {
    const hgt = p[d] * (B.h - 30), x = B.l + d * bw;
    ctx.fillStyle = d === k ? c.miss : c.sig; ctx.globalAlpha = d === k ? 0.95 : 0.55; ctx.fillRect(x + bw * 0.15, B.t + B.h - 16 - hgt, bw * 0.7, hgt); ctx.globalAlpha = 1;
    txt(ctx, String(d), x + bw / 2, B.t + B.h - 2, { color: d === hl ? c.ok : c.ink, size: 12, w: d === k ? "700" : "400", align: "center", mono: true });
    if (p[d] > 0.05) txt(ctx, Math.round(p[d] * 100) + "%", x + bw / 2, B.t + B.h - 20 - hgt, { color: c.ink, size: 10, align: "center", mono: true });
  }
}

/* ================= 1절 / Section 1 ================= */
// 사진 한 장을 한 줄로 펼친다: 28줄이 차례로 이어져 784칸
// Unroll one picture into a line: the 28 rows join one after another into 784 cells
const S1 = { i: 3, t: 0, timer: 0 };
const F1 = fig("cv1", w => (w >= 600 ? 250 : 330), (ctx, w, h, c) => {
  const a = sample28(S1.i), cell = 6, x0 = 16, y0 = 34, lines = w >= 600 ? 2 : 4, lineW = w - 40 - (w >= 600 ? 200 : 0), cw = lineW * lines / 784;
  const sx0 = w >= 600 ? 200 : 20, sy0 = w >= 600 ? 70 : 34 + 28 * cell + 30;
  panelTitle(ctx, c, tr(`28 × 28 사진 (숫자 ${SY[S1.i]})`, `28 × 28 picture (digit ${SY[S1.i]})`), x0, 18);
  txt(ctx, tr(`한 줄로 펼친 784칸 (${lines}줄로 접어 그림)`, `unrolled into 784 cells (drawn folded into ${lines} lines)`), sx0, sy0 - 12, { color: c.ink, size: 12, w: "600" });
  const e = S1.t * S1.t * (3 - 2 * S1.t), ink = rgb(c.ink), bg = rgb(c.bg);
  for (let r = 0; r < 28; r++) {
    const pos = r * 28, ln = Math.floor(pos * cw / lineW), sx = sx0 + (pos * cw - ln * lineW), sy = sy0 + ln * 34;
    for (let k = 0; k < 28; k++) {
      const v = a[r * 28 + k], ix = x0 + k * cell, iy = y0 + r * cell;
      const px = ix + (sx + k * cw - ix) * e, py = iy + (sy - iy) * e, pw = cell + (cw - cell) * e, ph = cell + (24 - cell) * e;
      ctx.fillStyle = cssRGB(mixRGB(bg, ink, v)); ctx.fillRect(px, py, Math.max(0.6, pw), ph);
    }
  }
  if (S1.t < 0.01) { ctx.strokeStyle = c.line; ctx.strokeRect(x0 - 0.5, y0 - 0.5, 28 * cell + 1, 28 * cell + 1); }
  if (S1.t > 0.99) for (let l = 0; l < lines; l++) { ctx.strokeStyle = c.line; ctx.strokeRect(sx0 - 0.5, sy0 + l * 34 - 0.5, lineW + 1, 25); }
});
function upd1() { paint(F1); $("r1").textContent = tr(`28 × 28 = 784개의 숫자 (0 = 흰색, 1 = 검정). FFN은 이 784개를 한 줄짜리 vector로 받는다. 이 장의 network는 2×2씩 평균 낸 14 × 14 = 196개를 쓴다.`, `28 × 28 = 784 numbers (0 = white, 1 = black). An FFN takes these 784 as a single vector. This chapter's network uses 14 × 14 = 196 numbers, averaging each 2×2 block.`); }
$("b1play").addEventListener("click", () => { const to = S1.t < 0.5 ? 1 : 0; runSteps(S1, 25, () => { S1.t = clamp(S1.t + (to ? 0.04 : -0.04), 0, 1); paint(F1); }, 30); });
$("b1next").addEventListener("click", () => { S1.i = (S1.i + 3) % NS; upd1(); });

/* ================= 2절 / Section 2 ================= */
// 숫자 그리기: 28×28 칸에 그리면 MNIST처럼 가운데 맞추고 14×14로 줄여 network에 넣는다
// Draw a digit: whatever you draw on the 28×28 grid is centred like MNIST, shrunk to 14×14 and fed to the network
const S2 = { a: sample28(7), lay: null, prev: null };
function prep(a) {
  let x0 = 28, x1 = -1, y0 = 28, y1 = -1, tot = 0;
  for (let r = 0; r < 28; r++) for (let k = 0; k < 28; k++) if (a[r * 28 + k] > 0.05) { x0 = Math.min(x0, k); x1 = Math.max(x1, k); y0 = Math.min(y0, r); y1 = Math.max(y1, r); }
  if (x1 < 0) return new Array(784).fill(0);
  const bw = x1 - x0 + 1, bh = y1 - y0 + 1, s = 20 / Math.max(bw, bh), out = new Array(784).fill(0);
  // 1) 글자 상자를 20칸 크기로 늘이거나 줄인다 (가까운 칸 평균)
  // 1) scale the ink box to 20 cells (average of nearby cells)
  const nw = Math.max(1, Math.round(bw * s)), nh = Math.max(1, Math.round(bh * s)), tmp = [];
  for (let r = 0; r < nh; r++) for (let k = 0; k < nw; k++) {
    const sx = x0 + (k + 0.5) / s - 0.5, sy = y0 + (r + 0.5) / s - 0.5, ix = Math.floor(sx), iy = Math.floor(sy), fx_ = sx - ix, fy = sy - iy;
    const g = (yy, xx) => (yy >= 0 && yy < 28 && xx >= 0 && xx < 28 ? a[yy * 28 + xx] : 0);
    tmp.push(g(iy, ix) * (1 - fx_) * (1 - fy) + g(iy, ix + 1) * fx_ * (1 - fy) + g(iy + 1, ix) * (1 - fx_) * fy + g(iy + 1, ix + 1) * fx_ * fy);
  }
  // 2) 무게 중심이 (14, 14)에 오도록 놓는다
  // 2) place it so the centre of mass sits at (14, 14)
  let cx = 0, cy = 0; tmp.forEach((v, i) => { cx += v * (i % nw); cy += v * Math.floor(i / nw); tot += v; });
  cx /= tot || 1; cy /= tot || 1;
  const ox = Math.round(14 - cx - 0.5), oy = Math.round(14 - cy - 0.5);
  tmp.forEach((v, i) => { const xx = (i % nw) + ox, yy = Math.floor(i / nw) + oy; if (xx >= 0 && xx < 28 && yy >= 0 && yy < 28) out[yy * 28 + xx] = Math.min(1, v); });
  return out;
}
const F2 = fig("cv2", w => (w >= 640 ? 300 : 620), (ctx, w, h, c) => {
  const wide = w >= 640, cell = wide ? Math.min(9.5, (w * 0.36) / 28) : Math.min(10, (w - 40) / 28), G = { l: 16, t: 30, s: cell * 28 };
  S2.lay = { G, cell };
  panelTitle(ctx, c, tr("여기에 숫자를 그린다 (끌기)", "Draw a digit here (drag)"), G.l, 18);
  drawImg(ctx, c, S2.a, 28, 28, G.l, G.t, cell, { grid: false });
  const x196 = pool14(prep(S2.a)), p = predict(x196);
  const mx = wide ? G.l + G.s + 30 : G.l, my = wide ? G.t : G.t + G.s + 40, mc = wide ? 6 : 6;
  panelTitle(ctx, c, tr("network가 보는 14 × 14", "what the network sees, 14 × 14"), mx, my - 12);
  drawImg(ctx, c, x196, 14, 14, mx, my, mc);
  const B = wide ? { l: mx + 215, t: G.t, w: w - (mx + 215) - 14, h: G.s } : { l: 16, t: my + 14 * mc + 30, w: w - 32, h: 190 };
  panelTitle(ctx, c, tr("softmax 확률", "softmax probabilities"), B.l, B.t - 12);
  probBars(ctx, c, p, B, -1);
});
function upd2() {
  paint(F2);
  const p = predict(pool14(prep(S2.a))), k = p.indexOf(Math.max(...p)), srt = p.map((v, d) => [v, d]).sort((a, b) => b[0] - a[0]);
  $("r2").textContent = tr(`답: ${k} (${Math.round(p[k] * 100)}%), 둘째: ${srt[1][1]} (${Math.round(srt[1][0] * 100)}%)\nnetwork 196-64-10 (parameter 13,258개), MNIST 시험 10,000장 정확도 ${(D8.net.acc * 100).toFixed(1)}%`,
    `answer: ${k} (${Math.round(p[k] * 100)}%), runner-up: ${srt[1][1]} (${Math.round(srt[1][0] * 100)}%)\nnetwork 196-64-10 (13,258 parameters), accuracy on the 10,000 MNIST test images ${(D8.net.acc * 100).toFixed(1)}%`);
}
function paint2(px, py) {
  const { G, cell } = S2.lay, cx = (px - G.l) / cell, cy = (py - G.t) / cell;
  for (let r = Math.floor(cy - 2); r <= Math.ceil(cy + 2); r++) for (let k = Math.floor(cx - 2); k <= Math.ceil(cx + 2); k++) {
    if (r < 0 || r >= 28 || k < 0 || k >= 28) continue;
    const d = Math.hypot(k + 0.5 - cx, r + 0.5 - cy); if (d < 1.7) S2.a[r * 28 + k] = Math.min(1, S2.a[r * 28 + k] + (d < 0.9 ? 1 : 0.45));
  }
}
dragOn(F2, (px, py) => { const L = S2.lay; if (!L) return null; return px >= L.G.l && px <= L.G.l + L.G.s && py >= L.G.t && py <= L.G.t + L.G.s ? 1 : null; }, (k, px, py) => {
  if (S2.prev) { const [qx, qy] = S2.prev, n = Math.ceil(Math.hypot(px - qx, py - qy) / 3); for (let i = 1; i <= n; i++) paint2(qx + (px - qx) * i / n, qy + (py - qy) * i / n); } else paint2(px, py);
  S2.prev = [px, py]; upd2();
});
F2.cv.addEventListener("pointerup", () => { S2.prev = null; });
$("b2clear").addEventListener("click", () => { S2.a = new Array(784).fill(0); upd2(); });
$("b2sample").addEventListener("click", () => { S2.si = ((S2.si || 7) + 4) % NS; S2.a = sample28(S2.si); upd2(); });

/* ================= 3절 / Section 3 ================= */
// 1층 weight: 학습 전후 분포, 그리고 weight를 14×14로 다시 접어 본 "무늬"
// Layer-1 weights: distribution before and after training, and the weights folded back to 14×14 "patterns"
const F3 = fig("cv3", w => (w >= 640 ? 260 : 520), (ctx, w, h, c) => {
  const wide = w >= 640, H = D8.hist, P = { x: 0, y: 0, w: wide ? w * 0.5 : w, h: 260 };
  const B = { l: P.x + 44, t: P.y + 30, w: P.w - 60, h: P.h - 76 }, mx = Math.max(...H.before, ...H.after);
  const A = axes(ctx, c, B, [-0.4, 0.4], [0, mx * 1.05], { xt: [-0.4, -0.2, 0, 0.2, 0.4], yt: ticksR(0, mx, 3), xlabel: "weight" });
  panelTitle(ctx, c, tr("1층 weight 12,544개의 분포", "Distribution of the 12,544 layer-1 weights"), B.l, P.y + 14);
  const bw = (A.X(H.bins[1]) - A.X(H.bins[0]));
  H.before.forEach((v, i) => { ctx.fillStyle = c.muted; ctx.globalAlpha = 0.45; ctx.fillRect(A.X(H.bins[i]), A.Y(v), bw - 1, B.t + B.h - A.Y(v)); });
  H.after.forEach((v, i) => { ctx.fillStyle = c.sig; ctx.globalAlpha = 0.6; ctx.fillRect(A.X(H.bins[i]) + bw * 0.25, A.Y(v), bw * 0.5, B.t + B.h - A.Y(v)); });
  ctx.globalAlpha = 1;
  txtBg(ctx, c, tr(`회색 = 학습 전 (표준편차 ${H.sd_before})`, `grey = before training (SD ${H.sd_before})`), B.l + 6, B.t + 14, { color: c.muted, size: 11.5, w: "600" });
  txtBg(ctx, c, tr(`파랑 = 학습 뒤 (표준편차 ${H.sd_after})`, `blue = after training (SD ${H.sd_after})`), B.l + 6, B.t + 31, { color: c.sig, size: 11.5, w: "600" });
  const tx = wide ? w * 0.5 + 24 : 16, ty = wide ? 30 : 290, cell = wide ? Math.min(5.5, (w * 0.5 - 60) / 4 / 15) : Math.min(5.5, (w - 40) / 4 / 15);
  panelTitle(ctx, c, tr("숨은 neuron 8개의 weight를 14 × 14로 (파랑 +, 빨강 −)", "8 hidden neurons' weights as 14 × 14 (blue +, red −)"), tx, ty - 14);
  D8.templates.forEach((t, i) => drawImg(ctx, c, t, 14, 14, tx + (i % 4) * (14 * cell + 12), ty + Math.floor(i / 4) * (14 * cell + 14), cell, { div: true }));
});

/* ================= 4절 / Section 4 ================= */
// pixel 섞기: 사람에게는 잡음이지만 FFN에게는 순서가 상관없다
// Scrambled pixels: noise to a person, but the order does not matter to an FFN
const S4 = { i: 12 };
const PERM = D8.scramble.perm;
const F4a = fig("cv4a", 200, (ctx, w, h, c) => {
  const a = pool14(sample28(S4.i)), sc = PERM.map(j => a[j]), cell = Math.min(9, (w - 80) / 28), x0 = 16, y0 = 30;
  panelTitle(ctx, c, tr(`원래 (숫자 ${SY[S4.i]})`, `original (digit ${SY[S4.i]})`), x0, 18);
  drawImg(ctx, c, a, 14, 14, x0, y0, cell);
  const x1 = x0 + 14 * cell + 50;
  arrowS(ctx, x0 + 14 * cell + 10, y0 + 7 * cell, x1 - 10, y0 + 7 * cell, c.muted, 1.6, 7);
  panelTitle(ctx, c, tr("196칸을 정해진 순서로 섞음", "the 196 cells in a fixed shuffled order"), x1, 18);
  drawImg(ctx, c, sc, 14, 14, x1, y0, cell);
});
$("b4next").addEventListener("click", () => { S4.i = (S4.i + 3) % NS; paint(F4a); });
// 옆으로 밀기: 학습한 network는 가운데 맞춘 숫자만 보았다
// Shifting sideways: the trained network has only seen centred digits
const S4b = { s: 0, i: 21 };
function shifted(a, s) { const o = new Array(784).fill(0); for (let r = 0; r < 28; r++) for (let k = 0; k < 28; k++) if (k + s < 28) o[r * 28 + k + s] = a[r * 28 + k]; return o; }
const F4b = fig("cv4b", w => (w >= 640 ? 260 : 560), (ctx, w, h, c) => {
  const wide = w >= 640, a = shifted(sample28(S4b.i), S4b.s), cell = 6, x0 = 16, y0 = 30;
  panelTitle(ctx, c, tr(`${S4b.s}칸 민 숫자`, `digit shifted by ${S4b.s} px`), x0, 18);
  drawImg(ctx, c, a, 28, 28, x0, y0, cell);
  const p = predict(pool14(a)), B = wide ? { l: x0 + 28 * cell + 24, t: y0, w: w * 0.3, h: 28 * cell } : { l: 16, t: y0 + 28 * cell + 30, w: w - 32, h: 150 };
  panelTitle(ctx, c, tr("network의 답", "the network's answer"), B.l, B.t - 12);
  probBars(ctx, c, p, B, SY[S4b.i]);
  const C = wide ? { l: B.l + B.w + 54, t: y0, w: w - (B.l + B.w + 54) - 14, h: 28 * cell - 30 } : { l: 50, t: B.t + B.h + 50, w: w - 64, h: h - (B.t + B.h + 50) - 40 };
  const A = axes(ctx, c, C, [-0.3, 6.3], [0, 1], { xt: [0, 1, 2, 3, 4, 5, 6], yt: [[0, "0"], [0.5, "50%"], [1, "100%"]], xlabel: tr("민 칸 수", "pixels shifted") });
  panelTitle(ctx, c, tr("시험 10,000장 정확도", "accuracy on 10,000 test images"), C.l, C.t - 12);
  line(ctx, D8.shift.map((v, s) => [A.X(s), A.Y(v)]), c.fa, 2); D8.shift.forEach((v, s) => dot(ctx, A.X(s), A.Y(v), 3, c.fa));
  dot(ctx, A.X(S4b.s), A.Y(D8.shift[S4b.s]), 6.5, c.miss, c.bg);
});
function upd4b() {
  paint(F4b);
  const p = predict(pool14(shifted(sample28(S4b.i), S4b.s))), k = p.indexOf(Math.max(...p));
  $("r4b").textContent = tr(`정답 ${SY[S4b.i]}, ${S4b.s}칸 민 숫자의 답 ${k} (${Math.round(p[k] * 100)}%)\n시험 10,000장을 ${S4b.s}칸 밀면 정확도 ${(D8.shift[S4b.s] * 100).toFixed(1)}% (안 밀면 ${(D8.shift[0] * 100).toFixed(1)}%)`,
    `answer ${SY[S4b.i]}, the shifted digit is read as ${k} (${Math.round(p[k] * 100)}%)\nshifting all 10,000 test images by ${S4b.s} px: accuracy ${(D8.shift[S4b.s] * 100).toFixed(1)}% (unshifted ${(D8.shift[0] * 100).toFixed(1)}%)`);
}
bindRange("i4s", "o4s", v => String(v), v => { S4b.s = v; upd4b(); });
$("b4bnext").addEventListener("click", () => { S4b.i = (S4b.i + 3) % NS; upd4b(); });

/* ================= 5절 / Section 5 ================= */
// universal approximation: 숨은 neuron N개의 sigmoid 계단을 더해 곡선 하나를 흉내 낸다 (출력 weight는 최소 제곱)
// Universal approximation: add up N hidden sigmoid steps to imitate a curve (output weights by least squares)
const target5 = x => Math.sin(2 * Math.PI * x) * 0.6 + 0.35 * Math.exp(-((x - 0.7) ** 2) / 0.004);
const XS5 = Array.from({ length: 200 }, (_, i) => i / 199);
const S5 = { n: 6 };
function fit5(n) {
  const cs = Array.from({ length: n }, (_, i) => (i + 0.5) / n), k = 4 * n + 4;
  const feat = x => [1, ...cs.map(cc => 1 / (1 + Math.exp(-k * (x - cc))))];
  const F = XS5.map(feat), m = n + 1, A = Array.from({ length: m }, (_, i) => Array.from({ length: m }, (_, j) => F.reduce((s, r) => s + r[i] * r[j], 0) + (i === j ? 1e-6 : 0)));
  const coef = (function solve(A, b) { const N = b.length, M = A.map((r, i) => [...r, b[i]]); for (let col = 0; col < N; col++) { let p = col; for (let r = col + 1; r < N; r++) if (Math.abs(M[r][col]) > Math.abs(M[p][col])) p = r; [M[col], M[p]] = [M[p], M[col]]; for (let r = col + 1; r < N; r++) { const f = M[r][col] / M[col][col]; for (let q = col; q <= N; q++) M[r][q] -= f * M[col][q]; } } const x = new Array(N).fill(0); for (let r = N - 1; r >= 0; r--) { let s = M[r][N]; for (let q = r + 1; q < N; q++) s -= M[r][q] * x[q]; x[r] = s / M[r][r]; } return x; })(A, Array.from({ length: m }, (_, i) => F.reduce((s, r, t) => s + r[i] * target5(XS5[t]), 0)));
  return { f: x => feat(x).reduce((s, v, i) => s + v * coef[i], 0), parts: cs.map((cc, i) => x => coef[i + 1] / (1 + Math.exp(-k * (x - cc)))), err: Math.max(...XS5.map(x => Math.abs(feat(x).reduce((s, v, i) => s + v * coef[i], 0) - target5(x)))) };
}
const F5 = fig("cv5", 300, (ctx, w, h, c) => {
  const B = { l: 44, t: 30, w: w - 60, h: h - 74 }, r = fit5(S5.n);
  const A = axes(ctx, c, B, [0, 1], [-1.2, 1.2], { xt: [0, 0.25, 0.5, 0.75, 1], yt: [-1, 0, 1], xlabel: "x" });
  panelTitle(ctx, c, tr(`숨은 neuron ${S5.n}개 (sigmoid 계단 ${S5.n}개의 합)`, `${S5.n} hidden neurons (a sum of ${S5.n} sigmoid steps)`), B.l, 16);
  clipBox(ctx, B);
  r.parts.forEach(pf => line(ctx, curve(A, pf, [0, 1], 150), c.line, 1.2));
  line(ctx, curve(A, target5, [0, 1], 300), c.ink, 1.6, [6, 4]);
  line(ctx, curve(A, r.f, [0, 1], 300), c.miss, 2.6);
  ctx.restore();
  txtBg(ctx, c, tr("점선 = 목표 곡선, 주황 = network, 회색 = 계단 하나하나", "dashed = target curve, orange = network, grey = each step"), B.l + 6, B.t + B.h - 8, { color: c.muted, size: 11 });
});
function upd5() { paint(F5); const e = fit5(S5.n).err; $("r5").textContent = tr(`숨은 neuron ${S5.n}개: 가장 큰 오차 ${fx(e, 3)}${e < 0.05 ? " → 거의 같다" : ""}`, `${S5.n} hidden neurons: largest error ${fx(e, 3)}${e < 0.05 ? " → almost identical" : ""}`); }
bindRange("i5n", "o5n", v => String(v), v => { S5.n = v; upd5(); });

/* ================= 6절 / Section 6 ================= */
// data 크기 × network 너비 (PyTorch로 미리 돌림, 3000걸음씩)
// Data size × network width (precomputed in PyTorch, 3000 steps each)
const F6 = fig("cv6", 290, (ctx, w, h, c) => {
  const G = D8.sizeGrid, B = { l: 50, t: 30, w: w - 66, h: h - 74 }, lx = v => Math.log10(v), cols = [c.muted, c.sig, c.miss];
  const A = axes(ctx, c, B, [2.4, 4.9], [0.75, 1], { xt: [[lx(300), "300"], [3, "1,000"], [lx(3000), "3,000"], [4, "10,000"], [lx(60000), "60,000"]], yt: [[0.8, "80%"], [0.9, "90%"], [1, "100%"]], xlabel: tr("학습 data 수 (log)", "training images (log)") });
  panelTitle(ctx, c, tr("시험 정확도: data 수 × 숨은 neuron 수", "Test accuracy: data size × hidden neurons"), B.l, 16);
  G.widths.forEach((wd, k) => {
    line(ctx, G.sizes.map((n, i) => [A.X(lx(n)), A.Y(G.acc[k][i])]), cols[k], 2.2); G.sizes.forEach((n, i) => dot(ctx, A.X(lx(n)), A.Y(G.acc[k][i]), 3.2, cols[k]));
    txtBg(ctx, c, tr(`숨은 ${wd}개`, `${wd} hidden`), A.X(lx(60000)) - 4, A.Y(G.acc[k][4]) + (k === 0 ? 16 : -8), { color: cols[k], size: 11.5, w: "700", align: "right" });
  });
});

/* ================= 7절 / Section 7 ================= */
// 불균형 data: 한 class가 아주 적다. 정확도는 높아 보여도 적은 class를 놓친다. oversampling으로 맞추기
// Unbalanced data: one class is rare. Accuracy looks high, yet the rare class is missed. Balance it with oversampling
const S7 = { minor: 5, over: false, res: null };
function data7() {
  const r = makeRng(77), X = [], Y = [], n1 = Math.round(400 * S7.minor / 100), n0 = 400 - n1;
  for (let i = 0; i < n0; i++) { X.push([r.n() * 0.7 - 0.4, r.n() * 0.7]); Y.push(0); }
  for (let i = 0; i < n1; i++) { X.push([r.n() * 0.5 + 0.7, r.n() * 0.5 + 0.2]); Y.push(1); }
  return { X, Y };
}
function run7() {
  const D = data7(); let X = D.X, Y = D.Y;
  if (S7.over) { const idx1 = Y.map((v, i) => (v ? i : -1)).filter(i => i >= 0), need = Y.length - 2 * idx1.length; const add = Array.from({ length: Math.max(0, need) }, (_, k) => idx1[k % idx1.length]); X = X.concat(add.map(i => X[i])); Y = Y.concat(add.map(() => 1)); }
  const net = new TinyNet([2, 8, 1], { hidden: "tanh", seed: 3, opt: { name: "adam", lr: 0.02 } });
  for (let e = 0; e < 40; e++) net.epoch(X, Y, 32);
  const pr = D.X.map(x => (net.predict(x)[0] > 0.5 ? 1 : 0)), tp = pr.filter((p, i) => p && D.Y[i]).length, n1 = D.Y.filter(v => v).length;
  S7.res = { net, D, acc: pr.filter((p, i) => p === D.Y[i]).length / D.Y.length, recall: n1 ? tp / n1 : 0, n1, n: D.Y.length };
}
const F7a = fig("cv7a", w => Math.round(clamp(w * 0.5, 240, 340)), (ctx, w, h, c) => {
  const { net, D } = S7.res, B = { l: 30, t: 24, w: w - 40, h: h - 34 }, R = equalRange(B, 2, 0.1, 0);
  drawField(ctx, c, B, R.xr, R.yr, (x, y) => net.probs([x, y]), { alpha: 0.35 });
  const A = axes(ctx, c, B, R.xr, R.yr, { xt: [], yt: [], grid: false });
  clipBox(ctx, B); drawPoints(ctx, c, A, D.X, D.Y, { r: 3 }); ctx.restore();
  panelTitle(ctx, c, tr(`많은 class ${S7.res.n - S7.res.n1}개 : 적은 class ${S7.res.n1}개${S7.over ? " (oversampling)" : ""}`, `common class ${S7.res.n - S7.res.n1} : rare class ${S7.res.n1}${S7.over ? " (oversampled)" : ""}`), B.l, 16);
});
function upd7() {
  run7(); paint(F7a);
  const R = S7.res;
  $("r7").textContent = tr(`정확도 ${Math.round(R.acc * 100)}%, 적은 class를 찾은 비율(recall) ${Math.round(R.recall * 100)}% (${Math.round(R.recall * R.n1)} / ${R.n1})\n→ ${R.recall < 0.6 ? "정확도는 높지만 적은 class를 거의 놓친다: \"전부 많은 class\"라고만 해도 정확도가 높기 때문" : "적은 class도 대부분 찾는다"}`,
    `accuracy ${Math.round(R.acc * 100)}%, share of the rare class found (recall) ${Math.round(R.recall * 100)}% (${Math.round(R.recall * R.n1)} / ${R.n1})\n→ ${R.recall < 0.6 ? "accuracy is high but the rare class is mostly missed: answering \"common\" every time already scores high" : "most of the rare class is found too"}`);
}
bindRange("i7m", "o7m", v => v + "%", v => { S7.minor = v; upd7(); });
bindSeg("g7o", v => { S7.over = v === "on"; upd7(); });
// augmentation 미리보기: 숫자 하나를 조금씩 움직이고 돌리고 잡음을 넣는다
// Augmentation preview: shift, rotate and add noise to one digit, a little each time
const S7b = { seed: 5, i: 15 };
function augment(a, r) {
  const ang = (r.u() - 0.5) * 0.5, sc = 0.9 + r.u() * 0.2, dx = (r.u() - 0.5) * 4, dy = (r.u() - 0.5) * 4, noise = r.u() < 0.5 ? 0.15 : 0, o = new Array(784).fill(0), cs = Math.cos(ang), sn = Math.sin(ang);
  for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
    const u = ((x - 13.5 - dx) * cs + (y - 13.5 - dy) * sn) / sc + 13.5, v = (-(x - 13.5 - dx) * sn + (y - 13.5 - dy) * cs) / sc + 13.5, iu = Math.round(u), iv = Math.round(v);
    o[y * 28 + x] = iu >= 0 && iu < 28 && iv >= 0 && iv < 28 ? a[iv * 28 + iu] : 0;
    if (noise) o[y * 28 + x] = clamp(o[y * 28 + x] + noise * r.n(), 0, 1);
  }
  return o;
}
const F7b = fig("cv7b", w => (w >= 600 ? 130 : 230), (ctx, w, h, c) => {
  const a = sample28(S7b.i), r = makeRng(S7b.seed), n = w >= 600 ? 8 : 6, cell = Math.min(3.4, (w - 30 - (n + 1) * 10) / ((n + 1) * 28)), per = w >= 600 ? n + 1 : 4;
  panelTitle(ctx, c, tr("원래 1장 → 조금씩 바꾼 새 학습 그림", "1 original → slightly altered new training pictures"), 12, 16);
  for (let k = 0; k <= n; k++) { const img = k ? augment(a, r) : a, x = 12 + (k % per) * (28 * cell + 12), y = 28 + Math.floor(k / per) * (28 * cell + 12); drawImg(ctx, c, img, 28, 28, x, y, cell); if (!k) { ctx.strokeStyle = c.miss; ctx.lineWidth = 2; ctx.strokeRect(x - 2, y - 2, 28 * cell + 4, 28 * cell + 4); } }
});
$("b7aug").addEventListener("click", () => { S7b.seed++; paint(F7b); });

/* ================= 8절 / Section 8 ================= */
// 가장 좋은 model 남기기: dev 정확도가 가장 높았던 epoch의 weight를 저장
// Keep the best model: save the weights from the epoch with the highest dev accuracy
const F8 = fig("cv8", 280, (ctx, w, h, c) => {
  const L = D8.ckpt.log, best = D8.ckpt.best, B = { l: 48, t: 30, w: w - 62, h: h - 74 };
  const A = axes(ctx, c, B, [0, 40], [0.7, 1], { xt: [0, 10, 20, 30, 40], yt: [[0.7, "70%"], [0.8, "80%"], [0.9, "90%"], [1, "100%"]], xlabel: "epoch" });
  panelTitle(ctx, c, tr("학습 1000장 · dev 5000장, 196-256-10", "1000 training · 5000 dev images, 196-256-10"), B.l, 16);
  line(ctx, L.map(r => [A.X(r[0]), A.Y(r[1])]), c.sig, 2); line(ctx, L.map(r => [A.X(r[0]), A.Y(r[2])]), c.miss, 2);
  line(ctx, [[A.X(best[0]), B.t], [A.X(best[0]), B.t + B.h]], c.ok, 1.6, [5, 4]); dot(ctx, A.X(best[0]), A.Y(best[2]), 6, c.ok, c.bg);
  txtBg(ctx, c, tr(`가장 좋은 dev: epoch ${best[0]}, ${(best[2] * 100).toFixed(1)}% → 이 weight를 저장`, `best dev: epoch ${best[0]}, ${(best[2] * 100).toFixed(1)}% → save these weights`), A.X(best[0]) - 6, A.Y(best[2]) + 22, { color: c.ok, size: 11.5, w: "700", align: "right" });
  txtBg(ctx, c, tr("학습 정확도", "training accuracy"), A.X(40) - 4, A.Y(L[L.length - 1][1]) - 8, { color: c.sig, size: 11.5, w: "700", align: "right" });
  txtBg(ctx, c, tr("dev 정확도", "dev accuracy"), A.X(40) - 4, A.Y(L[L.length - 1][2]) - 8, { color: c.miss, size: 11.5, w: "700", align: "right" });
});

// 처음 그리기
// First paint
upd1(); upd2(); paint(F3); paint(F4a); upd4b(); upd5(); paint(F6); upd7(); paint(F7b); paint(F8);
repaintAll();
