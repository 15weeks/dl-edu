/* 10장 그림과 조작. 한글판(ch10-autoencoders-KR.html)과 영어판(ch10-autoencoders.html)이 같이 쓴다
   Chapter 10 figures and interactions, shared by the Korean and English pages
   학습한 autoencoder weight와 결과는 assets/data/ch10_data.js (tools/ch10_numbers.py가 만든다)
   The trained autoencoder weights and results live in assets/data/ch10_data.js (made by tools/ch10_numbers.py) */
"use strict";

const D10 = CH10_DATA;
const b64bytes = s => Uint8Array.from(atob(s), ch => ch.charCodeAt(0));
function deq(d) { const q = new Int8Array(b64bytes(d.q).buffer), [r, c] = d.shape; return Array.from({ length: r }, (_, i) => Array.from({ length: c }, (_, j) => q[i * c + j] * d.s)); }
const layers = P => P.map(L => ({ W: deq(L.W), b: L.b }));
const lin = (L, x) => L.W.map((row, j) => row.reduce((s, w, i) => s + w * x[i], L.b[j]));
const relu = v => v.map(a => (a > 0 ? a : 0)), sigv = v => v.map(a => 1 / (1 + Math.exp(-a)));
const ENC = layers(D10.dae.enc), DEC = layers(D10.dae.dec), DEC2 = layers(D10.latent.dec);
const encode = x => relu(lin(ENC[1], relu(lin(ENC[0], x))));
const decode = z => sigv(lin(DEC[1], relu(lin(DEC[0], z))));
const decode2 = z => sigv(lin(DEC2[1], relu(lin(DEC2[0], z))));
const SX = b64bytes(D10.samples.x), SY = D10.samples.y, SZ = D10.samples.z, NS = SY.length;
const samp = i => Array.from(SX.subarray(i * 196, (i + 1) * 196), v => v / 255);
const mseOf = (a, b) => meanOf(a.map((v, i) => (v - b[i]) ** 2));
// 숫자 10개의 색 (밝은 화면, 어두운 화면 따로)
// Colours for the 10 digits (separately for light and dark screens)
const HUE10 = [210, 25, 140, 0, 270, 45, 180, 330, 95, 300];
const digitCol = (c, d) => { const dark = rgb(c.bg).reduce((s, v) => s + v, 0) < 380; return `hsl(${HUE10[d]}, ${dark ? 60 : 65}%, ${dark ? 66 : 42}%)`; };

/* ================= 1절 / Section 1 ================= */
// 원래 숫자, 16개 숫자로 줄인 code, 다시 푼 숫자
// The original digit, the code shrunk to 16 numbers, and the rebuilt digit
const S1 = { start: 0 };
const F1 = fig("cv1", w => (w >= 640 ? 300 : 600), (ctx, w, h, c) => {
  const n = w >= 640 ? 6 : 3, rowsets = w >= 640 ? 1 : 2, LM = 86, colW = (w - LM - 6) / n, cell = Math.min(6.5, (colW - 14) / 14);
  for (let set = 0; set < rowsets; set++) for (let k = 0; k < n; k++) {
    const i = (S1.start + set * n + k) % NS, x = samp(i), z = encode(x), r = decode(z), x0 = LM + k * colW + (colW - 14 * cell) / 2, y0 = 26 + set * 290;
    drawImg(ctx, c, x, 14, 14, x0, y0, cell);
    const bw = 14 * cell / 16, zmax = Math.max(1e-6, ...z), by = y0 + 14 * cell + 14, bh = 40;
    z.forEach((v, j) => { const hh = v / zmax * bh; ctx.fillStyle = c.vio; ctx.globalAlpha = 0.75; ctx.fillRect(x0 + j * bw + 0.5, by + bh - hh, bw - 1, hh); ctx.globalAlpha = 1; });
    ctx.strokeStyle = c.line; ctx.strokeRect(x0 - 0.5, by - 0.5, 14 * cell + 1, bh + 1);
    drawImg(ctx, c, r, 14, 14, x0, by + bh + 14, cell);
    txt(ctx, `MSE ${fx(mseOf(x, r), 3)}`, x0 + 7 * cell, by + bh + 14 + 14 * cell + 15, { color: c.muted, size: 10.5, mono: true, align: "center" });
    if (k === 0) {
      txt(ctx, tr("원래 (196)", "original (196)"), x0 - 6, y0 + 7 * cell, { color: c.ink, size: 10.5, align: "right" });
      txt(ctx, tr("code (16)", "code (16)"), x0 - 6, by + bh / 2 + 4, { color: c.vio, size: 10.5, align: "right", w: "700" });
      txt(ctx, tr("복원 (196)", "rebuilt (196)"), x0 - 6, by + bh + 14 + 7 * cell, { color: c.ink, size: 10.5, align: "right" });
    }
  }
});
function upd1() { paint(F1); $("r1").textContent = tr(`숫자 196개 → code 16개(보라 막대) → 다시 196개. 시험 10,000장 평균 복원 오차 MSE ${D10.dae.mse} (pixel 0~1).\nnetwork 196-64-16-64-196, parameter 27,476개. 정답 label 없이 "입력 = 정답"으로 학습한다.`,
  `196 numbers → a 16-number code (violet bars) → 196 again. Mean rebuild error over 10,000 test images: MSE ${D10.dae.mse} (pixels 0–1).\nNetwork 196-64-16-64-196, 27,476 parameters. Trained with no labels: "input = target".`); }
$("b1next").addEventListener("click", () => { S1.start = (S1.start + 7) % NS; upd1(); });

/* ================= 2절 / Section 2 ================= */
// 잡음 넣기: 같은 잡음 무늬에 크기만 바꾼다
// Adding noise: one fixed noise pattern, only its size changes
const S2 = { s: 0.4, i: 9 };
const NOISE = (() => { const r = makeRng(77); return Array.from({ length: NS }, () => Array.from({ length: 196 }, () => r.n())); })();
const noisy2 = () => samp(S2.i).map((v, j) => clamp(v + S2.s * NOISE[S2.i][j], 0, 1));
const F2 = fig("cv2", w => (w >= 560 ? 200 : 420), (ctx, w, h, c) => {
  const wide = w >= 560, cell = wide ? Math.min(9, (w - 120) / 42) : Math.min(9, (w - 40) / 28), x = samp(S2.i), xn = noisy2(), r = decode(encode(xn));
  [[xn, tr(`잡음 넣은 입력 (크기 ${fx(S2.s, 2)})`, `noisy input (size ${fx(S2.s, 2)})`)], [r, tr("autoencoder 출력", "autoencoder output")], [x, tr("원래 (정답)", "original (target)")]].forEach(([img, name], k) => {
    const x0 = wide ? 16 + k * (14 * cell + 50) : 16 + (k % 2) * (14 * cell + 30), y0 = wide ? 30 : 30 + Math.floor(k / 2) * (14 * cell + 40);
    panelTitle(ctx, c, name, x0, y0 - 10); drawImg(ctx, c, img, 14, 14, x0, y0, cell);
    if (wide && k === 0) arrowS(ctx, x0 + 14 * cell + 10, y0 + 7 * cell, x0 + 14 * cell + 40, y0 + 7 * cell, c.muted, 1.6, 7);
  });
});
function upd2() {
  paint(F2);
  const x = samp(S2.i), xn = noisy2(), r = decode(encode(xn));
  $("r2").textContent = tr(`잡음 넣은 입력과 원래의 MSE ${fx(mseOf(xn, x), 4)} → autoencoder 출력과 원래의 MSE ${fx(mseOf(r, x), 4)}`, `MSE of the noisy input vs the original ${fx(mseOf(xn, x), 4)} → MSE of the autoencoder output vs the original ${fx(mseOf(r, x), 4)}`);
}
bindRange("i2s", "o2s", v => fx(v, 2), v => { S2.s = v; upd2(); });
$("b2next").addEventListener("click", () => { S2.i = (S2.i + 5) % NS; upd2(); });

/* ================= 3절 / Section 3 ================= */
// bottleneck 크기 1~64: 복원 오차와 복원 모습 (PyTorch로 미리 학습)
// Bottleneck size 1 to 64: rebuild error and rebuilt pictures (trained in advance in PyTorch)
const BN = D10.bottleneck, BO = b64bytes(BN.orig), BR = BN.recs.map(b64bytes);
const F3 = fig("cv3", w => (w >= 640 ? 300 : 560), (ctx, w, h, c) => {
  const wide = w >= 640, B = wide ? { l: 52, t: 30, w: w * 0.38 - 60, h: h - 74 } : { l: 52, t: 30, w: w - 66, h: 200 }, lx = Math.log2;
  const A = axes(ctx, c, B, [-0.3, 6.3], [0, 0.045], { xt: BN.sizes.map(s => [lx(s), String(s)]), yt: [0, 0.01, 0.02, 0.03, 0.04], xlabel: tr("bottleneck unit 수", "bottleneck units") });
  panelTitle(ctx, c, tr("복원 오차 (MSE)", "Rebuild error (MSE)"), B.l, 16);
  line(ctx, BN.sizes.map((s, i) => [A.X(lx(s)), A.Y(BN.mse[i])]), c.fa, 2.2); BN.sizes.forEach((s, i) => dot(ctx, A.X(lx(s)), A.Y(BN.mse[i]), 3.2, c.fa));
  const gx = wide ? w * 0.38 + 20 : 16, gy = wide ? 52 : B.t + B.h + 84, cols = BN.sizes.length + 1, cell = Math.min(4.6, (w - gx - 20) / cols / 15.5);
  panelTitle(ctx, c, tr("복원 모습 (열 = bottleneck 크기)", "Rebuilt digits (column = bottleneck size)"), gx, gy - 30);
  txt(ctx, tr("원래", "orig."), gx + 7 * cell, gy - 8, { color: c.ink, size: 10.5, align: "center", w: "700" });
  BN.sizes.forEach((s, k) => txt(ctx, String(s), gx + (k + 1) * 15.5 * cell + 7 * cell, gy - 8, { color: c.muted, size: 10.5, align: "center", mono: true }));
  for (let r = 0; r < 3; r++) {
    const y = gy + r * (14 * cell + 10);
    drawImg(ctx, c, Array.from(BO.subarray(r * 196, (r + 1) * 196), v => v / 255), 14, 14, gx, y, cell);
    BR.forEach((R, k) => drawImg(ctx, c, Array.from(R.subarray(r * 196, (r + 1) * 196), v => v / 255), 14, 14, gx + (k + 1) * 15.5 * cell, y, cell));
  }
});

/* ================= 4절 / Section 4 ================= */
// 가린 부분 복원: 몇 줄을 가리고 넣으면 나머지를 보고 채운다
// Restoring a hidden part: hide some rows, and the network fills them in from the rest
const S4 = { r0: 5, hgt: 4, i: 37 };
const occl = () => samp(S4.i).map((v, j) => { const row = Math.floor(j / 14); return row >= S4.r0 && row < S4.r0 + S4.hgt ? 0 : v; });
const F4 = fig("cv4", w => (w >= 560 ? 200 : 420), (ctx, w, h, c) => {
  const wide = w >= 560, cell = wide ? Math.min(9, (w - 120) / 42) : Math.min(9, (w - 40) / 28), x = samp(S4.i), xo = occl(), r = decode(encode(xo));
  [[xo, tr("가린 입력", "input with rows hidden")], [r, tr("autoencoder 출력", "autoencoder output")], [x, tr("원래", "original")]].forEach(([img, name], k) => {
    const x0 = wide ? 16 + k * (14 * cell + 50) : 16 + (k % 2) * (14 * cell + 30), y0 = wide ? 30 : 30 + Math.floor(k / 2) * (14 * cell + 40);
    panelTitle(ctx, c, name, x0, y0 - 10); drawImg(ctx, c, img, 14, 14, x0, y0, cell);
    if (k === 0) { ctx.save(); ctx.strokeStyle = c.fa; ctx.lineWidth = 1.6; ctx.setLineDash([4, 3]); ctx.strokeRect(x0, y0 + S4.r0 * cell, 14 * cell, S4.hgt * cell); ctx.restore(); }
    if (wide && k === 0) arrowS(ctx, x0 + 14 * cell + 10, y0 + 7 * cell, x0 + 14 * cell + 40, y0 + 7 * cell, c.muted, 1.6, 7);
  });
});
function upd4() {
  paint(F4);
  const x = samp(S4.i), xo = occl(), r = decode(encode(xo));
  $("r4").textContent = tr(`${S4.r0 + 1}~${Math.min(14, S4.r0 + S4.hgt)}째 줄을 가림: 가린 입력의 MSE ${fx(mseOf(xo, x), 4)} → 복원 ${fx(mseOf(r, x), 4)}`, `rows ${S4.r0 + 1}–${Math.min(14, S4.r0 + S4.hgt)} hidden: MSE of the hidden input ${fx(mseOf(xo, x), 4)} → restored ${fx(mseOf(r, x), 4)}`);
}
bindRange("i4r", "o4r", v => String(v + 1), v => { S4.r0 = v; upd4(); });
bindRange("i4h", "o4h", v => String(v), v => { S4.hgt = v; upd4(); });
$("b4next").addEventListener("click", () => { S4.i = (S4.i + 5) % NS; upd4(); });

/* ================= 5절 / Section 5 ================= */
// latent 공간: code 2개짜리 autoencoder. 시험 숫자 1000개의 자리, 누르면 그 자리를 decoder가 그린다
// Latent space: an autoencoder with a 2-number code. Where 1000 test digits land; click anywhere and the decoder draws that spot
// 보이는 범위 = 1~99 percentile (멀리 떨어진 몇 점 때문에 무리가 한쪽에 몰리지 않게)
// Visible range = 1st to 99th percentile (so a few far-off points do not squeeze the groups into a corner)
const LT = D10.latent, pct = (a, q) => { const b = [...a].sort((u, v) => u - v); return b[Math.round(q * (b.length - 1))]; };
const ZX0 = pct(LT.z.map(p => p[0]), 0.01), ZX1 = pct(LT.z.map(p => p[0]), 0.99), ZY0 = pct(LT.z.map(p => p[1]), 0.01), ZY1 = pct(LT.z.map(p => p[1]), 0.99);
const PX = 0.06 * (ZX1 - ZX0), PY = 0.06 * (ZY1 - ZY0);
const centre = d => { const pts = LT.z.filter((_, i) => LT.y[i] === d); return [meanOf(pts.map(p => p[0])), meanOf(pts.map(p => p[1]))]; };
const S5 = { z: centre(0), lay: null };
function latentMap(ctx, c, B, extra) {
  rrect(ctx, B.l, B.t, B.w, B.h, 6, c.bg, c.line);
  const X = v => B.l + (v - ZX0 + PX) / (ZX1 - ZX0 + 2 * PX) * B.w, Y = v => B.t + B.h - (v - ZY0 + PY) / (ZY1 - ZY0 + 2 * PY) * B.h;
  clipBox(ctx, B);
  LT.z.forEach((p, i) => { ctx.globalAlpha = 0.7; dot(ctx, X(p[0]), Y(p[1]), 2.4, digitCol(c, LT.y[i])); }); ctx.globalAlpha = 1;
  for (let d = 0; d < 10; d++) { const [mx, my] = centre(d); txtBg(ctx, c, String(d), X(mx), Y(my) + 5, { color: digitCol(c, d), size: 14, w: "700", align: "center" }); }
  if (extra) extra(X, Y);
  ctx.restore();
  return { X, Y, inv: (px, py) => [ZX0 - PX + (px - B.l) / B.w * (ZX1 - ZX0 + 2 * PX), ZY0 - PY + (B.t + B.h - py) / B.h * (ZY1 - ZY0 + 2 * PY)] };
}
const F5 = fig("cv5", w => (w >= 600 ? 340 : 600), (ctx, w, h, c) => {
  const wide = w >= 600, side = wide ? h - 40 : Math.min(w - 32, 360), B = { l: 16, t: 28, w: side, h: side };
  panelTitle(ctx, c, tr("latent 공간: 시험 숫자 1000개의 code (z₁, z₂)", "Latent space: codes (z₁, z₂) of 1000 test digits"), B.l, 16);
  const M = latentMap(ctx, c, B, (X, Y) => { const px = X(S5.z[0]), py = Y(S5.z[1]); line(ctx, [[px - 9, py], [px + 9, py]], c.ink, 2.2); line(ctx, [[px, py - 9], [px, py + 9]], c.ink, 2.2); });
  S5.lay = { B, inv: M.inv };
  const cell = wide ? Math.min(12, (w - side - 70) / 14) : 10, x0 = wide ? B.l + side + 40 : 16, y0 = wide ? 60 : B.t + side + 40;
  panelTitle(ctx, c, tr(`+ 자리를 decoder가 그린 것`, `what the decoder draws at +`), x0, y0 - 12);
  drawImg(ctx, c, decode2(S5.z), 14, 14, x0, y0, cell);
  txt(ctx, `z = (${fx(S5.z[0], 1)}, ${fx(S5.z[1], 1)})`, x0, y0 + 14 * cell + 18, { color: c.muted, size: 11.5, mono: true });
});
function upd5() { paint(F5); $("r5").textContent = tr(`z = (${fx(S5.z[0], 2)}, ${fx(S5.z[1], 2)}): 숫자 2개만으로 196 pixel 그림 하나를 그린다. 같은 숫자는 latent 공간에서 모여 있고, 무리 사이를 지나면 그림이 서서히 바뀐다.`, `z = (${fx(S5.z[0], 2)}, ${fx(S5.z[1], 2)}): two numbers are enough to draw a 196-pixel picture. Digits of the same kind gather together in latent space, and moving between groups changes the picture gradually.`); }
dragOn(F5, (px, py) => { const L = S5.lay; return L && px >= L.B.l && px <= L.B.l + L.B.w && py >= L.B.t && py <= L.B.t + L.B.h ? 1 : null; }, (k, px, py) => { S5.z = S5.lay.inv(px, py); upd5(); });

/* ================= 7절 / Section 7 ================= */
// latent 공간에서 두 숫자 사이를 곧게 걸으며 decoder로 그린다. 무작위 점은 빈 곳일 수 있다
// Walk in a straight line between two digits in latent space and draw with the decoder. A random point may land in an empty region
const S7 = { a: 4, b: 28, rand: null, seed: 3 };
const F7 = fig("cv7", w => (w >= 600 ? 300 : 520), (ctx, w, h, c) => {
  const wide = w >= 600, side = wide ? 240 : Math.min(w - 32, 260), B = { l: 16, t: 28, w: side, h: side }, za = SZ[S7.a], zb = SZ[S7.b];
  panelTitle(ctx, c, tr("latent 공간의 직선", "A straight line in latent space"), B.l, 16);
  latentMap(ctx, c, B, (X, Y) => {
    line(ctx, [[X(za[0]), Y(za[1])], [X(zb[0]), Y(zb[1])]], c.ink, 2.2); dot(ctx, X(za[0]), Y(za[1]), 6, c.ink, c.bg); dot(ctx, X(zb[0]), Y(zb[1]), 6, c.ink, c.bg);
    if (S7.rand) S7.rand.forEach(z => { line(ctx, [[X(z[0]) - 5, Y(z[1]) - 5], [X(z[0]) + 5, Y(z[1]) + 5]], c.fa, 2); line(ctx, [[X(z[0]) - 5, Y(z[1]) + 5], [X(z[0]) + 5, Y(z[1]) - 5]], c.fa, 2); });
  });
  const x0 = wide ? B.l + side + 30 : 16, y0 = wide ? 50 : B.t + side + 40, n = 9, cell = Math.min(5.5, (w - x0 - 20) / n / 15.5);
  panelTitle(ctx, c, tr(`${SY[S7.a]}에서 ${SY[S7.b]}까지 9걸음`, `From ${SY[S7.a]} to ${SY[S7.b]} in 9 steps`), x0, y0 - 12);
  for (let k = 0; k < n; k++) { const t = k / (n - 1), z = [za[0] + (zb[0] - za[0]) * t, za[1] + (zb[1] - za[1]) * t]; drawImg(ctx, c, decode2(z), 14, 14, x0 + k * 15.5 * cell, y0, cell); }
  if (S7.rand) {
    panelTitle(ctx, c, tr("무작위 점 (빨간 ×)을 decoder로", "Random points (red ×) through the decoder"), x0, y0 + 14 * cell + 40);
    S7.rand.forEach((z, k) => drawImg(ctx, c, decode2(z), 14, 14, x0 + k * 15.5 * cell, y0 + 14 * cell + 52, cell));
  }
});
function upd7() { paint(F7); $("r7").textContent = tr(`A = 숫자 ${SY[S7.a]}, B = 숫자 ${SY[S7.b]}. 직선 위의 점을 decoder에 넣으면 한 숫자가 다른 숫자로 서서히 바뀐다.${S7.rand ? "\n무작위 점: data 무리 위에 떨어지면 숫자처럼 보이지만, 무리 사이나 가장자리면 흐리거나 두 숫자가 섞인다. 보통 autoencoder는 latent의 어디에서 뽑아야 하는지 정해 두지 않는다 (15장 VAE가 정한다)." : ""}`, `A = digit ${SY[S7.a]}, B = digit ${SY[S7.b]}. Feeding points along the line to the decoder morphs one digit gradually into the other.${S7.rand ? "\nRandom points: one that lands on a group looks like a digit, but one between groups or near the edge comes out blurry or as a blend of two digits. A plain autoencoder never says where in latent space to sample (the VAE of chapter 15 does)." : ""}`); }
$("b7a").addEventListener("click", () => { S7.a = (S7.a + 5) % NS; upd7(); });
$("b7b").addEventListener("click", () => { S7.b = (S7.b + 7) % NS; upd7(); });
$("b7r").addEventListener("click", () => { const r = makeRng(S7.seed++); S7.rand = Array.from({ length: 6 }, () => [ZX0 + r.u() * (ZX1 - ZX0), ZY0 + r.u() * (ZY1 - ZY0)]); upd7(); });

// 처음 그리기
// First paint
upd1(); upd2(); paint(F3); upd4(); upd5(); upd7();
repaintAll();
