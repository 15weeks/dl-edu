/* 3장 그림과 조작. 한글판(ch03-neural-networks-KR.html)과 영어판(ch03-neural-networks.html)이 같이 쓴다. 공통 도구는 assets/dlfig.js
   Chapter 3 figures and interactions, shared by the Korean and English pages. Shared helpers live in assets/dlfig.js */
"use strict";

const sigm = z => 1 / (1 + Math.exp(-z));

/* ================= 2절 / Section 2 ================= */
// neuron 하나: 입력 × weight를 더하고(Σ) activation을 지나 출력
// One neuron: inputs × weights are summed (Σ), then pass through the activation to the output
const S2 = { x: [1, 2], w: [0.5, -0.25], b: 0.5, act: "sig" };
const F2 = fig("cv2", 250, (ctx, w, h, c) => {
  const z = S2.w[0] * S2.x[0] + S2.w[1] * S2.x[1] + S2.b, a = S2.act === "sig" ? sigm(z) : z > 0 ? 1 : 0;
  const xi = Math.max(34, w * 0.07), xs = w * 0.42, xa = w * 0.64, xo = w - Math.max(40, w * 0.08), yc = 132, ys = [70, 194];
  ys.forEach((y, i) => {
    const wv = S2.w[i], col = wv >= 0 ? c.sig : c.fa;
    line(ctx, [[xi + 20, y], [xs - 28, yc]], col, 1 + Math.min(5, Math.abs(wv) * 3));
    txtBg(ctx, c, `w${SUB(i + 1)} = ${fx(wv)}`, (xi + xs) / 2, (y + yc) / 2 + (i ? 16 : -8), { color: col, size: 12, w: "700", align: "center" });
    dot(ctx, xi, y, 20, c.bg, c.ink); txt(ctx, fx(S2.x[i], 1), xi, y + 4, { color: c.ink, size: 12, mono: true, align: "center", w: "600" });
    txt(ctx, `x${SUB(i + 1)}`, xi, y - 26, { color: c.muted, size: 11.5, align: "center" });
  });
  line(ctx, [[xs, 40], [xs, yc - 28]], c.vio, 1.6, [4, 3]);
  txtBg(ctx, c, `+ b = ${fx(S2.b)}`, xs, 34, { color: c.vio, size: 12, w: "700", align: "center" });
  dot(ctx, xs, yc, 28, c.wash, c.ink); txt(ctx, "Σ", xs, yc - 4, { color: c.ink, size: 17, align: "center", w: "700" });
  txtBg(ctx, c, `z = ${fx(z)}`, xs, yc + 46, { color: c.ink, size: 11.5, mono: true, align: "center", w: "600" });
  arrowS(ctx, xs + 30, yc, xa - 50, yc, c.muted, 1.6, 7);
  // activation 상자: 작은 그래프 위에 z의 위치
  // Activation box: where z sits on a small plot
  const bw = 96, bh = 76, bx = xa - bw / 2, by = yc - bh / 2;
  rrect(ctx, bx, by, bw, bh, 8, c.surface || c.bg, c.ink);
  const X = v => bx + 8 + (v + 4) / 8 * (bw - 16), Y = v => by + bh - 10 - v * (bh - 24);
  line(ctx, [[bx + 6, Y(0)], [bx + bw - 6, Y(0)]], c.line, 1);
  const pts = []; for (let i = 0; i <= 60; i++) { const v = -4 + 8 * i / 60; pts.push([X(v), Y(S2.act === "sig" ? sigm(v) : v > 0 ? 1 : 0)]); }
  line(ctx, pts, c.ok, 2);
  const zc = clamp(z, -4, 4); dot(ctx, X(zc), Y(a), 4.5, c.miss);
  txt(ctx, S2.act === "sig" ? "sigmoid" : "step", xa, by - 6, { color: c.ok, size: 11.5, w: "700", align: "center" });
  arrowS(ctx, xa + bw / 2 + 2, yc, xo - 26, yc, c.muted, 1.6, 7);
  dot(ctx, xo, yc, 24, c.okwash, c.ok); txt(ctx, fx(a, S2.act === "sig" ? 3 : 0), xo, yc + 4, { color: c.ink, size: 12, mono: true, align: "center", w: "700" });
  txt(ctx, "ŷ", xo, yc - 30, { color: c.muted, size: 12, align: "center" });
});
function upd2() {
  paint(F2);
  const [x1, x2] = S2.x, [w1, w2] = S2.w, z = w1 * x1 + w2 * x2 + S2.b, a = S2.act === "sig" ? sigm(z) : z > 0 ? 1 : 0;
  $("r2").textContent = tr(
    `z = w₁x₁ + w₂x₂ + b = ${fx(w1)} × ${paren(x1, 1)} + ${paren(w2)} × ${paren(x2, 1)} + ${paren(S2.b)} = ${fx(z)}\nŷ = ${S2.act === "sig" ? `σ(${fx(z)}) = 1 ÷ (1 + e^${paren(-z)}) = ${fx(a, 3)}` : `step(${fx(z)}) = ${a}  (z > 0이면 1, 아니면 0)`}`,
    `z = w₁x₁ + w₂x₂ + b = ${fx(w1)} × ${paren(x1, 1)} + ${paren(w2)} × ${paren(x2, 1)} + ${paren(S2.b)} = ${fx(z)}\nŷ = ${S2.act === "sig" ? `σ(${fx(z)}) = 1 ÷ (1 + e^${paren(-z)}) = ${fx(a, 3)}` : `step(${fx(z)}) = ${a}  (1 if z > 0, else 0)`}`);
}
bindRange("i2x1", "o2x1", v => fx(v, 1), v => { S2.x[0] = v; upd2(); });
bindRange("i2x2", "o2x2", v => fx(v, 1), v => { S2.x[1] = v; upd2(); });
bindRange("i2w1", "o2w1", v => fx(v), v => { S2.w[0] = v; upd2(); });
bindRange("i2w2", "o2w2", v => fx(v), v => { S2.w[1] = v; upd2(); });
bindRange("i2b", "o2b", v => fx(v), v => { S2.b = v; upd2(); });
bindSeg("g2a", v => { S2.act = v; upd2(); });

/* ================= 3절 / Section 3 ================= */
// perceptron 경계선: w 화살표 끝을 끌면 선이 돈다. 선은 언제나 w에 수직
// Perceptron boundary: drag the tip of the w arrow to turn the line. The line is always perpendicular to w
const DATA3 = { blobs: makeData("blobs", 60, makeRng(11), 0.11), xor: makeData("xor", 60, makeRng(12), 0.06) };
const S3 = { d: "blobs", w: [0.4, -1], b: 0.2, timer: 0, idx: 0, pick: -1, ups: 0, lay: null };
const pred3 = p => (S3.w[0] * p[0] + S3.w[1] * p[1] + S3.b > 0 ? 1 : 0);
const F3 = fig("cv3", w => Math.round(clamp(w * 0.62, 260, 400)), (ctx, w, h, c) => {
  const B = { l: 36, t: 12, w: w - 50, h: h - 42 }, R = equalRange(B, 1.7);
  S3.lay = { B, R };
  const D = DATA3[S3.d], A0 = { X: v => B.l + (v - R.xr[0]) / (R.xr[1] - R.xr[0]) * B.w, Y: v => B.t + B.h - (v - R.yr[0]) / (R.yr[1] - R.yr[0]) * B.h };
  drawField(ctx, c, B, R.xr, R.yr, (x, y) => { const p = sigm(6 * (S3.w[0] * x + S3.w[1] * y + S3.b)); return [1 - p, p]; }, { alpha: 0.3 });
  const A = axes(ctx, c, B, R.xr, R.yr, { xt: intTicks(R.xr), yt: [-1, 0, 1], grid: false, xlabel: "" });
  clipBox(ctx, B);
  // 경계선 w₁x + w₂y + b = 0
  // Boundary w₁x + w₂y + b = 0
  const [w1, w2] = S3.w, n2 = w1 * w1 + w2 * w2;
  if (n2 > 1e-9) {
    const fx0 = -S3.b * w1 / n2, fy0 = -S3.b * w2 / n2, dx = -w2, dy = w1, L = 10;
    line(ctx, [[A.X(fx0 - dx * L), A.Y(fy0 - dy * L)], [A.X(fx0 + dx * L), A.Y(fy0 + dy * L)]], c.ink, 2.2);
  }
  ctx.restore();
  drawPoints(ctx, c, A, D.X, D.Y);
  D.X.forEach((p, i) => { if (pred3(p) !== D.Y[i]) dot(ctx, A.X(p[0]), A.Y(p[1]), 7.5, null, c.fa); });
  if (S3.pick >= 0) dot(ctx, A.X(D.X[S3.pick][0]), A.Y(D.X[S3.pick][1]), 11, null, c.ink);
  arrow(ctx, A.X(0), A.Y(0), A.X(w1), A.Y(w2), c.vio, 2.6);
  dot(ctx, A.X(w1), A.Y(w2), 7, c.vio, c.bg);
  txtBg(ctx, c, "w", A.X(w1) + 10, A.Y(w2) - 8, { color: c.vio, size: 13, w: "700" });
  txtBg(ctx, c, tr("보라 점을 끌어 w를 바꾼다", "drag the violet dot to change w"), B.l + 6, B.t + 14, { color: c.vio, size: 11, w: "600" });
});
function errs3() { const D = DATA3[S3.d]; return D.X.reduce((s, p, i) => s + (pred3(p) !== D.Y[i] ? 1 : 0), 0); }
function upd3(note) {
  paint(F3);
  const e = errs3(), [w1, w2] = S3.w;
  $("r3").textContent = tr(
    `경계선: ${fx(w1)}·x₁ + ${paren(w2)}·x₂ + ${paren(S3.b)} = 0   (w = (${fx(w1)}, ${fx(w2)})에 수직)\n틀린 점 (빨간 테두리): ${e}개 / ${DATA3[S3.d].X.length}개   perceptron 고침: ${S3.ups}번${note ? "\n→ " + note : ""}`,
    `boundary: ${fx(w1)}·x₁ + ${paren(w2)}·x₂ + ${paren(S3.b)} = 0   (perpendicular to w = (${fx(w1)}, ${fx(w2)}))\nwrong points (red ring): ${e} of ${DATA3[S3.d].X.length}   perceptron updates: ${S3.ups}${note ? "\n→ " + note : ""}`);
}
function reset3() { stopRun(S3); S3.w = [0.4, -1]; S3.b = 0.2; S3.idx = 0; S3.pick = -1; S3.ups = 0; setRange("i3b", S3.b); upd3(); }
// perceptron 학습 규칙: 틀린 점을 만나면 w ← w + η(y − ŷ)x, b ← b + η(y − ŷ)
// Perceptron learning rule: on a wrong point, w ← w + η(y − ŷ)x, b ← b + η(y − ŷ)
function learn3() {
  const D = DATA3[S3.d], n = D.X.length; let clean = 0;
  runSteps(S3, 400, () => {
    for (let k = 0; k < n; k++) {
      const i = (S3.idx + k) % n, p = D.X[i], e = D.Y[i] - pred3(p);
      if (e !== 0) {
        S3.w = [S3.w[0] + 0.3 * e * p[0], S3.w[1] + 0.3 * e * p[1]]; S3.b += 0.3 * e; S3.ups++;
        S3.idx = (i + 1) % n; S3.pick = i; setRange("i3b", clamp(S3.b, -2, 2)); upd3(); return S3.ups < 60;
      }
    }
    S3.pick = -1; upd3(tr("틀린 점이 없다: 직선 하나로 나뉜다", "no wrong points: one straight line separates them")); return false;
  }, 260, () => { if (errs3() > 0) upd3(tr("60번 고쳐도 틀린 점이 남는다: 이 data는 직선 하나로 나눌 수 없다 (XOR). 그래서 층을 쌓는다", "still wrong after 60 updates: no single straight line separates this data (XOR). That is why we stack layers")); });
}
bindRange("i3b", "o3b", v => fx(v), v => { stopRun(S3); S3.b = v; upd3(); });
bindSeg("g3d", v => { S3.d = v; reset3(); });
$("b3learn").addEventListener("click", learn3);
$("b3reset").addEventListener("click", reset3);
dragOn(F3, (px, py) => {
  const L = S3.lay; if (!L) return null;
  const x = L.R.xr[0] + (px - L.B.l) / L.B.w * (L.R.xr[1] - L.R.xr[0]), y = L.R.yr[1] - (py - L.B.t) / L.B.h * (L.R.yr[1] - L.R.yr[0]);
  return Math.hypot(x - S3.w[0], y - S3.w[1]) < 0.35 ? 1 : null;
}, (k, px, py) => {
  const L = S3.lay; stopRun(S3);
  S3.w = [L.R.xr[0] + (px - L.B.l) / L.B.w * (L.R.xr[1] - L.R.xr[0]), L.R.yr[1] - (py - L.B.t) / L.B.h * (L.R.yr[1] - L.R.yr[0])].map(v => Math.round(clamp(v, -1.6, 1.6) * 20) / 20);
  S3.pick = -1; upd3();
});

/* ================= 4절, 6절 / Sections 4 and 6 ================= */
// 2-2-1 network: x → h = σ(W₁x + b₁) → ŷ = σ(W₂h + b₂). 4절은 앞으로, 6절은 거꾸로
// 2-2-1 network: x → h = σ(W₁x + b₁) → ŷ = σ(W₂h + b₂). Section 4 goes forward, section 6 goes backward
const NET = { x: [1, 0.5], y: 1, W1: [[0.4, -0.6], [0.7, 0.2]], b1: [0.1, -0.3], W2: [[0.9, -0.5]], b2: [0.2] };
(function () {
  const N = NET;
  N.z1 = N.W1.map((r, j) => r[0] * N.x[0] + r[1] * N.x[1] + N.b1[j]); N.h = N.z1.map(sigm);
  N.z2 = [N.W2[0][0] * N.h[0] + N.W2[0][1] * N.h[1] + N.b2[0]]; N.yh = N.z2.map(sigm);
  N.L = -Math.log(N.yh[0]); N.d2 = [N.yh[0] - N.y];
  N.gW2 = [N.h.map(v => N.d2[0] * v)]; N.back = [N.W2[0][0] * N.d2[0], N.W2[0][1] * N.d2[0]];
  N.sp = N.h.map(v => v * (1 - v)); N.d1 = N.back.map((v, j) => v * N.sp[j]);
  N.gW1 = N.d1.map(d => N.x.map(v => d * v));
})();
// 망 그림과 그 단계의 행렬 계산을 함께 그린다
// Draw the network and that step's matrix arithmetic together
function netPanel(ctx, c, w, h, o) {
  const wide = w >= 640, NB = wide ? { l: 44, t: 30, w: w * 0.42 - 60, h: h - 60 } : { l: 44, t: 30, w: w - 90, h: 190 };
  const pos = drawNet(ctx, c, [2, 2, 1], NB, { r: 25, W: [NET.W1, NET.W2], wScale: 2.2, vals: o.vals, dp: 2, valSize: 10.5, hl: o.hl, fill: o.fill });
  [["x₁", 0, 0], ["x₂", 0, 1], ["h₁", 1, 0], ["h₂", 1, 1], ["ŷ", 2, 0]].forEach(([s, l, i]) => txt(ctx, s, pos[l][i][0], pos[l][i][1] - 31, { color: c.muted, size: 11.5, align: "center" }));
  if (o.edgeLab) o.edgeLab(pos);
  const mx = wide ? w * 0.46 : 20, my = wide ? 40 : NB.t + NB.h + 40;
  if (o.mats) o.mats(mx, my);
  if (o.title) panelTitle(ctx, c, o.title, 12, 16);
}
const S4 = { k: 0, timer: 0 };
const STEPS4 = 4;
const F4 = fig("cv4", w => (w >= 640 ? 270 : 420), (ctx, w, h, c) => {
  const k = S4.k, N = NET;
  const vals = [N.x, k >= 2 ? N.h : k >= 1 ? N.z1 : [null, null], k >= 4 ? N.yh : k >= 3 ? N.z2 : [null]];
  netPanel(ctx, c, w, h, {
    vals: vals.map(v => v.map(u => (u === null ? "" : u))), valSize: 10,
    hl: (l, j, i) => (k === 0 ? null : (k <= 2 && l === 1) || (k >= 3 && l === 2)),
    fill: (l, i) => (l === 1 && k >= 1 && k <= 2) || (l === 2 && k >= 3) ? c.sigwash : c.bg,
    edgeLab: pos => {
      const y = pos[0][1][1] + 44;
      if (k >= 1) { arrowS(ctx, pos[0][0][0] + 8, y, pos[1][0][0] - 8, y, c.ok, 2, 8); txtBg(ctx, c, "W₁", (pos[0][0][0] + pos[1][0][0]) / 2, y - 6, { color: c.ok, size: 11, w: "700", align: "center" }); }
      if (k >= 3) { arrowS(ctx, pos[1][0][0] + 8, y, pos[2][0][0] - 8, y, c.ok, 2, 8); txtBg(ctx, c, "W₂", (pos[1][0][0] + pos[2][0][0]) / 2, y - 6, { color: c.ok, size: 11, w: "700", align: "center" }); }
    },
    title: [tr("0. 입력 x", "0. Input x"), tr("1. z₁ = W₁x + b₁", "1. z₁ = W₁x + b₁"), tr("2. h = σ(z₁)", "2. h = σ(z₁)"), tr("3. z₂ = W₂h + b₂", "3. z₂ = W₂h + b₂"), tr("4. ŷ = σ(z₂)", "4. ŷ = σ(z₂)")][k],
    mats: (x, y) => {
      const op = (s, xx) => txt(ctx, s, xx, y + 30, { color: c.muted, size: 15, align: "center", w: "600" });
      if (k === 0) { const m = drawMat(ctx, c, N.x.map(v => [v]), x, y, { label: "x (2×1)", col: c.ink }); txt(ctx, tr("입력 2개 = shape (2, 1)", "two inputs = shape (2, 1)"), x + m.w + 16, y + 30, { color: c.muted, size: 11.5 }); }
      if (k === 1 || k === 2) {
        let xx = x; xx += drawMat(ctx, c, N.W1, xx, y, { label: "W₁ (2×2)", col: c.sig }).w + 24; op("·", xx - 12);
        xx += drawMat(ctx, c, N.x.map(v => [v]), xx, y, { label: "x (2×1)" }).w + 24; op("+", xx - 12);
        xx += drawMat(ctx, c, N.b1.map(v => [v]), xx, y, { label: "b₁ (2×1)", col: c.vio }).w + 24; op("=", xx - 12);
        xx += drawMat(ctx, c, N.z1.map(v => [v]), xx, y, { label: "z₁ (2×1)", col: c.ink, hl: () => k === 1 }).w + 30;
        if (k === 2) { op("σ→", xx - 12); drawMat(ctx, c, N.h.map(v => [v]), xx + 12, y, { label: "h (2×1)", col: c.ok, hl: () => true, dp: 4 }); }
      }
      if (k === 3 || k === 4) {
        let xx = x; xx += drawMat(ctx, c, N.W2, xx, y + 12, { label: "W₂ (1×2)", col: c.sig }).w + 24; op("·", xx - 12);
        xx += drawMat(ctx, c, N.h.map(v => [v]), xx, y, { label: "h (2×1)", dp: 4 }).w + 24; op("+", xx - 12);
        xx += drawMat(ctx, c, [N.b2], xx, y + 12, { label: "b₂", col: c.vio }).w + 24; op("=", xx - 12);
        xx += drawMat(ctx, c, [N.z2], xx, y + 12, { label: "z₂ (1×1)", hl: () => k === 3, dp: 4 }).w + 30;
        if (k === 4) { op("σ→", xx - 12); drawMat(ctx, c, [N.yh], xx + 12, y + 12, { label: "ŷ", col: c.ok, hl: () => true, dp: 4 }); }
      }
    }
  });
});
function upd4() {
  paint(F4);
  const N = NET, k = S4.k;
  const lines = [
    tr(`x = (${N.x.join(", ")}), 정답 y = 1`, `x = (${N.x.join(", ")}), answer y = 1`),
    `z₁ = (0.4·1 − 0.6·0.5 + 0.1, 0.7·1 + 0.2·0.5 − 0.3) = (${fx(N.z1[0])}, ${fx(N.z1[1])})`,
    `h = σ(z₁) = (σ(${fx(N.z1[0])}), σ(${fx(N.z1[1])})) = (${fx(N.h[0], 4)}, ${fx(N.h[1], 4)})`,
    `z₂ = 0.9 × ${fx(N.h[0], 4)} − 0.5 × ${fx(N.h[1], 4)} + 0.2 = ${fx(N.z2[0], 4)}`,
    tr(`ŷ = σ(${fx(N.z2[0], 4)}) = ${fx(N.yh[0], 4)}  → 정답 1에 59% 확신`, `ŷ = σ(${fx(N.z2[0], 4)}) = ${fx(N.yh[0], 4)}  → 59% sure of the answer 1`)
  ];
  $("r4").textContent = lines.slice(0, k + 1).join("\n");
  $("b4next").disabled = k >= STEPS4;
}
$("b4next").addEventListener("click", () => { stopRun(S4); if (S4.k < STEPS4) { S4.k++; upd4(); } });
$("b4reset").addEventListener("click", () => { stopRun(S4); S4.k = 0; upd4(); });
$("b4play").addEventListener("click", () => { S4.k = 0; upd4(); runSteps(S4, STEPS4, () => { S4.k++; upd4(); }, 1100); });

/* ================= 5절 / Section 5 ================= */
// 왼쪽 = BCE loss 곡선 두 개 (정답 1, 정답 0). 오른쪽 = sample 4개의 loss와 그 평균(cost)
// Left = the two BCE loss curves (answer 1, answer 0). Right = the loss of four samples and their mean (cost)
const S5 = { p: 0.5947, y: 1 };
const OTHERS5 = [[0, 0.2], [1, 0.9], [0, 0.7]];
const bce = (y, p) => -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
const F5 = fig("cv5", w => (stacked(w, 280, 2, 28) ? 560 : 290), (ctx, w, h, c) => {
  const P = panels(w, h, 2, 28, 280);
  {
    const p = P[0], B = { l: p.x + 40, t: p.y + 30, w: p.w - 52, h: p.h - 76 };
    const A = axes(ctx, c, B, [0, 1], [0, 4.6], { xt: [0, 0.25, 0.5, 0.75, 1], yt: [0, 1, 2, 3, 4], xlabel: tr("예측 ŷ", "prediction ŷ") });
    panelTitle(ctx, c, tr("한 sample의 loss (BCE)", "Loss of one sample (BCE)"), B.l, p.y + 14);
    clipBox(ctx, B);
    line(ctx, curve(A, q => -Math.log(q), [0.008, 1]), c.sig, S5.y === 1 ? 2.8 : 1.4);
    line(ctx, curve(A, q => -Math.log(1 - q), [0, 0.992]), c.miss, S5.y === 0 ? 2.8 : 1.4);
    ctx.restore();
    txtBg(ctx, c, tr("정답 1: −ln ŷ", "answer 1: −ln ŷ"), A.X(0.06), A.Y(3.9), { color: c.sig, size: 11.5, w: "700" });
    txtBg(ctx, c, tr("정답 0: −ln(1 − ŷ)", "answer 0: −ln(1 − ŷ)"), A.X(0.94), A.Y(3.9), { color: c.miss, size: 11.5, w: "700", align: "right" });
    const L = bce(S5.y, S5.p);
    dot(ctx, A.X(S5.p), A.Y(Math.min(L, 4.6)), 6.5, S5.y ? c.sig : c.miss, c.bg);
    line(ctx, [[A.X(S5.p), A.Y(0)], [A.X(S5.p), A.Y(Math.min(L, 4.6))]], c.muted, 1, [3, 3]);
    line(ctx, [[A.X(S5.y), B.t + B.h - 4], [A.X(S5.p), B.t + B.h - 4]], c.fa, 4);
    txtBg(ctx, c, tr(`오차 ŷ − y = ${fx(S5.p - S5.y)}`, `error ŷ − y = ${fx(S5.p - S5.y)}`), (A.X(S5.y) + A.X(S5.p)) / 2, B.t + B.h - 12, { color: c.fa, size: 11, w: "700", align: "center" });
  }
  {
    const p = P[1], B = { l: p.x + 40, t: p.y + 30, w: p.w - 52, h: p.h - 76 }, S = [[S5.y, S5.p], ...OTHERS5], L = S.map(([y, q]) => bce(y, q)), cost = L.reduce((a, b) => a + b, 0) / L.length;
    const top = Math.max(2, Math.max(...L) * 1.15);
    const A = axes(ctx, c, B, [0, 4], [0, top], { xt: [], yt: ticksR(0, top, 4) });
    panelTitle(ctx, c, tr("sample 4개의 loss와 cost(평균)", "Loss of 4 samples and the cost (mean)"), B.l, p.y + 14);
    const bw = B.w / 4 * 0.55;
    S.forEach(([y, q], i) => {
      const x = A.X(i + 0.5), yy = A.Y(L[i]);
      ctx.fillStyle = y ? c.sig : c.miss; ctx.globalAlpha = i === 0 ? 0.95 : 0.55; ctx.fillRect(x - bw / 2, yy, bw, B.t + B.h - yy); ctx.globalAlpha = 1;
      txt(ctx, fx(L[i], 2), x, yy - 5, { color: c.ink, size: 10.5, mono: true, align: "center" });
      txt(ctx, `y=${y}, ŷ=${fx(q, 2)}`, x, B.t + B.h + 14, { color: i === 0 ? c.ink : c.muted, size: 10, align: "center", mono: true });
    });
    line(ctx, [[B.l, A.Y(cost)], [B.l + B.w, A.Y(cost)]], c.ok, 2, [6, 4]);
    txtBg(ctx, c, `cost = ${fx(cost, 3)}`, B.l + B.w / 2, A.Y(cost) - 6, { color: c.ok, size: 11.5, w: "700", align: "center" });
  }
});
function upd5() {
  paint(F5);
  const S = [[S5.y, S5.p], ...OTHERS5], L = S.map(([y, q]) => bce(y, q)), cost = L.reduce((a, b) => a + b, 0) / 4;
  $("r5").textContent = tr(
    `오차 (error) = ŷ − y = ${fx(S5.p, 3)} − ${S5.y} = ${fx(S5.p - S5.y, 3)}\nloss (이 sample) = ${S5.y ? `−ln ${fx(S5.p, 3)}` : `−ln(1 − ${fx(S5.p, 3)})`} = ${fx(L[0], 3)}\ncost (sample 4개 평균) = (${L.map(v => fx(v, 3)).join(" + ")}) ÷ 4 = ${fx(cost, 3)}`,
    `error = ŷ − y = ${fx(S5.p, 3)} − ${S5.y} = ${fx(S5.p - S5.y, 3)}\nloss (this sample) = ${S5.y ? `−ln ${fx(S5.p, 3)}` : `−ln(1 − ${fx(S5.p, 3)})`} = ${fx(L[0], 3)}\ncost (mean of 4 samples) = (${L.map(v => fx(v, 3)).join(" + ")}) ÷ 4 = ${fx(cost, 3)}`);
}
bindRange("i5p", "o5p", v => fx(v, 3), v => { S5.p = v; upd5(); });
bindSeg("g5y", v => { S5.y = +v; upd5(); });

/* ================= 6절 / Section 6 ================= */
const S6 = { k: 0, timer: 0 };
const STEPS6 = 4;
const F6 = fig("cv6", w => (w >= 640 ? 270 : 420), (ctx, w, h, c) => {
  const k = S6.k, N = NET;
  const vals = [N.x, k >= 3 ? N.d1.map(v => fx(v, 3)) : N.h, k >= 1 ? [fx(N.d2[0], 3)] : N.yh];
  netPanel(ctx, c, w, h, {
    vals: vals.map(v => v.map(u => (typeof u === "number" ? fx(u, 2) : u))), valSize: 9.5,
    hl: (l, j, i) => (k === 0 ? null : (k <= 2 && l === 2) || (k >= 3 && l === 1)),
    fill: (l, i) => (l === 2 && k >= 1) || (l === 1 && k >= 3) ? c.okwash : c.bg,
    title: [tr("0. forward 결과 ŷ = 0.595, 정답 y = 1", "0. Forward result ŷ = 0.595, answer y = 1"), tr("1. 출력의 δ₂ = ŷ − y", "1. Output δ₂ = ŷ − y"), tr("2. ∂L/∂W₂ = δ₂ · hᵀ", "2. ∂L/∂W₂ = δ₂ · hᵀ"), tr("3. 거꾸로: δ₁ = (W₂ᵀδ₂) ⊙ σ′(z₁)", "3. Going back: δ₁ = (W₂ᵀδ₂) ⊙ σ′(z₁)"), tr("4. ∂L/∂W₁ = δ₁ · xᵀ", "4. ∂L/∂W₁ = δ₁ · xᵀ")][k],
    edgeLab: pos => {
      const y = pos[0][1][1] + 44;
      if (k >= 1) { arrowS(ctx, pos[2][0][0] - 8, y, pos[1][0][0] + 8, y, c.fa, 2, 8); txtBg(ctx, c, "δ₂", (pos[1][0][0] + pos[2][0][0]) / 2, y - 6, { color: c.fa, size: 11, w: "700", align: "center" }); }
      if (k >= 3) { arrowS(ctx, pos[1][0][0] - 8, y, pos[0][0][0] + 8, y, c.fa, 2, 8); txtBg(ctx, c, "δ₁", (pos[0][0][0] + pos[1][0][0]) / 2, y - 6, { color: c.fa, size: 11, w: "700", align: "center" }); }
    },
    mats: (x, y) => {
      const op = (s, xx, yy = y + 30) => txt(ctx, s, xx, yy, { color: c.muted, size: 15, align: "center", w: "600" });
      if (k === 0) [tr("loss L = −ln 0.5947 = 0.520", "loss L = −ln 0.5947 = 0.520"), tr("이 loss를 줄이려면 weight를 어느 쪽으로", "Which way should each weight move"), tr("고쳐야 하나? 출력에서 거꾸로 계산한다.", "to lower it? Work back from the output.")].forEach((s, i) => txt(ctx, s, x, y + 16 + i * 20, { color: i ? c.muted : c.ink, size: 12, w: i ? "400" : "600" }));
      if (k === 1) { let xx = x; xx += drawMat(ctx, c, [N.yh], xx, y + 12, { label: "ŷ", dp: 4 }).w + 24; op("−", xx - 12, y + 42); xx += drawMat(ctx, c, [[1]], xx, y + 12, { label: "y", dp: 0 }).w + 24; op("=", xx - 12, y + 42); drawMat(ctx, c, [N.d2], xx, y + 12, { label: "δ₂", col: c.fa, hl: () => c.okwash, dp: 4 }); }
      if (k === 2) { let xx = x; xx += drawMat(ctx, c, [N.d2], xx, y + 12, { label: "δ₂ (1×1)", col: c.fa, dp: 4 }).w + 24; op("·", xx - 12, y + 42); xx += drawMat(ctx, c, [N.h], xx, y + 12, { label: "hᵀ (1×2)", dp: 4 }).w + 24; op("=", xx - 12, y + 42); drawMat(ctx, c, N.gW2, xx, y + 12, { label: "∂L/∂W₂ (1×2)", col: c.fa, hl: () => c.okwash, dp: 4, cw: 58 }); }
      if (k === 3) {
        let xx = x; xx += drawMat(ctx, c, N.back.map(v => [v]), xx, y, { label: "W₂ᵀδ₂", dp: 4, cw: 58 }).w + 24; op("⊙", xx - 12);
        xx += drawMat(ctx, c, N.sp.map(v => [v]), xx, y, { label: "σ′(z₁) = h(1−h)", dp: 4, cw: 58 }).w + 24; op("=", xx - 12);
        drawMat(ctx, c, N.d1.map(v => [v]), xx, y, { label: "δ₁ (2×1)", col: c.fa, hl: () => c.okwash, dp: 4, cw: 58 });
      }
      if (k === 4) {
        let xx = x; xx += drawMat(ctx, c, N.d1.map(v => [v]), xx, y, { label: "δ₁ (2×1)", col: c.fa, dp: 4, cw: 58 }).w + 24; op("·", xx - 12);
        xx += drawMat(ctx, c, [N.x], xx, y + 12, { label: "xᵀ (1×2)", dp: 1 }).w + 24; op("=", xx - 12);
        drawMat(ctx, c, N.gW1, xx, y, { label: "∂L/∂W₁ (2×2)", col: c.fa, hl: () => c.okwash, dp: 4, cw: 58 });
      }
    }
  });
});
function upd6() {
  paint(F6);
  const N = NET, k = S6.k;
  const lines = [
    `L = −ln ŷ = −ln ${fx(N.yh[0], 4)} = ${fx(N.L, 4)}`,
    tr(`δ₂ = ŷ − y = ${fx(N.yh[0], 4)} − 1 = ${fx(N.d2[0], 4)}   (음수 → ŷ를 키워야 한다)`, `δ₂ = ŷ − y = ${fx(N.yh[0], 4)} − 1 = ${fx(N.d2[0], 4)}   (negative → ŷ should go up)`),
    `∂L/∂W₂ = δ₂ · hᵀ = ${fx(N.d2[0], 4)} × (${fx(N.h[0], 4)}, ${fx(N.h[1], 4)}) = (${fx(N.gW2[0][0], 4)}, ${fx(N.gW2[0][1], 4)})`,
    `δ₁ = (W₂ᵀδ₂) ⊙ h(1 − h) = (${fx(N.back[0], 4)}, ${fx(N.back[1], 4)}) ⊙ (${fx(N.sp[0], 4)}, ${fx(N.sp[1], 4)}) = (${fx(N.d1[0], 4)}, ${fx(N.d1[1], 4)})`,
    `∂L/∂W₁ = δ₁ · xᵀ = [[${fx(N.gW1[0][0], 4)}, ${fx(N.gW1[0][1], 4)}], [${fx(N.gW1[1][0], 4)}, ${fx(N.gW1[1][1], 4)}]]`
  ];
  $("r6").textContent = lines.slice(0, k + 1).join("\n");
  $("b6next").disabled = k >= STEPS6;
}
$("b6next").addEventListener("click", () => { stopRun(S6); if (S6.k < STEPS6) { S6.k++; upd6(); } });
$("b6reset").addEventListener("click", () => { stopRun(S6); S6.k = 0; upd6(); });
$("b6play").addEventListener("click", () => { S6.k = 0; upd6(); runSteps(S6, STEPS6, () => { S6.k++; upd6(); }, 1300); });

/* ================= 7절 / Section 7 ================= */
// browser 안에서 학습: 왼쪽 = 경계 (점선 = 숨은 neuron 하나하나의 선), 오른쪽 = epoch마다 loss
// Training in the browser: left = the boundary (dashed = each hidden neuron's own line), right = the loss at each epoch
const DATA7 = { blobs: makeData("blobs", 120, makeRng(21), 0.08), xor: makeData("xor", 120, makeRng(21), 0.08), circles: makeData("circles", 120, makeRng(21), 0.08) };
const S7 = { d: "xor", h: 4, eta: 1.5, lines: true, timer: 0, net: null, hist: [], ep: 0 };
function new7() { S7.net = new TinyNet([2, S7.h, 1], { hidden: "sigmoid", seed: 3, opt: { name: "sgd", lr: S7.eta } }); S7.hist = [S7.net.loss(DATA7[S7.d].X, DATA7[S7.d].Y)]; S7.ep = 0; }
const F7 = fig("cv7", w => (w >= 640 ? 330 : 640), (ctx, w, h, c) => {
  const wide = w >= 640, side = wide ? Math.min(h - 40, w * 0.52) : Math.min(w - 50, 330);
  const B = { l: 36, t: 24, w: side, h: side }, R = { xr: [-1.6, 1.6], yr: [-1.6, 1.6] }, D = DATA7[S7.d], net = S7.net;
  drawField(ctx, c, B, R.xr, R.yr, (x, y) => net.probs([x, y]), { alpha: 0.4 });
  const A = axes(ctx, c, B, R.xr, R.yr, { xt: [-1, 0, 1], yt: [-1, 0, 1], grid: false });
  panelTitle(ctx, c, tr(`경계 (숨은 neuron ${S7.h}개)`, `Boundary (${S7.h} hidden neurons)`), B.l, 14);
  if (S7.lines) {
    clipBox(ctx, B);
    net.W[0].forEach((r, j) => {
      const [a, b] = r, cc = net.b[0][j], n2 = a * a + b * b; if (n2 < 1e-9) return;
      const fx0 = -cc * a / n2, fy0 = -cc * b / n2, L = 10;
      line(ctx, [[A.X(fx0 + b * L), A.Y(fy0 - a * L)], [A.X(fx0 - b * L), A.Y(fy0 + a * L)]], c.ink, 1.2, [5, 4]);
    });
    ctx.restore();
  }
  drawPoints(ctx, c, A, D.X, D.Y, { r: 3.2 });
  const C = wide ? { l: B.l + B.w + 64, t: 24, w: w - (B.l + B.w + 64) - 14, h: side } : { l: 44, t: B.t + B.h + 50, w: w - 58, h: h - (B.t + B.h + 50) - 40 };
  const n = Math.max(100, S7.hist.length), top = Math.max(0.8, ...S7.hist) * 1.05;
  const A2 = axes(ctx, c, C, [0, n], [0, top], { xt: ticksR(0, n, 4), yt: ticksR(0, top, 4), xlabel: "epoch" });
  panelTitle(ctx, c, tr("epoch마다 loss", "Loss per epoch"), C.l, C.t - 10);
  line(ctx, S7.hist.map((v, i) => [A2.X(i), A2.Y(v)]), c.fa, 2);
});
function upd7() {
  paint(F7);
  const D = DATA7[S7.d], net = S7.net, L = S7.hist[S7.hist.length - 1], acc = net.accuracy(D.X, D.Y);
  $("r7").textContent = tr(
    `epoch ${S7.ep}: loss ${fx(L, 3)}, 정확도 ${Math.round(acc * 100)}%\nnetwork 2-${S7.h}-1, parameter ${4 * S7.h + 1}개 (W₁ ${2 * S7.h} + b₁ ${S7.h} + W₂ ${S7.h} + b₂ 1), η = ${fx(S7.eta)}`,
    `epoch ${S7.ep}: loss ${fx(L, 3)}, accuracy ${Math.round(acc * 100)}%\nnetwork 2-${S7.h}-1, ${4 * S7.h + 1} parameters (W₁ ${2 * S7.h} + b₁ ${S7.h} + W₂ ${S7.h} + b₂ 1), η = ${fx(S7.eta)}`);
  $("b7run").textContent = S7.timer ? tr("멈춤", "Pause") : tr("학습", "Train");
}
function train7(n) { const D = DATA7[S7.d]; for (let i = 0; i < n; i++) { S7.hist.push(S7.net.step(D.X, D.Y)); S7.ep++; } S7.hist[S7.hist.length - 1] = S7.net.loss(D.X, D.Y); }
function reset7() { stopRun(S7); new7(); upd7(); }
$("b7run").addEventListener("click", () => {
  if (S7.timer) { stopRun(S7); upd7(); return; }
  runSteps(S7, 300, () => { train7(10); upd7(); if (S7.ep >= 3000) return false; }, 40, upd7); upd7();
});
$("b7ep").addEventListener("click", () => { stopRun(S7); train7(100); upd7(); });
$("b7reset").addEventListener("click", reset7);
bindSeg("g7d", v => { S7.d = v; reset7(); });
bindSeg("g7l", v => { S7.lines = v === "on"; paint(F7); });
bindRange("i7h", "o7h", v => String(v), v => { S7.h = v; reset7(); });
bindRange("i7e", "o7e", v => fx(v, 1), v => { S7.eta = v; S7.net.opt.lr = v; upd7(); });

// 처음 그리기
// First paint
upd2(); upd3(); upd4(); upd5(); upd6(); new7(); upd7();
repaintAll();
