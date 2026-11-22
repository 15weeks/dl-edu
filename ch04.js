/* 4장 그림과 조작. 한글판(ch04-ann-practice-KR.html)과 영어판(ch04-ann-practice.html)이 같이 쓴다
   Chapter 4 figures and interactions, shared by the Korean and English pages
   미리 돌린 실험 결과와 iris data는 assets/data/ch04_data.js (tools/ch04_numbers.py가 만든다)
   Precomputed experiments and the iris data live in assets/data/ch04_data.js (made by tools/ch04_numbers.py) */
"use strict";

const D4 = CH04_DATA;
function corr(a, b) { const ma = meanOf(a), mb = meanOf(b); let sab = 0, saa = 0, sbb = 0; a.forEach((v, i) => { sab += (v - ma) * (b[i] - mb); saa += (v - ma) ** 2; sbb += (b[i] - mb) ** 2; }); return sab / Math.sqrt(saa * sbb || 1); }

/* ================= 1절 / Section 1 ================= */
// 회귀: y = m·x + 잡음에 neuron 하나(ŷ = w·x + b)를 gradient descent로 맞춘다
// Regression: fit one neuron (ŷ = w·x + b) to y = m·x + noise with gradient descent
const R1 = makeRng(4), X1 = [], E1 = [];
for (let i = 0; i < 60; i++) { X1.push(R1.n()); E1.push(R1.n()); }
const S1 = { m: 1, s: 0.5, w: 0, b: 0, hist: [], timer: 0 };
const ys1 = () => X1.map((x, i) => S1.m * x + S1.s * E1[i]);
const mse1 = () => { const y = ys1(); return meanOf(X1.map((x, i) => (S1.w * x + S1.b - y[i]) ** 2)); };
function step1() {
  const y = ys1(); let gw = 0, gb = 0;
  X1.forEach((x, i) => { const e = S1.w * x + S1.b - y[i]; gw += 2 * e * x; gb += 2 * e; });
  S1.w -= 0.1 * gw / X1.length; S1.b -= 0.1 * gb / X1.length; S1.hist.push(mse1());
}
const F1 = fig("cv1", w => (stacked(w, 260, 2, 28) ? 560 : 290), (ctx, w, h, c) => {
  const P = panels(w, h, 2, 28, 260), y = ys1();
  {
    const p = P[0], B = { l: p.x + 40, t: p.y + 30, w: p.w - 52, h: p.h - 76 };
    const A = axes(ctx, c, B, [-3, 3], [-6, 6], { xt: [-3, -2, -1, 0, 1, 2, 3], yt: [-6, -3, 0, 3, 6], xlabel: "x" });
    panelTitle(ctx, c, tr("data와 neuron 하나의 직선", "Data and one neuron's line"), B.l, p.y + 14);
    clipBox(ctx, B);
    line(ctx, [[A.X(-3), A.Y(-3 * S1.m)], [A.X(3), A.Y(3 * S1.m)]], c.muted, 1.4, [6, 4]);
    line(ctx, [[A.X(-3), A.Y(-3 * S1.w + S1.b)], [A.X(3), A.Y(3 * S1.w + S1.b)]], c.sig, 2.6);
    X1.forEach((x, i) => dot(ctx, A.X(x), A.Y(y[i]), 3.4, c.miss));
    ctx.restore();
    txtBg(ctx, c, tr("점선 = 진짜 기울기 m, 파랑 = 학습한 직선", "dashed = true slope m, blue = learned line"), B.l + 6, B.t + 13, { color: c.muted, size: 11 });
  }
  {
    const p = P[1], B = { l: p.x + 40, t: p.y + 30, w: p.w - 52, h: p.h - 76 }, n = Math.max(60, S1.hist.length);
    const top = Math.max(1, ...S1.hist, mse1()) * 1.05;
    const A = axes(ctx, c, B, [0, n], [0, top], { xt: ticksR(0, n, 4), yt: ticksR(0, top, 4), xlabel: "epoch" });
    panelTitle(ctx, c, tr("epoch마다 MSE", "MSE per epoch"), B.l, p.y + 14);
    if (S1.hist.length) line(ctx, S1.hist.map((v, i) => [A.X(i + 1), A.Y(v)]), c.fa, 2);
  }
});
function upd1() {
  paint(F1);
  const y = ys1(), yh = X1.map(x => S1.w * x + S1.b), r = corr(yh, y), L = mse1();
  const note = !S1.hist.length ? tr("학습을 누르면 시작", "press Train to start") : Math.abs(r) < 0.5 ? tr("MSE는 작아도 r이 작다: x로 y를 거의 설명하지 못한다", "the MSE is small but r is small too: x barely explains y") : tr("r이 크다: 직선이 data를 잘 설명한다", "r is large: the line explains the data well");
  $("r1").textContent = tr(
    `진짜: y = ${fx(S1.m, 1)}·x + ${fx(S1.s, 1)}·잡음   학습: ŷ = ${fx(S1.w)}·x + ${paren(S1.b)}   (${S1.hist.length} epoch)\nMSE = ${fx(L, 3)},  r(ŷ, y) = ${fx(Math.abs(S1.hist.length ? r : 0), 2)}\n→ ${note}`,
    `true: y = ${fx(S1.m, 1)}·x + ${fx(S1.s, 1)}·noise   learned: ŷ = ${fx(S1.w)}·x + ${paren(S1.b)}   (${S1.hist.length} epochs)\nMSE = ${fx(L, 3)},  r(ŷ, y) = ${fx(Math.abs(S1.hist.length ? r : 0), 2)}\n→ ${note}`);
}
function reset1() { stopRun(S1); S1.w = 0; S1.b = 0; S1.hist = []; upd1(); }
$("b1run").addEventListener("click", () => { reset1(); runSteps(S1, 40, () => { step1(); step1(); upd1(); }, 50); });
$("b1reset").addEventListener("click", reset1);
bindRange("i1m", "o1m", v => fx(v, 1), v => { S1.m = v; reset1(); });
bindRange("i1s", "o1s", v => fx(v, 1), v => { S1.s = v; reset1(); });

/* ================= 2절 / Section 2 ================= */
// qwerties: 두 무리 (중심 (1, 1)과 (1 + d, 1), 퍼짐 1). d를 바꾸면 그 자리에서 다시 학습
// Qwerties: two groups (centres (1, 1) and (1 + d, 1), spread 1). Changing d retrains on the spot
function qwerties(d, seed = 31, n = 200) {
  const r = makeRng(seed), X = [], Y = [];
  for (let i = 0; i < n; i++) { const k = i % 2; X.push([(k ? 1 + d : 1) + r.n(), 1 + r.n()]); Y.push(k); }
  return { X, Y };
}
function trainQ(d) {
  const D = qwerties(d), net = new TinyNet([2, 8, 1], { hidden: "tanh", seed: 5, opt: { name: "adam", lr: 0.05 } });
  for (let e = 0; e < 150; e++) net.step(D.X, D.Y);
  return { D, net, acc: net.accuracy(D.X, D.Y) };
}
const DS2 = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6], ACC2 = DS2.map(d => trainQ(d).acc);
const S2 = { d: 4, cur: null };
const F2 = fig("cv2", w => (stacked(w, 260, 2, 28) ? 580 : 300), (ctx, w, h, c) => {
  const P = panels(w, h, 2, 28, 260), { D, net } = S2.cur;
  {
    const p = P[0], B = { l: p.x + 36, t: p.y + 30, w: p.w - 46, h: p.h - 70 }, xr = [-2.5, 10], yr = [-2.5, 4.5];
    drawField(ctx, c, B, xr, yr, (x, y) => net.probs([x, y]), { alpha: 0.35 });
    const A = axes(ctx, c, B, xr, yr, { xt: [0, 2, 4, 6, 8, 10], yt: [-2, 0, 2, 4], grid: false });
    panelTitle(ctx, c, tr(`두 무리, 중심 거리 d = ${fx(S2.d, 1)}`, `Two groups, centre distance d = ${fx(S2.d, 1)}`), B.l, p.y + 14);
    drawPoints(ctx, c, A, D.X, D.Y, { r: 3 });
  }
  {
    const p = P[1], B = { l: p.x + 44, t: p.y + 30, w: p.w - 56, h: p.h - 70 };
    const A = axes(ctx, c, B, [0, 6.5], [0.4, 1.02], { xt: [0, 1, 2, 3, 4, 5, 6], yt: [[0.5, "50%"], [0.75, "75%"], [1, "100%"]], xlabel: tr("거리 d", "distance d") });
    panelTitle(ctx, c, tr("거리에 따른 정확도", "Accuracy vs distance"), B.l, p.y + 14);
    line(ctx, DS2.map((d, i) => [A.X(d), A.Y(ACC2[i])]), c.sig, 2);
    DS2.forEach((d, i) => dot(ctx, A.X(d), A.Y(ACC2[i]), 3, c.sig));
    dot(ctx, A.X(S2.d), A.Y(S2.cur.acc), 6.5, c.miss, c.bg);
  }
});
function upd2() {
  S2.cur = trainQ(S2.d); paint(F2);
  $("r2").textContent = tr(
    `d = ${fx(S2.d, 1)}: network 2-8-1, Adam 150 epoch → 정확도 ${Math.round(S2.cur.acc * 100)}%\n${S2.d <= 1.5 ? "두 무리가 겹친다: 어떤 경계로도 다 맞힐 수 없다" : S2.d >= 4 ? "두 무리가 멀다: 거의 다 맞힌다" : "조금 겹친다: 경계 근처 점은 틀린다"}`,
    `d = ${fx(S2.d, 1)}: network 2-8-1, Adam 150 epochs → accuracy ${Math.round(S2.cur.acc * 100)}%\n${S2.d <= 1.5 ? "the groups overlap: no boundary can get them all" : S2.d >= 4 ? "the groups are far apart: almost all correct" : "they overlap a little: points near the boundary are wrong"}`);
}
bindRange("i2d", "o2d", v => fx(v, 1), v => { S2.d = v; upd2(); });

/* ================= 3절 / Section 3 ================= */
// learning rate 실험: PyTorch로 미리 돌린 결과(점 = seed 5개) + 고른 learning rate를 지금 browser에서 한 번
// Learning-rate experiment: results precomputed in PyTorch (dots = 5 seeds) + one live run in the browser at the chosen rate
const LR3 = D4.lrExp, QD = qwerties(4);
const S3 = { i: 14, live: null };
function live3() {
  const lr = LR3.lrs[S3.i], net = new TinyNet([2, 16, 1], { hidden: "relu", seed: 2, opt: { name: "sgd", lr } }), hist = [];
  for (let e = 0; e < 1000; e++) { const L = net.step(QD.X, QD.Y); hist.push(isFinite(L) ? L : NaN); }
  S3.live = { hist, acc: net.accuracy(QD.X, QD.Y) };
}
const F3 = fig("cv3", w => (stacked(w, 260, 2, 28) ? 580 : 300), (ctx, w, h, c) => {
  const P = panels(w, h, 2, 28, 260);
  {
    const p = P[0], B = { l: p.x + 44, t: p.y + 30, w: p.w - 56, h: p.h - 76 }, lx = v => Math.log10(v);
    const A = axes(ctx, c, B, [-3.1, 0.6], [0.4, 1.02], { xt: [[-3, "0.001"], [-2, "0.01"], [-1, "0.1"], [0, "1"]], yt: [[0.5, "50%"], [0.75, "75%"], [1, "100%"]], xlabel: "learning rate (log)" });
    panelTitle(ctx, c, tr("PyTorch 실험: 1000 epoch 뒤 정확도", "PyTorch experiment: accuracy after 1000 epochs"), B.l, p.y + 14);
    const mean = LR3.acc.map(r => meanOf(r));
    line(ctx, LR3.lrs.map((v, i) => [A.X(lx(v)), A.Y(mean[i])]), c.sig, 2);
    LR3.acc.forEach((row, i) => row.forEach((a, k) => dot(ctx, A.X(lx(LR3.lrs[i])) + (k - 2) * 1.6, A.Y(a), 2.4, c.muted)));
    const x = A.X(lx(LR3.lrs[S3.i]));
    line(ctx, [[x, B.t], [x, B.t + B.h]], c.miss, 1.4, [4, 3]);
    dot(ctx, x, A.Y(mean[S3.i]), 6, c.miss, c.bg);
  }
  {
    const p = P[1], B = { l: p.x + 44, t: p.y + 30, w: p.w - 56, h: p.h - 76 }, H = S3.live.hist, top = Math.max(0.8, ...H.filter(isFinite).slice(0, 5));
    const A = axes(ctx, c, B, [0, 1000], [0, top], { xt: [0, 250, 500, 750, 1000], yt: ticksR(0, top, 4), xlabel: "epoch" });
    panelTitle(ctx, c, tr(`지금 browser에서: lr = ${sci(LR3.lrs[S3.i])}의 loss`, `Live in the browser: loss at lr = ${sci(LR3.lrs[S3.i])}`), B.l, p.y + 14);
    clipBox(ctx, B); line(ctx, H.map((v, i) => [A.X(i), A.Y(isFinite(v) ? v : top)]), c.fa, 1.8); ctx.restore();
  }
});
function upd3() {
  live3(); paint(F3);
  const lr = LR3.lrs[S3.i], m = meanOf(LR3.acc[S3.i]), L = S3.live.hist[S3.live.hist.length - 1];
  $("r3").textContent = tr(
    `lr = ${sci(lr)}: PyTorch seed 5개의 평균 정확도 ${Math.round(m * 100)}%   (${LR3.acc[S3.i].map(a => Math.round(a * 100) + "%").join(", ")})\nbrowser 한 번: 1000 epoch 뒤 loss ${isFinite(L) ? fx(L, 3) : "∞"}, 정확도 ${Math.round(S3.live.acc * 100)}%\n→ ${lr < 0.005 ? "너무 작다: 1000 epoch로는 덜 배웠다" : lr > 1.5 ? "크다: 아직 되지만 조금 흔들린다" : "넓은 범위에서 잘 된다"}`,
    `lr = ${sci(lr)}: mean accuracy over 5 PyTorch seeds ${Math.round(m * 100)}%   (${LR3.acc[S3.i].map(a => Math.round(a * 100) + "%").join(", ")})\none browser run: loss ${isFinite(L) ? fx(L, 3) : "∞"} after 1000 epochs, accuracy ${Math.round(S3.live.acc * 100)}%\n→ ${lr < 0.005 ? "too small: 1000 epochs are not enough" : lr > 1.5 ? "large: still works but shakes a little" : "works well over a wide range"}`);
}
bindRange("i3lr", "o3lr", v => sci(LR3.lrs[v]), v => { S3.i = v; upd3(); });

/* ================= 4절 / Section 4 ================= */
// 선형 층 두 개 = 행렬 하나: W = W₂W₁을 칸마다 계산하는 animation
// Two linear layers = one matrix: animation computing W = W₂W₁ cell by cell
const W14 = [[1, 0.5], [-0.5, 1]], B14 = [0.2, 0], W24 = [[0.8, -0.6], [0.6, 0.8]], B24 = [0, 0.1];
const W4 = matMul(W24, W14);
const S4 = { k: 4, timer: 0 };
const F4a = fig("cv4a", w => (w >= 600 ? 140 : 210), (ctx, w, h, c) => {
  const wide = w >= 600, k = S4.k, cur = k < 4 ? [Math.floor(k / 2), k % 2] : null;
  const y0 = 46, cw = 52;
  let x = 16;
  panelTitle(ctx, c, tr("W₂ · W₁ = W (선형 층 두 개를 행렬 하나로)", "W₂ · W₁ = W (two linear layers as one matrix)"), 16, 18);
  const m2 = drawMat(ctx, c, W24, x, y0, { label: "W₂", col: c.sig, cw, hl: (i) => cur && i === cur[0] ? c.sigwash : false });
  x += m2.w + 26; txt(ctx, "·", x - 13, y0 + 30, { color: c.muted, size: 18, align: "center", w: "700" });
  const m1 = drawMat(ctx, c, W14, x, y0, { label: "W₁", col: c.ok, cw, hl: (i, j) => cur && j === cur[1] ? c.okwash : false });
  x += m1.w + 26; txt(ctx, "=", x - 13, y0 + 30, { color: c.muted, size: 18, align: "center", w: "700" });
  drawMat(ctx, c, W4.map((r, i) => r.map((v, j) => (i * 2 + j <= k - (k < 4 ? 0 : 1) || k >= 4 ? v : "?"))), x, y0, { label: "W", col: c.miss, cw, hl: (i, j) => cur && i === cur[0] && j === cur[1] ? c.sigwash : false });
  const tx = wide ? x + 2 * cw + 30 : 16, ty = wide ? y0 + 6 : y0 + 100;
  if (cur) {
    const [i, j] = cur;
    txt(ctx, `W[${i + 1},${j + 1}] = ${fx(W24[i][0], 1)}×${paren(W14[0][j], 1)} + ${paren(W24[i][1], 1)}×${paren(W14[1][j], 1)} = ${fx(W4[i][j], 2)}`, tx, ty + 14, { color: c.ink, size: 12, mono: true });
  } else txt(ctx, tr("bias도 하나로: b = W₂b₁ + b₂ = (0.16, 0.22)", "the biases merge too: b = W₂b₁ + b₂ = (0.16, 0.22)"), tx, ty + 14, { color: c.ink, size: 12 });
  txt(ctx, tr("→ 층이 둘이어도 하는 일은 W·x + b 하나와 같다", "→ two layers do no more than a single W·x + b"), tx, ty + 40, { color: c.muted, size: 12 });
});
$("b4play").addEventListener("click", () => { S4.k = 0; paint(F4a); runSteps(S4, 4, () => { S4.k++; paint(F4a); }, 900); });

// 격자 휘기: 입력 격자 → 1층(W₁x + b₁, activation) → 2층(W₂h + b₂). activation이 없으면 선은 곧은 채로 남는다
// Bending a grid: input grid → layer 1 (W₁x + b₁, activation) → layer 2 (W₂h + b₂). Without an activation the lines stay straight
const S4b = { act: "none" };
const actF = v => (S4b.act === "relu" ? Math.max(0, v) : S4b.act === "tanh" ? Math.tanh(v) : v);
const lay1 = p => { const z = matVec(W14, p).map((v, i) => actF(v + B14[i])); return z; };
const lay2 = p => matVec(W24, lay1(p)).map((v, i) => v + B24[i]);
const F4b = fig("cv4b", w => (w >= 600 ? Math.round((w - 40) / 3) + 40 : 3 * Math.min(w - 40, 260) + 110), (ctx, w, h, c) => {
  const wide = w >= 600, side = wide ? (w - 40) / 3 - 14 : Math.min(w - 40, 260);
  const maps = [[p => p, tr("입력", "input")], [lay1, tr("1층 뒤", "after layer 1")], [lay2, tr("2층 뒤 (출력)", "after layer 2 (output)")]];
  const lines = []; for (let k = -6; k <= 6; k++) { const v = k / 4, a = [], b = []; for (let t = 0; t <= 40; t++) { const u = -1.5 + 3 * t / 40; a.push([v, u]); b.push([u, v]); } lines.push([a, k], [b, k]); }
  const pts = [[1, 0.5], [-1, -0.8], [0.6, -1.1]];
  maps.forEach(([f, name], m) => {
    const x0 = wide ? 20 + m * (side + 21) : 20, y0 = wide ? 26 : 26 + m * (side + 36), B = { l: x0, t: y0, w: side, h: side };
    rrect(ctx, x0, y0, side, side, 6, c.bg, c.line);
    const all = lines.flatMap(([L]) => L.map(f)), xs = all.map(p => p[0]), ys = all.map(p => p[1]);
    const lo = Math.min(...xs, ...ys, -0.2), hi = Math.max(...xs, ...ys, 0.2), pad = (hi - lo) * 0.08;
    const X = v => x0 + (v - lo + pad) / (hi - lo + 2 * pad) * side, Y = v => y0 + side - (v - lo + pad) / (hi - lo + 2 * pad) * side;
    line(ctx, [[X(0), y0 + 4], [X(0), y0 + side - 4]], c.line, 1); line(ctx, [[x0 + 4, Y(0)], [x0 + side - 4, Y(0)]], c.line, 1);
    lines.forEach(([L, k], i) => line(ctx, L.map(f).map(p => [X(p[0]), Y(p[1])]), i % 2 ? c.sig : c.ok, k === 0 ? 2 : 1.1));
    pts.forEach((p, i) => { const q = f(p); dot(ctx, X(q[0]), Y(q[1]), 5, [c.fa, c.miss, c.vio][i], c.bg); });
    txt(ctx, name, x0, y0 - 8, { color: c.ink, size: 12, w: "600" });
    if (m < 2 && wide) arrowS(ctx, x0 + side + 4, y0 + side / 2, x0 + side + 18, y0 + side / 2, c.muted, 1.4, 6);
  });
});
function upd4b() {
  paint(F4b);
  $("r4b").textContent = S4b.act === "none" ? tr("activation 없음: 출력 격자도 곧은 평행선. 두 층 = 행렬 하나로 늘이고 돌리기만 했다", "no activation: the output grid is still straight parallel lines. Two layers only stretched and turned, like one matrix")
    : S4b.act === "relu" ? tr("ReLU: 음수를 0으로 → 격자가 축에 접힌다. 이렇게 접은 면을 다음 층이 나누면 굽은 경계가 된다", "ReLU: negatives become 0 → the grid folds onto the axes. The next layer splits this folded sheet, which gives curved boundaries")
    : tr("tanh: 바깥쪽을 −1~1로 누른다 → 격자가 휜다", "tanh: squeezes the outside into −1 to 1 → the grid bends");
}
bindSeg("g4act", v => { S4b.act = v; upd4b(); });

/* ================= 5절 / Section 5 ================= */
// iris 3 class: 꽃잎 길이·너비로 학습. 그림을 누르면 그 자리의 softmax 확률
// Iris, 3 classes: trained on petal length and width. Click the figure for the softmax probabilities at that spot
const IR = D4.iris, IN = D4.irisNorm, IX = IR.map(r => [(r[0] - IN.mu[0]) / IN.sd[0], (r[1] - IN.mu[1]) / IN.sd[1]]), IY = IR.map(r => r[2]);
const NAMES5 = ["setosa", "versicolor", "virginica"];
const norm5 = (pl, pw) => [(pl - IN.mu[0]) / IN.sd[0], (pw - IN.mu[1]) / IN.sd[1]];
const S5 = { net: null, ep: 0, hist: [], probe: [4.8, 1.6], timer: 0, lay: null };
function new5() { S5.net = new TinyNet([2, 8, 3], { hidden: "tanh", seed: 9, opt: { name: "adam", lr: 0.02 } }); S5.ep = 0; S5.hist = [S5.net.loss(IX, IY)]; }
const F5 = fig("cv5", w => (stacked(w, 270, 2, 28) ? 600 : 310), (ctx, w, h, c) => {
  const P = panels(w, h, 2, 28, 270), net = S5.net;
  const p = P[0], B = { l: p.x + 40, t: p.y + 30, w: p.w - 50, h: p.h - 74 }, xr = [0.5, 7.5], yr = [-0.1, 2.8];
  S5.lay = { B, xr, yr };
  drawField(ctx, c, B, xr, yr, (x, y) => net.probs(norm5(x, y)), { alpha: 0.38 });
  const A = axes(ctx, c, B, xr, yr, { xt: [1, 2, 3, 4, 5, 6, 7], yt: [0, 0.5, 1, 1.5, 2, 2.5], grid: false, xlabel: tr("꽃잎 길이 (cm)", "petal length (cm)") });
  panelTitle(ctx, c, tr("iris 150송이: 꽃잎 길이 × 너비", "Iris, 150 flowers: petal length × width"), B.l, p.y + 14);
  drawPoints(ctx, c, A, IR.map(r => [r[0], r[1]]), IY, { r: 3.2 });
  const [px, py] = S5.probe; line(ctx, [[A.X(px) - 9, A.Y(py)], [A.X(px) + 9, A.Y(py)]], c.ink, 2); line(ctx, [[A.X(px), A.Y(py) - 9], [A.X(px), A.Y(py) + 9]], c.ink, 2);
  const q = P[1], C = { l: q.x + 20, t: q.y + 40, w: q.w - 30, h: q.h - 90 }, pr = net.probs(norm5(px, py)), cols = classColors(c);
  panelTitle(ctx, c, tr(`+ 자리의 softmax 출력 (길이 ${fx(px, 1)}, 너비 ${fx(py, 1)})`, `Softmax output at + (length ${fx(px, 1)}, width ${fx(py, 1)})`), C.l, q.y + 14);
  const bh = Math.min(40, C.h / 3 - 14);
  pr.forEach((v, k) => {
    const y = C.t + k * (bh + 22);
    txt(ctx, NAMES5[k], C.l, y + bh / 2 + 4, { color: cols[k], size: 12, w: "700" });
    const x0 = C.l + 86, W = C.w - 140;
    ctx.fillStyle = c.wash; ctx.fillRect(x0, y, W, bh); ctx.fillStyle = cols[k]; ctx.fillRect(x0, y, W * v, bh);
    txt(ctx, fx(v, 3), x0 + W + 8, y + bh / 2 + 4, { color: c.ink, size: 12, mono: true });
  });
  txt(ctx, tr("세 막대의 합 = 1 (softmax)", "the three bars add up to 1 (softmax)"), C.l, C.t + 3 * (bh + 22) + 6, { color: c.muted, size: 11 });
});
function upd5() {
  paint(F5);
  const acc = S5.net.accuracy(IX, IY), pr = S5.net.probs(norm5(...S5.probe)), k = pr.indexOf(Math.max(...pr));
  $("r5").textContent = tr(
    `epoch ${S5.ep}: loss ${fx(S5.hist[S5.hist.length - 1], 3)}, 정확도 ${Math.round(acc * 100)}%   network 2-8-3 (출력 3개 = class 3개)\n+ 자리 예측: ${NAMES5[k]} (${Math.round(pr[k] * 100)}%)`,
    `epoch ${S5.ep}: loss ${fx(S5.hist[S5.hist.length - 1], 3)}, accuracy ${Math.round(acc * 100)}%   network 2-8-3 (3 outputs = 3 classes)\nprediction at +: ${NAMES5[k]} (${Math.round(pr[k] * 100)}%)`);
  $("b5run").textContent = S5.timer ? tr("멈춤", "Pause") : tr("학습", "Train");
}
$("b5run").addEventListener("click", () => {
  if (S5.timer) { stopRun(S5); upd5(); return; }
  runSteps(S5, 60, () => { for (let i = 0; i < 5; i++) { S5.hist.push(S5.net.step(IX, IY)); S5.ep++; } upd5(); if (S5.ep >= 300) return false; }, 60, upd5); upd5();
});
$("b5reset").addEventListener("click", () => { stopRun(S5); new5(); upd5(); });
dragOn(F5, (px, py) => { const L = S5.lay; return L && px >= L.B.l && px <= L.B.l + L.B.w && py >= L.B.t && py <= L.B.t + L.B.h ? 1 : null; }, (k, px, py) => {
  const L = S5.lay; S5.probe = [clamp(L.xr[0] + (px - L.B.l) / L.B.w * (L.xr[1] - L.xr[0]), 0.6, 7.4), clamp(L.yr[1] - (py - L.B.t) / L.B.h * (L.yr[1] - L.yr[0]), 0, 2.7)]; upd5();
});

/* ================= 6절 / Section 6 ================= */
// parameter 계산기: 층마다 (입력 × 출력 + 출력)
// Parameter counter: per layer (inputs × outputs + outputs)
const S6 = { d: 2, w: 16, out: 3 };
const sizes6 = () => [2, ...Array(S6.d).fill(S6.w), S6.out];
const F6a = fig("cv6a", 250, (ctx, w, h, c) => {
  const sz = sizes6(), L = sz.length, B = { l: 40, t: 30, w: w - 80, h: 130 };
  const show = sz.map(n => Math.min(n, 8));
  const pos = drawNet(ctx, c, show, B, { r: 7, showVals: false });
  sz.forEach((n, l) => {
    const x = pos[l][0][0];
    txt(ctx, String(n), x, B.t - 10, { color: c.ink, size: 11.5, w: "700", align: "center", mono: true });
    if (n > 8) txt(ctx, "⋮", x, B.t + B.h + 8, { color: c.muted, size: 13, align: "center" });
    if (l > 0) {
      const p = sz[l - 1] * n + n, xm = (pos[l - 1][0][0] + x) / 2;
      const bw = Math.min(68, B.w / (L - 1) - 6);
      rrect(ctx, xm - bw / 2, B.t + B.h + 22, bw, 40, 6, c.wash, c.line);
      txt(ctx, `${sz[l - 1]}×${n}+${n}`, xm, B.t + B.h + 38, { color: c.muted, size: 10, align: "center", mono: true });
      txt(ctx, commas(p), xm, B.t + B.h + 55, { color: c.ink, size: 12, align: "center", mono: true, w: "700" });
    }
  });
  const tot = sz.slice(1).reduce((s, n, l) => s + sz[l] * n + n, 0);
  txt(ctx, tr(`합계 parameter ${commas(tot)}개`, `total ${commas(tot)} parameters`), w / 2, h - 10, { color: c.miss, size: 13, w: "700", align: "center" });
});
function upd6a() {
  paint(F6a);
  const sz = sizes6(), parts = sz.slice(1).map((n, l) => `${sz[l]}×${n} + ${n} = ${sz[l] * n + n}`), tot = sz.slice(1).reduce((s, n, l) => s + sz[l] * n + n, 0);
  $("r6a").textContent = tr(`network ${sz.join("-")}\n층마다 (입력 × 출력 weight) + (출력 bias): ${parts.join(",  ")}\n합계 = ${commas(tot)}`,
    `network ${sz.join("-")}\nper layer (inputs × outputs weights) + (outputs biases): ${parts.join(",  ")}\ntotal = ${commas(tot)}`);
  paint(F6b);
}
bindRange("i6d", "o6d", v => String(v), v => { S6.d = v; upd6a(); });
bindRange("i6w", "o6w", v => String(v), v => { S6.w = v; upd6a(); });
bindSeg("g6o", v => { S6.out = +v; upd6a(); });
// depth × breadth: 미리 돌린 spiral 실험의 시험 정확도 (seed 3개 평균)
// Depth × breadth: test accuracy of the precomputed spiral experiment (mean of 3 seeds)
const DB = D4.depthBreadth, SP = D4.spiral, S6b = { lay: null };
const F6b = fig("cv6b", w => (w >= 600 ? 300 : 560), (ctx, w, h, c) => {
  const wide = w >= 600, side = wide ? Math.min(260, w * 0.33) : Math.min(w - 40, 260);
  const A0 = { l: 20, t: 30, w: side, h: side }, R = [-1.4, 1.4];
  panelTitle(ctx, c, tr("spiral data (300개)", "Spiral data (300 points)"), A0.l, 18);
  rrect(ctx, A0.l, A0.t, side, side, 6, c.bg, c.line);
  const A = { X: v => A0.l + (v - R[0]) / (R[1] - R[0]) * side, Y: v => A0.t + side - (v - R[0]) / (R[1] - R[0]) * side };
  drawPoints(ctx, c, A, SP.X, SP.Y, { r: 2.6 });
  const gx = wide ? A0.l + side + 70 : 70, gy = wide ? 40 : A0.t + side + 50, cw = Math.min(64, (w - gx - 20) / 5), ch = wide ? Math.min(46, (h - 70) / 5) : 40;
  S6b.lay = { gx, gy, cw, ch };
  panelTitle(ctx, c, tr("시험 정확도: 깊이 × 너비", "Test accuracy: depth × width"), gx, gy - 22);
  const ok = rgb(c.ok), fa = rgb(c.fa), bg = rgb(c.bg);
  DB.grid.forEach((row, i) => row.forEach((cell, j) => {
    const t = clamp((cell.acc - 0.5) / 0.5, 0, 1), col = t > 0.5 ? mixRGB(bg, ok, (t - 0.5) * 2 * 0.8) : mixRGB(bg, fa, (0.5 - t) * 2 * 0.6);
    const x = gx + j * cw, y = gy + i * ch, cur = DB.depths[i] === S6.d && DB.widths[j] === S6.w;
    ctx.fillStyle = cssRGB(col); ctx.fillRect(x + 1, y + 1, cw - 2, ch - 2);
    if (cur) { ctx.strokeStyle = c.ink; ctx.lineWidth = 2.4; ctx.strokeRect(x + 2, y + 2, cw - 4, ch - 4); }
    txt(ctx, Math.round(cell.acc * 100) + "%", x + cw / 2, y + ch / 2 + 1, { color: c.ink, size: 11.5, align: "center", mono: true, w: "700" });
    txt(ctx, commas(cell.params), x + cw / 2, y + ch / 2 + 14, { color: c.ink, size: 9.5, align: "center", mono: true });
  }));
  DB.widths.forEach((v, j) => txt(ctx, String(v), gx + j * cw + cw / 2, gy - 6, { color: c.muted, size: 11, align: "center", mono: true }));
  DB.depths.forEach((v, i) => txt(ctx, String(v), gx - 8, gy + i * ch + ch / 2 + 4, { color: c.muted, size: 11, align: "right", mono: true }));
  txt(ctx, tr("너비 (층마다 neuron) →", "width (neurons per layer) →"), gx + 2.5 * cw, gy + 5 * ch + 18, { color: c.muted, size: 11, align: "center" });
  ctx.save(); ctx.translate(gx - 34, gy + 2.5 * ch); ctx.rotate(-Math.PI / 2); txt(ctx, tr("깊이 (숨은 층 수) →", "depth (hidden layers) →"), 0, 0, { color: c.muted, size: 11, align: "center" }); ctx.restore();
});
dragOn(F6b, (px, py) => { const L = S6b.lay; if (!L) return null; const j = Math.floor((px - L.gx) / L.cw), i = Math.floor((py - L.gy) / L.ch); return i >= 0 && i < 5 && j >= 0 && j < 5 ? [i, j] : null; }, (k) => {
  const [i, j] = k; S6.d = DB.depths[i]; S6.w = DB.widths[j]; setRange("i6d", S6.d, 0); setRange("i6w", S6.w, 0); $("o6d").textContent = S6.d; $("o6w").textContent = S6.w; upd6a();
});

/* ================= 8절 / Section 8 ================= */
// 학습된 iris network의 weight를 그대로 본다: 숫자는 다 보이지만 뜻은 바로 안 보인다
// Look straight at the trained iris network's weights: every number is visible, the meaning is not
const NET8 = new TinyNet([2, 8, 3], { hidden: "tanh", seed: 9, opt: { name: "adam", lr: 0.02 } });
for (let e = 0; e < 300; e++) NET8.step(IX, IY);
const F8 = fig("cv8", w => (w >= 640 ? 260 : 520), (ctx, w, h, c) => {
  const wide = w >= 640, NB = wide ? { l: 50, t: 34, w: w * 0.4 - 70, h: h - 60 } : { l: 50, t: 34, w: w - 100, h: 220 };
  panelTitle(ctx, c, tr("학습된 2-8-3 network (선 = weight)", "The trained 2-8-3 network (lines = weights)"), 16, 18);
  const pos = drawNet(ctx, c, [2, 8, 3], NB, { r: 8, W: NET8.W, wScale: 0.9, showVals: false });
  [tr("길이", "length"), tr("너비", "width")].forEach((s, i) => txt(ctx, s, pos[0][i][0] - 12, pos[0][i][1] + 4, { color: c.muted, size: 11, align: "right" }));
  NAMES5.forEach((s, i) => txt(ctx, s, pos[2][i][0] + 12, pos[2][i][1] + 4, { color: classColors(c)[i], size: 11, w: "700" }));
  const mx = wide ? w * 0.47 : 16, my = wide ? 46 : NB.t + NB.h + 50, cell = wide ? Math.min(26, (w - mx - 40) / 12) : Math.min(24, (w - 40) / 12);
  const heat = (M, x, y, name) => {
    const rows = M.length, cols = M[0].length, mxv = Math.max(...M.flat().map(Math.abs));
    M.forEach((r, i) => r.forEach((v, j) => { ctx.fillStyle = cssRGB(mixRGB(rgb(c.bg), rgb(v >= 0 ? c.sig : c.fa), Math.abs(v) / mxv)); ctx.fillRect(x + j * cell, y + i * cell, cell - 1, cell - 1); }));
    txt(ctx, `${name} (${rows}×${cols})`, x, y - 6, { color: c.ink, size: 11.5, w: "600" });
  };
  heat(NET8.W[0], mx, my, "W₁");
  heat(NET8.W[1], mx + 3 * cell + 30, my, "W₂");
  txt(ctx, tr("파랑 = 양수, 빨강 = 음수, 진할수록 크다", "blue = positive, red = negative, darker = larger"), mx, my + 8 * cell + 22, { color: c.muted, size: 11 });
});
function upd8() {
  paint(F8);
  const n = NET8.nParams(), w1 = NET8.W[0].map(r => "(" + r.map(v => fx(v, 2)).join(", ") + ")").slice(0, 3).join(" ");
  $("r8").textContent = tr(`parameter ${n}개 (W₁ 16 + b₁ 8 + W₂ 24 + b₂ 3), 정확도 ${Math.round(NET8.accuracy(IX, IY) * 100)}%\nW₁의 첫 세 행: ${w1} …\n→ 숫자는 다 보이지만 "숨은 neuron 3번은 무엇을 보나?"는 바로 답할 수 없다`,
    `${n} parameters (W₁ 16 + b₁ 8 + W₂ 24 + b₂ 3), accuracy ${Math.round(NET8.accuracy(IX, IY) * 100)}%\nfirst three rows of W₁: ${w1} …\n→ every number is visible, but "what does hidden neuron 3 look for?" has no direct answer`);
}

// 처음 그리기
// First paint
upd1(); upd2(); upd3(); paint(F4a); upd4b(); new5(); upd5(); upd6a(); upd8();
repaintAll();
