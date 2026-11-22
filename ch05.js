/* 5장 그림과 조작. 한글판(ch05-overfitting-regularization-KR.html)과 영어판(ch05-overfitting-regularization.html)이 같이 쓴다
   Chapter 5 figures and interactions, shared by the Korean and English pages
   다항식 계산은 tools/ch05_numbers.py와 같다 (같은 숫자가 나온다)
   The polynomial maths matches tools/ch05_numbers.py (same numbers) */
"use strict";

/* ================= 다항식 회귀 도구 / Polynomial regression tools ================= */
// data: x는 −1~1 고르게, y = 0.8 sin(πx) + 0.25·잡음. 학습 15개(seed 8), 시험 200개(seed 9)
// Data: x uniform in −1..1, y = 0.8 sin(πx) + 0.25·noise. 15 training points (seed 8), 200 test points (seed 9)
const truth = x => 0.8 * Math.sin(Math.PI * x);
function makePoly(seed, n) { const r = makeRng(seed), X = [], Y = []; for (let i = 0; i < n; i++) { const x = r.u() * 2 - 1; X.push(x); Y.push(truth(x) + 0.25 * r.n()); } return { X, Y }; }
const TRN = makePoly(8, 15), TST = makePoly(9, 200);
// Legendre 다항식 P₀..P_d (−1~1에서 숫자가 안정적이다)
// Legendre polynomials P₀..P_d (numerically stable on −1..1)
function leg(x, d) { const P = [1, x]; for (let k = 1; k < d; k++) P.push(((2 * k + 1) * x * P[k] - k * P[k - 1]) / (k + 1)); return P.slice(0, d + 1); }
function solve(A, b) {
  const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = c + 1; r < n; r++) { const f = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) { let s = M[r][n]; for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k]; x[r] = s / M[r][r]; }
  return x;
}
// 최소 제곱 + ridge: (ΦᵀΦ + λI′)c = Φᵀy (상수항 c₀는 벌하지 않는다)
// Least squares + ridge: (ΦᵀΦ + λI′)c = Φᵀy (the constant c₀ is not penalised)
function fitPoly(X, Y, d, lam = 0) {
  const F = X.map(x => leg(x, d)), A = Array.from({ length: d + 1 }, (_, i) => Array.from({ length: d + 1 }, (_, j) => F.reduce((s, r) => s + r[i] * r[j], 0) + (i === j ? (i ? lam : 0) + 1e-10 : 0)));
  return solve(A, Array.from({ length: d + 1 }, (_, i) => F.reduce((s, r, k) => s + r[i] * Y[k], 0)));
}
const evalPoly = (c, x) => leg(x, c.length - 1).reduce((s, v, i) => s + v * c[i], 0);
const msePoly = (c, D) => meanOf(D.X.map((x, i) => (evalPoly(c, x) - D.Y[i]) ** 2));
const lg = v => Math.log10(Math.max(v, 1e-12));
const MSE_TICKS = [[-3, "0.001"], [-2, "0.01"], [-1, "0.1"], [0, "1"], [1, "10"], [2, "100"]];
// 왼쪽 = data와 맞춘 곡선, 오른쪽 = 오차 곡선. 두 그림(1절, 5절)이 같이 쓴다
// Left = data and fitted curve, right = error curves. Shared by the figures in sections 1 and 5
function polyPanel(ctx, c, p, coef, title) {
  const B = { l: p.x + 40, t: p.y + 30, w: p.w - 52, h: p.h - 76 };
  const A = axes(ctx, c, B, [-1.05, 1.05], [-2, 2], { xt: [-1, -0.5, 0, 0.5, 1], yt: [-2, -1, 0, 1, 2], xlabel: "x" });
  panelTitle(ctx, c, title, B.l, p.y + 14);
  clipBox(ctx, B);
  TST.X.forEach((x, i) => { ctx.globalAlpha = 0.35; dot(ctx, A.X(x), A.Y(TST.Y[i]), 2.2, c.miss); ctx.globalAlpha = 1; });
  line(ctx, curve(A, truth, [-1.05, 1.05]), c.muted, 1.4, [6, 4]);
  line(ctx, curve(A, x => clamp(evalPoly(coef, x), -50, 50), [-1.05, 1.05], 400), c.fa, 2.4);
  TRN.X.forEach((x, i) => dot(ctx, A.X(x), A.Y(TRN.Y[i]), 4.5, c.sig, c.bg));
  ctx.restore();
  txtBg(ctx, c, tr("● 학습 15개  · 시험 200개  -- 진짜 함수", "● 15 training  · 200 test  -- true function"), B.l + 6, B.t + 13, { color: c.muted, size: 11 });
}
function errPanel(ctx, c, p, xs, tr_, te_, cur, o) {
  const B = { l: p.x + 48, t: p.y + 30, w: p.w - 60, h: p.h - 76 };
  const A = axes(ctx, c, B, o.xr, [-3.3, 2.3], { xt: o.xt, yt: MSE_TICKS, xlabel: o.xlabel });
  panelTitle(ctx, c, o.title, B.l, p.y + 14);
  clipBox(ctx, B);
  [[tr_, c.sig], [te_, c.miss]].forEach(([v, col]) => { line(ctx, xs.map((x, i) => [A.X(x), A.Y(clamp(lg(v[i]), -3.5, 2.5))]), col, 2); xs.forEach((x, i) => dot(ctx, A.X(x), A.Y(clamp(lg(v[i]), -3.5, 2.5)), 2.6, col)); });
  ctx.restore();
  const x = A.X(xs[cur]); line(ctx, [[x, B.t], [x, B.t + B.h]], c.ink, 1.2, [4, 3]);
  const lab = o.labels || [tr("학습 오차", "training error"), tr("시험 오차", "test error")];
  txtBg(ctx, c, lab[0], B.l + 6, B.t + B.h - 8, { color: c.sig, size: 11.5, w: "700" });
  txtBg(ctx, c, lab[1], B.l + 6, B.t + 14, { color: c.miss, size: 11.5, w: "700" });
  if (te_[cur] > 200) txtBg(ctx, c, "↑ " + sci(te_[cur]), x + 5, B.t + 14, { color: c.miss, size: 11, w: "700", mono: true });
}

/* ================= 1절 / Section 1 ================= */
const DEG = Array.from({ length: 15 }, (_, d) => d), FITS = DEG.map(d => fitPoly(TRN.X, TRN.Y, d));
const TR1 = FITS.map(cf => msePoly(cf, TRN)), TE1 = FITS.map(cf => msePoly(cf, TST));
const S1 = { d: 3 };
const F1 = fig("cv1", w => (stacked(w, 270, 2, 28) ? 580 : 300), (ctx, w, h, c) => {
  const P = panels(w, h, 2, 28, 270);
  polyPanel(ctx, c, P[0], FITS[S1.d], tr(`차수 ${S1.d} 다항식`, `Degree-${S1.d} polynomial`));
  errPanel(ctx, c, P[1], DEG, TR1, TE1, S1.d, { xr: [-0.5, 14.5], xt: [0, 2, 4, 6, 8, 10, 12, 14], xlabel: tr("다항식 차수 (복잡도)", "polynomial degree (complexity)"), title: tr("차수마다 오차 (log 눈금)", "Error per degree (log scale)") });
});
function upd1() {
  paint(F1);
  const d = S1.d, a = TR1[d], b = TE1[d];
  const st = d <= 1 ? tr("너무 단순: 학습 오차도 크다 (underfitting)", "too simple: even the training error is large (underfitting)") : b > 1 ? tr("학습 점은 거의 다 지나가지만 그 사이에서 날뛴다 (overfitting)", "passes almost every training point but goes wild in between (overfitting)") : tr("두 오차가 모두 작다: 잘 일반화한다", "both errors are small: it generalises well");
  $("r1").textContent = tr(`차수 ${d}: parameter ${d + 1}개, 학습 MSE ${fx(a, 4)}, 시험 MSE ${sci(b)}\n→ ${st}`, `degree ${d}: ${d + 1} parameters, training MSE ${fx(a, 4)}, test MSE ${sci(b)}\n→ ${st}`);
}
bindRange("i1d", "o1d", v => String(v), v => { S1.d = v; upd1(); });

/* ================= 2절 / Section 2 ================= */
// 100개를 섞어 train / dev / test로 나눈다
// Shuffle 100 items and split them into train / dev / test
const S2 = { tr: 70, dev: 15, seed: 3 };
const F2 = fig("cv2", w => (w >= 560 ? 230 : 400), (ctx, w, h, c) => {
  const wide = w >= 560, cell = wide ? Math.min(19, (w * 0.45) / 10) : Math.min(22, (w - 40) / 10), te = 100 - S2.tr - S2.dev;
  const order = shuffled(100, makeRng(S2.seed)), role = new Array(100);
  order.forEach((idx, pos) => { role[idx] = pos < S2.tr ? 0 : pos < S2.tr + S2.dev ? 1 : 2; });
  const cols = [c.sig, c.ok, c.miss], x0 = 16, y0 = 30;
  panelTitle(ctx, c, tr("data 100개 (칸 하나 = sample 하나)", "100 samples (one cell = one sample)"), x0, 16);
  for (let i = 0; i < 100; i++) { ctx.fillStyle = cols[role[i]]; ctx.globalAlpha = 0.85; ctx.fillRect(x0 + (i % 10) * cell, y0 + Math.floor(i / 10) * cell, cell - 2, cell - 2); ctx.globalAlpha = 1; }
  const lx = wide ? x0 + 10 * cell + 30 : x0, ly = wide ? y0 + 4 : y0 + 10 * cell + 20;
  const rows = [[0, `train ${S2.tr}%`, tr("weight를 학습한다 (매 epoch)", "learns the weights (every epoch)")], [1, `dev ${S2.dev}%`, tr("설정(learning rate, 차수, λ …)을 고른다", "chooses settings (learning rate, degree, λ …)")], [2, `test ${te}%`, tr("마지막에 딱 한 번: 새 data에서의 성적", "once, at the very end: the score on new data")]];
  rows.forEach(([k, a, b], i) => {
    const y = ly + i * 48;
    ctx.fillStyle = cols[k]; ctx.fillRect(lx, y, 16, 16);
    txt(ctx, a, lx + 24, y + 13, { color: c.ink, size: 13, w: "700" });
    txt(ctx, b, lx + 24, y + 31, { color: c.muted, size: 11.5 });
  });
});
function upd2() {
  paint(F2);
  const te = 100 - S2.tr - S2.dev;
  $("r2").textContent = tr(`train ${S2.tr}개 · dev ${S2.dev}개 · test ${te}개 (무작위로 섞은 뒤 나눈다)\n→ test로 설정을 고르면 test도 "본 data"가 된다. 그래서 고르는 일은 dev로만 한다`,
    `train ${S2.tr} · dev ${S2.dev} · test ${te} (shuffled at random, then split)\n→ choosing settings with the test set turns it into "seen data". So choosing is done on dev only`);
}
bindRange("i2tr", "o2tr", v => v + "%", v => { S2.tr = Math.min(v, 95 - S2.dev); upd2(); });
bindRange("i2dev", "o2dev", v => v + "%", v => { S2.dev = Math.min(v, 95 - S2.tr); upd2(); });
$("b2shuffle").addEventListener("click", () => { S2.seed += 1; upd2(); });

/* ================= 3절 / Section 3 ================= */
// k-fold: 학습 15개를 k 묶음으로. 묶음 하나씩 돌아가며 검증, 나머지로 학습. 차수마다 검증 오차의 평균
// k-fold: split the 15 training points into k groups. Each group takes a turn as validation while the rest train. Mean validation error per degree
const ORDER3 = shuffled(15, makeRng(5));
function cvErr(d, k) {
  const errs = [];
  for (let f = 0; f < k; f++) {
    const val = ORDER3.filter((_, i) => i % k === f), trn = [...Array(15).keys()].filter(i => !val.includes(i));
    const cf = fitPoly(trn.map(i => TRN.X[i]), trn.map(i => TRN.Y[i]), d);
    errs.push(meanOf(val.map(i => (evalPoly(cf, TRN.X[i]) - TRN.Y[i]) ** 2)));
  }
  return meanOf(errs);
}
const S3 = { k: 5, f: -1, timer: 0, cv: null };
const DEG3 = Array.from({ length: 11 }, (_, d) => d);
function comp3() { S3.cv = DEG3.map(d => cvErr(d, S3.k)); }
const F3 = fig("cv3", w => (stacked(w, 270, 2, 28) ? 640 : 330), (ctx, w, h, c) => {
  const P = panels(w, h, 2, 28, 270), k = S3.k;
  {
    const p = P[0], rowH = Math.min(22, (p.h - 70) / k), cw = Math.min(24, (p.w - 90) / 15), x0 = p.x + 64, y0 = p.y + 34;
    panelTitle(ctx, c, tr(`${k}-fold: 줄 하나 = 한 번의 학습`, `${k}-fold: one row = one training run`), p.x + 8, p.y + 14);
    for (let f = 0; f < k; f++) {
      const y = y0 + f * rowH, on = S3.f === -1 || S3.f === f;
      txt(ctx, tr(`${f + 1}번째`, `run ${f + 1}`), x0 - 8, y + rowH / 2 + 4, { color: on ? c.ink : c.muted, size: 11, align: "right" });
      ORDER3.forEach((idx, pos) => { const val = pos % k === f; ctx.globalAlpha = on ? 0.9 : 0.25; ctx.fillStyle = val ? c.ok : c.sig; ctx.fillRect(x0 + pos * cw, y + 2, cw - 2, rowH - 4); ctx.globalAlpha = 1; });
    }
    const yl = y0 + k * rowH + 18;
    ctx.fillStyle = c.sig; ctx.fillRect(x0, yl - 10, 12, 12); txt(ctx, tr("학습", "train"), x0 + 18, yl, { color: c.ink, size: 11.5 });
    ctx.fillStyle = c.ok; ctx.fillRect(x0 + 80, yl - 10, 12, 12); txt(ctx, tr("검증 (돌아가며)", "validation (takes turns)"), x0 + 98, yl, { color: c.ink, size: 11.5 });
  }
  {
    const best = S3.cv.indexOf(Math.min(...S3.cv)), te = DEG3.map(d => TE1[d]);
    errPanel(ctx, c, P[1], DEG3, S3.cv, te, best, { xr: [-0.5, 10.5], xt: [0, 2, 4, 6, 8, 10], xlabel: tr("다항식 차수", "polynomial degree"), title: tr("k-fold 검증 오차 vs 진짜 시험 오차", "k-fold validation error vs true test error"), labels: [tr("검증 오차 (k-fold 평균)", "validation error (k-fold mean)"), tr("시험 오차 (200개)", "test error (200 points)")] });
  }
});
function upd3() {
  comp3(); paint(F3);
  const best = S3.cv.indexOf(Math.min(...S3.cv)), tbest = DEG3.reduce((b, d) => (TE1[d] < TE1[b] ? d : b), 0);
  $("r3").textContent = tr(`k = ${S3.k}: 검증 오차가 가장 작은 차수 = ${best} (검증 MSE ${fx(S3.cv[best], 3)})\n참고: 시험 data 200개로 본 가장 좋은 차수 = ${tbest} (시험 MSE ${fx(TE1[tbest], 3)}), 고른 차수의 시험 MSE = ${fx(TE1[best], 3)}\n→ 학습 data만으로 고른 차수가 시험에서도 좋다. ${S3.k === 15 ? "k = 15는 하나씩 빼 보기 (leave-one-out)" : ""}`,
    `k = ${S3.k}: degree with the smallest validation error = ${best} (validation MSE ${fx(S3.cv[best], 3)})\nfor reference: the best degree on the 200 test points = ${tbest} (test MSE ${fx(TE1[tbest], 3)}); test MSE of the chosen degree = ${fx(TE1[best], 3)}\n→ a degree chosen from the training data alone also does well on the test data. ${S3.k === 15 ? "k = 15 is leave-one-out" : ""}`);
}
bindRange("i3k", "o3k", v => String(v), v => { stopRun(S3); S3.k = v; S3.f = -1; upd3(); });
$("b3play").addEventListener("click", () => { S3.f = 0; paint(F3); runSteps(S3, S3.k, () => { S3.f++; if (S3.f >= S3.k) { S3.f = -1; } paint(F3); }, 700); });

/* ================= 5절 / Section 5 ================= */
// 차수 14 + ridge: λ가 크면 계수가 작아지고 곡선이 얌전해진다
// Degree 14 + ridge: a larger λ shrinks the coefficients and calms the curve
const LAMS = Array.from({ length: 29 }, (_, i) => -6 + i * 0.25), FIT5 = LAMS.map(l => fitPoly(TRN.X, TRN.Y, 14, 10 ** l));
const TR5 = FIT5.map(cf => msePoly(cf, TRN)), TE5 = FIT5.map(cf => msePoly(cf, TST));
const S5 = { i: 20 };
const F5 = fig("cv5", w => (stacked(w, 270, 2, 28) ? 580 : 300), (ctx, w, h, c) => {
  const P = panels(w, h, 2, 28, 270);
  polyPanel(ctx, c, P[0], FIT5[S5.i], tr(`차수 14, λ = ${sci(10 ** LAMS[S5.i])}`, `Degree 14, λ = ${sci(10 ** LAMS[S5.i])}`));
  errPanel(ctx, c, P[1], LAMS, TR5, TE5, S5.i, { xr: [-6.3, 1.3], xt: [[-6, "10⁻⁶"], [-4, "10⁻⁴"], [-2, "0.01"], [0, "1"]], xlabel: tr("λ (log 눈금)", "λ (log scale)"), title: tr("λ마다 오차", "Error per λ") });
});
function upd5() {
  paint(F5);
  const i = S5.i, cf = FIT5[i], big = Math.max(...cf.slice(1).map(Math.abs));
  $("r5").textContent = tr(`λ = ${sci(10 ** LAMS[i])}: 학습 MSE ${fx(TR5[i], 4)}, 시험 MSE ${sci(TE5[i])}, 가장 큰 계수 |c| = ${sci(big)}\n→ ${TE5[i] > 1 ? "λ가 너무 작다: 계수가 커져 곡선이 날뛴다" : TR5[i] > 0.05 ? "λ가 너무 크다: 곡선이 납작해져 data도 못 따라간다" : "알맞다: 차수 14인데도 얌전하다"}`,
    `λ = ${sci(10 ** LAMS[i])}: training MSE ${fx(TR5[i], 4)}, test MSE ${sci(TE5[i])}, largest coefficient |c| = ${sci(big)}\n→ ${TE5[i] > 1 ? "λ too small: big coefficients make the curve go wild" : TR5[i] > 0.05 ? "λ too large: the curve flattens and misses the data" : "about right: calm even at degree 14"}`);
}
bindRange("i5l", "o5l", v => sci(10 ** LAMS[v]), v => { S5.i = v; upd5(); });

/* ================= 7절 / Section 7 ================= */
// dropout 그림: 학습 step마다 숨은 unit을 무작위로 끈다
// Dropout picture: each training step switches off random hidden units
const S7a = { p: 0.5, mask: null, timer: 0, step: 0, rng: makeRng(17) };
function newMask() { S7a.mask = [8, 8].map(n => Array.from({ length: n }, () => S7a.rng.u() >= S7a.p)); S7a.step++; }
const F7a = fig("cv7a", 230, (ctx, w, h, c) => {
  const B = { l: 60, t: 34, w: w - 120, h: h - 60 };
  panelTitle(ctx, c, tr(`학습 step ${S7a.step}: 꺼진 unit (회색 ×)은 이번 step에서 없는 셈`, `Training step ${S7a.step}: switched-off units (grey ×) do not exist for this step`), 12, 18);
  const on = (l, i) => l === 0 || l === 3 || S7a.mask[l - 1][i];
  const pos = drawNet(ctx, c, [2, 8, 8, 1], B, { r: 9, showVals: false, hl: (l, j, i) => (on(l, j) && on(l - 1, i) ? null : false), nodeOn: on });
  pos.forEach((col, l) => col.forEach(([x, y], i) => { if (!on(l, i)) { line(ctx, [[x - 6, y - 6], [x + 6, y + 6]], c.muted, 2); line(ctx, [[x - 6, y + 6], [x + 6, y - 6]], c.muted, 2); } }));
});
$("b7a").addEventListener("click", () => runSteps(S7a, 12, () => { newMask(); paint(F7a); }, 500));
bindRange("i7p", "o7p", v => fx(v, 1), v => { S7a.p = v; newMask(); paint(F7a); S7.p = v; reset7(); });
// 같은 network 둘을 나란히 학습: 왼쪽 dropout 없음, 오른쪽 dropout p
// Train two identical networks side by side: left without dropout, right with dropout p
const TR7 = makeData("moons", 60, makeRng(31), 0.25), TE7 = makeData("moons", 400, makeRng(32), 0.25);
const S7 = { p: 0.5, nets: null, ep: 0, timer: 0 };
function reset7() {
  stopRun(S7);
  S7.nets = [0, S7.p].map(p => new TinyNet([2, 24, 24, 1], { hidden: "relu", seed: 1, opt: { name: "adam", lr: 0.01, drop: p } }));
  S7.ep = 0; upd7();
}
const F7b = fig("cv7b", w => (stacked(w, 240, 2, 24) ? 600 : 310), (ctx, w, h, c) => {
  const P = panels(w, h, 2, 24, 240), R = [-2.1, 2.1];
  S7.nets.forEach((net, k) => {
    const p = P[k], side = Math.min(p.w - 30, p.h - 50), B = { l: p.x + 24, t: p.y + 30, w: side, h: side };
    drawField(ctx, c, B, R, R, (x, y) => net.probs([x, y]), { alpha: 0.4 });
    const A = axes(ctx, c, B, R, R, { xt: [], yt: [], grid: false });
    clipBox(ctx, B);
    ctx.globalAlpha = 0.28; drawPoints(ctx, c, A, TE7.X, TE7.Y, { r: 2 }); ctx.globalAlpha = 1;
    drawPoints(ctx, c, A, TR7.X, TR7.Y, { r: 3.6 });
    ctx.restore();
    panelTitle(ctx, c, k ? tr(`dropout p = ${fx(S7.p, 1)}`, `dropout p = ${fx(S7.p, 1)}`) : tr("dropout 없음", "no dropout"), B.l, p.y + 16);
    txtBg(ctx, c, tr(`학습 ${Math.round(net.accuracy(TR7.X, TR7.Y) * 100)}% · 시험 ${Math.round(net.accuracy(TE7.X, TE7.Y) * 100)}%`, `train ${Math.round(net.accuracy(TR7.X, TR7.Y) * 100)}% · test ${Math.round(net.accuracy(TE7.X, TE7.Y) * 100)}%`), B.l + 6, B.t + B.h - 8, { color: c.ink, size: 12, w: "700" });
  });
});
function upd7() {
  paint(F7b);
  const a = S7.nets.map(n => [n.accuracy(TR7.X, TR7.Y), n.accuracy(TE7.X, TE7.Y)]);
  $("r7").textContent = tr(`epoch ${S7.ep}  (network 2-24-24-1, Adam, 학습 60개 · 시험 400개)\ndropout 없음: 학습 ${Math.round(a[0][0] * 100)}%, 시험 ${Math.round(a[0][1] * 100)}%, 차이 ${Math.round((a[0][0] - a[0][1]) * 100)}%p\ndropout ${fx(S7.p, 1)}: 학습 ${Math.round(a[1][0] * 100)}%, 시험 ${Math.round(a[1][1] * 100)}%, 차이 ${Math.round((a[1][0] - a[1][1]) * 100)}%p`,
    `epoch ${S7.ep}  (network 2-24-24-1, Adam, 60 training · 400 test points)\nno dropout: train ${Math.round(a[0][0] * 100)}%, test ${Math.round(a[0][1] * 100)}%, gap ${Math.round((a[0][0] - a[0][1]) * 100)} points\ndropout ${fx(S7.p, 1)}: train ${Math.round(a[1][0] * 100)}%, test ${Math.round(a[1][1] * 100)}%, gap ${Math.round((a[1][0] - a[1][1]) * 100)} points`);
  $("b7run").textContent = S7.timer ? tr("멈춤", "Pause") : tr("둘 다 학습", "Train both");
}
$("b7run").addEventListener("click", () => {
  if (S7.timer) { stopRun(S7); upd7(); return; }
  runSteps(S7, 50, () => { for (let i = 0; i < 10; i++) S7.nets.forEach(n => n.step(TR7.X, TR7.Y)); S7.ep += 10; upd7(); if (S7.ep >= 500) return false; }, 40, upd7); upd7();
});
$("b7reset").addEventListener("click", reset7);

/* ================= 8절 / Section 8 ================= */
// L1과 L2: loss L(w) = ½(w − w*)ᵀA(w − w*)에 벌을 더한 해. L1은 모서리에서 만나 w₂ = 0을 만든다
// L1 and L2: the minimiser of L(w) = ½(w − w*)ᵀA(w − w*) plus a penalty. L1 meets at a corner and makes w₂ = 0
const A8 = [[1, 0.4], [0.4, 0.5]], WS = [2, 0.6];
const loss8 = (x, y) => { const d = [x - WS[0], y - WS[1]]; return 0.5 * (d[0] * (A8[0][0] * d[0] + A8[0][1] * d[1]) + d[1] * (A8[1][0] * d[0] + A8[1][1] * d[1])); };
function l2sol(lam) { const M = [[A8[0][0] + lam, A8[0][1]], [A8[1][0], A8[1][1] + lam]], r = matVec(A8, WS); return solve(M, r); }
function l1sol(lam) {
  let w = WS.slice(); const t = 1 / 1.22;
  for (let k = 0; k < 3000; k++) { const g = matVec(A8, [w[0] - WS[0], w[1] - WS[1]]); w = w.map((v, i) => { const u = v - t * g[i]; return Math.sign(u) * Math.max(Math.abs(u) - t * lam, 0); }); }
  return w;
}
const LAM8 = Array.from({ length: 61 }, (_, i) => i * 0.025), P2 = LAM8.map(l2sol), P1 = LAM8.map(l1sol);
const S8 = { lam: 0.5 };
const F8 = fig("cv8", w => (stacked(w, 260, 2, 24) ? 620 : 320), (ctx, w, h, c) => {
  const P = panels(w, h, 2, 24, 260), R = [-0.6, 2.6], w1 = l1sol(S8.lam), w2 = l2sol(S8.lam);
  [[tr("L2 (ridge): 원", "L2 (ridge): circle"), w2, P2, "l2"], [tr("L1 (lasso): 마름모", "L1 (lasso): diamond"), w1, P1, "l1"]].forEach(([name, sol, path, kind], k) => {
    const p = P[k], side = Math.min(p.w - 40, p.h - 56), B = { l: p.x + 34, t: p.y + 28, w: side, h: side };
    contour(ctx, c, B, loss8, R, R, "l1l2");
    const A = axes(ctx, c, B, R, R, { xt: [0, 1, 2], yt: [0, 1, 2], grid: false });
    panelTitle(ctx, c, name, B.l, p.y + 16);
    line(ctx, [[A.X(0), B.t], [A.X(0), B.t + B.h]], c.muted, 1); line(ctx, [[B.l, A.Y(0)], [B.l + B.w, A.Y(0)]], c.muted, 1);
    // 해의 크기와 같은 원·마름모
    // Circle or diamond with the same size as the solution
    clipBox(ctx, B); ctx.save(); ctx.beginPath();
    if (kind === "l2") { const r = Math.hypot(...sol); ctx.ellipse(A.X(0), A.Y(0), A.X(r) - A.X(0), A.X(r) - A.X(0), 0, 0, 2 * Math.PI); }
    else { const r = Math.abs(sol[0]) + Math.abs(sol[1]); ctx.moveTo(A.X(r), A.Y(0)); ctx.lineTo(A.X(0), A.Y(r)); ctx.lineTo(A.X(-r), A.Y(0)); ctx.lineTo(A.X(0), A.Y(-r)); ctx.closePath(); }
    ctx.fillStyle = c.okwash; ctx.globalAlpha = 0.55; ctx.fill(); ctx.globalAlpha = 1; ctx.strokeStyle = c.ok; ctx.lineWidth = 2; ctx.stroke(); ctx.restore(); ctx.restore();
    line(ctx, path.map(v => [A.X(v[0]), A.Y(v[1])]), c.miss, 1.6, [4, 3]);
    dot(ctx, A.X(WS[0]), A.Y(WS[1]), 5, c.ink);
    txtBg(ctx, c, "w*", A.X(WS[0]) + 8, A.Y(WS[1]) - 6, { color: c.ink, size: 11.5, w: "700" });
    dot(ctx, A.X(sol[0]), A.Y(sol[1]), 6.5, c.miss, c.bg);
    txt(ctx, "w₁", B.l + B.w - 4, A.Y(0) - 6, { color: c.muted, size: 11, align: "right" }); txt(ctx, "w₂", A.X(0) + 6, B.t + 12, { color: c.muted, size: 11 });
  });
});
function upd8() {
  paint(F8);
  const a = l2sol(S8.lam), b = l1sol(S8.lam);
  $("r8").textContent = tr(`λ = ${fx(S8.lam)}:  L2 해 w = (${fx(a[0], 3)}, ${fx(a[1], 3)})   L1 해 w = (${fx(b[0], 3)}, ${fx(b[1], 3)})\n→ ${Math.abs(b[1]) < 1e-9 ? "L1: w₂가 정확히 0. 마름모의 모서리(축 위)에서 등고선과 만났다" : "L1: 아직 w₂ ≠ 0. λ ≥ 0.34이면 w₂ = 0이 된다"}. L2는 둘 다 줄지만 0이 되지는 않는다`,
    `λ = ${fx(S8.lam)}:  L2 solution w = (${fx(a[0], 3)}, ${fx(a[1], 3)})   L1 solution w = (${fx(b[0], 3)}, ${fx(b[1], 3)})\n→ ${Math.abs(b[1]) < 1e-9 ? "L1: w₂ is exactly 0. The contours met the diamond at a corner (on an axis)" : "L1: w₂ ≠ 0 so far. From λ ≥ 0.34 on, w₂ = 0"}. L2 shrinks both but never makes them 0`);
}
bindRange("i8l", "o8l", v => fx(v), v => { S8.lam = v; upd8(); });

// 처음 그리기
// First paint
upd1(); upd2(); upd3(); upd5(); newMask(); paint(F7a); reset7(); upd8();
repaintAll();
