/* 6장 그림과 조작. 한글판(ch06-activation-loss-normalization-KR.html)과 영어판(ch06-activation-loss-normalization.html)이 같이 쓴다
   Chapter 6 figures and interactions, shared by the Korean and English pages
   wine data는 assets/data/ch06_data.js (tools/ch06_numbers.py가 만든다)
   The wine data live in assets/data/ch06_data.js (made by tools/ch06_numbers.py) */
"use strict";

/* ================= 1절 / Section 1 ================= */
// activation 탐색기: 왼쪽 = f(z), 오른쪽 = 기울기 f′(z). 회색 = 다른 함수들
// Activation explorer: left = f(z), right = slope f′(z). Grey = the other functions
const ACTS = { sigmoid: "sigmoid", tanh: "tanh", relu: "ReLU", leaky: "LeakyReLU(0.1)", elu: "ELU", gelu: "GELU" };
const S1 = { k: "sigmoid", z: 1 };
const dAct = (k, z) => ACT[k].d(z, ACT[k].f(z));
const F1 = fig("cv1", w => (stacked(w, 260, 2, 28) ? 560 : 290), (ctx, w, h, c) => {
  const P = panels(w, h, 2, 28, 260), zr = [-4, 4];
  [[tr("f(z)", "f(z)"), (k, z) => ACT[k].f(z), [-1.6, 4.2], [-1, 0, 1, 2, 3, 4]], [tr("기울기 f′(z)", "slope f′(z)"), dAct, [-0.3, 1.3], [0, 0.25, 0.5, 1]]].forEach(([name, F, yr, yt], k) => {
    const p = P[k], B = { l: p.x + 40, t: p.y + 30, w: p.w - 52, h: p.h - 76 };
    const A = axes(ctx, c, B, zr, yr, { xt: [-4, -2, 0, 2, 4], yt, xlabel: "z" });
    panelTitle(ctx, c, `${ACTS[S1.k]}: ${name}`, B.l, p.y + 14);
    clipBox(ctx, B);
    line(ctx, [[B.l, A.Y(0)], [B.l + B.w, A.Y(0)]], c.muted, 1);
    Object.keys(ACTS).forEach(a => { if (a !== S1.k) line(ctx, curve(A, z => F(a, z), zr, 200), c.line, 1.4); });
    line(ctx, curve(A, z => F(S1.k, z), zr, 300), k ? c.ok : c.sig, 2.8);
    ctx.restore();
    dot(ctx, A.X(S1.z), A.Y(F(S1.k, S1.z)), 6, c.miss, c.bg);
    if (k === 1 && S1.k === "sigmoid") { line(ctx, [[B.l, A.Y(0.25)], [B.l + B.w, A.Y(0.25)]], c.fa, 1.2, [5, 4]); txtBg(ctx, c, tr("최대 0.25", "max 0.25"), B.l + B.w - 4, A.Y(0.25) - 5, { color: c.fa, size: 11, w: "700", align: "right" }); }
  });
});
const NOTE1 = {
  sigmoid: [tr("출력 0~1. 기울기 최대 0.25 → 층이 깊으면 gradient가 사라진다", "output 0–1. Slope at most 0.25 → gradients vanish in deep networks"), "nn.Sigmoid()"],
  tanh: [tr("출력 −1~1, 0을 가운데로. 양 끝에서 기울기가 0에 가깝다", "output −1 to 1, centred on 0. Slope near 0 at both ends"), "nn.Tanh()"],
  relu: [tr("z > 0이면 기울기 1 (사라지지 않음). z < 0이면 0 → 영영 꺼진 neuron(dead ReLU)이 생길 수 있다", "slope 1 for z > 0 (does not vanish). 0 for z < 0 → neurons can die for good (dead ReLU)"), "nn.ReLU()"],
  leaky: [tr("z < 0에서도 기울기 0.1 (PyTorch 기본은 0.01): 꺼지지 않는다", "slope 0.1 even for z < 0 (PyTorch's default is 0.01): never dies"), "nn.LeakyReLU(0.1)"],
  elu: [tr("z < 0에서 부드럽게 −1로. 평균이 0에 가까워진다", "goes smoothly to −1 for z < 0. Keeps the mean nearer 0"), "nn.ELU()"],
  gelu: [tr("ReLU를 매끄럽게 한 모양. Transformer(14장)의 기본", "a smooth ReLU. The default in Transformers (chapter 14)"), "nn.GELU()"]
};
function upd1() {
  paint(F1);
  const k = S1.k, z = S1.z;
  $("r1").textContent = tr(`${ACTS[k]}: f(${fx(z)}) = ${fx(ACT[k].f(z), 4)}, 기울기 f′(${fx(z)}) = ${fx(dAct(k, z), 4)}   PyTorch: ${NOTE1[k][1]}\n→ ${NOTE1[k][0]}`,
    `${ACTS[k]}: f(${fx(z)}) = ${fx(ACT[k].f(z), 4)}, slope f′(${fx(z)}) = ${fx(dAct(k, z), 4)}   PyTorch: ${NOTE1[k][1]}\n→ ${NOTE1[k][0]}`);
}
bindSeg("g1f", v => { S1.k = v; upd1(); });
bindRange("i1z", "o1z", v => fx(v, 1), v => { S1.z = v; upd1(); });

/* ================= 2절 / Section 2 ================= */
// 숨은 층 10개짜리 network에서 층마다 gradient 크기 (log 눈금). 지금 browser에서 계산
// Gradient size per layer in a network with 10 hidden layers (log scale). Computed live in the browser
const R2 = makeRng(3), X2 = Array.from({ length: 32 }, () => Array.from({ length: 16 }, () => R2.n())), Y2 = X2.map((_, i) => i % 2);
const S2 = { a: "sigmoid", seed: 4, m: null };
function comp2() {
  const net = new TinyNet([16, ...Array(10).fill(32), 1], { hidden: S2.a, seed: S2.seed, init: S2.a === "relu" ? "kaiming" : "xavier" });
  const { gW } = net.grads(X2, Y2, false);
  S2.m = gW.map(W => { const f = W.flat(); return f.reduce((s, v) => s + Math.abs(v), 0) / f.length; });
}
const F2 = fig("cv2", 290, (ctx, w, h, c) => {
  const B = { l: 56, t: 30, w: w - 70, h: h - 80 }, n = S2.m.length, cw = B.w / n;
  const A = axes(ctx, c, B, [0, n], [-9, 0], { xt: [], yt: [[-8, "10⁻⁸"], [-6, "10⁻⁶"], [-4, "10⁻⁴"], [-2, "0.01"], [0, "1"]] });
  panelTitle(ctx, c, tr(`층마다 |∂L/∂W|의 평균 (${S2.a}, 숨은 층 10개 × 32)`, `Mean |∂L/∂W| per layer (${S2.a}, 10 hidden layers × 32)`), B.l, 16);
  S2.m.forEach((v, i) => {
    const x = B.l + i * cw + cw * 0.18, y = A.Y(clamp(Math.log10(v), -9, 0)), bad = v < 1e-4;
    ctx.fillStyle = bad ? c.fa : c.ok; ctx.globalAlpha = 0.8; ctx.fillRect(x, y, cw * 0.64, B.t + B.h - y); ctx.globalAlpha = 1;
    txt(ctx, i < n - 1 ? String(i + 1) : tr("출력", "out"), x + cw * 0.32, B.t + B.h + 15, { color: c.muted, size: 11, align: "center", mono: i < n - 1 });
    if (cw > 36) txt(ctx, sci(v), x + cw * 0.32, y - 5, { color: bad ? c.fa : c.ok, size: 9.5, align: "center", mono: true });
  });
  txt(ctx, tr("층 번호 (1 = 입력 쪽)", "layer (1 = input side)"), B.l + B.w / 2, B.t + B.h + 33, { color: c.muted, size: 11.5, align: "center" });
});
function upd2() {
  comp2(); paint(F2);
  const m = S2.m, r = m[0] / m[m.length - 2];
  $("r2").textContent = tr(`1층 / 10층 gradient 크기 비 = ${sci(r)}\n→ ${r < 1e-3 ? "1층은 거의 배우지 못한다 (sigmoid의 기울기 ≤ 0.25가 10번 곱해졌다)" : "층마다 비슷하다: 입력 쪽 층도 배운다"}`,
    `ratio of gradient size, layer 1 / layer 10 = ${sci(r)}\n→ ${r < 1e-3 ? "layer 1 barely learns (the sigmoid's slope ≤ 0.25 was multiplied ten times)" : "about the same at every layer: the input-side layers learn too"}`);
}
bindSeg("g2a", v => { S2.a = v; upd2(); });
$("b2seed").addEventListener("click", () => { S2.seed++; upd2(); });

/* ================= 3절 / Section 3 ================= */
// 정답 y = 1. 왼쪽 = loss, 오른쪽 = z에 대한 기울기의 크기. MSE는 크게 틀릴 때 기울기가 사라진다
// Answer y = 1. Left = loss, right = size of the slope with respect to z. MSE's slope vanishes exactly when it is very wrong
const S3 = { p: 0.01 };
const mse3 = p => (1 - p) ** 2, bce3 = p => -Math.log(p), gm3 = p => 2 * (1 - p) * p * (1 - p), gb3 = p => 1 - p;
const F3 = fig("cv3", w => (stacked(w, 260, 2, 28) ? 560 : 290), (ctx, w, h, c) => {
  const P = panels(w, h, 2, 28, 260), pr = [0.001, 1];
  [[tr("loss (정답 y = 1)", "Loss (answer y = 1)"), mse3, bce3, [0, 5], [0, 1, 2, 3, 4, 5]], [tr("|기울기| = 고치는 힘", "|slope| = how hard it pushes"), gm3, gb3, [0, 1.05], [0, 0.25, 0.5, 0.75, 1]]].forEach(([name, fm, fb, yr, yt], k) => {
    const p = P[k], B = { l: p.x + 40, t: p.y + 30, w: p.w - 52, h: p.h - 76 };
    const A = axes(ctx, c, B, [0, 1], yr, { xt: [0, 0.25, 0.5, 0.75, 1], yt, xlabel: tr("정답에 준 확률 p", "probability p given to the answer") });
    panelTitle(ctx, c, name, B.l, p.y + 14);
    clipBox(ctx, B);
    line(ctx, curve(A, fm, pr, 300), c.miss, 2.4); line(ctx, curve(A, fb, pr, 300), c.sig, 2.4);
    ctx.restore();
    dot(ctx, A.X(S3.p), A.Y(Math.min(fm(S3.p), yr[1])), 5.5, c.miss, c.bg); dot(ctx, A.X(S3.p), A.Y(Math.min(fb(S3.p), yr[1])), 5.5, c.sig, c.bg);
    txtBg(ctx, c, "MSE", A.X(0.75), A.Y(Math.min(fm(0.75), yr[1])) - 10, { color: c.miss, size: 12, w: "700", align: "center" });
    txtBg(ctx, c, tr("cross-entropy", "cross-entropy"), A.X(k ? 0.3 : 0.14), A.Y(Math.min(fb(k ? 0.3 : 0.14), yr[1])) - 10, { color: c.sig, size: 12, w: "700" });
  });
});
function upd3() {
  paint(F3);
  const p = S3.p;
  $("r3").textContent = tr(`p = ${fx(p, 3)} (정답 1에 ${fx(p * 100, 1)}%):\nMSE = ${fx(mse3(p), 4)}, 고치는 힘 |∂L/∂z| = ${fx(gm3(p), 4)}\ncross-entropy = ${fx(bce3(p), 4)}, 고치는 힘 |∂L/∂z| = ${fx(gb3(p), 4)}   (${fx(gb3(p) / Math.max(gm3(p), 1e-9), 1)}배)`,
    `p = ${fx(p, 3)} (${fx(p * 100, 1)}% on the answer 1):\nMSE = ${fx(mse3(p), 4)}, push |∂L/∂z| = ${fx(gm3(p), 4)}\ncross-entropy = ${fx(bce3(p), 4)}, push |∂L/∂z| = ${fx(gb3(p), 4)}   (${fx(gb3(p) / Math.max(gm3(p), 1e-9), 1)}×)`);
}
bindRange("i3p", "o3p", v => fx(v, 3), v => { S3.p = v; upd3(); });

/* ================= 5절 / Section 5 ================= */
// 특성 크기가 다르면 loss 등고선이 길쭉하다. z-score로 맞추면 둥글어져 GD가 곧게 간다
// Features of different sizes stretch the loss contours. z-scoring rounds them and GD goes straight
const R5 = makeRng(6), X5 = [], Y5 = [];
for (let i = 0; i < 100; i++) { const a = R5.n(), b = 5 * R5.n(); X5.push([a, b]); Y5.push(a + 0.2 * b + 0.3 * R5.n()); }
function moments(X, Y) {
  const n = X.length, mx = [0, 1].map(j => meanOf(X.map(r => r[j]))), my = meanOf(Y);
  const Xc = X.map(r => [r[0] - mx[0], r[1] - mx[1]]), Yc = Y.map(v => v - my);
  const S = [0, 1].map(i => [0, 1].map(j => Xc.reduce((s, r) => s + r[i] * r[j], 0) / n)), b = [0, 1].map(i => Xc.reduce((s, r, k) => s + r[i] * Yc[k], 0) / n), cc = Yc.reduce((s, v) => s + v * v, 0) / n;
  return { S, b, cc };
}
const SD5 = [0, 1].map(j => sdOf(X5.map(r => r[j])));
const MOM = { raw: moments(X5, Y5), z: moments(X5.map(r => [r[0] / SD5[0], r[1] / SD5[1]]), Y5) };
const lossQ = (M, w) => w[0] * (M.S[0][0] * w[0] + M.S[0][1] * w[1]) + w[1] * (M.S[1][0] * w[0] + M.S[1][1] * w[1]) - 2 * (M.b[0] * w[0] + M.b[1] * w[1]) + M.cc;
const gradQ = (M, w) => [2 * (M.S[0][0] * w[0] + M.S[0][1] * w[1] - M.b[0]), 2 * (M.S[1][0] * w[0] + M.S[1][1] * w[1] - M.b[1])];
const lamMax = M => { const a = M.S[0][0], d = M.S[1][1], b = M.S[0][1]; return (a + d) / 2 + Math.sqrt(((a - d) / 2) ** 2 + b * b); };
const wStar = M => { const a = M.S[0][0], b = M.S[0][1], d = M.S[1][1], det = a * d - b * b; return [(d * M.b[0] - b * M.b[1]) / det, (a * M.b[1] - b * M.b[0]) / det]; };
const S5 = { f: 0.5 };
function path5(M, n = 30) { const lr = S5.f * 2 / (2 * lamMax(M)), P = [[-1, -0.8]]; for (let i = 0; i < n; i++) { const w = P[P.length - 1], g = gradQ(M, w); P.push([w[0] - lr * g[0], w[1] - lr * g[1]]); } return { P, lr }; }
function stepsTo(M) { const lr = S5.f * 2 / (2 * lamMax(M)), ws = wStar(M), d0 = Math.hypot(-1 - ws[0], -0.8 - ws[1]); let w = [-1, -0.8]; for (let i = 1; i <= 5000; i++) { const g = gradQ(M, w); w = [w[0] - lr * g[0], w[1] - lr * g[1]]; if (Math.hypot(w[0] - ws[0], w[1] - ws[1]) < 0.01 * d0) return i; } return 5000; }
const F5 = fig("cv5", w => (stacked(w, 260, 2, 24) ? 620 : 320), (ctx, w, h, c) => {
  const P = panels(w, h, 2, 24, 260);
  [["raw", tr("원래 단위 (x₂가 x₁보다 5배 크다)", "Original units (x₂ is 5× larger than x₁)")], ["z", tr("z-score 뒤 (둘 다 표준편차 1)", "After z-score (both have SD 1)")]].forEach(([k, name], i) => {
    const p = P[i], side = Math.min(p.w - 40, p.h - 56), B = { l: p.x + 34, t: p.y + 28, w: side, h: side }, M = MOM[k], ws = wStar(M);
    const R = { xr: [-1.6, 2.2], yr: [-1.6, 2.2] };
    contour(ctx, c, B, (a, b) => Math.sqrt(lossQ(M, [a, b])), R.xr, R.yr, "norm" + k);
    const A = axes(ctx, c, B, R.xr, R.yr, { xt: [-1, 0, 1, 2], yt: [-1, 0, 1, 2], grid: false });
    panelTitle(ctx, c, name, B.l, p.y + 16);
    const { P: path } = path5(M);
    clipBox(ctx, B); line(ctx, path.map(v => [A.X(v[0]), A.Y(v[1])]), c.miss, 2); path.forEach(v => dot(ctx, A.X(v[0]), A.Y(v[1]), 2.4, c.miss)); ctx.restore();
    dot(ctx, A.X(ws[0]), A.Y(ws[1]), 5.5, c.ok, c.bg); dot(ctx, A.X(-1), A.Y(-0.8), 5, c.ink);
    txt(ctx, "w₁", B.l + B.w - 4, B.t + B.h - 6, { color: c.muted, size: 11, align: "right" }); txt(ctx, "w₂", B.l + 6, B.t + 12, { color: c.muted, size: 11 });
  });
});
function upd5() {
  paint(F5);
  const r = MOM.raw, z = MOM.z, cr = lamMax(r) / (r.S[0][0] + r.S[1][1] - lamMax(r)), cz = lamMax(z) / (z.S[0][0] + z.S[1][1] - lamMax(z));
  $("r5").textContent = tr(`같은 30걸음, η = 가능한 최대의 ${Math.round(S5.f * 100)}%\n원래 단위: 등고선 길이 비 ${fx(Math.sqrt(cr), 1)} : 1, w*까지 1% 안으로 ${stepsTo(r)}걸음\nz-score 뒤: 등고선 길이 비 ${fx(Math.sqrt(cz), 1)} : 1, w*까지 1% 안으로 ${stepsTo(z)}걸음`,
    `the same 30 steps, η = ${Math.round(S5.f * 100)}% of the largest stable value\noriginal units: contour axis ratio ${fx(Math.sqrt(cr), 1)} : 1, ${stepsTo(r)} steps to get within 1% of w*\nafter z-score: contour axis ratio ${fx(Math.sqrt(cz), 1)} : 1, ${stepsTo(z)} steps to get within 1% of w*`);
}
bindRange("i5f", "o5f", v => Math.round(v * 100) + "%", v => { S5.f = v; upd5(); });

/* ================= 6절 / Section 6 ================= */
// batch normalization: batch 16개의 평균·분산으로 맞추고 γ, β로 다시 늘이고 옮긴다
// Batch normalization: normalise with the 16-sample batch's mean and variance, then stretch and shift with γ, β
const S6 = { x: [], gamma: 1, beta: 0, t: 1, timer: 0, rm: 0, rv: 1, nb: 0, rng: makeRng(21) };
function batch6() { const mu = 3 + S6.rng.n() * 0.6, sd = 2 * (1 + S6.rng.n() * 0.15); S6.x = Array.from({ length: 16 }, () => mu + sd * S6.rng.n()); const m = meanOf(S6.x), v = meanOf(S6.x.map(a => (a - m) ** 2)); S6.rm = 0.9 * S6.rm + 0.1 * m; S6.rv = 0.9 * S6.rv + 0.1 * v * 16 / 15; S6.nb++; }
const bnStats = () => { const m = meanOf(S6.x), v = meanOf(S6.x.map(a => (a - m) ** 2)); return { m, v, sd: Math.sqrt(v + 1e-5) }; };
const F6 = fig("cv6", 300, (ctx, w, h, c) => {
  const B = { l: 130, t: 30, w: w - 150 }, xr = [-6, 10], X = v => B.l + (v - xr[0]) / (xr[1] - xr[0]) * B.w, { m, sd } = bnStats(), e = S6.t;
  const rows = [[tr("입력 x", "input x"), v => v, c.sig], [tr("정규화 x̂ = (x − μ)/σ", "normalised x̂ = (x − μ)/σ"), v => (v - m) / sd, c.ok], [tr("출력 γx̂ + β", "output γx̂ + β"), v => S6.gamma * (v - m) / sd + S6.beta, c.miss]];
  rows.forEach(([name, f, col], r) => {
    const y = B.t + 30 + r * 82;
    txt(ctx, name, 12, y + 4, { color: col, size: 11.5, w: "700" });
    line(ctx, [[X(xr[0]), y], [X(xr[1]), y]], c.muted, 1);
    for (let t = -6; t <= 10; t += 2) { line(ctx, [[X(t), y - 3], [X(t), y + 3]], c.muted, 1); txt(ctx, String(t), X(t), y + 16, { color: c.muted, size: 10, mono: true, align: "center" }); }
    const vals = S6.x.map(v => (r === 0 ? v : rows[0][1](v) + (f(v) - rows[0][1](v)) * e));
    vals.forEach((v, i) => dot(ctx, X(clamp(v, xr[0], xr[1])), y - 10 - (i % 3) * 5, 4, col, c.bg));
    const mm = meanOf(vals), ss = sdOf(vals);
    line(ctx, [[X(mm), y - 30], [X(mm), y + 4]], col, 2);
    line(ctx, [[X(mm - ss), y - 26], [X(mm + ss), y - 26]], col, 1.4, [3, 2]);
    txtBg(ctx, c, `μ ${fx(mm, 2)}, σ ${fx(ss, 2)}`, X(mm) + 6, y - 32, { color: col, size: 10.5, w: "600", mono: true });
  });
});
function upd6() {
  paint(F6);
  const { m, v } = bnStats();
  $("r6").textContent = tr(`batch ${S6.nb}: μ_B = ${fx(m)}, σ²_B = ${fx(v)} → x̂의 평균 0, 표준편차 1 → γ = ${fx(S6.gamma)}, β = ${fx(S6.beta)}로 출력의 평균 ${fx(S6.beta)}, 표준편차 ${fx(Math.abs(S6.gamma))}\n평가(eval)에 쓸 running 평균 ${fx(S6.rm)}, running 분산 ${fx(S6.rv)} (batch마다 0.9 × 이전 + 0.1 × 이번)`,
    `batch ${S6.nb}: μ_B = ${fx(m)}, σ²_B = ${fx(v)} → x̂ has mean 0, SD 1 → with γ = ${fx(S6.gamma)}, β = ${fx(S6.beta)} the output has mean ${fx(S6.beta)}, SD ${fx(Math.abs(S6.gamma))}\nrunning mean ${fx(S6.rm)}, running variance ${fx(S6.rv)} for evaluation (each batch: 0.9 × old + 0.1 × this batch)`);
}
$("b6next").addEventListener("click", () => { batch6(); S6.t = 0; upd6(); runSteps(S6, 12, () => { S6.t = Math.min(1, S6.t + 1 / 12); paint(F6); }, 50, upd6); });
bindRange("i6g", "o6g", v => fx(v, 1), v => { S6.gamma = v; upd6(); });
bindRange("i6b", "o6b", v => fx(v, 1), v => { S6.beta = v; upd6(); });

/* ================= 7절 / Section 7 ================= */
// wine quality: 특성 11개의 크기와, 정규화 없이 / z-score로 학습한 시험 정확도
// Wine quality: the sizes of the 11 features, and test accuracy trained without / with z-scoring
const WN = CH06_WINE, TES = new Set(WN.test), TRI = WN.X.map((_, i) => i).filter(i => !TES.has(i)), TEI = WN.test;
const MU7 = WN.names.map((_, j) => meanOf(TRI.map(i => WN.X[i][j]))), SDV7 = WN.names.map((_, j) => sdOf(TRI.map(i => WN.X[i][j])));
const DATA7 = [false, true].map(norm => { const f = x => (norm ? x.map((v, j) => (v - MU7[j]) / SDV7[j]) : x); return { Xt: TRI.map(i => f(WN.X[i])), Yt: TRI.map(i => WN.Y[i]), Xe: TEI.map(i => f(WN.X[i])), Ye: TEI.map(i => WN.Y[i]) }; });
const S7 = { nets: null, hist: [[], []], ep: 0, timer: 0 };
function reset7() { stopRun(S7); S7.nets = [0, 1].map(() => new TinyNet([11, 16, 1], { hidden: "relu", seed: 1, init: "kaiming", opt: { name: "sgd", lr: 0.01 } })); S7.hist = [[], []]; S7.ep = 0; upd7(); }
const F7 = fig("cv7", w => (stacked(w, 280, 2, 28) ? 640 : 330), (ctx, w, h, c) => {
  const P = panels(w, h, 2, 28, 280);
  {
    const p = P[0], B = { l: p.x + 128, t: p.y + 30, w: p.w - 140, h: p.h - 60 }, n = 11, rh = B.h / n;
    panelTitle(ctx, c, tr("특성 11개의 표준편차 (log 눈금)", "Standard deviation of the 11 features (log scale)"), p.x + 8, p.y + 14);
    const X = v => B.l + (Math.log10(v) + 3.2) / 5 * B.w;
    [-3, -2, -1, 0, 1].forEach(e => { line(ctx, [[X(10 ** e), B.t], [X(10 ** e), B.t + B.h]], c.line, 1); txt(ctx, e === 0 ? "1" : "10" + SUP(e), X(10 ** e), B.t + B.h + 13, { color: c.muted, size: 10, align: "center", mono: true }); });
    SDV7.forEach((s, j) => {
      const y = B.t + j * rh;
      txt(ctx, WN.names[j], B.l - 6, y + rh / 2 + 4, { color: c.ink, size: 10.5, align: "right" });
      ctx.fillStyle = c.miss; ctx.globalAlpha = 0.75; ctx.fillRect(B.l, y + rh * 0.2, X(s) - B.l, rh * 0.6); ctx.globalAlpha = 1;
    });
    line(ctx, [[X(1), B.t], [X(1), B.t + B.h]], c.ok, 2.4);
    txtBg(ctx, c, tr("z-score 뒤 = 모두 1", "after z-score = all 1"), X(1) + 5, B.t + 10, { color: c.ok, size: 11, w: "700" });
  }
  {
    const p = P[1], B = { l: p.x + 44, t: p.y + 30, w: p.w - 56, h: p.h - 76 };
    const A = axes(ctx, c, B, [0, 60], [0.45, 0.8], { xt: [0, 20, 40, 60], yt: [[0.5, "50%"], [0.6, "60%"], [0.7, "70%"], [0.8, "80%"]], xlabel: "epoch" });
    panelTitle(ctx, c, tr("시험 정확도 (320개)", "Test accuracy (320 wines)"), B.l, p.y + 14);
    [[S7.hist[0], c.miss, tr("원래 단위", "original units")], [S7.hist[1], c.ok, "z-score"]].forEach(([H, col, name], k) => {
      if (H.length) { clipBox(ctx, B); line(ctx, H.map((v, i) => [A.X(i + 1), A.Y(v)]), col, 2.2); ctx.restore(); txtBg(ctx, c, `${name} ${Math.round(H[H.length - 1] * 100)}%`, A.X(H.length), A.Y(H[H.length - 1]) + (k ? -10 : 18), { color: col, size: 11.5, w: "700", align: H.length > 40 ? "right" : "left" }); }
    });
  }
});
function upd7() {
  paint(F7);
  const a = S7.hist.map(H => (H.length ? Math.round(H[H.length - 1] * 100) + "%" : "–"));
  $("r7").textContent = tr(`epoch ${S7.ep} (network 11-16-1, SGD η 0.01, batch 32, 학습 1279개 · 시험 320개)\n원래 단위: 시험 ${a[0]}   z-score: 시험 ${a[1]}\n→ 큰 특성(total sulfur dioxide, 표준편차 33)이 작은 특성(density, 0.002)을 덮어 버린다. z-score로 맞추면 모든 특성이 고르게 배운다`,
    `epoch ${S7.ep} (network 11-16-1, SGD η 0.01, batch 32, 1279 training · 320 test wines)\noriginal units: test ${a[0]}   z-score: test ${a[1]}\n→ a large feature (total sulfur dioxide, SD 33) drowns out a small one (density, 0.002). With z-scoring every feature gets a fair say`);
  $("b7run").textContent = S7.timer ? tr("멈춤", "Pause") : tr("둘 다 학습", "Train both");
}
$("b7run").addEventListener("click", () => {
  if (S7.timer) { stopRun(S7); upd7(); return; }
  runSteps(S7, 30, () => { for (let r = 0; r < 2; r++) { S7.nets.forEach((n, k) => { n.epoch(DATA7[k].Xt, DATA7[k].Yt, 32); }); S7.ep++; S7.nets.forEach((n, k) => S7.hist[k].push(n.accuracy(DATA7[k].Xe, DATA7[k].Ye))); } upd7(); if (S7.ep >= 60) return false; }, 30, upd7); upd7();
});
$("b7reset").addEventListener("click", reset7);

// 처음 그리기
// First paint
upd1(); upd2(); upd3(); upd5(); batch6(); upd6(); reset7();
repaintAll();
