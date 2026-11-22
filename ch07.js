/* 7장 그림과 조작. 한글판(ch07-optimizers-KR.html)과 영어판(ch07-optimizers.html)이 같이 쓴다
   Chapter 7 figures and interactions, shared by the Korean and English pages */
"use strict";

// 2D 함수 위에서 optimizer 한 종류로 n걸음. 반환 = 지나간 점들
// n steps of one optimizer on a 2D function. Returns the points visited
function optPath(name, grad, start, lr, n, o = {}) {
  const b1 = o.beta === undefined ? 0.9 : o.beta, b2 = 0.999, rho = 0.9, eps = 1e-8;
  let w = start.slice(), m = [0, 0], v = [0, 0]; const P = [w.slice()];
  for (let t = 1; t <= n; t++) {
    const g = grad(w);
    if (name === "sgd") w = w.map((x, i) => x - lr * g[i]);
    else if (name === "momentum") { m = m.map((x, i) => b1 * x + g[i]); w = w.map((x, i) => x - lr * m[i]); }
    else if (name === "rmsprop") { v = v.map((x, i) => rho * x + (1 - rho) * g[i] * g[i]); w = w.map((x, i) => x - lr * g[i] / (Math.sqrt(v[i]) + eps)); }
    else { m = m.map((x, i) => b1 * x + (1 - b1) * g[i]); v = v.map((x, i) => b2 * x + (1 - b2) * g[i] * g[i]); w = w.map((x, i) => x - lr * (m[i] / (1 - b1 ** t)) / (Math.sqrt(v[i] / (1 - b2 ** t)) + eps)); }
    if (!w.every(isFinite) || Math.abs(w[0]) > 1e3 || Math.abs(w[1]) > 1e3) break;
    P.push(w.slice());
  }
  return P;
}
const OPT_COL = c => ({ sgd: c.muted, momentum: c.sig, rmsprop: c.ok, adam: c.miss });
const OPT_NAME = { sgd: "SGD", momentum: "momentum", rmsprop: "RMSprop", adam: "Adam" };
function pathOnMap(ctx, A, P, col, w = 2) { line(ctx, P.map(p => [A.X(p[0]), A.Y(p[1])]), col, w); }

/* ================= 1절 / Section 1 ================= */
// mini-batch gradient의 흔들림: 왼쪽 = batch마다 다른 gradient (점), 전체 gradient (화살표). 오른쪽 = 3 epoch 동안의 경로
// The wobble of mini-batch gradients: left = gradients of different batches (dots) and the full gradient (arrow). Right = the path over 3 epochs
const R1 = makeRng(8), N1 = 256, X1 = [], Y1 = [];
for (let i = 0; i < N1; i++) { const a = R1.n(), b = R1.n(); X1.push([a, b]); Y1.push(1.5 * a - 0.5 * b + 0.5 * R1.n()); }
function grad1(w, idx) { let g0 = 0, g1 = 0; idx.forEach(i => { const e = w[0] * X1[i][0] + w[1] * X1[i][1] - Y1[i]; g0 += 2 * e * X1[i][0]; g1 += 2 * e * X1[i][1]; }); return [g0 / idx.length, g1 / idx.length]; }
const ALL1 = [...Array(N1).keys()], loss1 = w => meanOf(ALL1.map(i => (w[0] * X1[i][0] + w[1] * X1[i][1] - Y1[i]) ** 2));
const S1 = { B: 8, cloud: null, path: null, seed: 1 };
function comp1() {
  const r = makeRng(S1.seed), w0 = [-0.5, 1];
  S1.cloud = Array.from({ length: 80 }, () => grad1(w0, Array.from({ length: S1.B }, () => Math.floor(r.u() * N1))));
  let w = [-0.5, 1]; const P = [w.slice()];
  for (let ep = 0; ep < 3; ep++) { const ord = shuffled(N1, r); for (let s = 0; s < N1; s += S1.B) { const g = grad1(w, ord.slice(s, s + S1.B)); w = [w[0] - 0.05 * g[0], w[1] - 0.05 * g[1]]; P.push(w.slice()); } }
  S1.path = P;
}
const F1 = fig("cv1", w => (stacked(w, 270, 2, 28) ? 600 : 310), (ctx, w, h, c) => {
  const P = panels(w, h, 2, 28, 270), gfull = grad1([-0.5, 1], ALL1);
  {
    const p = P[0], side = Math.min(p.w - 50, p.h - 60), B = { l: p.x + 40, t: p.y + 30, w: side, h: side }, R = [-9, 3], Ry = [-4, 8];
    const A = axes(ctx, c, B, R, Ry, { xt: [-8, -4, 0], yt: [-4, 0, 4, 8], xlabel: "∂L/∂w₁" });
    panelTitle(ctx, c, tr(`batch ${S1.B}개로 잰 gradient 80번`, `80 gradients from batches of ${S1.B}`), B.l, p.y + 14);
    clipBox(ctx, B); S1.cloud.forEach(g => { ctx.globalAlpha = 0.55; dot(ctx, A.X(g[0]), A.Y(g[1]), 3, c.sig); ctx.globalAlpha = 1; }); ctx.restore();
    arrow(ctx, A.X(0), A.Y(0), A.X(gfull[0]), A.Y(gfull[1]), c.fa, 2.6);
    txtBg(ctx, c, tr("전체 data의 gradient", "full-data gradient"), A.X(gfull[0]) + 8, A.Y(gfull[1]) - 8, { color: c.fa, size: 11.5, w: "700" });
  }
  {
    const p = P[1], side = Math.min(p.w - 50, p.h - 60), B = { l: p.x + 40, t: p.y + 30, w: side, h: side }, R = [-1, 2.2], Ry = [-1.6, 1.6];
    contour(ctx, c, B, (a, b) => loss1([a, b]), R, Ry, "ch7mb");
    const A = axes(ctx, c, B, R, Ry, { xt: [-1, 0, 1, 2], yt: [-1, 0, 1], grid: false });
    panelTitle(ctx, c, tr(`3 epoch의 경로 (${S1.path.length - 1}걸음)`, `Path over 3 epochs (${S1.path.length - 1} steps)`), B.l, p.y + 14);
    clipBox(ctx, B); pathOnMap(ctx, A, S1.path, c.miss, S1.B === 1 ? 1 : 1.8); ctx.restore();
    dot(ctx, A.X(-0.5), A.Y(1), 5, c.ink);
  }
});
function upd1() {
  comp1(); paint(F1);
  const gf = grad1([-0.5, 1], ALL1), sd = Math.sqrt(meanOf(S1.cloud.map(g => (g[0] - gf[0]) ** 2 + (g[1] - gf[1]) ** 2))), end = S1.path[S1.path.length - 1];
  $("r1").textContent = tr(`batch ${S1.B}개: epoch마다 ${Math.ceil(N1 / S1.B)}걸음, gradient의 흔들림(전체 gradient에서의 거리, 평균) = ${fx(sd)}\n3 epoch 뒤 w = (${fx(end[0])}, ${fx(end[1])}), loss ${fx(loss1(end), 3)}   (가장 좋은 w ≈ (1.5, −0.5))`,
    `batch of ${S1.B}: ${Math.ceil(N1 / S1.B)} steps per epoch, wobble of the gradient (mean distance from the full gradient) = ${fx(sd)}\nafter 3 epochs w = (${fx(end[0])}, ${fx(end[1])}), loss ${fx(loss1(end), 3)}   (best w ≈ (1.5, −0.5))`);
}
bindSeg("g1b", v => { S1.B = +v; upd1(); });
$("b1seed").addEventListener("click", () => { S1.seed++; upd1(); });

/* ================= 2절 / Section 2 ================= */
// 길쭉한 골짜기 f = (x² + 50y²)/2: SGD는 지그재그, momentum은 속도를 쌓아 빨리 간다
// A long valley f = (x² + 50y²)/2: SGD zigzags, momentum builds up speed
const fV = w => (w[0] ** 2 + 50 * w[1] ** 2) / 2, gV = w => [w[0], 50 * w[1]];
const S2 = { beta: 0.8, lr: 0.035, n: 40 };
const F2 = fig("cv2", w => Math.round(clamp(w * 0.45, 230, 340)) + 40, (ctx, w, h, c) => {
  const B = { l: 40, t: 14, w: w - 56, h: h - 44 }, R = equalRange(B, 2, -1.5);
  contour(ctx, c, B, (a, b) => fV([a, b]), R.xr, R.yr, "valley");
  const A = axes(ctx, c, B, R.xr, R.yr, { xt: intTicks(R.xr), yt: [-2, -1, 0, 1, 2], grid: false });
  const ps = optPath("sgd", gV, [-4.5, 1], S2.lr, S2.n), pm = optPath("momentum", gV, [-4.5, 1], S2.lr, S2.n, { beta: S2.beta });
  clipBox(ctx, B); pathOnMap(ctx, A, ps, c.muted, 2); pathOnMap(ctx, A, pm, c.sig, 2.4); ctx.restore();
  ps.forEach(p => dot(ctx, A.X(p[0]), A.Y(p[1]), 2, c.muted)); pm.forEach(p => dot(ctx, A.X(p[0]), A.Y(p[1]), 2.4, c.sig));
  dot(ctx, A.X(0), A.Y(0), 5, c.ok); dot(ctx, A.X(-4.5), A.Y(1), 5, c.ink);
  txtBg(ctx, c, tr(`회색 = SGD, 파랑 = momentum (β = ${fx(S2.beta, 2)}), 둘 다 η = ${fx(S2.lr, 3)}, ${S2.n}걸음`, `grey = SGD, blue = momentum (β = ${fx(S2.beta, 2)}), both η = ${fx(S2.lr, 3)}, ${S2.n} steps`), B.l + 6, B.t + 14, { color: c.ink, size: 11 });
});
function upd2() {
  paint(F2);
  const ps = optPath("sgd", gV, [-4.5, 1], S2.lr, S2.n), pm = optPath("momentum", gV, [-4.5, 1], S2.lr, S2.n, { beta: S2.beta });
  const a = ps[ps.length - 1], b = pm[pm.length - 1];
  $("r2").textContent = tr(`${S2.n}걸음 뒤 loss:  SGD ${sci(fV(a))}  (x = ${fx(a[0])})   momentum ${sci(fV(b))}  (x = ${fx(b[0])})\n→ ${fV(b) < fV(a) ? "momentum이 긴 방향으로 속도를 쌓아 더 멀리 갔다" : "β가 너무 크거나 작으면 이점이 줄어든다 (출렁임)"}`,
    `loss after ${S2.n} steps:  SGD ${sci(fV(a))}  (x = ${fx(a[0])})   momentum ${sci(fV(b))}  (x = ${fx(b[0])})\n→ ${fV(b) < fV(a) ? "momentum built up speed along the long direction and got further" : "with β too large or too small the advantage shrinks (overshooting)"}`);
}
bindRange("i2b", "o2b", v => fx(v, 2), v => { S2.beta = v; upd2(); });

/* ================= 3절 / Section 3 ================= */
// 같은 골짜기에서 RMSprop과 Adam: 방향마다 걸음 크기를 기울기의 크기로 나눠 맞춘다
// RMSprop and Adam in the same valley: each direction's step is divided by that direction's gradient size
const S3 = { n: 40 };
const F3 = fig("cv3", w => Math.round(clamp(w * 0.45, 230, 340)) + 40, (ctx, w, h, c) => {
  const B = { l: 40, t: 14, w: w - 56, h: h - 44 }, R = equalRange(B, 2, -1.5), col = OPT_COL(c);
  contour(ctx, c, B, (a, b) => fV([a, b]), R.xr, R.yr, "valley");
  const A = axes(ctx, c, B, R.xr, R.yr, { xt: intTicks(R.xr), yt: [-2, -1, 0, 1, 2], grid: false });
  const runs = [["sgd", 0.035], ["rmsprop", 0.2], ["adam", 0.2]];
  clipBox(ctx, B); runs.forEach(([k, lr]) => pathOnMap(ctx, A, optPath(k, gV, [-4.5, 1], lr, S3.n), col[k], 2.2)); ctx.restore();
  runs.forEach(([k, lr]) => optPath(k, gV, [-4.5, 1], lr, S3.n).forEach(p => dot(ctx, A.X(p[0]), A.Y(p[1]), 2.2, col[k])));
  dot(ctx, A.X(0), A.Y(0), 5, c.ink);
  let y = B.t + 14; runs.forEach(([k, lr]) => { txtBg(ctx, c, `${OPT_NAME[k]} (η ${lr})`, B.l + 6, y, { color: col[k], size: 11.5, w: "700" }); y += 17; });
});
function upd3() {
  paint(F3);
  const g = gV([-4.5, 1]);
  $("r3").textContent = tr(`출발점의 gradient = (${fx(g[0], 1)}, ${fx(g[1], 1)}): y 방향이 x 방향의 ${fx(Math.abs(g[1] / g[0]), 1)}배 가파르다\nRMSprop·Adam의 첫 걸음 ≈ η × (g / |g|) = 방향마다 거의 같은 크기 → 대각선으로 곧게 간다`,
    `gradient at the start = (${fx(g[0], 1)}, ${fx(g[1], 1)}): the y direction is ${fx(Math.abs(g[1] / g[0]), 1)}× steeper than x\nRMSprop's and Adam's first steps ≈ η × (g / |g|) = about the same size in each direction → they head straight along the diagonal`);
}

/* ================= 4절 / Section 4 ================= */
// optimizer 경주: 굽은 골짜기 f = (1 − x)² + 5(y − x²)² 위에서 네 가지를 동시에. 그림을 누르면 출발점이 바뀐다
// Optimizer race: all four at once on the curved valley f = (1 − x)² + 5(y − x²)². Click the map to move the start
const fR = w => (1 - w[0]) ** 2 + 5 * (w[1] - w[0] ** 2) ** 2, gR = w => [-2 * (1 - w[0]) - 20 * w[0] * (w[1] - w[0] ** 2), 10 * (w[1] - w[0] ** 2)];
const LR4 = { sgd: 0.02, momentum: 0.004, rmsprop: 0.02, adam: 0.05 };
const S4 = { start: [-1.5, 2.2], k: 300, timer: 0, lay: null };
const F4 = fig("cv4", w => (stacked(w, 270, 2, 28) ? 620 : 320), (ctx, w, h, c) => {
  const P = panels(w, h, 2, 28, 270), col = OPT_COL(c);
  const runs = Object.keys(LR4).map(k => [k, optPath(k, gR, S4.start, LR4[k], 300)]);
  {
    const p = P[0], side = Math.min(p.w - 46, p.h - 56), B = { l: p.x + 36, t: p.y + 28, w: side, h: side }, R = [-2, 2], Ry = [-1, 3];
    S4.lay = { B, R, Ry };
    contour(ctx, c, B, (a, b) => Math.log(1 + fR([a, b])), R, Ry, "rosen");
    const A = axes(ctx, c, B, R, Ry, { xt: [-2, -1, 0, 1, 2], yt: [-1, 0, 1, 2, 3], grid: false });
    panelTitle(ctx, c, tr("굽은 골짜기 (바닥 = (1, 1))", "A curved valley (bottom = (1, 1))"), B.l, p.y + 16);
    clipBox(ctx, B); runs.forEach(([k, Pp]) => pathOnMap(ctx, A, Pp.slice(0, S4.k + 1), col[k], 2.2)); ctx.restore();
    runs.forEach(([k, Pp]) => { const q = Pp[Math.min(S4.k, Pp.length - 1)]; dot(ctx, A.X(q[0]), A.Y(q[1]), 5, col[k], c.bg); });
    dot(ctx, A.X(1), A.Y(1), 5, c.ink); dot(ctx, A.X(S4.start[0]), A.Y(S4.start[1]), 5, c.ink);
  }
  {
    const p = P[1], B = { l: p.x + 48, t: p.y + 30, w: p.w - 60, h: p.h - 76 };
    const A = axes(ctx, c, B, [0, 300], [-4, 1.5], { xt: [0, 100, 200, 300], yt: [[-4, "10⁻⁴"], [-2, "0.01"], [0, "1"]], xlabel: tr("걸음", "step") });
    panelTitle(ctx, c, tr("걸음마다 loss (log)", "Loss per step (log)"), B.l, p.y + 14);
    clipBox(ctx, B); runs.forEach(([k, Pp]) => line(ctx, Pp.slice(0, S4.k + 1).map((q, i) => [A.X(i), A.Y(clamp(Math.log10(fR(q) + 1e-6), -4.5, 2))]), col[k], 2)); ctx.restore();
    let y = B.t + 14; runs.forEach(([k]) => { txtBg(ctx, c, `${OPT_NAME[k]} (η ${LR4[k]})`, B.l + B.w - 4, y, { color: col[k], size: 11.5, w: "700", align: "right" }); y += 17; });
  }
});
function upd4() {
  paint(F4);
  const res = Object.keys(LR4).map(k => { const Pp = optPath(k, gR, S4.start, LR4[k], 300), q = Pp[Pp.length - 1]; return `${OPT_NAME[k]} ${sci(fR(q))}`; });
  $("r4").textContent = tr(`출발 (${fx(S4.start[0], 1)}, ${fx(S4.start[1], 1)}), 300걸음 뒤 loss: ${res.join(",  ")}`, `start (${fx(S4.start[0], 1)}, ${fx(S4.start[1], 1)}), loss after 300 steps: ${res.join(",  ")}`);
}
$("b4play").addEventListener("click", () => { S4.k = 0; paint(F4); runSteps(S4, 60, () => { S4.k = Math.min(300, S4.k + 5); paint(F4); }, 40, upd4); });
dragOn(F4, (px, py) => { const L = S4.lay; return L && px >= L.B.l && px <= L.B.l + L.B.w && py >= L.B.t && py <= L.B.t + L.B.h ? 1 : null; }, (k, px, py) => {
  const L = S4.lay; stopRun(S4); S4.start = [L.R[0] + (px - L.B.l) / L.B.w * (L.R[1] - L.R[0]), L.Ry[1] - (py - L.B.t) / L.B.h * (L.Ry[1] - L.Ry[0])]; S4.k = 300; upd4();
});
// 진짜 network로 비교: moons data, 2-16-16-1, 같은 seed로 시작
// Comparison on a real network: moons data, 2-16-16-1, all from the same seed
const D4 = makeData("moons", 200, makeRng(41), 0.12);
const LRN = { sgd: 0.1, momentum: 0.05, rmsprop: 0.005, adam: 0.005 };
const S4b = { nets: null, hist: null, ep: 0, timer: 0 };
function reset4b() { stopRun(S4b); S4b.nets = Object.keys(LRN).map(k => new TinyNet([2, 16, 16, 1], { hidden: "tanh", seed: 2, opt: { name: k, lr: LRN[k] } })); S4b.hist = S4b.nets.map(n => [n.loss(D4.X, D4.Y)]); S4b.ep = 0; upd4b(); }
const F4b = fig("cv4b", 280, (ctx, w, h, c) => {
  const B = { l: 48, t: 30, w: w - 62, h: h - 74 }, col = OPT_COL(c), n = 120;
  const A = axes(ctx, c, B, [0, n], [0, 0.75], { xt: [0, 30, 60, 90, 120], yt: [0, 0.25, 0.5, 0.75], xlabel: "epoch" });
  panelTitle(ctx, c, tr("moons 분류 학습 loss (batch 20, 같은 출발 weight)", "Training loss on moons (batch 20, same starting weights)"), B.l, 16);
  clipBox(ctx, B); Object.keys(LRN).forEach((k, i) => line(ctx, S4b.hist[i].map((v, e) => [A.X(e), A.Y(v)]), col[k], 2.2)); ctx.restore();
  let y = B.t + 14; Object.keys(LRN).forEach((k, i) => { const H = S4b.hist[i]; txtBg(ctx, c, `${OPT_NAME[k]} (η ${LRN[k]}): ${fx(H[H.length - 1], 3)}`, B.l + B.w - 4, y, { color: col[k], size: 11.5, w: "700", align: "right" }); y += 17; });
});
function upd4b() { paint(F4b); $("b4brun").textContent = S4b.timer ? tr("멈춤", "Pause") : tr("네 가지 같이 학습", "Train all four"); }
$("b4brun").addEventListener("click", () => {
  if (S4b.timer) { stopRun(S4b); upd4b(); return; }
  runSteps(S4b, 60, () => { for (let r = 0; r < 2; r++) { S4b.nets.forEach((n, i) => S4b.hist[i].push(n.epoch(D4.X, D4.Y, 20))); S4b.ep++; } upd4b(); if (S4b.ep >= 120) return false; }, 30, upd4b); upd4b();
});
$("b4breset").addEventListener("click", reset4b);

/* ================= 5절 / Section 5 ================= */
// Adam + L2 vs AdamW: weight 두 개 중 하나는 gradient가 크다. 실제로 줄어드는 비율(걸음마다)을 비교
// Adam + L2 vs AdamW: one of two weights has large gradients. Compare the actual shrink rate per step
const S5 = { g2: 1 };
const F5 = fig("cv5", 270, (ctx, w, h, c) => {
  const lr = 0.001, lam = 0.01, gs = [0.01, S5.g2], B = { l: 160, t: 40, w: w - 260 };
  const vals = [gs.map(g => lr * lam / (g + 1e-8)), gs.map(() => lr * lam)];
  const mx = Math.max(...vals.flat());
  panelTitle(ctx, c, tr("걸음마다 weight가 줄어드는 비율 (η = 0.001, λ = 0.01)", "Fraction each weight shrinks per step (η = 0.001, λ = 0.01)"), 12, 18);
  [["Adam + L2", c.fa], ["AdamW", c.ok]].forEach(([name, col], r) => {
    gs.forEach((g, k) => {
      const y = B.t + 14 + r * 100 + k * 32, v = vals[r][k], W = Math.max(2, (Math.log10(v) + 8) / (Math.log10(mx) + 8) * B.w);
      if (k === 0) txt(ctx, name, 12, y - 6, { color: col, size: 13, w: "700" });
      txt(ctx, tr(`gradient 크기 ${g}`, `gradient size ${g}`), B.l - 8, y + 16, { color: c.muted, size: 11, align: "right" });
      ctx.fillStyle = col; ctx.globalAlpha = 0.75; ctx.fillRect(B.l, y + 4, W, 20); ctx.globalAlpha = 1;
      txt(ctx, sci(v), B.l + W + 6, y + 18, { color: c.ink, size: 11.5, mono: true });
    });
  });
  txt(ctx, tr("막대 길이 = log 눈금", "bar length = log scale"), w - 12, h - 8, { color: c.muted, size: 10.5, align: "right" });
});
function upd5() {
  paint(F5);
  const r = S5.g2 / 0.01;
  $("r5").textContent = tr(`Adam + L2: λw도 gradient에 섞여 √v로 나뉜다 → gradient가 ${fx(r, 0)}배 큰 weight는 ${fx(r, 0)}배 덜 줄어든다\nAdamW: weight decay를 따로 η·λ·w로 → 모든 weight가 같은 비율(${sci(1e-5)})로 줄어든다`,
    `Adam + L2: λw is mixed into the gradient and divided by √v → a weight whose gradients are ${fx(r, 0)}× larger shrinks ${fx(r, 0)}× less\nAdamW: weight decay applied separately as η·λ·w → every weight shrinks by the same fraction (${sci(1e-5)})`);
}
bindRange("i5g", "o5g", v => String(v), v => { S5.g2 = v; upd5(); });

/* ================= 6절 / Section 6 ================= */
// learning rate schedule 그래프
// Learning-rate schedule plots
const S6 = { k: "cos", warm: 0 };
function sched(k, t, T = 100, e0 = 0.1) {
  let e;
  if (k === "const") e = e0;
  else if (k === "step") e = e0 * 0.5 ** Math.floor(t / 25);
  else if (k === "exp") e = e0 * 0.97 ** t;
  else e = 0.001 + 0.5 * (e0 - 0.001) * (1 + Math.cos(Math.PI * t / T));
  if (S6.warm && t < S6.warm) e *= (t + 1) / S6.warm;
  return e;
}
const SCH = { const: tr("고정", "constant"), step: tr("계단 (25마다 ×0.5)", "step (×0.5 every 25)"), exp: tr("지수 (×0.97)", "exponential (×0.97)"), cos: "cosine" };
const F6 = fig("cv6", 280, (ctx, w, h, c) => {
  const B = { l: 48, t: 30, w: w - 62, h: h - 74 };
  const A = axes(ctx, c, B, [0, 100], [0, 0.11], { xt: [0, 25, 50, 75, 100], yt: [0, 0.025, 0.05, 0.075, 0.1], xlabel: "epoch" });
  panelTitle(ctx, c, tr("epoch마다 learning rate η", "Learning rate η per epoch"), B.l, 16);
  Object.keys(SCH).forEach(k => { if (k !== S6.k) line(ctx, Array.from({ length: 101 }, (_, t) => [A.X(t), A.Y(sched(k, t))]), c.line, 1.4); });
  line(ctx, Array.from({ length: 101 }, (_, t) => [A.X(t), A.Y(sched(S6.k, t))]), c.miss, 2.8);
  txtBg(ctx, c, SCH[S6.k] + (S6.warm ? tr(` + warmup ${S6.warm}`, ` + warmup ${S6.warm}`) : ""), B.l + B.w - 4, B.t + 14, { color: c.miss, size: 12, w: "700", align: "right" });
});
function upd6() {
  paint(F6);
  const v = [0, 10, 50, 99].map(t => `η${SUB(t)} = ${fx(sched(S6.k, t), 4)}`).join(", ");
  $("r6").textContent = tr(`${SCH[S6.k]}${S6.warm ? `, 처음 ${S6.warm} epoch은 0에서 천천히 올린다 (warmup)` : ""}: ${v}`, `${SCH[S6.k]}${S6.warm ? `, ramping up from 0 over the first ${S6.warm} epochs (warmup)` : ""}: ${v}`);
}
bindSeg("g6k", v => { S6.k = v; upd6(); });
bindRange("i6w", "o6w", v => String(v), v => { S6.warm = v; upd6(); });

/* ================= 7절 / Section 7 ================= */
// metaparameter 찾기: 격자(grid) vs 무작위(random). 가로 = 중요한 설정, 세로 = 거의 상관없는 설정
// Searching metaparameters: grid vs random. Horizontal = a setting that matters, vertical = one that barely matters
const S7 = { mode: "grid", seed: 3 };
const score7 = (a, b) => Math.exp(-((a - 0.62) ** 2) / 0.012) * (0.92 + 0.08 * b);
function pts7() {
  if (S7.mode === "grid") { const v = [1 / 6, 0.5, 5 / 6]; return v.flatMap(a => v.map(b => [a, b])); }
  const r = makeRng(S7.seed); return Array.from({ length: 9 }, () => [r.u(), r.u()]);
}
const F7 = fig("cv7", w => (w >= 560 ? 300 : 520), (ctx, w, h, c) => {
  const wide = w >= 560, side = wide ? Math.min(230, h - 70) : Math.min(w - 70, 230), B = { l: 50, t: 30, w: side, h: side }, pts = pts7();
  contour(ctx, c, B, (a, b) => -score7(a, b), [0, 1], [0, 1], "hp");
  const A = axes(ctx, c, B, [0, 1], [0, 1], { xt: [], yt: [], grid: false });
  panelTitle(ctx, c, S7.mode === "grid" ? tr("격자 9칸", "a 3 × 3 grid") : tr("무작위 9번", "9 random tries"), B.l, 16);
  txt(ctx, tr("중요한 설정 (예: learning rate) →", "a setting that matters (e.g. learning rate) →"), B.l + B.w / 2, B.t + B.h + 18, { color: c.muted, size: 11, align: "center" });
  ctx.save(); ctx.translate(B.l - 14, B.t + B.h / 2); ctx.rotate(-Math.PI / 2); txt(ctx, tr("덜 중요한 설정 →", "a setting that barely matters →"), 0, 0, { color: c.muted, size: 11, align: "center" }); ctx.restore();
  pts.forEach(p => dot(ctx, A.X(p[0]), A.Y(p[1]), 5.5, c.miss, c.bg));
  const C = wide ? { l: B.l + B.w + 60, t: 30, w: w - (B.l + B.w + 74), h: side } : { l: 50, t: B.t + B.h + 60, w: w - 64, h: h - (B.t + B.h + 60) - 34 };
  const A2 = axes(ctx, c, C, [0, 1], [0, 1.05], { xt: [], yt: [0, 0.5, 1], xlabel: tr("중요한 설정만 보면", "looking at the important setting only") });
  panelTitle(ctx, c, tr("성능 (진짜 곡선과 시도한 값)", "Performance (true curve and the values tried)"), C.l, C.t - 10);
  line(ctx, curve(A2, a => score7(a, 1), [0, 1], 200), c.sig, 2);
  pts.forEach(p => { line(ctx, [[A2.X(p[0]), C.t + C.h], [A2.X(p[0]), A2.Y(score7(p[0], p[1]))]], c.miss, 1, [3, 3]); dot(ctx, A2.X(p[0]), A2.Y(score7(p[0], p[1])), 4.5, c.miss); });
});
function upd7() {
  paint(F7);
  const pts = pts7(), distinct = new Set(pts.map(p => p[0].toFixed(3))).size, best = Math.max(...pts.map(p => score7(p[0], p[1])));
  $("r7").textContent = tr(`${S7.mode === "grid" ? "격자" : "무작위"} 9번: 중요한 설정의 서로 다른 값 ${distinct}개, 가장 좋은 성능 ${fx(best, 3)} (최고 1)\n→ ${S7.mode === "grid" ? "격자는 같은 값을 세 번씩 다시 본다" : "무작위는 9번 모두 다른 값을 본다: 중요한 설정을 더 촘촘히 훑는다"}`,
    `${S7.mode === "grid" ? "grid" : "random"}, 9 tries: ${distinct} distinct values of the important setting, best performance ${fx(best, 3)} (max 1)\n→ ${S7.mode === "grid" ? "the grid looks at each value three times over" : "random search sees 9 different values: it scans the important setting more finely"}`);
}
bindSeg("g7m", v => { S7.mode = v; upd7(); });
$("b7seed").addEventListener("click", () => { S7.seed++; S7.mode = "random"; document.querySelectorAll("#g7m button").forEach(b => b.setAttribute("aria-pressed", b.dataset.v === "random" ? "true" : "false")); upd7(); });

// 처음 그리기
// First paint
upd1(); upd2(); upd3(); upd4(); reset4b(); upd5(); upd6(); upd7();
repaintAll();
