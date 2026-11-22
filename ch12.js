/* 12장 그림과 조작. 한글판(ch12-transfer-style-KR.html)과 영어판(ch12-transfer-style.html)이 같이 쓴다
   Chapter 12 figures and interactions, shared by the Korean and English pages
   실험 결과와 style transfer 그림은 assets/data/ch12_data.js (tools/ch12_numbers.py, ch12_style.py, ch12_cifar.py가 만든다)
   Experiment results and style-transfer pictures live in assets/data/ch12_data.js (made by tools/ch12_numbers.py, ch12_style.py and ch12_cifar.py) */
"use strict";

const D12 = CH12_DATA;
// 반올림 오차 없이 소수 첫째 자리 % (0.8035 → 80.4%)
// Percent to one decimal without float rounding slips (0.8035 → 80.4%)
const pct = v => (Math.round(v * 1000 + 1e-6) / 10).toFixed(1) + "%";
// 막대 그림 공통: items = [[이름, 값, 색, 아래 작은 글]]
// Shared bar chart: items = [[name, value, colour, small note below]]
function bars(ctx, c, B, items, yr, o = {}) {
  const A = axes(ctx, c, B, [0, items.length], yr, { xt: [], yt: ticksR(yr[0], yr[1], 4).map(v => [v, Math.round(v * 100) + "%"]) });
  const bw = B.w / items.length;
  items.forEach(([name, v, col, note, sd], k) => {
    const x = B.l + k * bw + bw * 0.18, wd = bw * 0.64, y = A.Y(v);
    ctx.fillStyle = col; ctx.globalAlpha = o.hl === undefined || o.hl === k ? 0.85 : 0.35; ctx.fillRect(x, y, wd, B.t + B.h - y); ctx.globalAlpha = 1;
    if (o.hl === k) { ctx.strokeStyle = c.ink; ctx.lineWidth = 2; ctx.strokeRect(x - 1, y - 1, wd + 2, B.t + B.h - y + 1); }
    if (sd) line(ctx, [[x + wd / 2, A.Y(v - sd)], [x + wd / 2, A.Y(v + sd)]], c.ink, 1.4);
    txt(ctx, pct(v), x + wd / 2, y - (sd ? A.Y(v) - A.Y(v + sd) + 6 : 6), { color: c.ink, size: 11, mono: true, align: "center", w: "600" });
    name.split("\n").forEach((s, j) => txt(ctx, s, x + wd / 2, B.t + B.h + 15 + j * 13, { color: c.ink, size: 10.5, align: "center" }));
    if (note) txt(ctx, note, x + wd / 2, B.t + B.h + 15 + name.split("\n").length * 13, { color: c.vio, size: 10, mono: true, align: "center" });
  });
  return A;
}

/* ================= 2절 / Section 2 ================= */
// 층별 freeze 스위치: 출발 data로 학습한 값(또는 무작위)에서 시작해 도착 data 500장으로 다시 학습 (3번 평균)
// Per-layer freeze switches: start from source-trained values (or random) and retrain on 500 target images (mean of 3 runs)
const S2 = { dir: "mnist_fmnist", init: "pre", f1: 0, f2: 0 };
const PC = [208, 3216, 7850];
const key2 = (init, f1, f2) => `${init}${f1}${f2}`;
const F2 = fig("cv2", w => (w >= 700 ? 330 : 640), (ctx, w, h, c) => {
  const wide = w >= 700, T = D12.transfer[S2.dir], [src, dst] = S2.dir === "mnist_fmnist" ? ["MNIST", "FMNIST"] : ["FMNIST", "MNIST"];
  // 왼쪽: network 그림과 스위치 상태
  // Left: the network and the switch states
  const blocks = [["conv1", "5×5 × 8", S2.f1, PC[0]], ["conv2", "5×5 × 16", S2.f2, PC[1]], ["head", "Linear 784→10", 0, PC[2]]], bw = wide ? 92 : (w - 60) / 3, x0 = 16, y0 = 52;
  panelTitle(ctx, c, tr(`${src}로 배운 몸통 → ${dst} 500장`, `body learned on ${src} → 500 ${dst} images`), x0, 18);
  blocks.forEach(([name, sub, fr, p], k) => {
    const x = x0 + k * (bw + 12), isHead = k === 2, col = isHead ? c.miss : fr ? c.sig : c.ok;
    rrect(ctx, x, y0, bw, 96, 8, fr ? c.sigwash : isHead ? c.wash : c.okwash, col, 1.6);
    txt(ctx, name, x + bw / 2, y0 + 24, { color: c.ink, size: 12.5, w: "700", align: "center" });
    txt(ctx, sub, x + bw / 2, y0 + 42, { color: c.muted, size: 10.5, mono: true, align: "center" });
    txt(ctx, isHead ? tr("새로, 학습", "new, trained") : fr ? tr("🔒 얼림", "🔒 frozen") : tr("학습", "trained"), x + bw / 2, y0 + 64, { color: col, size: 12, w: "700", align: "center" });
    txt(ctx, isHead ? tr("무작위 시작", "random start") : S2.init === "pre" ? tr(`${src} 값으로 시작`, `starts from ${src}`) : tr("무작위 시작", "random start"), x + bw / 2, y0 + 82, { color: c.muted, size: 10, align: "center" });
    txt(ctx, `${commas(p)} p`, x + bw / 2, y0 + 112, { color: c.vio, size: 10.5, mono: true, align: "center" });
    if (k < 2) arrowS(ctx, x + bw + 1, y0 + 48, x + bw + 11, y0 + 48, c.muted, 1.3, 5);
  });
  // 오른쪽: 8가지 조합의 시험 정확도 (지금 고른 것 강조)
  // Right: test accuracy of all 8 combinations (the current one highlighted)
  const order = [];
  ["pre", "rand"].forEach(init => [[0, 0], [0, 1], [1, 0], [1, 1]].forEach(([a, b]) => order.push([init, a, b])));
  const mark = v => (v ? "●" : "○");
  const items = order.map(([init, a, b]) => [`${mark(a)}${mark(b)}`, T[key2(init, a, b)][0], init === "pre" ? c.ok : c.muted, null, T[key2(init, a, b)][1]]);
  const B = wide ? { l: x0 + 3 * (bw + 12) + 50, t: 40, w: w - (x0 + 3 * (bw + 12) + 50) - 14, h: h - 110 } : { l: 52, t: y0 + 170, w: w - 66, h: h - y0 - 250 };
  const lo = S2.dir === "mnist_fmnist" ? 0.6 : 0.8;
  const A = bars(ctx, c, B, items, [lo, 1], { hl: order.findIndex(([i, a, b]) => i === S2.init && a === S2.f1 && b === S2.f2) });
  line(ctx, [[B.l, A.Y(T.full)], [B.l + B.w, A.Y(T.full)]], c.fa, 1.4, [5, 4]);
  txtBg(ctx, c, tr(`${dst} 60,000장 전부로 학습: ${pct(T.full)}`, `trained on all 60,000 ${dst}: ${pct(T.full)}`), B.l + B.w - 4, A.Y(T.full) - 6, { color: c.fa, size: 11, w: "600", align: "right" });
  txt(ctx, tr(`${src} 값으로 시작`, `start from ${src}`), B.l + B.w / 4, B.t + B.h + 44, { color: c.ok, size: 11, w: "700", align: "center" });
  txt(ctx, tr("무작위로 시작", "random start"), B.l + 3 * B.w / 4, B.t + B.h + 44, { color: c.muted, size: 11, w: "700", align: "center" });
  txt(ctx, tr("● 얼림  ○ 학습 (1층, 2층)", "● frozen  ○ trained (layer 1, layer 2)"), B.l, B.t - 12, { color: c.muted, size: 10.5 });
});
function upd2() {
  paint(F2);
  const T = D12.transfer[S2.dir], k = key2(S2.init, S2.f1, S2.f2), [m, sd] = T[k], tp = (S2.f1 ? 0 : PC[0]) + (S2.f2 ? 0 : PC[1]) + PC[2];
  const base = T.rand00[0];
  $("r2").textContent = tr(`시험 정확도 ${pct(m)} ± ${(sd * 100).toFixed(1)} (3번 평균), 학습하는 parameter ${commas(tp)}개\n무작위로 시작해 모두 학습한 경우(${pct(base)})보다 ${m >= base ? "+" : "−"}${Math.abs((m - base) * 100).toFixed(1)}%p`,
    `test accuracy ${pct(m)} ± ${(sd * 100).toFixed(1)} (mean of 3 runs), ${commas(tp)} trained parameters\n${m >= base ? "+" : "−"}${Math.abs((m - base) * 100).toFixed(1)} points vs a random start with every layer trained (${pct(base)})`);
}
bindSeg("g2d", v => { S2.dir = v; upd2(); });
bindSeg("g2i", v => { S2.init = v; upd2(); });
bindSeg("g2a", v => { S2.f1 = +v; upd2(); });
bindSeg("g2b", v => { S2.f2 = +v; upd2(); });

/* ================= 3절 / Section 3 ================= */
// residual connection: 30층을 거꾸로 지나는 gradient의 크기 (너비 32, tanh, 무작위 weight 5번 평균)
// Residual connections: size of the gradient travelling back through 30 layers (width 32, tanh, mean of 5 random weight draws)
const S3 = { s: 0.5 };
function gradNorms(s, residual, L = 30, n = 32, reps = 5) {
  const acc = new Array(L + 1).fill(0);
  for (let rep = 0; rep < reps; rep++) {
    const r = makeRng(11 + rep), Ws = [], hs = [], ds = [];
    let h = Array.from({ length: n }, () => r.n());
    for (let l = 0; l < L; l++) {
      const W = Array.from({ length: n }, () => Array.from({ length: n }, () => r.n() * s / Math.sqrt(n))), a = matVec(W, h).map(Math.tanh);
      Ws.push(W); hs.push(h); ds.push(a.map(v => 1 - v * v)); h = residual ? h.map((v, i) => v + a[i]) : a;
    }
    let g = new Array(n).fill(1); acc[L] += Math.log10(Math.hypot(...g));
    for (let l = L - 1; l >= 0; l--) {
      const gd = g.map((v, i) => v * ds[l][i]), back = Ws[l][0].map((_, j) => Ws[l].reduce((t, row, i) => t + row[j] * gd[i], 0));
      g = residual ? back.map((v, j) => v + g[j]) : back; acc[l] += Math.log10(Math.max(1e-30, Math.hypot(...g)));
    }
  }
  return acc.map(v => v / reps);
}
const F3 = fig("cv3", 280, (ctx, w, h, c) => {
  const P = gradNorms(S3.s, false), R = gradNorms(S3.s, true), lo = Math.min(-8, Math.floor(Math.min(...P, ...R))), hi = Math.max(2, Math.ceil(Math.max(...P, ...R)));
  const B = { l: 58, t: 30, w: w - 76, h: h - 74 }, A = axes(ctx, c, B, [0, 30], [lo, hi], { xt: [0, 5, 10, 15, 20, 25, 30], yt: intTicks([lo, hi]).filter(v => v % 2 === 0).map(v => [v, v === 0 ? "1" : "10" + SUP(v)]), xlabel: tr("층 (0 = 입력 쪽, 30 = 출력 쪽)", "layer (0 = input side, 30 = output side)") });
  panelTitle(ctx, c, tr("gradient 크기 ‖∂L/∂h‖ (log)", "gradient size ‖∂L/∂h‖ (log)"), B.l, 16);
  line(ctx, P.map((v, l) => [A.X(l), A.Y(v)]), c.fa, 2.2); line(ctx, R.map((v, l) => [A.X(l), A.Y(v)]), c.ok, 2.2);
  txtBg(ctx, c, tr("보통 층: h ← tanh(Wh)", "plain: h ← tanh(Wh)"), A.X(4), A.Y(P[4]) + 18, { color: c.fa, size: 11.5, w: "700" });
  txtBg(ctx, c, tr("residual: h ← h + tanh(Wh)", "residual: h ← h + tanh(Wh)"), A.X(4), A.Y(R[4]) - 10, { color: c.ok, size: 11.5, w: "700" });
});
function upd3() {
  paint(F3);
  const P = gradNorms(S3.s, false), R = gradNorms(S3.s, true), f = v => "10" + SUP(Math.round(v));
  $("r3").textContent = tr(`weight 표준편차 ${fx(S3.s, 2)}/√32. 출력 쪽(30층)에서 입력 쪽(0층)까지 오면 gradient가: 보통 층 약 ×${f(P[0] - P[30])}, residual 약 ×${f(R[0] - R[30])}\n보통 층은 표준편차에 따라 gradient가 사라지거나 커진다. residual은 더하기 길(h + …)이 gradient를 적어도 1배로 넘겨 주어 사라지지 않는다.`,
    `weight SD ${fx(S3.s, 2)}/√32. From the output side (layer 30) back to the input side (layer 0) the gradient changes by: plain about ×${f(P[0] - P[30])}, residual about ×${f(R[0] - R[30])}\nIn plain layers the gradient vanishes or grows depending on the SD. With residuals the addition path (h + …) passes at least ×1 of the gradient along, so it does not vanish.`);
}
bindRange("i3s", "o3s", v => fx(v, 2), v => { S3.s = v; upd3(); });

/* ================= 4절 / Section 4 ================= */
// CIFAR10 5,000장: 처음부터 vs ImageNet으로 학습한 network 빌리기
// CIFAR10 with 5,000 images: from scratch vs borrowing ImageNet-trained networks
const F4 = fig("cv4", 290, (ctx, w, h, c) => {
  const C = D12.cifar;
  if (!C) { txt(ctx, tr("결과 없음 (tools/ch12_cifar.py를 실행)", "no results (run tools/ch12_cifar.py)"), 16, 40, { color: c.fa, size: 12.5 }); return; }
  const items = [[tr("작은 CNN\n처음부터", "small CNN\nfrom scratch"), C.scratch.acc, c.muted, `${commas(C.scratch.trained)} p`],
    [tr("ResNet-18\n몸통 얼림", "ResNet-18\nbody frozen"), C.resnet_frozen.acc, c.sig, `${commas(C.resnet_frozen.trained)} p`],
    [tr("ResNet-18\n전체 fine-tune", "ResNet-18\nfine-tuned"), C.resnet_tune.acc, c.ok, `${commas(C.resnet_tune.trained)} p`],
    [tr("VGG-16\n몸통 얼림", "VGG-16\nbody frozen"), C.vgg_frozen.acc, c.vio, `${commas(C.vgg_frozen.trained)} p`]];
  panelTitle(ctx, c, tr(`CIFAR10 학습 ${commas(C.n_train)}장 → 시험 ${commas(C.n_test)}장 정확도 (보라 = 학습한 parameter 수)`, `CIFAR10: ${commas(C.n_train)} training → accuracy on ${commas(C.n_test)} test images (violet = trained parameters)`), 16, 16);
  bars(ctx, c, { l: 52, t: 36, w: w - 66, h: h - 100 }, items, [0, 1]);
});

/* ================= 5절 / Section 5 ================= */
// autoencoder pretraining: FMNIST 500장의 정확도 (3번 평균)
// Autoencoder pretraining: accuracy with 500 FMNIST images (mean of 3 runs)
const F5 = fig("cv5", 270, (ctx, w, h, c) => {
  const P = D12.aepre, M = D12.transfer.mnist_fmnist.pre00;
  const items = [[tr("무작위로\n시작", "random\nstart"), P.rand[0], c.muted, null, P.rand[1]], [tr("AE 몸통\n얼림", "AE body\nfrozen"), P.ae_frozen[0], c.sig, null, P.ae_frozen[1]],
    [tr("AE 몸통\nfine-tune", "AE body\nfine-tuned"), P.ae_tune[0], c.ok, null, P.ae_tune[1]], [tr("MNIST 분류로\n배운 몸통", "body from\nMNIST classes"), M[0], c.vio, null, M[1]]];
  panelTitle(ctx, c, tr("FMNIST label 500장으로 학습한 시험 정확도 (막대 위 선 = 표준편차)", "Test accuracy after training on 500 labelled FMNIST images (line = SD)"), 16, 16);
  bars(ctx, c, { l: 52, t: 36, w: w - 66, h: h - 86 }, items, [0.7, 0.85]);
});

/* ================= 6절 / Section 6 ================= */
// Gram matrix: feature map 3장(4×4)의 칸을 누르면 켜고 끈다. G_ij = 두 지도가 같이 켜진 정도
// Gram matrix: click cells of the three 4×4 feature maps to switch them. G_ij = how much two maps are on together
const F6INIT = [[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]];
const S6 = { F: F6INIT.map(r => [...r]), lay: null, seed: 1 };
const gram6 = F => F.map(a => F.map(b => a.reduce((s, v, k) => s + v * b[k], 0)));
const F6 = fig("cv6", w => (w >= 640 ? 230 : 440), (ctx, w, h, c) => {
  const wide = w >= 640, cell = wide ? 22 : Math.min(22, (w - 60) / 13), cols = [c.sig, c.miss, c.ok], G = gram6(S6.F), y0 = 40;
  S6.lay = { cell, xs: [], y0 };
  S6.F.forEach((m, k) => {
    const x = 16 + k * (4 * cell + 24); S6.lay.xs.push(x);
    txt(ctx, tr(`지도 ${k + 1}`, `map ${k + 1}`), x, y0 - 10, { color: cols[k], size: 12, w: "700" });
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
      const v = m[i * 4 + j]; ctx.fillStyle = v ? cols[k] : c.bg; ctx.globalAlpha = v ? 0.75 : 1; ctx.fillRect(x + j * cell, y0 + i * cell, cell, cell); ctx.globalAlpha = 1;
      ctx.strokeStyle = c.line; ctx.lineWidth = 1; ctx.strokeRect(x + j * cell + 0.5, y0 + i * cell + 0.5, cell - 1, cell - 1);
    }
  });
  const gx = wide ? 16 + 3 * (4 * cell + 24) + 50 : 16, gy = wide ? y0 : y0 + 4 * cell + 60, gc = 44;
  if (wide) txt(ctx, "→", gx - 30, y0 + 2 * cell + 6, { color: c.muted, size: 18, align: "center" });
  txt(ctx, tr("Gram matrix G = F Fᵀ", "Gram matrix G = F Fᵀ"), gx, gy - 10, { color: c.ink, size: 12.5, w: "700" });
  for (let i = 0; i < 3; i++) {
    txt(ctx, String(i + 1), gx - 8, gy + i * gc + gc / 2 + 4, { color: cols[i], size: 11, w: "700", align: "right" });
    txt(ctx, String(i + 1), gx + i * gc + gc / 2, gy + 3 * gc + 14, { color: cols[i], size: 11, w: "700", align: "center" });
  }
  gridNums(ctx, c, G.flat(), 3, 3, gx, gy, gc, { max: 8 });
});
// 숫자가 적힌 칸 그림 (11장과 같은 모양)
// Grid of numbered cells (same look as chapter 11)
function gridNums(ctx, c, M, n, m, x, y, cell, o = {}) {
  const bg = rgb(c.bg), mx = o.max || Math.max(1e-9, ...M.map(Math.abs));
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) {
    const v = M[i * m + j], X = x + j * cell, Y = y + i * cell;
    ctx.fillStyle = cssRGB(mixRGB(bg, rgb(c.vio), Math.min(1, Math.abs(v) / mx) * 0.6)); ctx.fillRect(X, Y, cell, cell);
    ctx.strokeStyle = c.line; ctx.lineWidth = 1; ctx.strokeRect(X + 0.5, Y + 0.5, cell - 1, cell - 1);
    txt(ctx, String(v), X + cell / 2, Y + cell / 2 + 5, { color: c.ink, size: 14, mono: true, align: "center", w: "600" });
  }
}
function upd6() {
  paint(F6);
  const G = gram6(S6.F);
  $("r6").textContent = tr(`G₁₂ = ${G[0][1]}: 지도 1과 2가 같은 칸에서 같이 켜진 수. G₁₁ = ${G[0][0]}: 지도 1이 켜진 칸 수.\n"자리 섞기"는 세 지도의 칸을 똑같이 섞는다: 그림은 달라져도 G는 그대로다. Gram matrix는 "어디에"를 버리고 "무엇이 무엇과 같이"만 남긴다 = style.`,
    `G₁₂ = ${G[0][1]}: cells where maps 1 and 2 are on together. G₁₁ = ${G[0][0]}: cells where map 1 is on.\n"Shuffle positions" moves the cells of all three maps the same way: the pictures change but G stays the same. The Gram matrix drops "where" and keeps only "what goes with what" = style.`);
}
F6.cv.addEventListener("click", ev => {
  const L = S6.lay; if (!L) return; const r = F6.cv.getBoundingClientRect(), px = ev.clientX - r.left, py = ev.clientY - r.top;
  L.xs.forEach((x, k) => { const j = Math.floor((px - x) / L.cell), i = Math.floor((py - L.y0) / L.cell); if (i >= 0 && i < 4 && j >= 0 && j < 4) { S6.F[k][i * 4 + j] ^= 1; upd6(); } });
});
$("b6shuf").addEventListener("click", () => { const p = shuffled(16, makeRng(S6.seed++)); S6.F = S6.F.map(m => p.map(q => m[q])); upd6(); });
$("b6reset").addEventListener("click", () => { S6.F = F6INIT.map(r => [...r]); upd6(); });

/* ================= 7절 / Section 7 ================= */
// style transfer 결과: 내용 그림 + style 그림 → 결과 (β에 따라), 그리고 반복에 따른 변화
// Style-transfer results: content + style → result (by β), and how it changes over the iterations
const ST = D12.style, IMG = {};
const loadImg = (k, b64) => { const im = new Image(); im.onload = () => { paint(F7); paint(F7b); }; im.src = "data:image/png;base64," + b64; IMG[k] = im; };
loadImg("content", ST.content); ST.styles.forEach((s, i) => loadImg("style" + i, s)); ST.out.forEach((row, i) => row.forEach((s, j) => loadImg(`out${i}_${j}`, s)));
Object.entries(ST.snaps).forEach(([it, s]) => loadImg("snap" + it, s));
const S7 = { s: 0, b: 2 };
const putImg = (ctx, c, im, x, y, sz, label, col) => {
  if (im && im.complete && im.naturalWidth) { ctx.imageSmoothingEnabled = true; ctx.drawImage(im, x, y, sz, sz); }
  ctx.strokeStyle = c.line; ctx.lineWidth = 1; ctx.strokeRect(x - 0.5, y - 0.5, sz + 1, sz + 1);
  txt(ctx, label, x + sz / 2, y + sz + 16, { color: col || c.ink, size: 11.5, align: "center", w: "600" });
};
const F7 = fig("cv7", w => (w >= 640 ? 230 : 420), (ctx, w, h, c) => {
  const wide = w >= 640, sz = wide ? Math.min(170, (w - 120) / 3) : Math.min(150, (w - 50) / 2), y0 = 14;
  const pos = wide ? [[16, y0], [16 + sz + 40, y0], [16 + 2 * sz + 100, y0]] : [[16, y0], [16 + sz + 18, y0], [16 + (sz + 18) / 2, y0 + sz + 40]];
  putImg(ctx, c, IMG.content, pos[0][0], pos[0][1], sz, tr("내용 (content)", "content"), c.sig);
  putImg(ctx, c, IMG["style" + S7.s], pos[1][0], pos[1][1], sz, tr("무늬 (style)", "style"), c.miss);
  putImg(ctx, c, IMG[`out${S7.s}_${S7.b}`], pos[2][0], pos[2][1], sz, tr(`결과 (β = 10${SUP(Math.log10(ST.beta[S7.b]))})`, `result (β = 10${SUP(Math.log10(ST.beta[S7.b]))})`), c.ok);
  if (wide) { txt(ctx, "+", pos[0][0] + sz + 20, y0 + sz / 2 + 6, { color: c.muted, size: 20, align: "center" }); arrowS(ctx, pos[1][0] + sz + 14, y0 + sz / 2, pos[2][0] - 14, y0 + sz / 2, c.muted, 1.6, 7); }
});
const F7b = fig("cv7b", w => (w >= 640 ? 150 : 270), (ctx, w, h, c) => {
  const its = Object.keys(ST.snaps).map(Number).sort((a, b) => a - b), per = w >= 640 ? its.length : 3, sz = Math.min(120, (w - 32 - (per - 1) * 12) / per);
  its.forEach((it, k) => putImg(ctx, c, IMG["snap" + it], 16 + (k % per) * (sz + 12), 8 + Math.floor(k / per) * (sz + 34), sz, tr(`반복 ${it}`, `iteration ${it}`), c.muted));
});
function upd7() {
  paint(F7); paint(F7b);
  $("r7").textContent = tr(`β = 10${SUP(Math.log10(ST.beta[S7.b]))}: β가 클수록 style loss를 더 중요하게 본다 → 무늬가 짙어지고 내용은 흐려진다. 결과 그림은 network가 아니라 pixel을 gradient descent로 고친 것이다 (VGG-16 weight는 그대로).`,
    `β = 10${SUP(Math.log10(ST.beta[S7.b]))}: a larger β weighs the style loss more → the pattern gets stronger and the content fainter. The result is made by changing the pixels with gradient descent, not the network (VGG-16's weights stay fixed).`);
}
bindSeg("g7s", v => { S7.s = +v; upd7(); });
bindRange("i7b", "o7b", v => "10" + SUP(Math.log10(ST.beta[v])), v => { S7.b = v; upd7(); });

// 처음 그리기
// First paint
upd2(); upd3(); paint(F4); paint(F5); upd6(); upd7();
repaintAll();
