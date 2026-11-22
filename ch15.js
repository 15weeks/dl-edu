/* 15장 그림과 조작. 한글판(ch15-gan-diffusion-KR.html)과 영어판(ch15-gan-diffusion.html)이 같이 쓴다
   Chapter 15 figures and interactions, shared by the Korean and English pages
   GAN·diffusion 학습 결과는 assets/data/ch15_data.js (tools/ch15_gan.py, ch15_diffusion.py가 만든다)
   GAN and diffusion training results live in assets/data/ch15_data.js (made by tools/ch15_gan.py and ch15_diffusion.py) */
"use strict";

const D15 = CH15_DATA;
const IMG = {};
const loadImg = (k, b64, figsToPaint) => { const im = new Image(); im.onload = () => figsToPaint().forEach(f => f && paint(f)); im.src = "data:image/png;base64," + b64; IMG[k] = im; };
const putImg = (ctx, c, im, x, y, wd, ht) => { if (im && im.complete && im.naturalWidth) { ctx.imageSmoothingEnabled = false; ctx.drawImage(im, x, y, wd, ht); ctx.imageSmoothingEnabled = true; } ctx.strokeStyle = c.line; ctx.lineWidth = 1; ctx.strokeRect(x - 0.5, y - 0.5, wd + 1, ht + 1); };
const npdf = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));

/* ================= 1절 / Section 1 ================= */
// 생성 model = 분포를 배우고 거기서 뽑기. 예: 키의 분포(평균 170, 표준편차 7)에서 새 사람 20명
// Generative model = learn a distribution, then sample from it. Example: 20 new people from a height distribution (mean 170, SD 7)
const S1 = { seed: 3, pts: [] };
const draw1 = () => { const r = makeRng(S1.seed); S1.pts = Array.from({ length: 20 }, () => 170 + 7 * r.n()); };
const F1 = fig("cv1", 230, (ctx, w, h, c) => {
  const B = { l: 44, t: 30, w: w - 60, h: h - 74 }, A = axes(ctx, c, B, [145, 195], [0, 0.065], { xt: [150, 160, 170, 180, 190], yt: [], xlabel: tr("키 (cm)", "height (cm)") });
  ctx.fillStyle = c.sigwash; ctx.beginPath(); ctx.moveTo(A.X(145), A.Y(0)); for (let x = 145; x <= 195; x += 0.5) ctx.lineTo(A.X(x), A.Y(npdf(x, 170, 7))); ctx.lineTo(A.X(195), A.Y(0)); ctx.closePath(); ctx.fill();
  line(ctx, curve(A, x => npdf(x, 170, 7), [145, 195]), c.sig, 2);
  S1.pts.forEach((v, k) => dot(ctx, A.X(v), A.Y(0) - 8 - (k % 4) * 9, 4.5, c.miss, c.bg));
  panelTitle(ctx, c, tr("배운 분포 p(x) (파랑)에서 새로 뽑은 20명 (주황 점)", "20 new people (orange dots) drawn from the learned distribution p(x) (blue)"), B.l, 16);
});
function upd1() { paint(F1); const m = meanOf(S1.pts), s = sdOf(S1.pts); $("r1").textContent = tr(`뽑은 20명: 평균 ${fx(m, 1)} cm, 표준편차 ${fx(s, 1)} cm. 누를 때마다 다른 사람이 나오지만 모두 "그럴듯한 키"다.\n그림이라면 x가 784칸(28×28)이라 p(x)를 식으로 쓸 수 없다: 그래서 network가 분포를 배운다.`, `The 20 drawn: mean ${fx(m, 1)} cm, SD ${fx(s, 1)} cm. Each click gives different people, but all "plausible heights".\nFor a picture x has 784 cells (28×28), so p(x) cannot be written as a formula: a network has to learn the distribution.`); }
$("b1draw").addEventListener("click", () => { S1.seed++; draw1(); upd1(); });

/* ================= 2절 / Section 2 ================= */
// 1D GAN: data 분포(회색), G가 만든 것의 histogram(주황), D(x) = "진짜일 확률"(파랑)
// 1-D GAN: data distribution (grey), histogram of what G makes (orange), D(x) = "probability it is real" (blue)
const G1 = D15.gan1d, S2 = { d: "one", k: 0, timer: 0 };
const pdf2 = x => (S2.d === "one" ? npdf(x, 2, 0.5) : 0.5 * npdf(x, -2, 0.4) + 0.5 * npdf(x, 2, 0.4));
const F2 = fig("cv2", 280, (ctx, w, h, c) => {
  const s = G1[S2.d][S2.k], B = { l: 44, t: 30, w: w - 88, h: h - 74 }, A = axes(ctx, c, B, [-5, 5], [0, 1.1], { xt: [-4, -2, 0, 2, 4], yt: [0, 0.5, 1], xlabel: "x" });
  ctx.fillStyle = c.line; ctx.globalAlpha = 0.6; ctx.beginPath(); ctx.moveTo(A.X(-5), A.Y(0)); for (let x = -5; x <= 5; x += 0.05) ctx.lineTo(A.X(x), A.Y(pdf2(x))); ctx.lineTo(A.X(5), A.Y(0)); ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1;
  s.hist.forEach((v, k) => { const x0 = A.X(-5 + k * 0.2), x1 = A.X(-5 + (k + 1) * 0.2); ctx.fillStyle = c.miss; ctx.globalAlpha = 0.7; ctx.fillRect(x0 + 0.5, A.Y(Math.min(1.1, v)), x1 - x0 - 1, A.Y(0) - A.Y(Math.min(1.1, v))); ctx.globalAlpha = 1; });
  line(ctx, s.d.map((v, k) => [A.X(-5 + k * 0.1), A.Y(v)]), c.sig, 2.4);
  line(ctx, [[B.l, A.Y(0.5)], [B.l + B.w, A.Y(0.5)]], c.sig, 1, [3, 4]);
  txt(ctx, "D(x)", B.l + B.w + 6, A.Y(s.d[s.d.length - 1]) + 4, { color: c.sig, size: 11.5, w: "700" });
  panelTitle(ctx, c, tr(`학습 ${s.step}번: 회색 = 진짜 data, 주황 = G가 만든 것, 파랑 = D(x)`, `step ${s.step}: grey = real data, orange = what G makes, blue = D(x)`), B.l, 16);
});
function upd2() {
  paint(F2); setRange("i2s", S2.k, 0); $("o2s").textContent = String(G1[S2.d][S2.k].step);
  const s = G1[S2.d][S2.k], last = G1[S2.d][G1[S2.d].length - 1];
  $("r2").textContent = tr(`G가 만든 것: 평균 ${fx(s.mean, 2)}, 표준편차 ${fx(s.sd, 2)} (진짜: ${S2.d === "one" ? "평균 2, 표준편차 0.5" : "−2와 2에 봉우리 둘"}).\nD가 0.5 근처로 평평해지면 D가 더 이상 진짜와 가짜를 가르지 못한다는 뜻 = G가 잘 흉내 낸다.` + (S2.d === "two" ? ` 마지막(3000번) G: 평균 ${fx(last.mean, 2)}, 표준편차 ${fx(last.sd, 2)}.` : ""),
    `What G makes: mean ${fx(s.mean, 2)}, SD ${fx(s.sd, 2)} (real: ${S2.d === "one" ? "mean 2, SD 0.5" : "two peaks at −2 and 2"}).\nWhen D flattens near 0.5 it can no longer tell real from fake = G imitates well.` + (S2.d === "two" ? ` At the end (step 3000) G: mean ${fx(last.mean, 2)}, SD ${fx(last.sd, 2)}.` : ""));
}
bindSeg("g2d", v => { stopRun(S2); S2.d = v; upd2(); });
bindRange("i2s", null, v => v, v => { stopRun(S2); S2.k = v; upd2(); });
$("b2play").addEventListener("click", () => { S2.k = 0; upd2(); runSteps(S2, G1[S2.d].length - 1, () => { S2.k++; upd2(); }, 160); });

/* ================= 3절 / Section 3 ================= */
// 그림 GAN: 고정한 z 64개로 만든 8×8 격자, epoch마다
// Image GANs: 8×8 grids made from 64 fixed z, at chosen epochs
const GN = D15.gans, S3 = { m: "mnist_cnn", e: 2 };
Object.entries(GN).forEach(([k, v]) => Object.entries(v.png).forEach(([ep, b]) => loadImg(`${k}_${ep}`, b, () => [F3])));
const F3 = fig("cv3", w => Math.min(w, 420) + 40, (ctx, w, h, c) => {
  const eps = Object.keys(GN[S3.m].png).sort((a, b) => a - b), ep = eps[S3.e], sz = Math.min(w - 32, 400);
  panelTitle(ctx, c, tr(`${{ mnist_lin: "MNIST, linear GAN", mnist_cnn: "MNIST, CNN GAN", fmnist_cnn: "FMNIST, CNN GAN", cifar_cnn: "CIFAR10, CNN GAN" }[S3.m]} — epoch ${ep}`, `${{ mnist_lin: "MNIST, linear GAN", mnist_cnn: "MNIST, CNN GAN", fmnist_cnn: "FMNIST, CNN GAN", cifar_cnn: "CIFAR10, CNN GAN" }[S3.m]} — epoch ${ep}`), 16, 18);
  putImg(ctx, c, IMG[`${S3.m}_${ep}`], 16, 30, sz, sz);
});
function upd3() {
  const eps = Object.keys(GN[S3.m].png).sort((a, b) => a - b); S3.e = Math.min(S3.e, eps.length - 1);
  document.querySelectorAll("#g3e button").forEach((b, k) => { b.textContent = tr(`epoch ${eps[k]}`, `epoch ${eps[k]}`); });
  paint(F3);
  $("r3").textContent = tr(`G parameter ${commas(GN[S3.m].G)}개, D parameter ${commas(GN[S3.m].D)}개. 같은 z 64개로 그렸으니 epoch마다 같은 자리의 그림이 어떻게 바뀌는지 볼 수 있다.`, `G has ${commas(GN[S3.m].G)} parameters, D ${commas(GN[S3.m].D)}. The same 64 z are used, so you can follow how the picture in each spot changes from epoch to epoch.`);
}
bindSeg("g3m", v => { S3.m = v; upd3(); });
bindSeg("g3e", v => { S3.e = +v; upd3(); });

/* ================= 4절 / Section 4 ================= */
// forward process를 진짜 숫자에: x_t = √ᾱ_t x₀ + √(1−ᾱ_t) ε (ε는 고정)
// The forward process on a real digit: x_t = √ᾱ_t x₀ + √(1−ᾱ_t) ε (ε fixed)
const TT = 1000, BETA = Array.from({ length: TT }, (_, i) => 1e-4 + (0.02 - 1e-4) * i / (TT - 1)), ABAR = []; BETA.reduce((p, b, i) => (ABAR[i] = p * (1 - b)), 1);
const ab = t => (t <= 0 ? 1 : ABAR[t - 1]);
const REAL = Uint8Array.from(atob(D15.real), ch => ch.charCodeAt(0)), EPS = (() => { const r = makeRng(5); return Array.from({ length: 784 }, () => r.n()); })();
const S4 = { t: 250, i: 0, k: 0, timer: 0 };
const F4 = fig("cv4", w => (w >= 640 ? 180 : 420), (ctx, w, h, c) => {
  const x0 = Array.from(REAL.subarray(S4.i * 784, (S4.i + 1) * 784), v => v / 127.5 - 1), ts = [0, 50, 100, 250, 500, 750, 1000], per = w >= 640 ? 7 : 4, cell = Math.min(4, (w - 32 - (per - 1) * 12) / per / 28);
  ts.forEach((t, k) => {
    const a = ab(t), img = x0.map((v, j) => (Math.sqrt(a) * v + Math.sqrt(1 - a) * EPS[j] + 1) / 2), x = 16 + (k % per) * (28 * cell + 12), y = 30 + Math.floor(k / per) * (28 * cell + 40);
    drawImg(ctx, c, img.map(v => clamp(v, 0, 1)), 28, 28, x, y, cell);
    txt(ctx, `t = ${t}`, x + 14 * cell, y + 28 * cell + 15, { color: t === S4.t ? c.miss : c.muted, size: 11, mono: true, align: "center", w: t === S4.t ? "700" : "400" });
  });
  panelTitle(ctx, c, tr("진짜 숫자에 noise를 조금씩 더하기 (같은 ε)", "Adding noise to a real digit, a little at a time (the same ε)"), 16, 16);
});
const F4s = fig("cv4s", w => (w >= 640 ? 200 : 400), (ctx, w, h, c) => {
  const x0 = Array.from(REAL.subarray(S4.i * 784, (S4.i + 1) * 784), v => v / 127.5 - 1), a = ab(S4.t), cell = Math.min(6, (w - 60) / 56);
  const img = x0.map((v, j) => clamp((Math.sqrt(a) * v + Math.sqrt(1 - a) * EPS[j] + 1) / 2, 0, 1));
  panelTitle(ctx, c, `x₀`, 16, 16); drawImg(ctx, c, x0.map(v => (v + 1) / 2), 28, 28, 16, 26, cell);
  const x1 = w >= 640 ? 16 + 28 * cell + 40 : 16, y1 = w >= 640 ? 26 : 26 + 28 * cell + 30;
  panelTitle(ctx, c, `x_t (t = ${S4.t})`, x1, y1 - 10); drawImg(ctx, c, img, 28, 28, x1, y1, cell);
});
function upd4() { paint(F4); paint(F4s); const a = ab(S4.t); $("r4").textContent = tr(`t = ${S4.t}: ᾱ_t = ${fx(a, 4)} → x_t = ${fx(Math.sqrt(a), 3)} · x₀ + ${fx(Math.sqrt(1 - a), 3)} · ε. 원래 그림의 몫 √ᾱ_t가 1에서 0으로 줄고, noise의 몫이 0에서 1로 는다.`, `t = ${S4.t}: ᾱ_t = ${fx(a, 4)} → x_t = ${fx(Math.sqrt(a), 3)} · x₀ + ${fx(Math.sqrt(1 - a), 3)} · ε. The picture's share √ᾱ_t falls from 1 to 0 while the noise's share rises from 0 to 1.`); }
bindRange("i4t", "o4t", v => String(v), v => { S4.t = v; upd4(); paint(F5); });
$("b4next").addEventListener("click", () => { S4.i = (S4.i + 1) % 4; upd4(); });
// 2D 점 구름: 앞으로(퍼지기) 그리고 거꾸로(network로 되돌리기)
// 2-D point cloud: forward (spreading out) and reverse (brought back by the network)
const DF = D15.diff2d, ORDER = [...DF.keep.map(t => ["fwd", t]), ...DF.keep.slice().reverse().slice(1).map(t => ["rev", t])], S4b = { k: 0, timer: 0 };
const F4b = fig("cv4b", 300, (ctx, w, h, c) => {
  const [dir, t] = ORDER[S4b.k], P = (dir === "fwd" ? DF.fwd : DF.rev)[String(t)], side = Math.min(h - 40, w - 32), B = { l: 16, t: 30, w: side, h: side }, X = v => B.l + (v + 2) / 4 * B.w, Y = v => B.t + B.h - (v + 2) / 4 * B.h;
  rrect(ctx, B.l, B.t, B.w, B.h, 6, c.bg, c.line); clipBox(ctx, B);
  ctx.fillStyle = dir === "fwd" ? c.sig : c.miss; ctx.globalAlpha = 0.6; P.forEach(p => { ctx.fillRect(X(p[0]) - 1.5, Y(p[1]) - 1.5, 3, 3); }); ctx.globalAlpha = 1; ctx.restore();
  panelTitle(ctx, c, dir === "fwd" ? tr(`앞으로: t = ${t} (noise 더하기)`, `forward: t = ${t} (adding noise)`) : tr(`거꾸로: t = ${t} (network가 noise를 빼기)`, `reverse: t = ${t} (the network removes noise)`), B.l, 16);
  // 오른쪽: 진행 막대 / right: progress bar
  if (w - side > 150) {
    const bx = B.l + side + 30, bw = w - bx - 20; ORDER.forEach(([d, tt], k) => { ctx.fillStyle = k === S4b.k ? (d === "fwd" ? c.sig : c.miss) : c.line; ctx.fillRect(bx + k * bw / ORDER.length, B.t + B.h / 2, bw / ORDER.length - 2, 10); });
    txt(ctx, tr("data → noise", "data → noise"), bx, B.t + B.h / 2 - 8, { color: c.sig, size: 11 }); txt(ctx, tr("→ 다시 data", "→ data again"), bx + bw, B.t + B.h / 2 - 8, { color: c.miss, size: 11, align: "right" });
  }
});
function upd4b() { paint(F4b); const [dir, t] = ORDER[S4b.k]; $("r4b").textContent = dir === "fwd" ? tr(`점 800개(swiss roll)에 noise를 더한다. t = ${DF.T}이면 거의 정규분포 덩어리.`, `Noise is added to 800 points (a swiss roll). At t = ${DF.T} it is almost a normal-distribution blob.`) : tr(`새 noise 800개에서 출발해 학습한 network로 ${DF.T}걸음 되돌린다: 원래 data를 본 적 없는 새 점들이 roll 모양에 모인다.`, `Starting from 800 fresh noise points, the trained network steps back ${DF.T} times: new points, never seen in the data, gather into the roll shape.`); }
$("b4bplay").addEventListener("click", () => { S4b.k = 0; upd4b(); runSteps(S4b, ORDER.length - 1, () => { S4b.k++; upd4b(); }, 600); });
$("b4bstep").addEventListener("click", () => { stopRun(S4b); S4b.k = (S4b.k + 1) % ORDER.length; upd4b(); });

/* ================= 5절 / Section 5 ================= */
// noise schedule: β_t (곧게 증가), ᾱ_t, √ᾱ_t, √(1−ᾱ_t)
// Noise schedule: β_t (rising in a line), ᾱ_t, √ᾱ_t, √(1−ᾱ_t)
const F5 = fig("cv5", 260, (ctx, w, h, c) => {
  const B = { l: 44, t: 30, w: w - 120, h: h - 74 }, A = axes(ctx, c, B, [0, 1000], [0, 1], { xt: [0, 250, 500, 750, 1000], yt: [0, 0.5, 1], xlabel: "t" });
  const pts = f => Array.from({ length: 201 }, (_, k) => { const t = k * 5; return [A.X(t), A.Y(f(t))]; });
  line(ctx, pts(t => Math.sqrt(ab(t))), c.sig, 2.2); line(ctx, pts(t => Math.sqrt(1 - ab(t))), c.fa, 2.2); line(ctx, pts(t => ab(t)), c.ok, 1.6, [6, 3]); line(ctx, pts(t => (t ? BETA[t - 1] : 0) / 0.02), c.muted, 1.4, [2, 3]);
  line(ctx, [[A.X(S4.t), B.t], [A.X(S4.t), B.t + B.h]], c.ink, 1, [3, 3]);
  [["√(1−ᾱ_t)", A.Y(1) + 4, c.fa], ["β_t ÷ 0.02", A.Y(1) + 18, c.muted], ["√ᾱ_t", A.Y(0) - 16, c.sig], ["ᾱ_t", A.Y(0) - 2, c.ok]].forEach(([lab, y, col]) => txt(ctx, lab, B.l + B.w + 8, y, { color: col, size: 11, w: "700" }));
  panelTitle(ctx, c, tr("T = 1000, β: 0.0001 → 0.02 (곧게). 점선 세로줄 = 4절 slider의 t", "T = 1000, β: 0.0001 → 0.02 (linear). Vertical dashes = t from the section 4 slider"), B.l, 16);
});

/* ================= 7절 / Section 7 ================= */
// sampling: 순수한 noise(t = 1000)에서 숫자(t = 0)까지 1000걸음, 그 중간 모습
// Sampling: 1000 steps from pure noise (t = 1000) to digits (t = 0), with snapshots on the way
const DD = D15.ddpm, SN = Object.keys(DD.uncond).map(Number).sort((a, b) => b - a), S7 = { k: SN.length - 1, timer: 0 };
SN.forEach(t => loadImg("u" + t, DD.uncond[String(t)], () => [F7]));
const F7 = fig("cv7", w => (w >= 640 ? 260 : 470), (ctx, w, h, c) => {
  const wide = w >= 640, sz = wide ? 200 : Math.min(w - 32, 260), t = SN[S7.k];
  panelTitle(ctx, c, tr(`t = ${t}`, `t = ${t}`), 16, 18); putImg(ctx, c, IMG["u" + t], 16, 30, sz, sz);
  const sx = wide ? 16 + sz + 30 : 16, sy = wide ? 30 : 30 + sz + 30, n = SN.length, cs = Math.min(64, (w - sx - 16) / n - 6);
  SN.forEach((tt, k) => { putImg(ctx, c, IMG["u" + tt], sx + k * (cs + 6), sy, cs, cs); txt(ctx, String(tt), sx + k * (cs + 6) + cs / 2, sy + cs + 14, { color: k === S7.k ? c.miss : c.muted, size: 10, mono: true, align: "center", w: k === S7.k ? "700" : "400" }); });
});
function upd7() { paint(F7); $("o7k").textContent = String(SN[S7.k]); $("r7").textContent = tr(`조건 없이(class를 지운 채) 16장. t = ${SN[S7.k]}. 큰 t에서는 모양의 윤곽이, 작은 t에서는 획의 세부가 정해진다. U-Net parameter ${commas(DD.params)}개, MNIST 12 epoch.\n조건이 없으면 숫자처럼 보이지 않는 것(B·A 모양)도 섞인다: 8절처럼 조건을 주면 훨씬 낫다.`, `16 samples without a condition (class dropped). t = ${SN[S7.k]}. At large t the rough shape is decided, at small t the stroke details. U-Net with ${commas(DD.params)} parameters, MNIST, 12 epochs.\nWithout a condition some come out not digit-like (B- or A-shapes): with a condition, as in section 8, it does much better.`); }
bindRange("i7k", null, v => v, v => { stopRun(S7); S7.k = v; upd7(); });
$("b7play").addEventListener("click", () => { S7.k = 0; setRange("i7k", 0, 0); upd7(); runSteps(S7, SN.length - 1, () => { S7.k++; setRange("i7k", S7.k, 0); upd7(); }, 500); });

/* ================= 8절 / Section 8 ================= */
// classifier-free guidance: w = 0, 1, 3에서 숫자 0~9 (줄마다 4장)
// Classifier-free guidance: digits 0–9 at w = 0, 1, 3 (4 per row)
const GD = DD.guid, S8 = { w: 0 };
GD.png.forEach((b, k) => loadImg("g" + k, b, () => [F8]));
const F8 = fig("cv8", 330, (ctx, w, h, c) => {
  const wd = Math.min(112, (w - 92) / 3), ht = wd * 2.5;
  GD.w.forEach((wv, k) => { const x = 16 + k * (wd + 30); if (x + wd > w) return; txt(ctx, `w = ${wv}`, x + wd / 2, 18, { color: k === S8.w ? c.miss : c.ink, size: 12, w: "700", align: "center" }); putImg(ctx, c, IMG["g" + k], x, 28, wd, ht); if (k === S8.w) { ctx.strokeStyle = c.miss; ctx.lineWidth = 2.4; ctx.strokeRect(x - 3, 25, wd + 6, ht + 6); } txt(ctx, tr(`맞음 ${Math.round(GD.hit[k] * 100)}%`, `on target ${Math.round(GD.hit[k] * 100)}%`), x + wd / 2, 28 + ht + 16, { color: c.ok, size: 11, w: "700", align: "center" }); });
});
function upd8() { paint(F8); $("r8").textContent = tr(`w = ${GD.w[S8.w]}: 숫자마다 50장씩 500장 중 ${Math.round(GD.hit[S8.w] * 1000) / 10}%가 요청한 숫자로 읽혔다 (판정: 시험 정확도 ${(GD.judge * 100).toFixed(1)}%인 작은 CNN). 줄 = 0부터 9까지.\nw가 클수록 조건을 더 강하게 따르고, 대신 다양함이 줄어든다.`, `w = ${GD.w[S8.w]}: of 500 samples (50 per digit), ${Math.round(GD.hit[S8.w] * 1000) / 10}% were read as the requested digit (judged by a small CNN with ${(GD.judge * 100).toFixed(1)}% test accuracy). Rows = 0 to 9.\nThe larger w, the more strictly the condition is followed, at the cost of variety.`); }
bindSeg("g8w", v => { S8.w = +v; upd8(); });

// 처음 그리기
// First paint
draw1(); upd1(); upd2(); upd3(); upd4(); upd4b(); paint(F5); upd7(); upd8();
repaintAll();
