/* 2장 그림과 조작. 한글판(ch02-gradient-descent-KR.html)과 영어판(ch02-gradient-descent.html)이 같이 쓴다. 글자는 tr(한글, 영어)
   Chapter 2 figures and interactions, shared by the Korean and English pages. Text goes through tr(Korean, English) */
"use strict";

// 자주 쓰는 짧은 도구
// Small shared helpers
const $ = id => document.getElementById(id);
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
function fx(v, d = 2) { let s = (+v).toFixed(d); if (/^-0(\.0*)?$/.test(s)) s = s.slice(1); return s.replace("-", "−"); }
const paren = (v, d = 2) => (v < 0 ? "(" + fx(v, d) + ")" : fx(v, d));
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const SUP = n => String(n).replace(/./g, ch => "⁰¹²³⁴⁵⁶⁷⁸⁹"["0123456789".indexOf(ch)] || (ch === "-" ? "⁻" : ch));
function sci(v) {
  const a = Math.abs(v); if (a === 0) return "0";
  if (a >= 1e4 || a < 1e-3) { const e = Math.floor(Math.log10(a)); return fx(v / 10 ** e, 1) + "×10" + SUP(e); }
  return a >= 100 ? fx(v, 0) : a >= 1 ? fx(v, 2) : fx(v, 4);
}
function txtBg(ctx, c, s, x, y, o = {}) {
  ctx.save(); ctx.font = (o.w || "400") + " " + (o.size || 11.5) + "px " + (o.mono ? FM : FS);
  const wd = ctx.measureText(s).width, x0 = o.align === "right" ? x - wd : o.align === "center" ? x - wd / 2 : x;
  ctx.fillStyle = c.bg; ctx.globalAlpha = 0.9; ctx.fillRect(x0 - 3, y - (o.size || 11.5) + 1, wd + 6, (o.size || 11.5) + 5); ctx.restore();
  txt(ctx, s, x, y, o);
}
function panelTitle(ctx, c, s, x, y) { txt(ctx, s, x, y, { color: c.ink, w: "600", size: 12.5 }); }
// 함수 곡선의 점들
// Points along a function curve
function curve(A, f, xr, n = 240) { const p = []; for (let i = 0; i <= n; i++) { const x = xr[0] + (xr[1] - xr[0]) * i / n; p.push([A.X(x), A.Y(f(x))]); } return p; }
// 작은 화살표 (머리가 작다)
// Small arrow with a small head
function arrowS(ctx, x0, y0, x1, y1, color, width = 1.4, head = 5) {
  line(ctx, [[x0, y0], [x1, y1]], color, width);
  const a = Math.atan2(y1 - y0, x1 - x0);
  ctx.save(); ctx.fillStyle = color; ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x1 - head * Math.cos(a - 0.45), y1 - head * Math.sin(a - 0.45)); ctx.lineTo(x1 - head * Math.cos(a + 0.45), y1 - head * Math.sin(a + 0.45)); ctx.closePath(); ctx.fill(); ctx.restore();
}
// 걸음을 차례로 보여 주는 timer (움직임 줄이기 설정이면 한 번에)
// Timer that plays steps one after another (all at once with reduced motion)
function stopRun(S) { clearInterval(S.timer); S.timer = 0; }
function runSteps(S, n, step, ms = 140, done) {
  stopRun(S); let k = 0;
  if (reduceMotion) { while (k++ < n && step() !== false); if (done) done(); return; }
  S.timer = setInterval(() => { if (k++ >= n || step() === false) { stopRun(S); if (done) done(); } }, ms);
}
function setRange(id, v, d = 2) { const el = $(id); el.value = v; const o = $(id.replace(/^i/, "o")); if (o) o.textContent = fx(v, d); }

// 2D 함수의 등고선 그림: 낮을수록 진한 파랑, 띠의 경계 = 등고선. 같은 조건이면 다시 계산하지 않는다
// Contour picture of a 2D function: lower = deeper blue, band edges = contour lines. Cached for identical inputs
function rgb(hex) { let h = hex.trim().replace("#", ""); if (h.length === 3) h = h.split("").map(ch => ch + ch).join(""); const n = parseInt(h, 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; }
const contourCache = new Map();
function contour(ctx, c, B, f, xr, yr, key, levels = 13) {
  const cell = 2, nx = Math.max(2, Math.round(B.w / cell)), ny = Math.max(2, Math.round(B.h / cell));
  const k = [key, nx, ny, xr, yr, c.bg, c.sig].join("|");
  let off = contourCache.get(k);
  if (!off) {
    const v = new Float32Array(nx * ny); let lo = Infinity, hi = -Infinity;
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const z = f(xr[0] + (i + 0.5) / nx * (xr[1] - xr[0]), yr[1] - (j + 0.5) / ny * (yr[1] - yr[0]));
      v[j * nx + i] = z; if (z < lo) lo = z; if (z > hi) hi = z;
    }
    const top = Math.sqrt(hi - lo) || 1, band = new Int16Array(nx * ny);
    for (let i = 0; i < v.length; i++) band[i] = Math.min(levels - 1, Math.floor(Math.sqrt(v[i] - lo) / top * levels));
    off = document.createElement("canvas"); off.width = nx; off.height = ny;
    const ox = off.getContext("2d"), img = ox.createImageData(nx, ny), b0 = rgb(c.bg), b1 = rgb(c.sig), lc = rgb(c.muted);
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const q = j * nx + i, L = band[q], edge = (i < nx - 1 && band[q + 1] !== L) || (j < ny - 1 && band[q + nx] !== L);
      const a = edge ? 0.5 : 0.07 + 0.5 * (1 - L / (levels - 1)), to = edge ? lc : b1;
      for (let ch = 0; ch < 3; ch++) img.data[q * 4 + ch] = Math.round(b0[ch] + (to[ch] - b0[ch]) * a);
      img.data[q * 4 + 3] = 255;
    }
    ox.putImageData(img, 0, 0);
    if (contourCache.size > 40) contourCache.clear();
    contourCache.set(k, off);
  }
  ctx.save(); ctx.imageSmoothingEnabled = false; ctx.drawImage(off, B.l, B.t, B.w, B.h); ctx.restore();
}
// 같은 비율(1 단위 = 같은 pixel)의 2D 좌표 범위
// 2D range with equal scale on both axes
function equalRange(B, yHalf, cx = 0) { const xh = yHalf * B.w / B.h; return { xr: [cx - xh, cx + xh], yr: [-yHalf, yHalf] }; }
function intTicks(r) { const out = []; for (let v = Math.ceil(r[0]); v <= Math.floor(r[1]); v++) out.push(v); return out; }

/* ================= 1절 / Section 1 ================= */
// 점 4개에 직선 y = w x 맞추기. 왼쪽 = data와 직선과 오차, 오른쪽 = loss 곡선 L(w)
// Fit the line y = w x to four points. Left = data, line and errors; right = the loss curve L(w)
const X1 = [1, 2, 3, 4], Y1 = [2.1, 3.9, 6.2, 7.8], ETA1 = 0.03, W1BEST = 1.99;
const S1 = { w: 1, hist: [], timer: 0 };
const loss1 = w => X1.reduce((s, x, i) => s + (w * x - Y1[i]) ** 2, 0) / X1.length;
const grad1 = w => X1.reduce((s, x, i) => s + 2 * (w * x - Y1[i]) * x, 0) / X1.length;
const F1 = fig("cv1", w => (stacked(w, 260, 2, 28) ? 580 : 300), (ctx, w, h, c) => {
  const P = panels(w, h, 2, 28, 260);
  {
    const p = P[0], B = { l: p.x + 40, t: p.y + 30, w: p.w - 52, h: p.h - 76 };
    const A = axes(ctx, c, B, [0, 4.6], [0, 11], { xt: [0, 1, 2, 3, 4], yt: [0, 2, 4, 6, 8, 10], xlabel: "x" });
    panelTitle(ctx, c, tr("data와 직선 ŷ = w·x", "Data and the line ŷ = w·x"), B.l, p.y + 14);
    clipBox(ctx, B);
    X1.forEach((x, i) => {
      const yh = S1.w * x, side = Math.abs(A.Y(Y1[i]) - A.Y(yh));
      ctx.fillStyle = c.fa; ctx.globalAlpha = 0.1; ctx.fillRect(A.X(x) - side, Math.min(A.Y(Y1[i]), A.Y(yh)), side, side); ctx.globalAlpha = 1;
      line(ctx, [[A.X(x), A.Y(Y1[i])], [A.X(x), A.Y(yh)]], c.fa, 2.2);
    });
    line(ctx, [[A.X(0), A.Y(0)], [A.X(4.6), A.Y(4.6 * S1.w)]], c.sig, 2.4);
    ctx.restore();
    X1.forEach((x, i) => dot(ctx, A.X(x), A.Y(Y1[i]), 5, c.ink));
    txtBg(ctx, c, tr("● 정답 y   빨강 = 오차   연한 사각형 = 오차²", "● answer y   red = error   pale square = error²"), B.l + 6, B.t + 13, { color: c.fa, size: 11, w: "600" });
    txtBg(ctx, c, `w = ${fx(S1.w)}`, A.X(4.4), clamp(A.Y(4.4 * S1.w) + (S1.w > 2.2 ? 16 : -8), B.t + 30, B.t + B.h - 6), { color: c.sig, size: 12, w: "700", align: "right" });
  }
  {
    const p = P[1], B = { l: p.x + 40, t: p.y + 30, w: p.w - 52, h: p.h - 76 };
    const A = axes(ctx, c, B, [0, 4], [0, 32], { xt: [0, 1, 2, 3, 4], yt: [0, 10, 20, 30], xlabel: "w" });
    panelTitle(ctx, c, tr("loss L(w): w마다 오차²의 평균", "Loss L(w): mean error² for each w"), B.l, p.y + 14);
    clipBox(ctx, B);
    line(ctx, curve(A, loss1, [0, 4]), c.sig, 2.2);
    const w0 = S1.w, L0 = loss1(w0), g = grad1(w0), dw = 0.4;
    line(ctx, [[A.X(w0 - dw), A.Y(L0 - g * dw)], [A.X(w0 + dw), A.Y(L0 + g * dw)]], c.ok, 2.2);
    ctx.restore();
    S1.hist.forEach(v => dot(ctx, A.X(v), A.Y(loss1(v)), 3.5, null, c.muted));
    dot(ctx, A.X(W1BEST), A.Y(loss1(W1BEST)), 3, c.muted);
    line(ctx, [[A.X(W1BEST), A.Y(4.2)], [A.X(W1BEST), A.Y(0.6)]], c.muted, 1, [3, 3]);
    txt(ctx, tr("가장 낮은 곳 w = 1.99", "lowest at w = 1.99"), A.X(W1BEST), A.Y(5), { color: c.muted, size: 10.5, align: "center" });
    dot(ctx, A.X(w0), A.Y(L0), 6.5, c.miss);
    txtBg(ctx, c, tr(`기울기 ${fx(g)}`, `slope ${fx(g)}`), A.X(w0) + (w0 < 2.6 ? 12 : -12), A.Y(L0) - 8, { color: c.ok, size: 11.5, w: "600", align: w0 < 2.6 ? "left" : "right" });
    const wn = clamp(w0 - ETA1 * g, 0, 4), ya = B.t + 14;
    if (Math.abs(A.X(wn) - A.X(w0)) > 8) {
      arrow(ctx, A.X(w0), ya, A.X(wn), ya, c.ok, 2);
      txtBg(ctx, c, tr("다음 w", "next w"), A.X(wn) + (wn > w0 ? 6 : -6), ya + 4, { color: c.ok, size: 11, w: "600", align: wn > w0 ? "left" : "right" });
    }
  }
});
function upd1() {
  paint(F1);
  const w = S1.w, yh = X1.map(x => w * x), e = yh.map((v, i) => v - Y1[i]), L = loss1(w), g = grad1(w), wn = w - ETA1 * g;
  const dir = Math.abs(g) < 0.05 ? tr("거의 0 → 바닥에 왔다", "almost 0 → we are at the bottom") : g < 0 ? tr("음수 → w를 키우면 loss가 준다", "negative → making w larger lowers the loss") : tr("양수 → w를 줄이면 loss가 준다", "positive → making w smaller lowers the loss");
  $("r1").textContent = tr(
    `w = ${fx(w)} → 예측 ŷ = ${yh.map(v => fx(v)).join(", ")}\n오차 ŷ − y = ${e.map(v => fx(v)).join(", ")}\nloss = 평균(오차²) = (${e.map(v => fx(v * v)).join(" + ")}) ÷ 4 = ${fx(L, 3)}\n기울기 dL/dw = ${fx(g)} (${dir})\n다음 w = ${fx(w)} − ${ETA1} × ${paren(g)} = ${fx(wn)}   (지금까지 ${S1.hist.length} 걸음)`,
    `w = ${fx(w)} → predictions ŷ = ${yh.map(v => fx(v)).join(", ")}\nerrors ŷ − y = ${e.map(v => fx(v)).join(", ")}\nloss = mean(error²) = (${e.map(v => fx(v * v)).join(" + ")}) ÷ 4 = ${fx(L, 3)}\nslope dL/dw = ${fx(g)} (${dir})\nnext w = ${fx(w)} − ${ETA1} × ${paren(g)} = ${fx(wn)}   (${S1.hist.length} steps so far)`);
}
function step1() { S1.hist.push(S1.w); S1.w = clamp(S1.w - ETA1 * grad1(S1.w), 0, 4); setRange("i1w", S1.w); upd1(); }
bindRange("i1w", "o1w", v => fx(v), v => { stopRun(S1); S1.w = v; S1.hist = []; upd1(); });
$("b1step").addEventListener("click", () => { stopRun(S1); step1(); });
$("b1run").addEventListener("click", () => runSteps(S1, 10, step1, 280));
$("b1reset").addEventListener("click", () => { stopRun(S1); S1.w = 1; S1.hist = []; setRange("i1w", 1); upd1(); });

/* ================= 2절 / Section 2 ================= */
// 위 = 함수 f, 아래 = 그 기울기 f′. 위의 접선 기울기가 아래 그래프의 높이가 된다
// Top = the function f, bottom = its slope f′. The tangent slope on top becomes the height below
const FN2 = {
  x2: { f: x => x * x, d: x => 2 * x, nm: ["x²", "2x"], yr: [-1, 5], yt: [0, 2, 4], dr: [-5, 5], dt: [-4, -2, 0, 2, 4], ex: x => `2 × ${paren(x)}` },
  x3: { f: x => x ** 3, d: x => 3 * x * x, nm: ["x³", "3x²"], yr: [-11, 11], yt: [-10, -5, 0, 5, 10], dr: [-1.5, 15], dt: [0, 5, 10, 15], ex: x => `3 × ${paren(x)}²` },
  cub: { f: x => x ** 3 - 3 * x, d: x => 3 * x * x - 3, nm: ["x³ − 3x", "3x² − 3"], yr: [-4.6, 4.6], yt: [-4, -2, 0, 2, 4], dr: [-4, 12], dt: [-3, 0, 3, 6, 9, 12], ex: x => `3 × ${paren(x)}² − 3` }
};
const S2 = { k: "x3", x: 1, tr: null, timer: 0 };
const F2 = fig("cv2", 420, (ctx, w, h, c) => {
  const F = FN2[S2.k], xr = [-2.2, 2.2], x = S2.x;
  const B1 = { l: 44, t: 26, w: w - 60, h: 158 }, B2 = { l: 44, t: 240, w: w - 60, h: 140 };
  const A1 = axes(ctx, c, B1, xr, F.yr, { xt: [-2, -1, 0, 1, 2], yt: F.yt });
  panelTitle(ctx, c, `f(x) = ${F.nm[0]}`, B1.l, 16);
  clipBox(ctx, B1);
  line(ctx, [[B1.l, A1.Y(0)], [B1.l + B1.w, A1.Y(0)]], c.muted, 1);
  line(ctx, curve(A1, F.f, xr), c.sig, 2.4);
  const fx0 = F.f(x), m = F.d(x), dx = 0.55;
  line(ctx, [[A1.X(x - dx), A1.Y(fx0 - m * dx)], [A1.X(x + dx), A1.Y(fx0 + m * dx)]], c.ok, 2.6);
  ctx.restore();
  dot(ctx, A1.X(x), A1.Y(fx0), 5.5, c.ok);
  txt(ctx, tr(`초록 접선의 기울기 = ${fx(m)}`, `slope of the green tangent = ${fx(m)}`), B1.l + B1.w, 16, { color: c.ok, size: 11.5, w: "700", align: "right" });
  const A2 = axes(ctx, c, B2, xr, F.dr, { xt: [-2, -1, 0, 1, 2], yt: F.dt, xlabel: "x" });
  panelTitle(ctx, c, tr(`f′(x) = ${F.nm[1]}: 위 그래프의 기울기`, `f′(x) = ${F.nm[1]}: the slope of the graph above`), B2.l, B2.t - 12);
  clipBox(ctx, B2);
  if (F.dr[0] < -2) {
    ctx.fillStyle = c.okwash; ctx.fillRect(B2.l, A2.Y(0), B2.w, B2.t + B2.h - A2.Y(0));
    line(ctx, [[B2.l, A2.Y(0)], [B2.l + B2.w, A2.Y(0)]], c.muted, 1);
  }
  const xEnd = S2.tr === null ? xr[1] : S2.tr;
  if (xEnd > xr[0]) line(ctx, curve(A2, F.d, [xr[0], xEnd], 200), c.ok, 2.4);
  ctx.restore();
  if (F.dr[0] < -2) {
    txt(ctx, tr("+ 오르막", "+ uphill"), B2.l + 6, A2.Y(0) - 6, { color: c.fa, size: 11, w: "600" });
    txt(ctx, tr("− 내리막", "− downhill"), B2.l + 6, A2.Y(0) + 15, { color: c.ok, size: 11, w: "600" });
  }
  line(ctx, [[A1.X(x), A1.Y(clamp(fx0, F.yr[0], F.yr[1]))], [A2.X(x), A2.Y(clamp(m, F.dr[0], F.dr[1]))]], c.muted, 1.2, [4, 4]);
  if (m >= F.dr[0] && m <= F.dr[1]) dot(ctx, A2.X(x), A2.Y(m), 5.5, c.ok);
});
function upd2() {
  paint(F2);
  const F = FN2[S2.k], x = S2.x, m = F.d(x);
  const dir = Math.abs(m) < 0.005 ? tr("평평 → 바닥 또는 꼭대기", "flat → a bottom or a top") : m > 0 ? tr("오르막 (x를 늘리면 f가 는다)", "uphill (f grows as x grows)") : tr("내리막 (x를 늘리면 f가 준다)", "downhill (f shrinks as x grows)");
  $("r2").textContent = tr(
    `f(x) = ${F.nm[0]} → f′(x) = ${F.nm[1]}   (규칙: xⁿ → n·xⁿ⁻¹)\nx = ${fx(x)}: f = ${fx(F.f(x))}, 기울기 f′ = ${F.ex(x)} = ${fx(m)} → ${dir}`,
    `f(x) = ${F.nm[0]} → f′(x) = ${F.nm[1]}   (rule: xⁿ → n·xⁿ⁻¹)\nx = ${fx(x)}: f = ${fx(F.f(x))}, slope f′ = ${F.ex(x)} = ${fx(m)} → ${dir}`);
}
bindSeg("g2f", v => { S2.k = v; S2.tr = null; stopRun(S2); upd2(); });
bindRange("i2x", "o2x", v => fx(v), v => { stopRun(S2); S2.tr = null; S2.x = v; upd2(); });
$("b2trace").addEventListener("click", () => {
  S2.tr = -2.2; S2.x = -2.2;
  runSteps(S2, 88, () => { S2.tr = Math.min(2.2, S2.tr + 0.05); S2.x = S2.tr; setRange("i2x", S2.x); upd2(); }, 45, () => { S2.tr = null; upd2(); });
});

/* ================= 3절 / Section 3 ================= */
// 곡선을 f′의 부호로 칠한다: 초록 = 내리막(f′ < 0), 빨강 = 오르막(f′ > 0). f′ = 0인 곳이 바닥이나 꼭대기
// Colour the curve by the sign of f′: green = downhill (f′ < 0), red = uphill (f′ > 0). Where f′ = 0 is a bottom or a top
const FN3 = {
  a: { f: x => x * x - 4 * x + 5, d: x => 2 * x - 4, xr: [-0.6, 4.6], yr: [0, 10.5], xt: [0, 1, 2, 3, 4], yt: [0, 2, 4, 6, 8, 10], z: [2], nm: ["x² − 4x + 5", "2x − 4"], x0: 3.5 },
  b: { f: x => x ** 3 - 3 * x, d: x => 3 * x * x - 3, xr: [-2.3, 2.3], yr: [-4.8, 4.8], xt: [-2, -1, 0, 1, 2], yt: [-4, -2, 0, 2, 4], z: [-1, 1], nm: ["x³ − 3x", "3x² − 3"], x0: 1.8 },
  c: { f: x => x ** 4 / 4 - x * x, d: x => x ** 3 - 2 * x, xr: [-2.3, 2.3], yr: [-1.6, 2.2], xt: [-2, -1, 0, 1, 2], yt: [-1, 0, 1, 2], z: [-Math.SQRT2, 0, Math.SQRT2], nm: ["x⁴/4 − x²", "x³ − 2x"], x0: 1.9 }
};
const S3 = { k: "a", x: 3.5 };
const kind3 = (F, z) => (F.d(z - 0.01) < 0 && F.d(z + 0.01) > 0 ? "min" : F.d(z - 0.01) > 0 && F.d(z + 0.01) < 0 ? "max" : "flat");
const F3 = fig("cv3", 350, (ctx, w, h, c) => {
  const F = FN3[S3.k], B = { l: 44, t: 26, w: w - 60, h: h - 112 };
  const A = axes(ctx, c, B, F.xr, F.yr, { xt: F.xt, yt: F.yt });
  panelTitle(ctx, c, `f(x) = ${F.nm[0]}`, B.l, 16);
  clipBox(ctx, B);
  const n = 300; let run = [], sgn = 0;
  for (let i = 0; i <= n; i++) {
    const x = F.xr[0] + (F.xr[1] - F.xr[0]) * i / n, s = F.d(x) < 0 ? -1 : 1, p = [A.X(x), A.Y(F.f(x))];
    if (s !== sgn && run.length) { run.push(p); line(ctx, run, sgn < 0 ? c.ok : c.fa, 2.6); run = []; }
    sgn = s; run.push(p);
  }
  line(ctx, run, sgn < 0 ? c.ok : c.fa, 2.6);
  const x = S3.x, m = F.d(x), y = F.f(x), dx = (F.xr[1] - F.xr[0]) * 0.1;
  line(ctx, [[A.X(x - dx), A.Y(y - m * dx)], [A.X(x + dx), A.Y(y + m * dx)]], c.ink, 1.4, [5, 4]);
  ctx.restore();
  F.z.forEach(z => {
    const k = kind3(F, z), X = A.X(z), Y = A.Y(F.f(z));
    if (k === "min") { dot(ctx, X, Y, 6.5, c.ok); txtBg(ctx, c, tr("바닥: f′ = 0", "bottom: f′ = 0"), X, Y + 20, { color: c.ok, size: 11.5, w: "700", align: "center" }); }
    else if (k === "max") { dot(ctx, X, Y, 6.5, c.bg, c.fa); txtBg(ctx, c, tr("꼭대기: f′ = 0", "top: f′ = 0"), X, Y - 12, { color: c.fa, size: 11.5, w: "700", align: "center" }); }
  });
  dot(ctx, A.X(x), A.Y(y), 5, c.ink);
  // f′ 부호 띠
  // Sign strip of f′
  const sy = B.t + B.h + 40, sh = 18;
  txt(ctx, tr("f′ 부호", "sign of f′"), B.l - 4, sy - 6, { color: c.muted, size: 11, w: "600" });
  const cuts = [F.xr[0], ...F.z, F.xr[1]];
  for (let i = 0; i < cuts.length - 1; i++) {
    const a = cuts[i], b = cuts[i + 1], neg = F.d((a + b) / 2) < 0;
    ctx.fillStyle = neg ? c.okwash : c.wash; ctx.fillRect(A.X(a), sy, A.X(b) - A.X(a), sh);
    ctx.strokeStyle = neg ? c.ok : c.fa; ctx.lineWidth = 1; ctx.strokeRect(A.X(a) + 0.5, sy + 0.5, A.X(b) - A.X(a) - 1, sh - 1);
    if (A.X(b) - A.X(a) > 70) txt(ctx, neg ? tr("− 내리막", "− downhill") : tr("+ 오르막", "+ uphill"), (A.X(a) + A.X(b)) / 2, sy + 13, { color: neg ? c.ok : c.fa, size: 11, w: "600", align: "center" });
    else txt(ctx, neg ? "−" : "+", (A.X(a) + A.X(b)) / 2, sy + 13, { color: neg ? c.ok : c.fa, size: 12, w: "700", align: "center" });
  }
  F.z.forEach(z => txt(ctx, "0", A.X(z), sy + sh + 13, { color: c.ink, size: 11, mono: true, align: "center" }));
  arrowS(ctx, A.X(x), sy - 12, A.X(x), sy - 2, c.ink, 1.4);
});
function upd3() {
  paint(F3);
  const F = FN3[S3.k], x = S3.x, m = F.d(x);
  const where = F.z.map(z => { const k = kind3(F, z); return `x = ${fx(z)} (${k === "min" ? tr("바닥", "bottom") : k === "max" ? tr("꼭대기", "top") : tr("평평", "flat")}, f = ${fx(F.f(z))})`; }).join(", ");
  const dir = Math.abs(m) < 0.02 ? tr("거의 0 → 평평한 곳", "almost 0 → a flat spot") : m > 0 ? tr("양수 → 오르막. 더 낮은 곳은 왼쪽", "positive → uphill. Lower ground is to the left") : tr("음수 → 내리막. 더 낮은 곳은 오른쪽", "negative → downhill. Lower ground is to the right");
  $("r3").textContent = tr(
    `f′(x) = ${F.nm[1]}\nx = ${fx(x)}: f′ = ${fx(m)} → ${dir}\nf′ = 0인 곳: ${where}`,
    `f′(x) = ${F.nm[1]}\nx = ${fx(x)}: f′ = ${fx(m)} → ${dir}\nwhere f′ = 0: ${where}`);
}
bindSeg("g3f", v => {
  S3.k = v; const F = FN3[v], el = $("i3x");
  el.min = F.xr[0] + 0.1; el.max = F.xr[1] - 0.1; S3.x = F.x0; setRange("i3x", F.x0); upd3();
});
bindRange("i3x", "o3x", v => fx(v), v => { S3.x = v; upd3(); });

/* ================= 4절 / Section 4 ================= */
// product rule: 직사각형 u × v가 커질 때 늘어난 넓이 = 오른쪽 띠 + 위쪽 띠 + 작은 모서리
// Product rule: when the rectangle u × v grows, the new area = right strip + top strip + a tiny corner
const S4 = { x: 2, d: 0.3 };
const F4a = fig("cv4a", 300, (ctx, w, h, c) => {
  const x = S4.x, d = S4.d, u = x * x, v = x + 1, u2 = (x + d) ** 2, v2 = x + d + 1;
  const s = Math.min((w - 150) / 12.25, (h - 74) / 4.5), ox = 64, oy = h - 40;
  const R = (x0, y0, x1, y1, fill, stroke, a) => { ctx.save(); ctx.globalAlpha = a; ctx.fillStyle = fill; ctx.fillRect(ox + x0 * s, oy - y1 * s, (x1 - x0) * s, (y1 - y0) * s); ctx.globalAlpha = 1; ctx.strokeStyle = stroke; ctx.lineWidth = 1.4; ctx.strokeRect(ox + x0 * s, oy - y1 * s, (x1 - x0) * s, (y1 - y0) * s); ctx.restore(); };
  R(0, 0, u, v, c.sigwash, c.sig, 1);
  if (d > 0) { R(u, 0, u2, v, c.sig, c.sig, 0.35); R(0, v, u, v2, c.ok, c.ok, 0.35); R(u, v, u2, v2, c.miss, c.miss, 0.6); }
  txt(ctx, `u·v = ${fx(u)} × ${fx(v)} = ${fx(u * v)}`, ox + u * s / 2, oy - v * s / 2 + 4, { color: c.ink, size: 12.5, w: "600", align: "center" });
  txt(ctx, `u = x² = ${fx(u)}`, ox + u * s / 2, oy + 18, { color: c.sig, size: 11.5, w: "600", align: "center" });
  txt(ctx, `v = ${fx(v)}`, ox - 8, oy - v * s / 2 + 4, { color: c.ok, size: 11.5, w: "600", align: "right" });
  if (d > 0) {
    txtBg(ctx, c, `v·Δu = ${fx(v * (u2 - u))}`, ox + u2 * s + 8, oy - v * s / 2 + 4, { color: c.sig, size: 11.5, w: "700" });
    txtBg(ctx, c, `u·Δv = ${fx(u * (v2 - v))}`, ox + u * s / 2, oy - v2 * s - 8, { color: c.ok, size: 11.5, w: "700", align: "center" });
    txtBg(ctx, c, `Δu·Δv = ${fx((u2 - u) * (v2 - v))}`, ox + u2 * s + 8, oy - v2 * s - 4, { color: c.miss, size: 11.5, w: "700" });
  }
});
function upd4a() {
  paint(F4a);
  const x = S4.x, d = S4.d, u = x * x, v = x + 1, du = (x + d) ** 2 - u, dv = d, a = v * du, b = u * dv, cc = du * dv;
  $("r4a").textContent = tr(
    `u = x² = ${fx(u)}, v = x + 1 = ${fx(v)}, Δx = ${fx(d)} → Δu = ${fx(du)}, Δv = ${fx(dv)}\n늘어난 넓이 = v·Δu + u·Δv + Δu·Δv = ${fx(a)} + ${fx(b)} + ${fx(cc)} = ${fx(a + b + cc)}\n÷ Δx = ${d > 0 ? fx((a + b + cc) / d) : "–"}. Δx → 0이면 모서리는 사라지고 u′·v + u·v′ = ${fx(2 * x)} × ${fx(v)} + ${fx(u)} × 1 = ${fx(2 * x * v + u)}`,
    `u = x² = ${fx(u)}, v = x + 1 = ${fx(v)}, Δx = ${fx(d)} → Δu = ${fx(du)}, Δv = ${fx(dv)}\nnew area = v·Δu + u·Δv + Δu·Δv = ${fx(a)} + ${fx(b)} + ${fx(cc)} = ${fx(a + b + cc)}\n÷ Δx = ${d > 0 ? fx((a + b + cc) / d) : "–"}. As Δx → 0 the corner vanishes and u′·v + u·v′ = ${fx(2 * x)} × ${fx(v)} + ${fx(u)} × 1 = ${fx(2 * x * v + u)}`);
}
bindRange("i4x", "o4x", v => fx(v, 1), v => { S4.x = v; upd4a(); });
bindRange("i4d", "o4d", v => fx(v), v => { S4.d = v; upd4a(); });

// chain rule: 같은 눈금의 자 세 개. x의 작은 변화 dx가 u에서 3배, y에서 다시 2u배가 된다
// Chain rule: three rulers with the same scale. A small change dx becomes 3 times larger in u, then 2u times larger in y
const S4b = { x: 1, d: 0.1, timer: 0 };
const F4b = fig("cv4b", 290, (ctx, w, h, c) => {
  const x = S4b.x, dx = S4b.d, u = 3 * x + 1, du = 3 * dx, y = u * u, dy = (u + du) ** 2 - y;
  const lab = Math.min(118, w * 0.28), x0 = lab + 26, W = w - x0 - 16, ppu = Math.min(150, 0.72 * W / dy);
  const rows = [
    { nm: "x", v: x, d: dx, col: c.sig, dn: "dx" },
    { nm: "u = 3x + 1", v: u, d: du, col: c.ok, dn: "du" },
    { nm: "y = u²", v: y, d: dy, col: c.miss, dn: "dy" }
  ];
  const Y = [52, 148, 244], st = niceStepC(W / ppu / 5);
  rows.forEach((r, i) => {
    const y0 = Y[i], left = r.v - 18 / ppu, px = v => x0 + (v - left) * ppu;
    txt(ctx, r.nm, 8, y0 + 4, { color: r.col, size: 12.5, w: "700" });
    line(ctx, [[x0, y0], [x0 + W, y0]], c.muted, 1.2);
    for (let t = Math.ceil(left / st) * st; px(t) <= x0 + W; t += st) {
      line(ctx, [[px(t), y0 - 4], [px(t), y0 + 4]], c.muted, 1);
      txt(ctx, fmtTick(t, st), px(t), y0 + 18, { color: c.muted, size: 10, mono: true, align: "center" });
    }
    ctx.save(); ctx.globalAlpha = 0.3; ctx.fillStyle = r.col; ctx.fillRect(px(r.v), y0 - 9, r.d * ppu, 18); ctx.restore();
    ctx.strokeStyle = r.col; ctx.lineWidth = 2; ctx.strokeRect(px(r.v), y0 - 9, r.d * ppu, 18);
    txtBg(ctx, c, `${r.dn} = ${fx(r.d)}`, px(r.v) + Math.max(r.d * ppu, 0) + 8, y0 - 12, { color: r.col, size: 11.5, w: "700" });
    r.px = px;
  });
  for (let i = 0; i < 2; i++) {
    const a = rows[i], b = rows[i + 1];
    line(ctx, [[a.px(a.v), Y[i] + 9], [b.px(b.v), Y[i + 1] - 9]], c.muted, 1, [3, 3]);
    line(ctx, [[a.px(a.v + a.d), Y[i] + 9], [b.px(b.v + b.d), Y[i + 1] - 9]], c.muted, 1, [3, 3]);
    const m = i === 0 ? tr("× 3  (du/dx)", "× 3  (du/dx)") : tr(`× 2u = × ${fx(2 * u, 1)}  (dy/du)`, `× 2u = × ${fx(2 * u, 1)}  (dy/du)`);
    txtBg(ctx, c, m, 8, (Y[i] + Y[i + 1]) / 2 + 4, { color: i === 0 ? c.ok : c.miss, size: 12, w: "700" });
  }
});
function niceStepC(raw) { const p = 10 ** Math.floor(Math.log10(raw)), m = raw / p; return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p; }
function fmtTick(t, st) { const d = Math.max(0, -Math.floor(Math.log10(st) + 1e-9)); return fx(t, d); }
function upd4b() {
  paint(F4b);
  const x = S4b.x, dx = S4b.d, u = 3 * x + 1, du = 3 * dx, y = u * u, dy = (u + du) ** 2 - y;
  $("r4b").textContent = tr(
    `x: ${fx(x)} → ${fx(x + dx)}   dx = ${fx(dx)}\nu: ${fx(u)} → ${fx(u + du)}   du = 3 × dx = ${fx(du)}\ny: ${fx(y)} → ${fx(y + dy)}   dy = ${fx(dy)} ≈ 2u × du = ${fx(2 * u)} × ${fx(du)} = ${fx(2 * u * du)}\ndy/dx ≈ ${fx(dy)} ÷ ${fx(dx)} = ${fx(dy / dx, 1)}. dx → 0이면 정확히 3 × ${fx(2 * u)} = ${fx(6 * u)} (배율끼리 곱한다)`,
    `x: ${fx(x)} → ${fx(x + dx)}   dx = ${fx(dx)}\nu: ${fx(u)} → ${fx(u + du)}   du = 3 × dx = ${fx(du)}\ny: ${fx(y)} → ${fx(y + dy)}   dy = ${fx(dy)} ≈ 2u × du = ${fx(2 * u)} × ${fx(du)} = ${fx(2 * u * du)}\ndy/dx ≈ ${fx(dy)} ÷ ${fx(dx)} = ${fx(dy / dx, 1)}. As dx → 0 it is exactly 3 × ${fx(2 * u)} = ${fx(6 * u)} (the stretch factors multiply)`);
}
bindRange("i4bx", "o4bx", v => fx(v), v => { stopRun(S4b); S4b.x = v; upd4b(); });
bindRange("i4bd", "o4bd", v => fx(v), v => { S4b.d = v; upd4b(); });
$("b4bplay").addEventListener("click", () => {
  S4b.x = 0; runSteps(S4b, 30, () => { S4b.x = Math.min(1.5, S4b.x + 0.05); setRange("i4bx", S4b.x); upd4b(); }, 90);
});

/* ================= 5절 / Section 5 ================= */
// 1D gradient descent: 공이 f(x) = (x − 2)² + 1 위를 한 걸음씩 내려간다. 오른쪽 = 걸음마다 loss
// 1D gradient descent: a ball steps down f(x) = (x − 2)² + 1. Right = the loss at each step
const f5 = x => (x - 2) ** 2 + 1, d5 = x => 2 * (x - 2);
const S5 = { x: 5, eta: 0.1, hist: [], timer: 0, lay: null };
const F5 = fig("cv5", w => (w < 620 ? 560 : 320), (ctx, w, h, c) => {
  const wide = w >= 620, P = wide ? [{ x: 0, y: 0, w: w * 0.64, h }, { x: w * 0.64 + 22, y: 0, w: w * 0.36 - 22, h }] : [{ x: 0, y: 0, w, h: 320 }, { x: 0, y: 330, w, h: 230 }];
  const p = P[0], B = { l: p.x + 40, t: p.y + 30, w: p.w - 52, h: p.h - 78 }, xr = [-1.5, 6.5];
  const A = axes(ctx, c, B, xr, [0, 22], { xt: [-1, 0, 1, 2, 3, 4, 5, 6], yt: [0, 5, 10, 15, 20], xlabel: "x" });
  S5.lay = { B, xr };
  panelTitle(ctx, c, tr("loss f(x) = (x − 2)² + 1 위의 공", "A ball on the loss f(x) = (x − 2)² + 1"), B.l, p.y + 14);
  clipBox(ctx, B);
  line(ctx, curve(A, f5, xr), c.sig, 2.4);
  const pts = [...S5.hist, S5.x];
  for (let i = 1; i < pts.length; i++) line(ctx, [[A.X(pts[i - 1]), A.Y(f5(pts[i - 1]))], [A.X(pts[i]), A.Y(f5(pts[i]))]], c.miss, 1.4, [4, 3]);
  const x = S5.x, m = d5(x), dx = 0.7;
  line(ctx, [[A.X(x - dx), A.Y(f5(x) - m * dx)], [A.X(x + dx), A.Y(f5(x) + m * dx)]], c.ok, 2.4);
  ctx.restore();
  S5.hist.forEach((v, i) => dot(ctx, A.X(v), A.Y(f5(v)), 4, c.bg, c.miss));
  dot(ctx, A.X(x), A.Y(f5(x)), 8, c.miss);
  txtBg(ctx, c, tr(`기울기 f′ = ${fx(m)}`, `slope f′ = ${fx(m)}`), A.X(x) + (x > 2 ? -14 : 14), A.Y(f5(x)) - 10, { color: c.ok, size: 12, w: "700", align: x > 2 ? "right" : "left" });
  const xn = x - S5.eta * m, ya = B.t + B.h - 14;
  if (Math.abs(A.X(xn) - A.X(x)) > 6) {
    arrow(ctx, A.X(x), ya, A.X(clamp(xn, xr[0], xr[1])), ya, c.ok, 2.4);
    txtBg(ctx, c, tr(`한 걸음 = η × f′ = ${fx(S5.eta * m)}`, `one step = η × f′ = ${fx(S5.eta * m)}`), (A.X(x) + A.X(clamp(xn, xr[0], xr[1]))) / 2, ya - 8, { color: c.ok, size: 11.5, w: "600", align: "center" });
  }
  txt(ctx, tr("공을 끌어서 시작점을 바꾼다", "drag the ball to change the start"), B.l + 6, B.t + 12, { color: c.muted, size: 11 });
  // 걸음마다 loss
  // Loss at each step
  const q = P[1], C = { l: q.x + 36, t: q.y + 30, w: q.w - 46, h: q.h - 78 }, n = Math.max(10, pts.length);
  const L = pts.map(f5), top = Math.max(10, ...L.filter(isFinite)) * 1.05;
  const A2 = axes(ctx, c, C, [-0.5, n - 0.5], [0, top], { xt: [0, 5, 10, 15, 20, 25, 30].filter(v => v < n), yt: [0, 5, 10, 15, 20].filter(v => v <= top), xlabel: tr("걸음", "step") });
  panelTitle(ctx, c, tr("걸음마다 loss", "Loss at each step"), C.l, q.y + 14);
  const bw = Math.max(2, C.w / n * 0.7);
  L.forEach((v, i) => { const y = A2.Y(Math.min(v, top)); ctx.fillStyle = i === L.length - 1 ? c.miss : c.sig; ctx.globalAlpha = i === L.length - 1 ? 1 : 0.55; ctx.fillRect(A2.X(i) - bw / 2, y, bw, C.t + C.h - y); ctx.globalAlpha = 1; });
  line(ctx, [[C.l, A2.Y(1)], [C.l + C.w, A2.Y(1)]], c.ok, 1.2, [4, 3]);
  txtBg(ctx, c, tr("가장 낮은 loss = 1", "lowest loss = 1"), C.l + C.w - 2, A2.Y(1) - 5, { color: c.ok, size: 10.5, align: "right" });
});
function upd5() {
  paint(F5);
  const x = S5.x, m = d5(x), e = S5.eta, xn = x - e * m;
  const side = Math.abs(m) < 1e-6 ? tr("기울기 0 → 멈춘다", "slope 0 → it stops") : m > 0 ? tr("기울기가 양수 → 왼쪽으로", "positive slope → move left") : tr("기울기가 음수 → 오른쪽으로", "negative slope → move right");
  $("r5").textContent = tr(
    `x = ${fx(x)}, 기울기 f′(x) = 2 × (${fx(x)} − 2) = ${fx(m)}\n한 걸음 = η × f′ = ${fx(e)} × ${paren(m)} = ${fx(e * m)}   (${side})\n새 x = ${fx(x)} − ${paren(e * m)} = ${fx(xn)}, loss ${fx(f5(x))} → ${fx(f5(xn))}   (지금까지 ${S5.hist.length} 걸음)`,
    `x = ${fx(x)}, slope f′(x) = 2 × (${fx(x)} − 2) = ${fx(m)}\none step = η × f′ = ${fx(e)} × ${paren(m)} = ${fx(e * m)}   (${side})\nnew x = ${fx(x)} − ${paren(e * m)} = ${fx(xn)}, loss ${fx(f5(x))} → ${fx(f5(xn))}   (${S5.hist.length} steps so far)`);
}
function step5() { if (S5.hist.length >= 30) return false; S5.hist.push(S5.x); S5.x = S5.x - S5.eta * d5(S5.x); upd5(); }
bindRange("i5e", "o5e", v => fx(v), v => { S5.eta = v; upd5(); });
$("b5step").addEventListener("click", () => { stopRun(S5); step5(); });
$("b5run").addEventListener("click", () => runSteps(S5, 10, step5, 260));
$("b5reset").addEventListener("click", () => { stopRun(S5); S5.x = 5; S5.hist = []; upd5(); });
dragOn(F5, (px, py) => { const L = S5.lay; return L && px >= L.B.l && px <= L.B.l + L.B.w && py >= L.B.t && py <= L.B.t + L.B.h ? 1 : null; }, (k, px) => {
  const L = S5.lay; stopRun(S5);
  S5.x = Math.round(clamp(L.xr[0] + (px - L.B.l) / L.B.w * (L.xr[1] - L.xr[0]), -1.2, 6.2) * 10) / 10; S5.hist = []; upd5();
});

/* ================= 6절 / Section 6 ================= */
// learning rate 네 개를 나란히: 너무 작다, 알맞다, 지그재그, 발산
// Four learning rates side by side: too small, about right, zigzag, divergence
const ETAS6 = [[0.02, "slow"], [0.2, "good"], [0.9, "zig"], [1.05, "div"]];
const F6a = fig("cv6a", w => (w >= 720 ? 220 : 440), (ctx, w, h, c) => {
  const cols = w >= 720 ? 4 : 2, gap = 14, cw = (w - (cols - 1) * gap) / cols, chh = 220, xr = [-3, 7];
  ETAS6.forEach(([eta, kind], i) => {
    const px = (i % cols) * (cw + gap), py = Math.floor(i / cols) * chh;
    const B = { l: px + 28, t: py + 46, w: cw - 36, h: chh - 84 };
    const A = axes(ctx, c, B, xr, [0, 27], { xt: [-2, 2, 6], yt: [0, 25], grid: false });
    const col = kind === "good" ? c.ok : kind === "div" ? c.fa : c.miss;
    const name = { slow: tr("너무 작다: 느리다", "too small: slow"), good: tr("알맞다", "about right"), zig: tr("크다: 지그재그", "large: zigzag"), div: tr("너무 크다: 발산", "too large: diverges") }[kind];
    txt(ctx, `η = ${eta}`, px + 4, py + 14, { color: c.ink, size: 12.5, w: "700", mono: true });
    txt(ctx, name, px + 4, py + 31, { color: col, size: 12, w: "700" });
    clipBox(ctx, B);
    line(ctx, curve(A, f5, xr, 120), c.sig, 1.8);
    let x = 5; const pts = [[A.X(x), A.Y(f5(x))]];
    for (let k = 0; k < 10; k++) { x -= eta * d5(x); pts.push([A.X(x), A.Y(f5(x))]); }
    line(ctx, pts, col, 1.4);
    pts.forEach((p, k) => { ctx.globalAlpha = 0.35 + 0.65 * k / 10; dot(ctx, p[0], p[1], k ? 3 : 4.5, col); ctx.globalAlpha = 1; });
    ctx.restore();
    txt(ctx, tr(`10걸음 뒤 x = ${fx(x)}`, `after 10 steps x = ${fx(x)}`), B.l, B.t + B.h + 30, { color: c.muted, size: 11, mono: false });
    if (Math.abs(x) > 7) txt(ctx, tr("→ 그림 밖으로", "→ off the chart"), px + cw - 4, py + 31, { color: c.fa, size: 11, w: "600", align: "right" });
  });
});
// 걸음마다 x: 고정 learning rate 또는 점점 줄어드는 learning rate
// x at each step: a fixed learning rate or one that shrinks over time
const S6 = { eta: 0.9, mode: "fixed" };
const eta6 = t => (S6.mode === "fixed" ? S6.eta : S6.eta / (1 + t / 5));
function path6() { const xs = [5]; for (let t = 0; t < 30; t++) { const x = xs[t]; xs.push(Math.abs(x) > 1e6 ? x : x - eta6(t) * d5(x)); } return xs; }
const F6b = fig("cv6b", 290, (ctx, w, h, c) => {
  const B = { l: 44, t: 28, w: w - 60, h: h - 72 };
  const A = axes(ctx, c, B, [0, 30], [-4, 8], { xt: [0, 5, 10, 15, 20, 25, 30], yt: [-4, -2, 0, 2, 4, 6, 8], xlabel: tr("걸음 t", "step t") });
  panelTitle(ctx, c, tr("걸음마다 x (시작 x = 5)", "x at each step (start x = 5)"), B.l, 16);
  clipBox(ctx, B);
  line(ctx, [[B.l, A.Y(2)], [B.l + B.w, A.Y(2)]], c.ok, 1.6, [6, 4]);
  const xs = path6();
  line(ctx, xs.map((v, t) => [A.X(t), A.Y(clamp(v, -50, 50))]), c.miss, 2);
  xs.forEach((v, t) => { if (v > -4.5 && v < 8.5) dot(ctx, A.X(t), A.Y(v), 3, c.miss); });
  ctx.restore();
  txtBg(ctx, c, tr("바닥 x = 2", "bottom x = 2"), B.l + B.w - 4, A.Y(2) - 6, { color: c.ok, size: 11.5, w: "600", align: "right" });
});
function upd6() {
  paint(F6b);
  const e = S6.eta, k = 1 - 2 * e, xs = path6(), last = xs[30];
  const state = Math.abs(k) >= 1 ? tr("|배율| ≥ 1 → 거리가 커진다: 발산", "|factor| ≥ 1 → the distance grows: divergence")
    : k < 0 ? tr("음수 → 바닥을 넘나들며(지그재그) 줄어든다", "negative → it jumps across the bottom (zigzag) while shrinking")
    : k > 0.9 ? tr("1에 가깝다 → 아주 천천히 줄어든다", "close to 1 → it shrinks very slowly") : tr("0과 1 사이 → 곧게 줄어든다", "between 0 and 1 → it shrinks steadily");
  const end = Math.abs(last) > 1e5 ? tr("아주 멀리 (발산)", "very far away (diverged)") : fx(last, 3);
  $("r6").textContent = S6.mode === "fixed" ? tr(
    `η = ${fx(e)} → 걸음마다 바닥까지 거리 × (1 − 2η) = × ${paren(k)}\n${state}\n30걸음 뒤 x = ${end}`,
    `η = ${fx(e)} → each step multiplies the distance to the bottom by (1 − 2η) = ${paren(k)}\n${state}\nafter 30 steps x = ${end}`) : tr(
    `ηₜ = ${fx(e)} ÷ (1 + t/5): η₀ = ${fx(eta6(0))}, η₁₀ = ${fx(eta6(10))}, η₂₉ = ${fx(eta6(29))}\n처음에는 크게, 나중에는 작게 걷는다\n30걸음 뒤 x = ${end}`,
    `ηₜ = ${fx(e)} ÷ (1 + t/5): η₀ = ${fx(eta6(0))}, η₁₀ = ${fx(eta6(10))}, η₂₉ = ${fx(eta6(29))}\nbig steps first, small steps later\nafter 30 steps x = ${end}`);
}
bindRange("i6e", "o6e", v => fx(v), v => { S6.eta = v; upd6(); });
bindSeg("g6m", v => { S6.mode = v; upd6(); });

/* ================= 7절 / Section 7 ================= */
// 골짜기가 두 개인 loss. 아래 띠 = 시작점 지도 (그곳에서 출발하면 어느 골짜기에 도착하나)
// A loss with two valleys. Strip below = map of starting points (which valley you end up in from there)
const f7 = x => x ** 4 / 4 - x * x + x / 4, d7 = x => x ** 3 - 2 * x + 0.25;
const G7 = -1.4730, L7 = 1.3470, XR7 = [-2.4, 2.4];
const S7 = { eta: 0.05, x0: 0.5, balls: [0.5], timer: 0, map: null, lay: null };
const rng7 = makeRng(newSeed());
function end7(x, eta) { for (let i = 0; i < 400; i++) { x -= eta * d7(x); if (!isFinite(x) || Math.abs(x) > 50) return NaN; } return x; }
const cls7 = e => (Math.abs(e - G7) < 0.05 ? "g" : Math.abs(e - L7) < 0.05 ? "l" : "x");
function map7() { const n = 240, m = []; for (let i = 0; i < n; i++) { const x = XR7[0] + (XR7[1] - XR7[0]) * (i + 0.5) / n; m.push(cls7(end7(x, S7.eta))); } S7.map = m; }
const F7a = fig("cv7a", 340, (ctx, w, h, c) => {
  const B = { l: 44, t: 26, w: w - 60, h: h - 108 };
  const A = axes(ctx, c, B, XR7, [-2.3, 3.3], { xt: [-2, -1, 0, 1, 2], yt: [-2, -1, 0, 1, 2, 3] });
  S7.lay = { B };
  panelTitle(ctx, c, "f(x) = x⁴/4 − x² + x/4", B.l, 16);
  clipBox(ctx, B);
  line(ctx, curve(A, f7, XR7), c.sig, 2.4);
  ctx.restore();
  dot(ctx, A.X(G7), A.Y(f7(G7)), 6, c.ok); dot(ctx, A.X(L7), A.Y(f7(L7)), 6, c.miss);
  txtBg(ctx, c, tr("가장 낮은 바닥 (global minimum)", "lowest valley (global minimum)"), A.X(G7), A.Y(f7(G7)) + 20, { color: c.ok, size: 11.5, w: "700", align: "center" });
  txtBg(ctx, c, tr("옆 골짜기 (local minimum)", "side valley (local minimum)"), A.X(L7), A.Y(f7(L7)) + 20, { color: c.miss, size: 11.5, w: "700", align: "center" });
  S7.balls.forEach(x => { if (isFinite(x) && x > XR7[0] && x < XR7[1]) dot(ctx, A.X(x), A.Y(f7(x)), 5.5, c.ink, c.bg); });
  if (!S7.map) map7();
  const sy = B.t + B.h + 40, sh = 16, cw = B.w / S7.map.length;
  txt(ctx, tr("시작점 지도: 이 x에서 출발하면 어디에 도착하나 (누르면 그곳에서 출발)", "Start map: where you end up when you start at this x (click to start there)"), B.l, sy - 8, { color: c.muted, size: 11 });
  S7.map.forEach((k, i) => { ctx.fillStyle = k === "g" ? c.ok : k === "l" ? c.miss : c.line; ctx.globalAlpha = 0.8; ctx.fillRect(B.l + i * cw, sy, cw + 0.6, sh); ctx.globalAlpha = 1; });
  arrowS(ctx, A.X(S7.x0), sy + sh + 14, A.X(S7.x0), sy + sh + 2, c.ink, 1.6);
});
function upd7() {
  paint(F7a);
  const e = end7(S7.x0, S7.eta), k = cls7(e), frac = S7.map ? S7.map.filter(v => v === "l").length / S7.map.length : 0;
  const res = k === "g" ? tr("global minimum (가장 낮은 바닥)에 도착", "reached the global minimum (the lowest valley)") : k === "l" ? tr(`local minimum에 갇혔다 (가장 낮은 바닥은 x = ${fx(G7)}, loss ${fx(f7(G7))})`, `trapped in the local minimum (the lowest valley is at x = ${fx(G7)}, loss ${fx(f7(G7))})`) : tr("바닥에 닿지 못했다 (튀어 나감)", "did not settle in a valley (bounced away)");
  $("r7").textContent = tr(
    `시작 x = ${fx(S7.x0)} → 도착 x = ${isFinite(e) ? fx(e) : "∞"}${isFinite(e) ? `, loss ${fx(f7(e))}` : ""}\n→ ${res}\nη = ${fx(S7.eta)}일 때 시작점의 ${Math.round(frac * 100)}%가 local minimum으로 간다`,
    `start x = ${fx(S7.x0)} → end x = ${isFinite(e) ? fx(e) : "∞"}${isFinite(e) ? `, loss ${fx(f7(e))}` : ""}\n→ ${res}\nwith η = ${fx(S7.eta)}, ${Math.round(frac * 100)}% of starting points end in the local minimum`);
}
function roll7(starts) {
  S7.balls = starts.slice();
  runSteps(S7, 120, () => { S7.balls = S7.balls.map(x => (isFinite(x) && Math.abs(x) < 50 ? x - S7.eta * d7(x) : NaN)); paint(F7a); }, 40, upd7);
}
bindRange("i7e", "o7e", v => fx(v), v => { stopRun(S7); S7.eta = v; map7(); S7.balls = [S7.x0]; upd7(); });
$("b7roll").addEventListener("click", () => roll7([S7.x0]));
$("b7rand").addEventListener("click", () => roll7(Array.from({ length: 20 }, () => XR7[0] + 0.1 + rng7.u() * (XR7[1] - XR7[0] - 0.2))));
dragOn(F7a, (px, py) => { const L = S7.lay; return L && px >= L.B.l && px <= L.B.l + L.B.w ? 1 : null; }, (k, px) => {
  const L = S7.lay; S7.x0 = Math.round(clamp(XR7[0] + (px - L.B.l) / L.B.w * (XR7[1] - XR7[0]), -2.3, 2.3) * 100) / 100;
  stopRun(S7); S7.balls = [S7.x0]; upd7();
});
F7a.cv.addEventListener("pointerup", () => { if (!S7.timer) roll7([S7.x0]); });

// saddle point: f(x, y) = x² + y⁴/4 − y²/2. (0, 0)에서 기울기 0이지만 바닥이 아니다
// Saddle point: f(x, y) = x² + y⁴/4 − y²/2. The slope is 0 at (0, 0), but it is not a bottom
const f7b = (x, y) => x * x + y ** 4 / 4 - y * y / 2, g7b = (x, y) => [2 * x, y ** 3 - y];
const S7b = { p0: [1.5, 0], path: [[1.5, 0]], timer: 0, lay: null };
const F7b = fig("cv7b", w => Math.round(clamp(w * 0.55, 230, 360)) + 36, (ctx, w, h, c) => {
  const B = { l: 40, t: 12, w: w - 54, h: h - 44 }, R = equalRange(B, 1.6);
  S7b.lay = { B, R };
  contour(ctx, c, B, f7b, R.xr, R.yr, "saddle");
  const A = axes(ctx, c, B, R.xr, R.yr, { xt: intTicks(R.xr), yt: [-1, 0, 1], grid: false });
  const P = S7b.path;
  clipBox(ctx, B);
  line(ctx, P.map(p => [A.X(p[0]), A.Y(p[1])]), c.miss, 2.2);
  ctx.restore();
  [[0, 1], [0, -1]].forEach(p => dot(ctx, A.X(p[0]), A.Y(p[1]), 5.5, c.ok));
  txtBg(ctx, c, tr("바닥", "bottom"), A.X(0) + 10, A.Y(1) + 4, { color: c.ok, size: 11.5, w: "700" });
  txtBg(ctx, c, tr("바닥", "bottom"), A.X(0) + 10, A.Y(-1) + 4, { color: c.ok, size: 11.5, w: "700" });
  const s = 6, X0 = A.X(0), Y0 = A.Y(0);
  line(ctx, [[X0 - s, Y0 - s], [X0 + s, Y0 + s]], c.fa, 2.4); line(ctx, [[X0 - s, Y0 + s], [X0 + s, Y0 - s]], c.fa, 2.4);
  txtBg(ctx, c, "saddle point", X0 - 10, Y0 - 10, { color: c.fa, size: 11.5, w: "700", align: "right" });
  dot(ctx, A.X(P[0][0]), A.Y(P[0][1]), 5, c.ink);
  const e = P[P.length - 1]; dot(ctx, A.X(e[0]), A.Y(e[1]), 5.5, c.miss, c.bg);
});
function upd7b() {
  paint(F7b);
  const P = S7b.path, e = P[P.length - 1], n = P.length - 1, g = g7b(...e), gm = Math.hypot(...g);
  const near = P.findIndex(p => Math.hypot(p[0], p[1]) < 0.15), out = P.findIndex((p, i) => i > near && near >= 0 && Math.abs(p[1]) > 0.5);
  const stuck = gm < 1e-3 && Math.abs(e[1]) < 0.05;
  const msg = stuck ? tr("saddle point에 멈췄다: 기울기는 (0, 0)인데 바닥이 아니다 (y 쪽으로는 더 내려갈 수 있다)", "stopped at the saddle point: the slope is (0, 0), but it is not a bottom (going along y would still go down)")
    : Math.abs(Math.abs(e[1]) - 1) < 0.02 && Math.abs(e[0]) < 0.02 ? (near >= 0 && out > near ? tr(`바닥에 도착. saddle point 근처에서 ${out - near} 걸음 동안 거의 멈춰 있었다`, `reached a bottom, after crawling near the saddle point for ${out - near} steps`) : tr("바닥에 도착", "reached a bottom")) : tr("아직 가는 중", "still on the way");
  $("r7b").textContent = tr(
    `시작 (${fx(P[0][0])}, ${fx(P[0][1], 3)}) → ${n} 걸음 뒤 (${fx(e[0])}, ${fx(e[1], 3)}), loss ${fx(f7b(...e), 3)}\n→ ${msg}`,
    `start (${fx(P[0][0])}, ${fx(P[0][1], 3)}) → after ${n} steps (${fx(e[0])}, ${fx(e[1], 3)}), loss ${fx(f7b(...e), 3)}\n→ ${msg}`);
}
function roll7b(p0) {
  S7b.p0 = p0; S7b.path = [p0.slice()];
  runSteps(S7b, 100, () => {
    for (let r = 0; r < 3; r++) { const [x, y] = S7b.path[S7b.path.length - 1], g = g7b(x, y); S7b.path.push([x - 0.1 * g[0], y - 0.1 * g[1]]); }
    upd7b();
  }, 30);
}
$("b7s0").addEventListener("click", () => roll7b([1.5, 0]));
$("b7s1").addEventListener("click", () => roll7b([1.5, 0.001]));
dragOn(F7b, (px, py) => 1, (k, px, py) => {
  const L = S7b.lay; if (!L) return;
  const x = L.R.xr[0] + (px - L.B.l) / L.B.w * (L.R.xr[1] - L.R.xr[0]), y = L.R.yr[1] - (py - L.B.t) / L.B.h * (L.R.yr[1] - L.R.yr[0]);
  stopRun(S7b); S7b.p0 = [clamp(x, L.R.xr[0], L.R.xr[1]), clamp(y, -1.6, 1.6)]; S7b.path = [S7b.p0.slice()]; upd7b();
});
F7b.cv.addEventListener("pointerup", () => { if (!S7b.timer) roll7b(S7b.p0); });

/* ================= 8절 / Section 8 ================= */
// 2D gradient descent: 등고선 위에서 gradient 화살표와 걷는 경로. 작은 회색 화살표 = 그 자리의 내리막 방향
// 2D gradient descent: the gradient arrow and the path on a contour map. Small grey arrows = the downhill direction at each spot
const S8 = { k: 3, eta: 0.1, p0: [2, 1], dir: -1, path: [], timer: 0, lay: null };
const f8 = (x, y) => x * x + S8.k * y * y, g8 = (x, y) => [2 * x, 2 * S8.k * y];
const F8 = fig("cv8", w => Math.round(clamp(w * 0.58, 250, 400)) + 30, (ctx, w, h, c) => {
  const B = { l: 36, t: 12, w: w - 50, h: h - 42 }, R = equalRange(B, 2.2);
  S8.lay = { B, R };
  contour(ctx, c, B, f8, R.xr, R.yr, "bowl" + S8.k);
  const A = axes(ctx, c, B, R.xr, R.yr, { xt: intTicks(R.xr), yt: [-2, -1, 0, 1, 2], grid: false });
  // 내리막(또는 오르막) 방향 화살표 밭
  // Field of downhill (or uphill) arrows
  const sp = 44;
  for (let px = B.l + sp / 2; px < B.l + B.w; px += sp) for (let py = B.t + sp / 2; py < B.t + B.h; py += sp) {
    const x = R.xr[0] + (px - B.l) / B.w * (R.xr[1] - R.xr[0]), y = R.yr[1] - (py - B.t) / B.h * (R.yr[1] - R.yr[0]), g = g8(x, y), m = Math.hypot(...g);
    if (m < 1e-6) continue;
    const ux = S8.dir * g[0] / m, uy = S8.dir * g[1] / m;
    ctx.globalAlpha = 0.55; arrowS(ctx, px - ux * 7, py + uy * 7, px + ux * 7, py - uy * 7, c.muted, 1.2, 4); ctx.globalAlpha = 1;
  }
  const P = S8.path;
  clipBox(ctx, B);
  line(ctx, P.map(p => [A.X(p[0]), A.Y(p[1])]), c.miss, 2.2);
  P.forEach(p => dot(ctx, A.X(p[0]), A.Y(p[1]), 2.6, c.miss));
  ctx.restore();
  // 시작점의 gradient와 두 성분
  // Gradient at the start and its two components
  const [x0, y0] = S8.p0, g = g8(x0, y0), gm = Math.hypot(...g), sc = gm > 0 ? Math.min(0.2, 1.3 / gm) : 0;
  const X0 = A.X(x0), Y0 = A.Y(y0), gx = A.X(x0 + g[0] * sc) - X0, gy = A.Y(y0 + g[1] * sc) - Y0;
  if (gm > 1e-6) {
    arrowS(ctx, X0, Y0, X0 + gx, Y0, c.sig, 2, 7);
    arrowS(ctx, X0 + gx, Y0, X0 + gx, Y0 + gy, c.vio, 2, 7);
    arrowS(ctx, X0, Y0, X0 + gx, Y0 + gy, c.fa, 2.6, 9);
    txtBg(ctx, c, `∂f/∂x = ${fx(g[0], 1)}`, X0 + gx / 2, Y0 + (gy < 0 ? 16 : -8), { color: c.sig, size: 11.5, w: "700", align: "center" });
    const right = gx >= 0 ? X0 + gx + 96 < B.l + B.w : X0 + gx - 96 < B.l;
    txtBg(ctx, c, `∂f/∂y = ${fx(g[1], 1)}`, X0 + gx + (right ? 8 : -8), Y0 + gy / 2 + 4, { color: c.vio, size: 11.5, w: "700", align: right ? "left" : "right" });
    txtBg(ctx, c, "∇f", X0 + gx * 0.55 - (gx >= 0 ? 12 : -12), Y0 + gy * 0.55 - 6, { color: c.fa, size: 13, w: "700", align: gx >= 0 ? "right" : "left" });
  }
  dot(ctx, X0, Y0, 5.5, c.ink);
  txtBg(ctx, c, tr("누르면 그 자리에서 출발", "click to start there"), B.l + 6, B.t + B.h - 6, { color: c.ink, size: 11 });
});
function upd8() {
  paint(F8);
  const [x, y] = S8.p0, g = g8(x, y), e = S8.eta, s = S8.dir, n = [x + s * e * g[0], y + s * e * g[1]], P = S8.path, last = P[P.length - 1];
  const far = Math.hypot(...last) > 50;
  $("r8").textContent = tr(
    `시작 (${fx(x)}, ${fx(y)}), loss f = ${fx(x)}² + ${fx(S8.k, 1)} × ${paren(y)}² = ${fx(f8(x, y))}\n∇f = (2x, 2·${fx(S8.k, 1)}·y) = (${fx(g[0])}, ${fx(g[1])}) → 가장 가파른 오르막 방향 (빨강)\n한 걸음 = ${s < 0 ? "−" : "+"}η∇f = (${fx(s * e * g[0])}, ${fx(s * e * g[1])}) → (${fx(n[0])}, ${fx(n[1])}), loss ${fx(f8(...n))}\n${P.length - 1} 걸음 뒤 ${far ? "아주 멀리 (발산)" : `(${fx(last[0])}, ${fx(last[1])}), loss ${fx(f8(...last))}`}`,
    `start (${fx(x)}, ${fx(y)}), loss f = ${fx(x)}² + ${fx(S8.k, 1)} × ${paren(y)}² = ${fx(f8(x, y))}\n∇f = (2x, 2·${fx(S8.k, 1)}·y) = (${fx(g[0])}, ${fx(g[1])}) → the steepest uphill direction (red)\none step = ${s < 0 ? "−" : "+"}η∇f = (${fx(s * e * g[0])}, ${fx(s * e * g[1])}) → (${fx(n[0])}, ${fx(n[1])}), loss ${fx(f8(...n))}\nafter ${P.length - 1} steps ${far ? "very far away (diverged)" : `(${fx(last[0])}, ${fx(last[1])}), loss ${fx(f8(...last))}`}`);
}
function roll8() {
  S8.path = [S8.p0.slice()];
  runSteps(S8, 40, () => {
    const [x, y] = S8.path[S8.path.length - 1]; if (Math.hypot(x, y) > 50) return false;
    const g = g8(x, y); S8.path.push([x + S8.dir * S8.eta * g[0], y + S8.dir * S8.eta * g[1]]); upd8();
  }, 70, upd8);
}
bindRange("i8k", "o8k", v => fx(v, 1), v => { S8.k = v; roll8(); });
bindRange("i8e", "o8e", v => fx(v), v => { S8.eta = v; roll8(); });
bindSeg("g8d", v => { S8.dir = +v; roll8(); });
$("b8run").addEventListener("click", roll8);
dragOn(F8, () => 1, (k, px, py) => {
  const L = S8.lay; if (!L) return;
  S8.p0 = [L.R.xr[0] + (px - L.B.l) / L.B.w * (L.R.xr[1] - L.R.xr[0]), L.R.yr[1] - (py - L.B.t) / L.B.h * (L.R.yr[1] - L.R.yr[0])].map(v => Math.round(v * 10) / 10);
  stopRun(S8); S8.path = [S8.p0.slice()]; upd8();
});
F8.cv.addEventListener("pointerup", () => { if (!S8.timer) roll8(); });

/* ================= 9절 / Section 9 ================= */
// 층 10개를 거꾸로 지나며 gradient에 층마다 a가 곱해진다. 막대 = 각 층이 받는 gradient (log 눈금)
// Going back through 10 layers multiplies the gradient by a at each layer. Bars = the gradient each layer receives (log scale)
const S9 = { a: 0.25, shown: 11, timer: 0 };
const F9 = fig("cv9", 320, (ctx, w, h, c) => {
  const B = { l: 52, t: 40, w: w - 64, h: h - 104 }, n = 11, cw = B.w / n;
  const A = axes(ctx, c, B, [0, n], [-7, 2], { xt: [], yt: [[-6, "10⁻⁶"], [-4, "10⁻⁴"], [-2, "0.01"], [0, "1"], [2, "100"]] });
  panelTitle(ctx, c, tr("각 층이 받는 gradient의 크기 (세로는 log 눈금: 한 칸 = 100배)", "Size of the gradient each layer receives (log scale: one grid line = ×100)"), B.l, 16);
  arrowS(ctx, B.l + B.w - 6, B.t - 10, B.l + 6, B.t - 10, c.muted, 1.4, 6);
  txtBg(ctx, c, tr(`역전파: 층을 지날 때마다 × ${fx(S9.a)}`, `backprop: × ${fx(S9.a)} at every layer`), B.l + B.w / 2, B.t - 6, { color: c.muted, size: 11.5, w: "600", align: "center" });
  for (let k = 0; k < n; k++) {
    const v = S9.a ** k, lg = Math.log10(v), xc = B.l + (n - 1 - k + 0.5) * cw, bw = Math.min(34, cw * 0.62);
    const lab = k === 0 ? tr("출력", "out") : String(n - k);
    txt(ctx, lab, xc, B.t + B.h + 16, { color: c.muted, size: 11, align: "center", mono: k > 0 });
    if (k >= S9.shown) continue;
    const yTop = A.Y(clamp(lg, -7, 2)), bad = v < 1e-2 || v > 10, col = bad ? c.fa : c.ok;
    ctx.fillStyle = col; ctx.globalAlpha = 0.75; ctx.fillRect(xc - bw / 2, Math.min(yTop, A.Y(-7)), bw, Math.abs(A.Y(-7) - yTop)); ctx.globalAlpha = 1;
    if (cw > 44 || k === 0 || k === n - 1) txt(ctx, sci(v), xc, Math.max(B.t + 10, yTop - 5), { color: col, size: 10, mono: true, align: "center" });
  }
  txt(ctx, tr("층 번호 (1층 = 입력 쪽)", "layer (1 = the input side)"), B.l + B.w / 2, B.t + B.h + 34, { color: c.muted, size: 11.5, align: "center" });
});
function upd9() {
  paint(F9);
  const a = S9.a, v = a ** 10;
  const st = v < 1e-2 ? tr("거의 0 → 1층은 거의 배우지 못한다 (vanishing gradient)", "almost 0 → layer 1 barely learns (vanishing gradient)")
    : v > 10 ? tr("너무 크다 → 1층의 걸음이 너무 커서 튄다 (exploding gradient)", "far too large → layer 1 takes huge jumps (exploding gradient)") : tr("적당하다 → 모든 층이 비슷하게 배운다", "reasonable → every layer learns at a similar pace");
  $("r9").textContent = tr(
    `층마다 × ${fx(a)}${Math.abs(a - 0.25) < 1e-9 ? " (sigmoid의 기울기는 최대 0.25)" : ""}\n1층이 받는 gradient = ${fx(a)}¹⁰ = ${sci(v)}\n→ ${st}`,
    `× ${fx(a)} at every layer${Math.abs(a - 0.25) < 1e-9 ? " (the slope of a sigmoid is at most 0.25)" : ""}\ngradient reaching layer 1 = ${fx(a)}¹⁰ = ${sci(v)}\n→ ${st}`);
}
bindRange("i9a", "o9a", v => fx(v), v => { stopRun(S9); S9.a = v; S9.shown = 11; upd9(); });
bindSeg("g9p", v => { stopRun(S9); S9.a = +v; S9.shown = 11; setRange("i9a", S9.a); upd9(); });
$("b9play").addEventListener("click", () => { S9.shown = 0; upd9(); runSteps(S9, 11, () => { S9.shown++; paint(F9); }, 220); });

// 처음 그리기
// First paint
upd1(); upd2(); upd3(); upd4a(); upd4b(); upd5(); paint(F6a); upd6(); map7(); upd7(); roll7b([1.5, 0]); roll8(); upd9();
repaintAll();
