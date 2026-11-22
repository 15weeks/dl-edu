/* 1장 그림과 조작. 한글판(ch01-intro-KR.html)과 영어판(ch01-intro.html)이 같이 쓴다. 글자는 tr(한글, 영어)
   Chapter 1 figures and interactions, shared by the Korean and English pages. Text goes through tr(Korean, English) */
"use strict";

// 자주 쓰는 짧은 도구
// Small shared helpers
const $ = id => document.getElementById(id);
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const SUB = n => String(n).split("").map(d => String.fromCharCode(0x2080 + +d)).join("");
function fx(v, d = 2) { let s = (+v).toFixed(d); if (/^-0(\.0*)?$/.test(s)) s = s.slice(1); return s.replace("-", "−"); }
function fnum(v, d = 2) { let s = fx(v, d); if (s.includes(".")) s = s.replace(/0+$/, "").replace(/\.$/, ""); return s; }
const paren = v => (v < 0 ? "(" + fnum(v) + ")" : fnum(v));
const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
const varOf = (a, dd = 1) => { const m = mean(a); return a.reduce((s, v) => s + (v - m) ** 2, 0) / (a.length - dd); };
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const snap = (v, st) => Math.round(v / st) * st;
const normPdf = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
function niceStep(max, n = 4) {
  const raw = max / n, p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * p;
}
function ticksTo(max, n = 4) { const st = niceStep(max, n), out = []; for (let v = 0; v <= max + 1e-9; v += st) out.push(+v.toFixed(10)); return out; }
function txtBg(ctx, c, s, x, y, o = {}) {
  ctx.save(); ctx.font = (o.w || "400") + " " + (o.size || 11.5) + "px " + (o.mono ? FM : FS);
  const wd = ctx.measureText(s).width, x0 = o.align === "right" ? x - wd : o.align === "center" ? x - wd / 2 : x;
  ctx.fillStyle = c.bg; ctx.globalAlpha = 0.9; ctx.fillRect(x0 - 3, y - (o.size || 11.5) + 1, wd + 6, (o.size || 11.5) + 5); ctx.restore();
  txt(ctx, s, x, y, o);
}
function panelTitle(ctx, c, s, x, y) { txt(ctx, s, x, y, { color: c.ink, w: "600", size: 12.5 }); }

// t 분포: log gamma, 불완전 beta 함수, 양쪽 p-value
// Student t distribution: log gamma, incomplete beta, two-sided p-value
function lgammaFn(x) {
  const g = 7, cf = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgammaFn(1 - x);
  x -= 1; let a = cf[0]; const t = x + g + 0.5;
  for (let i = 1; i < 9; i++) a += cf[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}
function betacf(a, b, x) {
  const TINY = 1e-300; let c = 1, d = 1 - (a + b) * x / (a + 1);
  if (Math.abs(d) < TINY) d = TINY; d = 1 / d; let h = d;
  for (let m = 1; m <= 300; m++) {
    const m2 = 2 * m;
    let aa = m * (b - m) * x / ((a - 1 + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < TINY) d = TINY; c = 1 + aa / c; if (Math.abs(c) < TINY) c = TINY; d = 1 / d; h *= d * c;
    aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + 1 + m2));
    d = 1 + aa * d; if (Math.abs(d) < TINY) d = TINY; c = 1 + aa / c; if (Math.abs(c) < TINY) c = TINY; d = 1 / d;
    const del = d * c; h *= del; if (Math.abs(del - 1) < 3e-14) break;
  }
  return h;
}
function betai(a, b, x) {
  if (x <= 0) return 0; if (x >= 1) return 1;
  const bt = Math.exp(lgammaFn(a + b) - lgammaFn(a) - lgammaFn(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b;
}
const tTwoSided = (t, df) => betai(df / 2, 0.5, df / (df + t * t));
const tPdf = (t, df) => Math.exp(lgammaFn((df + 1) / 2) - lgammaFn(df / 2) - 0.5 * Math.log(df * Math.PI) - (df + 1) / 2 * Math.log(1 + t * t / df));
function tCrit(df) { let lo = 0, hi = 60; for (let i = 0; i < 60; i++) { const m = (lo + hi) / 2; if (tTwoSided(m, df) > 0.05) lo = m; else hi = m; } return (lo + hi) / 2; }

// 숫자 격자(DOM) 만들기: 2절, 4절, 12절에서 쓴다
// Build number grids in the DOM, used by sections 2, 4 and 12
function gridM(M, opts = {}) {
  const g = document.createElement("div"); g.className = "grid-m";
  g.style.gridTemplateColumns = `repeat(${M[0].length}, auto)`;
  if (opts.name) g.dataset.m = opts.name;
  M.forEach((row, i) => row.forEach((v, j) => {
    const s = document.createElement("span"); s.textContent = fnum(v); s.dataset.i = i; s.dataset.j = j;
    const k = opts.cls ? opts.cls(i, j) : ""; if (k) s.className = k;
    if (opts.click) { s.classList.add("click"); s.tabIndex = 0; s.setAttribute("role", "button"); }
    g.appendChild(s);
  }));
  return g;
}
function matBox(grid, label) { const d = document.createElement("div"); const l = document.createElement("div"); l.className = "lbl"; l.textContent = label; d.append(grid, l); return d; }
function opSign(s) { const d = document.createElement("div"); d.className = "op"; d.textContent = s; return d; }

/* ================= 1절 / Section 1 ================= */
// 8×8 pixel 사진과 그 숫자 matrix. 칸을 누르면 0 → 0.5 → 1 → 0
// An 8×8 pixel picture and its number matrix. A click cycles 0 → 0.5 → 1 → 0
const SEVEN = ["00000000", "01111110", "0000h110", "00001100", "000h1000", "00011000", "0001h000", "00010000"];
const S1 = { px: new Array(64).fill(0) };
function seven1() { SEVEN.forEach((row, r) => [...row].forEach((ch, k) => { S1.px[r * 8 + k] = ch === "1" ? 1 : ch === "h" ? 0.5 : 0; })); }
function lay1(w) {
  if (w >= 560) {
    const s = Math.min(40, Math.floor((w - 64) / 16)), g = 8 * s, gap = Math.min(110, w - 2 * g), x0 = (w - 2 * g - gap) / 2;
    return { s, side: true, a: { x: x0, y: 30 }, b: { x: x0 + g + gap, y: 30 }, h: 30 + g + 6 };
  }
  const s = Math.min(40, Math.floor((w - 4) / 8)), g = 8 * s, x0 = (w - g) / 2;
  return { s, side: false, a: { x: x0, y: 30 }, b: { x: x0, y: 30 + g + 44 }, h: 30 + 2 * g + 50 };
}
const F1 = fig("cv1", w => lay1(w).h, (ctx, w, h, c) => {
  const L = lay1(w), s = L.s, g = 8 * s;
  panelTitle(ctx, c, tr("사진 (사람이 보는 것)", "Picture (what people see)"), L.a.x, L.a.y - 10);
  panelTitle(ctx, c, tr("matrix (computer가 보는 것)", "Matrix (what the computer sees)"), L.b.x, L.b.y - 10);
  for (let i = 0; i < 64; i++) {
    const r = Math.floor(i / 8), k = i % 8, v = S1.px[i], gr = Math.round(255 * (1 - v));
    ctx.fillStyle = `rgb(${gr},${gr},${gr})`;
    ctx.fillRect(L.a.x + k * s, L.a.y + r * s, s, s);
    ctx.fillStyle = v > 0 ? c.sigwash : c.bg;
    ctx.fillRect(L.b.x + k * s, L.b.y + r * s, s, s);
    txt(ctx, v === 0.5 ? "0.5" : String(v), L.b.x + k * s + s / 2, L.b.y + r * s + s / 2 + 4, { color: v > 0 ? c.sig : c.muted, align: "center", mono: true, size: s >= 34 ? 11.5 : 10, w: v > 0 ? "600" : "400" });
  }
  ctx.strokeStyle = c.line; ctx.lineWidth = 1;
  for (const o of [L.a, L.b]) for (let i = 0; i <= 8; i++) {
    ctx.beginPath(); ctx.moveTo(o.x + i * s + 0.5, o.y); ctx.lineTo(o.x + i * s + 0.5, o.y + g); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(o.x, o.y + i * s + 0.5); ctx.lineTo(o.x + g, o.y + i * s + 0.5); ctx.stroke();
  }
  if (L.side) {
    arrow(ctx, L.a.x + g + 12, L.a.y + g / 2, L.b.x - 12, L.b.y + g / 2, c.muted, 1.5);
    txt(ctx, tr("숫자로", "as numbers"), (L.a.x + g + L.b.x) / 2, L.a.y + g / 2 - 10, { color: c.muted, align: "center", size: 11.5 });
  }
});
function cell1(x, y) {
  const L = lay1(F1.w);
  for (const o of [L.a, L.b]) {
    const k = Math.floor((x - o.x) / L.s), r = Math.floor((y - o.y) / L.s);
    if (k >= 0 && k < 8 && r >= 0 && r < 8) return r * 8 + k;
  }
  return -1;
}
function upd1() {
  paint(F1);
  const nz = S1.px.filter(v => v > 0).length, sum = S1.px.reduce((a, b) => a + b, 0);
  const flat = S1.px.slice(0, 24).map(v => (v === 0.5 ? "0.5" : String(v))).join(", ");
  $("r1").textContent = tr(
    `shape (8, 8) = 숫자 64개 · 0이 아닌 칸 ${nz}개 · 밝기 합 ${fnum(sum, 1)}\n한 줄로 펴면(flatten) shape (64,): [${flat}, …]\n→ 사람에게는 그림이지만 model에게는 숫자 64개짜리 vector다.`,
    `shape (8, 8) = 64 numbers · ${nz} non-zero cells · total ink ${fnum(sum, 1)}\nflattened into one row, shape (64,): [${flat}, …]\n→ A picture to people, a vector of 64 numbers to the model.`);
}
dragOn(F1, (x, y) => { const i = cell1(x, y); if (i < 0) return null; const v = S1.px[i]; return { v: v === 0 ? 0.5 : v === 0.5 ? 1 : 0 }; },
  (on, x, y) => { const i = cell1(x, y); if (i >= 0 && S1.px[i] !== on.v) { S1.px[i] = on.v; upd1(); } });
$("b1seven").addEventListener("click", () => { seven1(); upd1(); });
$("b1clear").addEventListener("click", () => { S1.px.fill(0); upd1(); });
$("b1inv").addEventListener("click", () => { S1.px = S1.px.map(v => 1 - v); upd1(); });
seven1();

/* ================= 2절 / Section 2 ================= */
// transpose: 짝이 되는 칸 칠하기와, 대각선을 거울 삼아 칸이 옮겨 가는 animation
// Transpose: highlight paired cells, and animate cells moving across the diagonal mirror
const A2 = [[1, 2, 3], [4, 5, 6]];
const gA2 = gridM(A2, { click: true }), gT2 = gridM(A2[0].map((_, j) => A2.map(r => r[j])), { click: true });
$("t2").append(matBox(gA2, "A (2 × 3)"), opSign("→ transpose →"), matBox(gT2, "Aᵀ (3 × 2)"));
function pick2(i, j) {
  gA2.querySelectorAll("span").forEach(s => { const a = +s.dataset.i, b = +s.dataset.j; s.classList.toggle("hl", a === i && b === j); s.classList.toggle("hl2", a === i && b !== j); });
  gT2.querySelectorAll("span").forEach(s => { const a = +s.dataset.i, b = +s.dataset.j; s.classList.toggle("hl", a === j && b === i); s.classList.toggle("hl2", b === i && a !== j); });
  const v = A2[i][j], row = A2[i].join(", ");
  $("r2").textContent = tr(
    `A의 ${i + 1}행 ${j + 1}열  A${SUB(i + 1)}${SUB(j + 1)} = ${v}\n= Aᵀ의 ${j + 1}행 ${i + 1}열  Aᵀ${SUB(j + 1)}${SUB(i + 1)} = ${v}\n초록 = A의 ${i + 1}행 (${row}) → Aᵀ의 ${i + 1}열이 된다.`,
    `row ${i + 1}, column ${j + 1} of A:  A${SUB(i + 1)}${SUB(j + 1)} = ${v}\n= row ${j + 1}, column ${i + 1} of Aᵀ:  Aᵀ${SUB(j + 1)}${SUB(i + 1)} = ${v}\nGreen = row ${i + 1} of A (${row}) → becomes column ${i + 1} of Aᵀ.`);
}
function hook2(g, isT) {
  const h = e => { const s = e.target.closest("span"); if (!s) return; const i = +s.dataset.i, j = +s.dataset.j; if (isT) pick2(j, i); else pick2(i, j); };
  g.addEventListener("mouseover", h); g.addEventListener("click", h); g.addEventListener("focusin", h);
}
hook2(gA2, false); hook2(gT2, true);
const S2 = { t: 0, raf: 0 };
const F2 = fig("cv2", 200, (ctx, w, h, c) => {
  const s = 44, x0 = Math.round((w - 3 * s) / 2), y0 = 34, e = S2.t * S2.t * (3 - 2 * S2.t);
  line(ctx, [[x0 - 10, y0 - 10], [x0 + 3 * s + 10, y0 + 3 * s + 10]], c.muted, 1.2, [4, 4]);
  txt(ctx, tr("대각선 = 거울", "diagonal = mirror"), x0 + 3 * s + 14, y0 + 3 * s + 8, { color: c.muted, size: 11 });
  A2.forEach((row, i) => row.forEach((v, j) => {
    const x = x0 + ((1 - e) * j + e * i) * s, y = y0 + ((1 - e) * i + e * j) * s;
    ctx.fillStyle = i === 0 ? c.sigwash : c.okwash; ctx.fillRect(x + 2, y + 2, s - 4, s - 4);
    ctx.strokeStyle = i === 0 ? c.sig : c.ok; ctx.lineWidth = 1.5; ctx.strokeRect(x + 2.5, y + 2.5, s - 5, s - 5);
    txt(ctx, String(v), x + s / 2, y + s / 2 + 5, { color: c.ink, align: "center", mono: true, size: 14, w: "600" });
  }));
  txt(ctx, S2.t < 0.5 ? "A  (2 × 3)" : "Aᵀ  (3 × 2)", x0, y0 - 14, { color: c.ink, w: "600", size: 12.5 });
});
const i2t = bindRange("i2t", "o2t", v => fx(v, 2), v => { S2.t = v; paint(F2); });
$("b2play").addEventListener("click", () => {
  cancelAnimationFrame(S2.raf);
  const from = S2.t, to = S2.t < 0.5 ? 1 : 0, t0 = performance.now(), dur = 1400;
  const step = now => {
    const k = reduceMotion ? 1 : Math.min(1, (now - t0) / dur);
    S2.t = from + (to - from) * k; i2t.value = S2.t; $("o2t").textContent = fx(S2.t, 2); paint(F2);
    if (k < 1) S2.raf = requestAnimationFrame(step);
  };
  S2.raf = requestAnimationFrame(step);
});

/* ================= 3절 / Section 3 ================= */
// dot product: 두 vector를 끌어 보며 곱의 합, 각도, 그림자(projection)를 본다
// Dot product: drag two vectors and watch the sum of products, the angle and the projection
const S3 = { a: [2, 1], b: [1, 3] };
function lay3(w) {
  const wide = w >= 620, P = wide ? Math.min(400, w - 250) : Math.min(380, w), ox = wide ? 0 : (w - P) / 2;
  return { wide, plane: { l: ox + 30, t: 10, w: P - 40, h: P - 40 }, bars: wide ? { l: P + 50, t: 50, w: w - P - 60, h: P - 116 } : { l: 50, t: P + 40, w: w - 66, h: 130 }, h: wide ? P : P + 214 };
}
const R3 = 4;
const X3 = (B, v) => B.l + (v + R3) / (2 * R3) * B.w, Y3 = (B, v) => B.t + B.h - (v + R3) / (2 * R3) * B.h;
const F3 = fig("cv3", w => lay3(w).h, (ctx, w, h, c) => {
  const L = lay3(w), B = L.plane, X = v => X3(B, v), Y = v => Y3(B, v);
  ctx.lineWidth = 1;
  for (let v = -R3; v <= R3; v++) {
    ctx.strokeStyle = v === 0 ? c.muted : c.line;
    ctx.beginPath(); ctx.moveTo(X(v) + 0.5, B.t); ctx.lineTo(X(v) + 0.5, B.t + B.h); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(B.l, Y(v) + 0.5); ctx.lineTo(B.l + B.w, Y(v) + 0.5); ctx.stroke();
  }
  [-4, -2, 0, 2, 4].forEach(v => {
    txt(ctx, fnum(v), X(v), B.t + B.h + 14, { color: c.muted, align: "center", mono: true, size: 10.5 });
    if (v) txt(ctx, fnum(v), B.l - 6, Y(v) + 4, { color: c.muted, align: "right", mono: true, size: 10.5 });
  });
  const [a1, a2] = S3.a, [b1, b2] = S3.b, d = a1 * b1 + a2 * b2, na = Math.hypot(a1, a2);
  const k = d / (na * na), px = k * a1, py = k * a2;
  ctx.save(); ctx.globalAlpha = 0.45; line(ctx, [[X(0), Y(0)], [X(px), Y(py)]], d >= 0 ? c.ok : c.fa, 8); ctx.restore();
  line(ctx, [[X(b1), Y(b2)], [X(px), Y(py)]], c.muted, 1.3, [4, 4]);
  const angA = Math.atan2(a2, a1), angB = Math.atan2(b2, b1);
  let del = angB - angA; while (del > Math.PI) del -= 2 * Math.PI; while (del <= -Math.PI) del += 2 * Math.PI;
  ctx.save(); ctx.strokeStyle = c.vio; ctx.lineWidth = 1.6; ctx.beginPath(); ctx.arc(X(0), Y(0), 24, -angA, -angB, del > 0); ctx.stroke(); ctx.restore();
  const mid = angA + del / 2;
  txt(ctx, "θ", X(0) + 36 * Math.cos(mid), Y(0) - 36 * Math.sin(mid) + 4, { color: c.vio, align: "center", size: 13, w: "600" });
  [["a", S3.a, c.sig], ["b", S3.b, c.miss]].forEach(([nm, v, k2]) => {
    arrow(ctx, X(0), Y(0), X(v[0]), Y(v[1]), k2, 2.6);
    dot(ctx, X(v[0]), Y(v[1]), 7, null, k2);
    const n = Math.hypot(v[0], v[1]), ux = v[0] / n, uy = -v[1] / n;
    txt(ctx, `${nm} (${fnum(v[0])}, ${fnum(v[1])})`, X(v[0]) + ux * 26, Y(v[1]) + uy * 20 + 4, { color: k2, align: "center", w: "700", size: 12.5 });
  });
  if (Math.abs(d) > 1e-9) txt(ctx, tr("그림자", "shadow"), (X(0) + X(px)) / 2, (Y(0) + Y(py)) / 2 + 16, { color: d >= 0 ? c.ok : c.fa, align: "center", size: 11, w: "600" });
  const BB = L.bars, vals = [a1 * b1, a2 * b2, d], labs = ["a₁ × b₁", "a₂ × b₂", tr("a · b (합)", "a · b (sum)")];
  const ym = niceStep(Math.max(4, ...vals.map(Math.abs)) * 1.2, 2) * 2;
  panelTitle(ctx, c, `a · b = ${paren(a1)}×${paren(b1)} + ${paren(a2)}×${paren(b2)} = ${fnum(d)}`, BB.l - 40, BB.t - 22);
  const A = axes(ctx, c, BB, [0, 3], [-ym, ym], { yt: [-ym, -ym / 2, 0, ym / 2, ym] });
  line(ctx, [[BB.l, A.Y(0)], [BB.l + BB.w, A.Y(0)]], c.muted, 1.2);
  vals.forEach((v, i) => {
    const x = A.X(i + 0.5), bw = BB.w / 3 * 0.5;
    ctx.fillStyle = i < 2 ? c.sig : v > 0 ? c.ok : v < 0 ? c.fa : c.muted; ctx.globalAlpha = i < 2 ? 0.45 : 0.9;
    ctx.fillRect(x - bw / 2, Math.min(A.Y(0), A.Y(v)), bw, Math.abs(A.Y(v) - A.Y(0))); ctx.globalAlpha = 1;
    txt(ctx, fnum(v), x, v >= 0 ? A.Y(v) - 5 : A.Y(v) + 14, { color: c.ink, align: "center", mono: true, size: 11.5, w: "600" });
    txt(ctx, labs[i], x, BB.t + BB.h + 16, { color: c.muted, align: "center", size: 11.5 });
  });
});
function upd3() {
  paint(F3);
  const [a1, a2] = S3.a, [b1, b2] = S3.b, d = a1 * b1 + a2 * b2, na = Math.hypot(a1, a2), nb = Math.hypot(b1, b2), cs = d / (na * nb);
  const th = Math.acos(clamp(cs, -1, 1)) * 180 / Math.PI;
  const say = Math.abs(d) < 1e-9 ? tr("θ = 90°: 직각이다. 서로 상관없는 방향 (a · b = 0)", "θ = 90°: perpendicular, unrelated directions (a · b = 0)")
    : d > 0 ? tr("θ < 90°: 같은 쪽을 본다 (a · b > 0)", "θ < 90°: pointing the same way (a · b > 0)")
    : tr("θ > 90°: 반대쪽을 본다 (a · b < 0)", "θ > 90°: pointing opposite ways (a · b < 0)");
  $("r3").textContent = `a = (${fnum(a1)}, ${fnum(a2)}), b = (${fnum(b1)}, ${fnum(b2)})\na · b = ${paren(a1)}×${paren(b1)} + ${paren(a2)}×${paren(b2)} = ${fnum(d)}\n|a| = ${fx(na, 3)}, |b| = ${fx(nb, 3)}, cos θ = ${fnum(d)} ÷ (${fx(na, 3)} × ${fx(nb, 3)}) = ${fx(cs, 3)}, θ = ${fx(th, 1)}°\n→ ${say}\n${tr("그림자 길이", "shadow length")} = a · b ÷ |a| = ${fx(d / na, 3)}`;
}
dragOn(F3, (x, y) => {
  const B = lay3(F3.w).plane; let best = null, bd = 24;
  for (const k of ["b", "a"]) { const v = S3[k], dd = Math.hypot(x - X3(B, v[0]), y - Y3(B, v[1])); if (dd < bd) { bd = dd; best = k; } }
  return best;
}, (k, x, y) => {
  const B = lay3(F3.w).plane;
  const vx = clamp(snap((x - B.l) / B.w * 2 * R3 - R3, 0.5), -R3, R3), vy = clamp(snap(R3 - (y - B.t) / B.h * 2 * R3, 0.5), -R3, R3);
  if (vx === 0 && vy === 0) return;
  if (S3[k][0] !== vx || S3[k][1] !== vy) { S3[k] = [vx + 0, vy + 0]; upd3(); }
});
$("b3r").addEventListener("click", () => { S3.a = [2, 1]; S3.b = [1, 3]; upd3(); });

/* ================= 4절 / Section 4 ================= */
// matrix 곱: 결과 칸을 누르면 그 칸을 만든 행과 열이 칠해진다
// Matrix multiplication: clicking a result cell highlights the row and column that made it
const A4o = [[1, 2, 3], [4, 5, 6]], B4o = [[1, 0], [0, 1], [1, 1]];
const S4 = { A: A4o, B: B4o, sel: null, timer: 0 };
const rng4 = makeRng(newSeed());
const matmul = (A, B) => A.map(row => B[0].map((_, j) => row.reduce((s, v, k) => s + v * B[k][j], 0)));
function build4(focus) {
  const C = matmul(S4.A, S4.B), sel = S4.sel;
  $("t4").replaceChildren(
    matBox(gridM(S4.A, { cls: i => (sel && i === sel[0] ? "hl" : "") }), "A (2 × 3)"), opSign("@"),
    matBox(gridM(S4.B, { cls: (i, j) => (sel && j === sel[1] ? "hl2" : "") }), "B (3 × 2)"), opSign("="),
    matBox(gridM(C, { name: "C", click: true, cls: (i, j) => (sel && i === sel[0] && j === sel[1] ? "pick" : "") }), "C = A @ B (2 × 2)"));
  if (!sel) {
    $("r4").textContent = tr("shape 확인: (2 × 3) @ (3 × 2) → 안쪽 3 = 3이라 곱할 수 있고, 결과는 바깥 숫자 (2 × 2).\n결과 C의 칸을 눌러 보세요.",
      "Shape check: (2 × 3) @ (3 × 2) → the inner sizes match (3 = 3), so the product exists, and its shape is the outer numbers (2 × 2).\nClick a cell of the result C.");
    return;
  }
  const [i, j] = sel, a = S4.A[i], b = S4.B.map(r => r[j]);
  const head = tr(`C의 ${i + 1}행 ${j + 1}열 = A의 ${i + 1}행 · B의 ${j + 1}열   (Python에서는 C[${i}, ${j}])`, `C row ${i + 1}, column ${j + 1} = (row ${i + 1} of A) · (column ${j + 1} of B)   (C[${i}, ${j}] in Python)`);
  $("r4").textContent = `${head}\n= ${a.map((v, k) => `${paren(v)}×${paren(b[k])}`).join(" + ")}\n= ${a.map((v, k) => paren(v * b[k])).join(" + ")} = ${fnum(C[i][j])}`;
  if (focus) { const s = $("t4").querySelector(`[data-m="C"] span[data-i="${i}"][data-j="${j}"]`); if (s) s.focus(); }
}
function stop4() { clearInterval(S4.timer); S4.timer = 0; $("b4play").textContent = tr("차례로 보기", "Step through"); }
$("t4").addEventListener("click", e => { const s = e.target.closest('[data-m="C"] span'); if (!s) return; stop4(); S4.sel = [+s.dataset.i, +s.dataset.j]; build4(false); });
$("t4").addEventListener("keydown", e => { if (e.key !== "Enter" && e.key !== " ") return; const s = e.target.closest('[data-m="C"] span'); if (!s) return; e.preventDefault(); stop4(); S4.sel = [+s.dataset.i, +s.dataset.j]; build4(true); });
$("b4play").addEventListener("click", () => {
  if (S4.timer) { stop4(); return; }
  const order = [[0, 0], [0, 1], [1, 0], [1, 1]]; let k = 0;
  const go = () => { S4.sel = order[k % 4]; k++; build4(false); };
  go(); S4.timer = setInterval(go, 1600); $("b4play").textContent = tr("멈추기", "Stop");
});
$("b4new").addEventListener("click", () => {
  const ri = () => Math.floor(rng4.u() * 8) - 2;
  S4.A = [[ri(), ri(), ri()], [ri(), ri(), ri()]]; S4.B = [[ri(), ri()], [ri(), ri()], [ri(), ri()]]; S4.sel = null; stop4(); build4(false);
});
$("b4reset").addEventListener("click", () => { S4.A = A4o; S4.B = B4o; S4.sel = null; stop4(); build4(false); });

/* ================= 5절 / Section 5 ================= */
// softmax: 점수 막대와 확률 막대. 점선 = T가 1일 때의 확률
// Softmax: score bars and probability bars. Dashed = probabilities at T = 1
const S5 = { z: [2, 1, 0], T: 1 };
const NAMES5 = [tr("고양이", "cat"), tr("개", "dog"), tr("새", "bird")];
function softmaxT(z, T) { const m = Math.max(...z), e = z.map(v => Math.exp((v - m) / T)), s = e.reduce((a, b) => a + b, 0); return e.map(v => v / s); }
const argmaxOf = a => a.reduce((bi, v, i) => (v > a[bi] ? i : bi), 0);
const F5 = fig("cv5", w => (stacked(w, 250, 2, 28) ? 490 : 250), (ctx, w, h, c) => {
  const P = panels(w, h, 2, 28, 250), p = softmaxT(S5.z, S5.T), p1 = softmaxT(S5.z, 1), am = argmaxOf(p);
  const bL = { l: P[0].x + 36, t: P[0].y + 40, w: P[0].w - 44, h: P[0].h - 40 - 30 };
  const bR = { l: P[1].x + 40, t: P[1].y + 40, w: P[1].w - 48, h: P[1].h - 40 - 30 };
  panelTitle(ctx, c, tr("점수 z (logit)", "Score z (logit)"), P[0].x + 4, P[0].y + 16);
  const L = axes(ctx, c, bL, [0, 3], [-3, 8], { yt: [-2, 0, 2, 4, 6, 8] });
  line(ctx, [[bL.l, L.Y(0)], [bL.l + bL.w, L.Y(0)]], c.muted, 1.2);
  S5.z.forEach((z, i) => {
    const x = L.X(i + 0.5), bw = bL.w / 3 * 0.5;
    ctx.globalAlpha = i === am ? 1 : 0.4; ctx.fillStyle = c.sig;
    ctx.fillRect(x - bw / 2, Math.min(L.Y(0), L.Y(z)), bw, Math.abs(L.Y(z) - L.Y(0))); ctx.globalAlpha = 1;
    txt(ctx, fx(z, 1), x, z >= 0 ? L.Y(z) - 5 : L.Y(z) + 14, { color: c.ink, align: "center", mono: true, size: 11 });
    txt(ctx, NAMES5[i], x, bL.t + bL.h + 16, { color: c.ink, align: "center", size: 11.5, w: i === am ? "700" : "400" });
  });
  const dashed = Math.abs(S5.T - 1) > 1e-9;
  panelTitle(ctx, c, tr("softmax 확률 (합 = 1)", "softmax probability (sum = 1)") + (dashed ? tr(" · 점선 = T 1일 때", " · dashed = at T = 1") : ""), P[1].x + 4, P[1].y + 16);
  const R = axes(ctx, c, bR, [0, 3], [0, 1], { yt: [0, 0.25, 0.5, 0.75, 1] });
  p.forEach((v, i) => {
    const x = R.X(i + 0.5), bw = bR.w / 3 * 0.5;
    ctx.globalAlpha = i === am ? 1 : 0.4; ctx.fillStyle = c.sig; ctx.fillRect(x - bw / 2, R.Y(v), bw, R.Y(0) - R.Y(v)); ctx.globalAlpha = 1;
    if (dashed) { ctx.save(); ctx.setLineDash([4, 3]); ctx.strokeStyle = c.ink; ctx.lineWidth = 1.2; ctx.strokeRect(x - bw / 2 - 4.5, R.Y(p1[i]) + 0.5, bw + 9, R.Y(0) - R.Y(p1[i])); ctx.restore(); }
    txt(ctx, (v * 100).toFixed(1) + "%", x, R.Y(v) - 5, { color: c.ink, align: "center", mono: true, size: 11, w: "600" });
    txt(ctx, NAMES5[i], x, bR.t + bR.h + 16, { color: c.ink, align: "center", size: 11.5, w: i === am ? "700" : "400" });
  });
});
function upd5() {
  paint(F5);
  const zt = S5.z.map(v => v / S5.T), e = zt.map(Math.exp), s = e.reduce((a, b) => a + b, 0), p = softmaxT(S5.z, S5.T), am = argmaxOf(p);
  const f = v => (v >= 1e4 || v < 1e-3 ? fmtNum(v) : fx(v, 3));
  $("r5").textContent = tr(
    `z ÷ T = (${zt.map(v => fx(v, 2)).join(", ")})\ne^(z/T) = (${e.map(f).join(", ")}), 합 = ${f(s)}\nsoftmax = 각 값 ÷ 합 = (${p.map(v => fx(v, 3)).join(", ")}), 합 = 1\n가장 큰 확률: ${NAMES5[am]} (argmax = ${am})`,
    `z ÷ T = (${zt.map(v => fx(v, 2)).join(", ")})\ne^(z/T) = (${e.map(f).join(", ")}), sum = ${f(s)}\nsoftmax = each value ÷ sum = (${p.map(v => fx(v, 3)).join(", ")}), sum = 1\nLargest probability: ${NAMES5[am]} (argmax = ${am})`);
}
[["i5a", "o5a", 0], ["i5b", "o5b", 1], ["i5c", "o5c", 2]].forEach(([id, out, k]) => bindRange(id, out, v => fx(v, 1), v => { S5.z[k] = v; upd5(); }));
bindRange("i5t", "o5t", v => fx(v, 1), v => { S5.T = v; upd5(); });

/* ================= 6절 / Section 6 ================= */
// ln과 exp: y = x를 거울로 한 짝
// ln and exp: a mirrored pair across y = x
const S6 = { x: 2 };
function lay6(w) { const S = Math.min(w - 56, 360); return { B: { l: Math.max(40, (w - S) / 2), t: 12, w: S, h: S }, h: S + 48 }; }
const F6 = fig("cv6", w => lay6(w).h, (ctx, w, h, c) => {
  const { B } = lay6(w), r = [-3, 4.5], tk = [-3, -2, -1, 0, 1, 2, 3, 4];
  const A = axes(ctx, c, B, r, r, { xt: tk, yt: tk });
  clipBox(ctx, B);
  line(ctx, [[A.X(0), B.t], [A.X(0), B.t + B.h]], c.muted, 1); line(ctx, [[B.l, A.Y(0)], [B.l + B.w, A.Y(0)]], c.muted, 1);
  line(ctx, [[A.X(-3), A.Y(-3)], [A.X(4.5), A.Y(4.5)]], c.muted, 1.2, [5, 4]);
  const lnP = [], exP = [];
  for (let i = 0; i <= 240; i++) { const u = -3.1 + (Math.log(4.5) + 3.1) * i / 240; lnP.push([A.X(Math.exp(u)), A.Y(u)]); exP.push([A.X(u), A.Y(Math.exp(u))]); }
  line(ctx, lnP, c.sig, 2.4); line(ctx, exP, c.miss, 2.4);
  const x = S6.x, lx = Math.log(x);
  line(ctx, [[A.X(x), A.Y(lx)], [A.X(lx), A.Y(x)]], c.vio, 1.4, [2, 3]);
  line(ctx, [[A.X(x), A.Y(0)], [A.X(x), A.Y(lx)]], c.sig, 1, [3, 3]);
  ctx.restore();
  dot(ctx, A.X(x), A.Y(lx), 5.5, c.sig); dot(ctx, A.X(lx), A.Y(x), 5.5, c.miss);
  txt(ctx, "y = ln x", A.X(3.3), A.Y(Math.log(3.3)) + 18, { color: c.sig, w: "700", size: 12, align: "center" });
  txt(ctx, "y = eˣ", A.X(Math.log(3.6)) - 10, A.Y(3.6), { color: c.miss, w: "700", size: 12, align: "right" });
  txt(ctx, tr("y = x (거울)", "y = x (mirror)"), A.X(3.2) + 8, A.Y(3.2) + 16, { color: c.muted, size: 11.5 });
  txt(ctx, `(${fx(x, 2)}, ${fx(lx, 3)})`, A.X(x) + 8, A.Y(lx) + (lx < -2 ? -8 : 16), { color: c.sig, mono: true, size: 11 });
  txtBg(ctx, c, `(${fx(lx, 3)}, ${fx(x, 2)})`, A.X(lx) - 12, A.Y(x) + 4, { color: c.miss, mono: true, size: 11, align: "right" });
});
function upd6() {
  paint(F6);
  const x = S6.x, lx = Math.log(x);
  const tail = x <= 1
    ? tr(`x를 확률로 보면: ln ${fx(x, 2)} = ${fx(lx, 3)} (0 이하). 놀람 −ln x = ${fx(-lx, 3)} (7절)`, `Read x as a probability: ln ${fx(x, 2)} = ${fx(lx, 3)} (zero or below). Surprise −ln x = ${fx(-lx, 3)} (section 7)`)
    : tr("x > 1이면 ln x > 0. 확률은 1을 넘지 않으므로 딥러닝의 log 확률은 늘 0 이하다.", "For x > 1, ln x > 0. Probabilities never exceed 1, so log-probabilities in deep learning are always zero or below.");
  $("r6").textContent = tr(
    `x = ${fx(x, 2)} → ln x = ${fx(lx, 3)}   (e를 ${fx(lx, 3)}번 곱하면 ${fx(x, 2)})\n되돌리기: e^${fx(lx, 3)} = ${fx(Math.exp(lx), 2)}\n${tail}`,
    `x = ${fx(x, 2)} → ln x = ${fx(lx, 3)}   (e raised to the power ${fx(lx, 3)} gives ${fx(x, 2)})\nUndo: e^${fx(lx, 3)} = ${fx(Math.exp(lx), 2)}\n${tail}`);
}
bindRange("i6x", "o6x", v => fx(v, 2), v => { S6.x = v; upd6(); });

/* ================= 7절 / Section 7 ================= */
// entropy 곡선과 cross-entropy 곡선
// Entropy curve and cross-entropy curve
const S7 = { p: 0.9, q: 0.7 };
const Hc = p => -(p * Math.log(p) + (1 - p) * Math.log(1 - p));
const F7 = fig("cv7", w => (stacked(w, 250, 2, 28) ? 500 : 260), (ctx, w, h, c) => {
  const P = panels(w, h, 2, 28, 250);
  const b0 = { l: P[0].x + 36, t: P[0].y + 34, w: P[0].w - 44, h: P[0].h - 34 - 44 };
  const b1 = { l: P[1].x + 36, t: P[1].y + 34, w: P[1].w - 44, h: P[1].h - 34 - 44 };
  const xt = [0, 0.25, 0.5, 0.75, 1];
  panelTitle(ctx, c, tr("동전의 entropy H(p)", "Entropy of a coin, H(p)"), P[0].x + 4, P[0].y + 16);
  const A = axes(ctx, c, b0, [0, 1], [0, 0.8], { xt, yt: [0, 0.2, 0.4, 0.6, 0.8], xlabel: tr("앞면 확률 p", "probability of heads p") });
  line(ctx, [[b0.l, A.Y(Math.LN2)], [b0.l + b0.w, A.Y(Math.LN2)]], c.muted, 1, [4, 3]);
  txt(ctx, tr("최대 0.693 nat = 1 bit", "max 0.693 nat = 1 bit"), b0.l + b0.w - 4, A.Y(Math.LN2) - 5, { color: c.muted, align: "right", size: 10.5 });
  const pts = []; for (let i = 1; i < 200; i++) { const p = i / 200; pts.push([A.X(p), A.Y(Hc(p))]); }
  line(ctx, pts, c.sig, 2.4);
  line(ctx, [[A.X(S7.p), A.Y(0)], [A.X(S7.p), A.Y(Hc(S7.p))]], c.sig, 1, [3, 3]);
  dot(ctx, A.X(S7.p), A.Y(Hc(S7.p)), 5.5, c.sig);
  txt(ctx, fx(Hc(S7.p), 3), A.X(S7.p) + (S7.p > 0.8 ? -8 : 8), A.Y(Hc(S7.p)) - 8, { color: c.sig, mono: true, size: 11, w: "600", align: S7.p > 0.8 ? "right" : "left" });
  panelTitle(ctx, c, "cross-entropy = −ln q", P[1].x + 4, P[1].y + 16);
  const B = axes(ctx, c, b1, [0, 1], [0, 5], { xt, yt: [0, 1, 2, 3, 4, 5], xlabel: tr("model이 정답에 준 확률 q", "probability the model gave the right answer, q") });
  const q0 = Math.exp(-5), pts2 = [];
  for (let i = 0; i <= 200; i++) { const q = q0 + (1 - q0) * i / 200; pts2.push([B.X(q), B.Y(-Math.log(q))]); }
  line(ctx, pts2, c.fa, 2.4);
  [[0.7, "A"], [0.3, "B"]].forEach(([q, nm]) => { dot(ctx, B.X(q), B.Y(-Math.log(q)), 4.5, null, c.muted); txt(ctx, `model ${nm}`, B.X(q), B.Y(-Math.log(q)) - 10, { color: c.muted, size: 10.5, align: "center" }); });
  const near = Math.abs(S7.q - 0.7) < 0.08 || Math.abs(S7.q - 0.3) < 0.08;
  line(ctx, [[B.X(S7.q), B.Y(0)], [B.X(S7.q), B.Y(-Math.log(S7.q))]], c.fa, 1, [3, 3]);
  dot(ctx, B.X(S7.q), B.Y(-Math.log(S7.q)), 5.5, c.fa);
  txt(ctx, fx(-Math.log(S7.q), 3), B.X(S7.q) + (S7.q > 0.8 ? -8 : 8), B.Y(-Math.log(S7.q)) - (near ? 24 : 8), { color: c.fa, mono: true, size: 11, w: "600", align: S7.q > 0.8 ? "right" : "left" });
});
function upd7() {
  paint(F7);
  const p = S7.p, s1 = -Math.log(p), s0 = -Math.log(1 - p), H = Hc(p), q = S7.q;
  $("r7").textContent = tr(
    `동전 p = ${fx(p, 2)}: 앞면 놀람 −ln ${fx(p, 2)} = ${fx(s1, 3)}, 뒷면 놀람 −ln ${fx(1 - p, 2)} = ${fx(s0, 3)}\nH = ${fx(p, 2)}×${fx(s1, 3)} + ${fx(1 - p, 2)}×${fx(s0, 3)} = ${fx(H, 3)} nat = ${fx(H / Math.LN2, 3)} bit\n정답 확률 q = ${fx(q, 2)}: cross-entropy = −ln ${fx(q, 2)} = ${fx(-Math.log(q), 3)}${q >= 0.995 ? "  (정답에 확률 1 → loss 0)" : ""}`,
    `coin p = ${fx(p, 2)}: surprise at heads −ln ${fx(p, 2)} = ${fx(s1, 3)}, at tails −ln ${fx(1 - p, 2)} = ${fx(s0, 3)}\nH = ${fx(p, 2)}×${fx(s1, 3)} + ${fx(1 - p, 2)}×${fx(s0, 3)} = ${fx(H, 3)} nat = ${fx(H / Math.LN2, 3)} bit\nprobability on the right answer q = ${fx(q, 2)}: cross-entropy = −ln ${fx(q, 2)} = ${fx(-Math.log(q), 3)}${q >= 0.995 ? "  (probability 1 on the answer → loss 0)" : ""}`);
}
bindRange("i7p", "o7p", v => fx(v, 2), v => { S7.p = v; upd7(); });
bindRange("i7q", "o7q", v => fx(v, 2), v => { S7.q = v; upd7(); });

/* ================= 9절 / Section 9 ================= */
// 평균과 분산: 점을 끌면 거리와 제곱 막대가 바뀐다
// Mean and variance: dragging a point changes its distance and its squared-distance bar
const S9 = { x: [2, 4, 4, 4, 5, 5, 7, 9] };
const LANE9 = 17;
function lay9(w) { const top = { l: 40, t: 24, w: w - 56, h: 8 * LANE9 + 8 }; const bot = { l: 40, t: top.t + top.h + 64, w: w - 56, h: 150 }; return { top, bot, h: bot.t + bot.h + 34 }; }
const laneY9 = (top, i) => top.t + 4 + (i + 0.5) * LANE9;
const F9 = fig("cv9", w => lay9(w).h, (ctx, w, h, c) => {
  const { top, bot } = lay9(w), xs = S9.x, m = mean(xs), sq = xs.map(v => (v - m) ** 2), vr = mean(sq);
  const A = axes(ctx, c, top, [0, 10], [0, 1], { xt: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] });
  line(ctx, [[A.X(m), top.t - 4], [A.X(m), top.t + top.h]], c.ok, 2.2);
  txt(ctx, `${tr("평균", "mean")} ${fx(m, 2)}`, A.X(m), top.t - 8, { color: c.ok, align: "center", w: "700", size: 12 });
  xs.forEach((v, i) => {
    const y = laneY9(top, i);
    txt(ctx, `x${SUB(i + 1)}`, top.l - 8, y + 4, { color: c.muted, align: "right", size: 11 });
    ctx.save(); ctx.globalAlpha = 0.55; line(ctx, [[A.X(v), y], [A.X(m), y]], c.sig, 2.2); ctx.restore();
    dot(ctx, A.X(v), y, 6.5, c.sig, c.bg);
  });
  const ym = Math.max(10, Math.max(...sq) * 1.15), st = niceStep(ym, 4), yt = [];
  for (let v = 0; v <= ym + 1e-9; v += st) yt.push(+v.toFixed(6));
  panelTitle(ctx, c, tr("거리의 제곱 (x − 평균)²", "Squared distance (x − mean)²"), bot.l - 36, bot.t - 24);
  const B = axes(ctx, c, bot, [0, 8], [0, ym], { yt });
  sq.forEach((v, i) => {
    const x = B.X(i + 0.5), bw = bot.w / 8 * 0.55;
    ctx.globalAlpha = 0.45; ctx.fillStyle = c.sig; ctx.fillRect(x - bw / 2, B.Y(v), bw, B.Y(0) - B.Y(v)); ctx.globalAlpha = 1;
    ctx.strokeStyle = c.sig; ctx.lineWidth = 1.2; ctx.strokeRect(x - bw / 2 + 0.5, B.Y(v) + 0.5, bw - 1, Math.max(0, B.Y(0) - B.Y(v) - 1));
    txt(ctx, fnum(v), x, B.Y(v) - 5, { color: c.ink, align: "center", mono: true, size: 11 });
    txt(ctx, `x${SUB(i + 1)}=${fnum(xs[i], 1)}`, x, bot.t + bot.h + 15, { color: c.muted, align: "center", mono: true, size: bot.w / 8 < 52 ? 9.5 : 10.5 });
  });
  line(ctx, [[bot.l, B.Y(vr)], [bot.l + bot.w, B.Y(vr)]], c.ok, 2, [6, 4]);
  txt(ctx, `${tr("초록 점선 = 막대 평균 = 분산", "green dashed = mean of the bars = variance")} ${fx(vr, 2)}`, bot.l + bot.w, bot.t - 10, { color: c.ok, align: "right", w: "700", size: 12 });
});
function upd9() {
  paint(F9);
  const xs = S9.x, n = xs.length, s = xs.reduce((a, b) => a + b, 0), m = s / n, sq = xs.map(v => (v - m) ** 2), ss = sq.reduce((a, b) => a + b, 0);
  const list = xs.map(v => fnum(v, 1)).join(", "), sql = sq.map(v => fnum(v)).join(", ");
  $("r9").textContent = tr(
    `data: ${list}  (n = ${n})\n평균 = ${fnum(s, 1)} ÷ ${n} = ${fx(m, 2)}\n(x − 평균)² = ${sql} → 합 ${fx(ss, 2)}\n분산 (÷n) = ${fx(ss, 2)} ÷ ${n} = ${fx(ss / n, 2)}, SD = ${fx(Math.sqrt(ss / n), 2)}\n표본 분산 (÷(n−1)) = ${fx(ss, 2)} ÷ ${n - 1} = ${fx(ss / (n - 1), 2)}, SD = ${fx(Math.sqrt(ss / (n - 1)), 2)}`,
    `data: ${list}  (n = ${n})\nmean = ${fnum(s, 1)} ÷ ${n} = ${fx(m, 2)}\n(x − mean)² = ${sql} → sum ${fx(ss, 2)}\nvariance (÷n) = ${fx(ss, 2)} ÷ ${n} = ${fx(ss / n, 2)}, SD = ${fx(Math.sqrt(ss / n), 2)}\nsample variance (÷(n−1)) = ${fx(ss, 2)} ÷ ${n - 1} = ${fx(ss / (n - 1), 2)}, SD = ${fx(Math.sqrt(ss / (n - 1)), 2)}`);
}
dragOn(F9, (x, y) => {
  const { top } = lay9(F9.w); let best = null, bd = 16;
  S9.x.forEach((v, i) => { const dd = Math.hypot(x - (top.l + v / 10 * top.w), y - laneY9(top, i)); if (dd < bd) { bd = dd; best = i; } });
  return best;
}, (i, x) => {
  const { top } = lay9(F9.w), v = clamp(snap((x - top.l) / top.w * 10, 0.5), 0, 10);
  if (S9.x[i] !== v) { S9.x[i] = v; upd9(); }
});
$("b9r").addEventListener("click", () => { S9.x = [2, 4, 4, 4, 5, 5, 7, 9]; upd9(); });

/* ================= 10절 / Section 10 ================= */
// 표본 평균 쌓기: population 곡선과 표본 평균 histogram
// Collecting sample means: the population curve and a histogram of sample means
const S10 = { n: 10, means: [], last: null, lastSeed: null };
const rng10 = makeRng(newSeed());
function sample10(r) { const v = []; for (let i = 0; i < S10.n; i++) v.push(70 + 10 * r.n()); return v; }
const F10 = fig("cv10", w => (stacked(w, 260, 2, 28) ? 520 : 270), (ctx, w, h, c) => {
  const P = panels(w, h, 2, 28, 260), xr = [35, 105], xt = [40, 50, 60, 70, 80, 90, 100];
  const b0 = { l: P[0].x + 12, t: P[0].y + 34, w: P[0].w - 22, h: P[0].h - 34 - 44 };
  panelTitle(ctx, c, tr("population: 점수 (평균 70, SD 10)", "Population: scores (mean 70, SD 10)"), P[0].x + 4, P[0].y + 16);
  const A = axes(ctx, c, b0, xr, [0, 0.046], { xt, xlabel: tr("점수", "score"), grid: false });
  ctx.beginPath(); ctx.moveTo(A.X(35), A.Y(0));
  for (let i = 0; i <= 200; i++) { const x = 35 + 70 * i / 200; ctx.lineTo(A.X(x), A.Y(normPdf(x, 70, 10))); }
  ctx.lineTo(A.X(105), A.Y(0)); ctx.closePath(); ctx.globalAlpha = 0.18; ctx.fillStyle = c.muted; ctx.fill(); ctx.globalAlpha = 1; ctx.strokeStyle = c.muted; ctx.lineWidth = 1.5; ctx.stroke();
  if (S10.last) {
    S10.last.forEach(v => { const x = A.X(clamp(v, 35, 105)); line(ctx, [[x, b0.t + b0.h], [x, b0.t + b0.h - 14]], c.sig, 1.4); });
    const m = mean(S10.last), x = A.X(m);
    line(ctx, [[x, b0.t + 4], [x, b0.t + b0.h]], c.sig, 2.4);
    txt(ctx, `${tr("이 표본의 평균", "this sample's mean")} ${fx(m, 1)}`, x + (m < 75 ? 6 : -6), b0.t + 14, { color: c.sig, w: "700", size: 11.5, align: m < 75 ? "left" : "right" });
  }
  const b1 = { l: P[1].x + 40, t: P[1].y + 34, w: P[1].w - 48, h: P[1].h - 34 - 44 };
  const N = S10.means.length, bw = S10.n >= 100 ? 0.5 : 1, nb = Math.round(70 / bw), cnt = new Array(nb).fill(0);
  S10.means.forEach(m => { const k = Math.floor((m - 35) / bw); if (k >= 0 && k < nb) cnt[k]++; });
  const se = 10 / Math.sqrt(S10.n), peak = N * bw * normPdf(70, 70, se), ym = Math.max(4, ...cnt, peak) * 1.15;
  panelTitle(ctx, c, tr(`표본 평균 ${N.toLocaleString()}개 (표본마다 n = ${S10.n}명)`, `${N.toLocaleString()} sample mean${N === 1 ? "" : "s"} (n = ${S10.n} people per sample)`), P[1].x + 4, P[1].y + 16);
  const B = axes(ctx, c, b1, xr, [0, ym], { xt, yt: ticksTo(ym, 3), xlabel: tr("표본 평균", "sample mean") });
  const px = b1.w * bw / 70;
  ctx.fillStyle = c.sig; ctx.globalAlpha = 0.6;
  cnt.forEach((k, i) => { if (k) ctx.fillRect(B.X(35 + i * bw) + (px > 3 ? 0.5 : 0), B.Y(k), Math.max(1, px - (px > 3 ? 1 : 0)), B.Y(0) - B.Y(k)); });
  ctx.globalAlpha = 1;
  if (N) {
    const pts = []; for (let i = 0; i <= 300; i++) { const x = 35 + 70 * i / 300; pts.push([B.X(x), B.Y(N * bw * normPdf(x, 70, se))]); }
    clipBox(ctx, b1); line(ctx, pts, c.fa, 2); ctx.restore();
  } else txt(ctx, tr("아직 없음: 위의 뽑기 버튼을 누르세요", "Nothing yet: press a draw button above"), b1.l + b1.w / 2, b1.t + b1.h / 2, { color: c.muted, align: "center", size: 12 });
});
function upd10() {
  paint(F10);
  const N = S10.means.length, lines = [];
  if (S10.last) {
    const L = S10.last, show = L.slice(0, 10).map(v => fx(v, 1)).join(", ") + (L.length > 10 ? tr(`, … (${L.length}개)`, `, … (${L.length} values)`) : "");
    const sd = S10.lastSeed !== null ? `, seed ${S10.lastSeed}` : "";
    lines.push(tr(`마지막 표본 (n = ${L.length}${sd}): ${show}\n→ 이 표본의 평균 ${fx(mean(L), 2)}`, `last sample (n = ${L.length}${sd}): ${show}\n→ this sample's mean ${fx(mean(L), 2)}`));
  }
  if (N > 1) lines.push(tr(
    `쌓인 표본 평균 ${N.toLocaleString()}개: 평균 ${fx(mean(S10.means), 2)}, SD ${fx(Math.sqrt(varOf(S10.means)), 2)}  (이론 SE = 10 ÷ √${S10.n} = ${fx(10 / Math.sqrt(S10.n), 2)})`,
    `${N.toLocaleString()} sample means collected: mean ${fx(mean(S10.means), 2)}, SD ${fx(Math.sqrt(varOf(S10.means)), 2)}  (theory SE = 10 ÷ √${S10.n} = ${fx(10 / Math.sqrt(S10.n), 2)})`));
  $("r10").textContent = lines.length ? lines.join("\n") : tr(`표본 하나 = population에서 ${S10.n}명을 무작위로 뽑은 것. 뽑기 버튼을 눌러 보세요.`, `One sample = ${S10.n} people drawn at random from the population. Press a draw button.`);
}
function add10(r, seed) { const v = sample10(r); S10.last = v; S10.lastSeed = seed; S10.means.push(mean(v)); }
bindSeg("g10n", v => { S10.n = +v; S10.means = []; S10.last = null; upd10(); });
$("b10one").addEventListener("click", () => { add10(rng10, null); upd10(); });
$("b10many").addEventListener("click", () => { for (let i = 0; i < 1000; i++) add10(rng10, null); upd10(); });
$("b10clear").addEventListener("click", () => { S10.means = []; S10.last = null; upd10(); });
$("b10seed").addEventListener("click", () => { const sd = Math.max(0, Math.floor(+$("i10seed").value || 0)); add10(makeRng(sd), sd); upd10(); });

/* ================= 11절 / Section 11 ================= */
// t-test: 두 model의 실행별 정확도와 t 분포 위의 p-value 넓이
// t-test: per-run accuracies of two models and the p-value area under the t distribution
const S11 = { d: 0.03, s: 0.016, n: 5, zA: [], zB: [], rep: null };
function fillZ11(first) {
  const r = makeRng(newSeed());
  for (let i = 0; i < 30; i++) { S11.zA[i] = r.n(); S11.zB[i] = r.n(); }
  if (first) [-0.625, 0.625, -1.25, 0, 1.25].forEach((z, i) => { S11.zA[i] = z; S11.zB[i] = z; });
}
function data11() { const A = [], B = []; for (let i = 0; i < S11.n; i++) { A.push(0.82 + S11.s * S11.zA[i]); B.push(0.82 + S11.d + S11.s * S11.zB[i]); } return { A, B }; }
function welch(A, B) {
  const nA = A.length, nB = B.length, mA = mean(A), mB = mean(B), vA = varOf(A), vB = varOf(B), qa = vA / nA, qb = vB / nB, se = Math.sqrt(qa + qb), t = (mB - mA) / se;
  const df = (qa + qb) ** 2 / (qa * qa / (nA - 1) + qb * qb / (nB - 1));
  return { mA, mB, sA: Math.sqrt(vA), sB: Math.sqrt(vB), seA: Math.sqrt(qa), seB: Math.sqrt(qb), se, t, df, p: tTwoSided(t, df) };
}
const fp = p => (p < 0.001 ? "< 0.001" : fx(p, 3));
const F11 = fig("cv11", w => (stacked(w, 280, 2, 28) ? 540 : 270), (ctx, w, h, c) => {
  const P = panels(w, h, 2, 28, 280), D = data11(), W = welch(D.A, D.B);
  const b0 = { l: P[0].x + 28, t: P[0].y + 34, w: P[0].w - 40, h: P[0].h - 34 - 44 };
  panelTitle(ctx, c, tr("두 model의 실행별 정확도", "Accuracy of each run, two models"), P[0].x + 4, P[0].y + 16);
  const A = axes(ctx, c, b0, [0.68, 1.0], [0, 1], { xt: [[0.7, "0.70"], [0.75, "0.75"], [0.8, "0.80"], [0.85, "0.85"], [0.9, "0.90"], [0.95, "0.95"], [1, "1.00"]], xlabel: tr("정확도 (점 1개 = 실행 1번)", "accuracy (one dot = one run)") });
  const rows = [["A", D.A, b0.t + b0.h * 0.27, c.miss, W.mA, W.seA], ["B", D.B, b0.t + b0.h * 0.73, c.sig, W.mB, W.seB]];
  rows.forEach(([nm, vals, y, k, m, se]) => {
    txt(ctx, nm, b0.l - 8, y + 4, { color: k, align: "right", w: "700", size: 13 });
    ctx.globalAlpha = 0.75; vals.forEach((v, i) => dot(ctx, A.X(clamp(v, 0.68, 1)), y - 4 + ((i * 7) % 5 - 2) * 3.5, 4.2, k)); ctx.globalAlpha = 1;
    line(ctx, [[A.X(m), y - 22], [A.X(m), y + 14]], k, 3);
    line(ctx, [[A.X(m - se), y + 22], [A.X(m + se), y + 22]], k, 2);
    line(ctx, [[A.X(m - se), y + 18], [A.X(m - se), y + 26]], k, 2); line(ctx, [[A.X(m + se), y + 18], [A.X(m + se), y + 26]], k, 2);
  });
  const ym = b0.t + b0.h * 0.5 + 2, xa = A.X(W.mA), xb = A.X(W.mB);
  if (Math.abs(xb - xa) > 12) arrow(ctx, xa, ym, xb, ym, c.ink, 1.4);
  txt(ctx, `${tr("차이", "difference")} ${fx(W.mB - W.mA, 3)}`, (xa + xb) / 2, ym - 6, { color: c.ink, align: "center", size: 11.5, w: "600" });
  txt(ctx, tr("굵은 선 = 평균, 아래 막대 = 평균 ± SE", "thick line = mean, bar below = mean ± SE"), b0.l + b0.w, b0.t + 10, { color: c.muted, align: "right", size: 10.5 });

  const b1 = { l: P[1].x + 12, t: P[1].y + 34, w: P[1].w - 22, h: P[1].h - 34 - 44 };
  panelTitle(ctx, c, tr(`t 분포 (df ${fx(W.df, 1)}): 차이가 0일 때 t가 나오는 모양`, `t distribution (df ${fx(W.df, 1)}): where t falls when there is no real difference`), P[1].x + 4, P[1].y + 16);
  const T = axes(ctx, c, b1, [-6, 6], [0, 0.42], { xt: [-6, -4, -2, 0, 2, 4, 6], xlabel: "t", grid: false });
  const at = Math.abs(W.t), tc = tCrit(W.df);
  if (at < 6) {
    ctx.fillStyle = c.fa; ctx.globalAlpha = 0.35;
    [1, -1].forEach(sg => {
      ctx.beginPath(); ctx.moveTo(T.X(sg * at), T.Y(0));
      for (let i = 0; i <= 80; i++) { const t = sg * (at + (6 - at) * i / 80); ctx.lineTo(T.X(t), T.Y(tPdf(t, W.df))); }
      ctx.lineTo(T.X(sg * 6), T.Y(0)); ctx.closePath(); ctx.fill();
    });
    ctx.globalAlpha = 1;
  }
  const pts = []; for (let i = 0; i <= 240; i++) { const t = -6 + 12 * i / 240; pts.push([T.X(t), T.Y(tPdf(t, W.df))]); }
  line(ctx, pts, c.ink, 1.8);
  [-tc, tc].forEach(t => line(ctx, [[T.X(t), b1.t + 20], [T.X(t), T.Y(0)]], c.muted, 1.2, [4, 3]));
  txtBg(ctx, c, `${tr("p = 0.05 경계", "p = 0.05 cut-off")} ±${fx(tc, 2)}`, T.X(-tc) - 4, b1.t + 30, { color: c.muted, align: "right", size: 10.5 });
  if (at < 6) {
    line(ctx, [[T.X(W.t), b1.t + 40], [T.X(W.t), T.Y(0)]], c.fa, 2.2);
    const out = W.t > 0 ? T.X(W.t) + 70 < b1.l + b1.w : T.X(W.t) - 70 > b1.l;
    txtBg(ctx, c, `t = ${fx(W.t, 2)}`, T.X(W.t) + (W.t > 0 === out ? 5 : -5), b1.t + 52, { color: c.fa, align: W.t > 0 === out ? "left" : "right", w: "700", size: 12 });
  } else txt(ctx, `t = ${fx(W.t, 2)} ${W.t > 0 ? "→" : "←"} ${tr("(그림 밖)", "(off the chart)")}`, W.t > 0 ? b1.l + b1.w : b1.l, b1.t + 52, { color: c.fa, align: W.t > 0 ? "right" : "left", w: "700", size: 12 });
  txt(ctx, `${tr("빨간 넓이", "red area")} = p = ${fp(W.p)}`, b1.l + 2, b1.t + 10, { color: c.fa, w: "700", size: 12 });
});
function upd11() {
  paint(F11);
  const D = data11(), W = welch(D.A, D.B), n = S11.n;
  const list = a => a.slice(0, 8).map(v => fx(v, 3)).join(", ") + (a.length > 8 ? ", …" : "");
  const verdict = W.p < 0.05 ? tr("0.05보다 작다: 우연이라 보기 어렵다", "below 0.05: hard to explain by chance") : tr("0.05보다 크다: 우연일 수도 있다", "above 0.05: could be chance");
  const M = tr("평균", "mean"), lines = [
    `A: ${list(D.A)} → ${M} ${fx(W.mA, 4)}, SD ${fx(W.sA, 4)}`,
    `B: ${list(D.B)} → ${M} ${fx(W.mB, 4)}, SD ${fx(W.sB, 4)}`,
    `${tr("차이", "difference")} = ${fx(W.mB - W.mA, 4)}, SE = √(${fx(W.sA, 4)}²/${n} + ${fx(W.sB, 4)}²/${n}) = ${fx(W.se, 4)}`,
    `t = ${fx(W.mB - W.mA, 4)} ÷ ${fx(W.se, 4)} = ${fx(W.t, 2)}, df = ${fx(W.df, 1)}, p = ${fp(W.p)} → ${verdict}`];
  if (S11.rep) {
    const R = S11.rep, pc = Math.round(R.k / 2);
    lines.push(tr(
      `200번 반복 (참 차이 ${fx(R.d, 3)}, 잡음 ${fx(R.s, 3)}, n ${R.n}): p < 0.05가 ${R.k}번 (${pc}%) → ${R.d === 0 ? "차이가 없는데도 나온 '우연한 유의' 비율" : "진짜 차이를 알아챈 비율"}`,
      `200 repeats (true difference ${fx(R.d, 3)}, noise ${fx(R.s, 3)}, n ${R.n}): p < 0.05 in ${R.k} (${pc}%) → ${R.d === 0 ? "the rate of 'significant by chance' when there is no difference" : "the rate of catching the real difference"}`));
  }
  $("r11").textContent = lines.join("\n");
}
const reset11 = () => { S11.rep = null; upd11(); };
bindRange("i11d", "o11d", v => fx(v, 3), v => { S11.d = v; reset11(); });
bindRange("i11s", "o11s", v => fx(v, 3), v => { S11.s = v; reset11(); });
bindRange("i11n", "o11n", v => String(v), v => { S11.n = v; reset11(); });
$("b11").addEventListener("click", () => { fillZ11(false); reset11(); });
$("b11many").addEventListener("click", () => {
  const r = makeRng(newSeed()); let k = 0;
  for (let rep = 0; rep < 200; rep++) {
    const A = [], B = [];
    for (let i = 0; i < S11.n; i++) { A.push(0.82 + S11.s * r.n()); B.push(0.82 + S11.d + S11.s * r.n()); }
    if (welch(A, B).p < 0.05) k++;
  }
  S11.rep = { k, d: S11.d, s: S11.s, n: S11.n }; upd11();
});
fillZ11(true);
add10(makeRng(42), 42);

/* ================= 12절 / Section 12 ================= */
// broadcasting: 크기 1인 축을 복사해 늘리는 모습(옅은 칸)
// Broadcasting: size-1 axes copied to stretch (faded cells)
const CASES12 = [
  { a: [[0], [10], [20]], b: [[1, 2, 3, 4]], as: "(3, 1)", bs: "(1, 4)" },
  { a: [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], b: [[100, 200, 300, 400]], as: "(3, 4)", bs: "(4,)" },
  { a: [[1, 2, 3], [4, 5, 6]], b: [[1, 0], [0, 1], [1, 1]], as: "(2, 3)", bs: "(3, 2)", err: true }
];
const TEXT12 = [
  tr("오른쪽 축부터 짝짓기:\n  뒤 축: a 1, b 4 → a의 1을 4로 늘린다 (열 복사)\n  앞 축: a 3, b 1 → b의 1을 3으로 늘린다 (행 복사)\n결과 shape (3, 4). 칸 (i, j) = a[i] + b[j]",
     "Pair the axes from the right:\n  last axis: a 1, b 4 → stretch a's 1 to 4 (copy the column)\n  first axis: a 3, b 1 → stretch b's 1 to 3 (copy the row)\nResult shape (3, 4). Cell (i, j) = a[i] + b[j]"),
  tr("b의 shape (4,)는 앞에 1을 붙여 (1, 4)로 본다.\n  뒤 축: 4, 4 → 같다\n  앞 축: 3, 1 → b의 1을 3으로 늘린다\n결과 shape (3, 4). 행마다 b를 더한다 (bias를 더할 때 늘 이렇게 된다)",
     "b's shape (4,) is read as (1, 4) by adding a 1 in front.\n  last axis: 4, 4 → equal\n  first axis: 3, 1 → stretch b's 1 to 3\nResult shape (3, 4). b is added to every row (this is what happens every time a bias is added)"),
  tr("  뒤 축: a 3, b 2 → 다르고 둘 다 1이 아니다 → error\nValueError: operands could not be broadcast together with shapes (2,3) (3,2)\n(matrix 곱 a @ b는 된다: 안쪽 3 = 3. 덧셈과 matrix 곱은 규칙이 다르다.)",
     "  last axis: a 3, b 2 → different, and neither is 1 → error\nValueError: operands could not be broadcast together with shapes (2,3) (3,2)\n(The matrix product a @ b does work: inner 3 = 3. Addition and matrix multiplication follow different rules.)")
];
const stretch12 = (M, R, C) => Array.from({ length: R }, (_, i) => Array.from({ length: C }, (_, j) => M[M.length === 1 ? 0 : i][M[0].length === 1 ? 0 : j]));
function show12(k) {
  const cs = CASES12[k];
  if (cs.err) $("t12").replaceChildren(matBox(gridM(cs.a), "a " + cs.as), opSign("+"), matBox(gridM(cs.b), "b " + cs.bs), opSign("= error"));
  else {
    const R = Math.max(cs.a.length, cs.b.length), C = Math.max(cs.a[0].length, cs.b[0].length), sa = stretch12(cs.a, R, C), sb = stretch12(cs.b, R, C);
    const ghost = M => (i, j) => (i >= M.length || j >= M[0].length ? "ghost" : "");
    $("t12").replaceChildren(
      matBox(gridM(sa, { cls: ghost(cs.a) }), `a ${cs.as} → (${R}, ${C})`), opSign("+"),
      matBox(gridM(sb, { cls: ghost(cs.b) }), `b ${cs.bs} → (${R}, ${C})`), opSign("="),
      matBox(gridM(sa.map((row, i) => row.map((v, j) => v + sb[i][j]))), `${tr("결과", "result")} (${R}, ${C})`));
  }
  $("r12").textContent = TEXT12[k];
}
bindSeg("g12", v => show12(+v));

/* ================= 13절 / Section 13 ================= */
// 미분: 접선(초록)과 할선(주황). h가 작아지면 둘이 붙는다
// Derivative: tangent (green) and secant (orange). They meet as h shrinks
const S13 = { x: 3, h: 2 };
const F13 = fig("cv13", 320, (ctx, w, h, c) => {
  const B = { l: 44, t: 16, w: w - 60, h: 256 }, xr = [-2, 6.2], yr = [-5, 38];
  const A = axes(ctx, c, B, xr, yr, { xt: [-2, -1, 0, 1, 2, 3, 4, 5, 6], yt: [0, 10, 20, 30], xlabel: "x" });
  const x0 = S13.x, y0 = x0 * x0, x1 = x0 + S13.h, y1 = x1 * x1, mt = 2 * x0, ms = (y1 - y0) / S13.h;
  clipBox(ctx, B);
  line(ctx, [[B.l, A.Y(0)], [B.l + B.w, A.Y(0)]], c.muted, 1);
  const pts = []; for (let i = 0; i <= 200; i++) { const x = xr[0] + (xr[1] - xr[0]) * i / 200; pts.push([A.X(x), A.Y(x * x)]); }
  line(ctx, pts, c.sig, 2.4);
  line(ctx, [[A.X(xr[0]), A.Y(y0 + mt * (xr[0] - x0))], [A.X(xr[1]), A.Y(y0 + mt * (xr[1] - x0))]], c.ok, 2);
  line(ctx, [[A.X(xr[0]), A.Y(y0 + ms * (xr[0] - x0))], [A.X(xr[1]), A.Y(y0 + ms * (xr[1] - x0))]], c.miss, 2);
  line(ctx, [[A.X(x0), A.Y(y0)], [A.X(x1), A.Y(y0)], [A.X(x1), A.Y(y1)]], c.miss, 1.2, [3, 3]);
  ctx.restore();
  dot(ctx, A.X(x0), A.Y(y0), 5.5, c.ok); dot(ctx, A.X(x1), A.Y(y1), 4.5, c.miss);
  if (A.X(x1) - A.X(x0) > 22) txt(ctx, "h", (A.X(x0) + A.X(x1)) / 2, A.Y(y0) + 14, { color: c.miss, align: "center", size: 11.5, w: "600" });
  if (A.Y(y0) - A.Y(y1) > 22) txt(ctx, "Δy", A.X(x1) + 5, (A.Y(y0) + A.Y(y1)) / 2 + 4, { color: c.miss, size: 11.5, w: "600" });
  txt(ctx, "y = x²", A.X(-1.75), A.Y(3.06) - 8, { color: c.sig, w: "700", size: 12 });
  txt(ctx, tr("초록 = 접선 (기울기 2x)", "green = tangent (slope 2x)"), B.l + 8, B.t + 14, { color: c.ok, w: "600", size: 11.5 });
  txt(ctx, tr("주황 = 할선 (x와 x + h를 잇는 선)", "orange = secant (through x and x + h)"), B.l + 8, B.t + 30, { color: c.miss, w: "600", size: 11.5 });
  const ya = A.Y(-3.4);
  if (Math.abs(mt) > 1e-9) {
    const dir = -Math.sign(mt), xe = clamp(x0 + dir * 1.1, xr[0] + 0.1, xr[1] - 0.1);
    arrow(ctx, A.X(x0), ya, A.X(xe), ya, c.ok, 2);
    txtBg(ctx, c, tr("y가 줄어드는 쪽", "y gets smaller this way"), A.X(xe) + (dir < 0 ? -6 : 6), ya + 4, { color: c.ok, align: dir < 0 ? "right" : "left", size: 11, w: "600" });
  } else txtBg(ctx, c, tr("기울기 0: 바닥 (최솟값)", "slope 0: the bottom (minimum)"), A.X(x0), ya + 4, { color: c.ok, align: "center", size: 11, w: "600" });
});
function upd13() {
  paint(F13);
  const x = S13.x, hh = S13.h, f0 = x * x, f1 = (x + hh) ** 2, sec = (f1 - f0) / hh, tan = 2 * x;
  const dir = Math.abs(tan) < 1e-9 ? tr("기울기 0 → 여기가 바닥이다. 더 내려갈 곳이 없다", "slope 0 → this is the bottom; there is nowhere lower to go")
    : tan > 0 ? tr("기울기가 양수 → y를 줄이려면 x를 줄인다 (왼쪽으로)", "positive slope → to make y smaller, make x smaller (move left)")
    : tr("기울기가 음수 → y를 줄이려면 x를 늘린다 (오른쪽으로)", "negative slope → to make y smaller, make x larger (move right)");
  $("r13").textContent = tr(
    `x = ${fx(x, 1)}, h = ${fx(hh, 2)}\n할선 기울기 = (f(${fx(x + hh, 2)}) − f(${fx(x, 1)})) ÷ ${fx(hh, 2)} = (${fx(f1, hh < 0.1 ? 4 : 2)} − ${fx(f0, 2)}) ÷ ${fx(hh, 2)} = ${fx(sec, 2)}\n접선 기울기 f′(${fx(x, 1)}) = 2 × ${paren(x)} = ${fx(tan, 2)}   (차이 = h = ${fx(hh, 2)})\n→ ${dir}`,
    `x = ${fx(x, 1)}, h = ${fx(hh, 2)}\nsecant slope = (f(${fx(x + hh, 2)}) − f(${fx(x, 1)})) ÷ ${fx(hh, 2)} = (${fx(f1, hh < 0.1 ? 4 : 2)} − ${fx(f0, 2)}) ÷ ${fx(hh, 2)} = ${fx(sec, 2)}\ntangent slope f′(${fx(x, 1)}) = 2 × ${paren(x)} = ${fx(tan, 2)}   (difference = h = ${fx(hh, 2)})\n→ ${dir}`);
}
bindRange("i13x", "o13x", v => fx(v, 1), v => { S13.x = v; upd13(); });
bindRange("i13h", "o13h", v => fx(v, 2), v => { S13.h = v; upd13(); });

// 처음 그리기
// First paint
upd1(); pick2(1, 2); upd3(); build4(false); upd5(); upd6(); upd7(); upd9(); upd10(); upd11(); show12(0); upd13();
repaintAll();
