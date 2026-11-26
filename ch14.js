/* 14장 그림과 조작. 한글판(ch14-transformer-KR.html)과 영어판(ch14-transformer.html)이 같이 쓴다
   Chapter 14 figures and interactions, shared by the Korean and English pages
   GPT-2 small의 token·attention·확률, 작은 ViT 결과는 assets/data/ch14_data.js (tools/ch14_numbers.py, ch14_gpt2.py가 만든다)
   GPT-2 small tokens, attention and probabilities, and the small ViT results live in assets/data/ch14_data.js (made by tools/ch14_numbers.py and ch14_gpt2.py) */
"use strict";

const D14 = CH14_DATA, G2 = D14.gpt2, AT = G2.att, TOK = G2.tokens.en.pieces;
const attW = (l, h) => AT.w[l - 1][h - 1].map(r => r.map(v => v / 1000));
const tokLab = s => s.replace(/^ /, "␣");
// 가장 첫 token을 많이 보는 head (attention sink)
// The head that looks at the first token the most (attention sink)
const FIRST = (() => { let best = [1, 1, -1]; for (let l = 1; l <= 12; l++) for (let h = 1; h <= 12; h++) { const W = attW(l, h), m = meanOf(W.slice(1).map(r => r[0])); if (m > best[2]) best = [l, h, m]; } return best; })();

/* ================= 1절 / Section 1 ================= */
// 단어를 누르면 그 단어가 앞의 단어들을 얼마나 보는지 (GPT-2, 층·head 선택)
// Click a word to see how much it looks at the words before it (GPT-2, chosen layer and head)
const S1 = { q: TOK.indexOf(" it"), l: AT.it_animal[0], h: AT.it_animal[1], lay: null };
const F1 = fig("cv1", w => (w >= 640 ? 250 : 360), (ctx, w, h, c) => {
  const W = attW(S1.l, S1.h), widths = TOK.map(t => textW(ctx, tokLab(t), 13, "600", true) + 14);
  let x = 16, y = 150, row = 0; const pos = [];
  TOK.forEach((t, k) => { if (x + widths[k] > w - 16) { x = 16; row++; } pos.push([x, (w >= 640 ? 150 : 120) + row * 70, widths[k]]); x += widths[k] + 8; });
  S1.lay = pos;
  const qx = pos[S1.q][0] + pos[S1.q][2] / 2, qy = pos[S1.q][1];
  // 고른 단어에서 앞 단어들로 굽은 선 (굵기 = attention)
  // Curves from the chosen word to the earlier words (thickness = attention)
  for (let k = 0; k <= S1.q; k++) {
    const a = W[S1.q][k]; if (a < 0.01) continue;
    const tx = pos[k][0] + pos[k][2] / 2, ty = pos[k][1], mid = Math.min(qy, ty) - 30 - Math.abs(qx - tx) * 0.18;
    ctx.save(); ctx.strokeStyle = c.miss; ctx.globalAlpha = 0.25 + 0.75 * a; ctx.lineWidth = 1 + 14 * a;
    ctx.beginPath(); ctx.moveTo(qx, qy - 14); ctx.quadraticCurveTo((qx + tx) / 2, mid, tx, ty - 14); ctx.stroke(); ctx.restore();
  }
  TOK.forEach((t, k) => {
    const [px, py, pw] = pos[k], a = k <= S1.q ? W[S1.q][k] : 0, sel = k === S1.q;
    ctx.fillStyle = sel ? c.sigwash : k > S1.q ? c.bg : cssRGB(mixRGB(rgb(c.bg), rgb(c.miss), Math.min(1, a) * 0.8));
    ctx.fillRect(px, py - 12, pw, 26); ctx.strokeStyle = sel ? c.sig : c.line; ctx.lineWidth = sel ? 2.4 : 1; ctx.strokeRect(px, py - 12, pw, 26);
    txt(ctx, tokLab(t), px + pw / 2, py + 6, { color: k > S1.q ? c.muted : c.ink, size: 13, mono: true, align: "center", w: "600" });
    if (k <= S1.q && a >= 0.05) txt(ctx, Math.round(a * 100) + "%", px + pw / 2, py + 30, { color: c.miss, size: 10.5, mono: true, align: "center" });
  });
  panelTitle(ctx, c, tr(`GPT-2 small, ${S1.l}층 head ${S1.h}: 파랑 단어가 앞 단어들을 보는 비율 (단어를 누른다)`, `GPT-2 small, layer ${S1.l} head ${S1.h}: how much the blue word looks at each earlier word (click a word)`), 16, 18);
});
function upd1() {
  paint(F1);
  const W = attW(S1.l, S1.h)[S1.q], top = W.map((a, k) => [a, k]).slice(0, S1.q + 1).sort((p, q) => q[0] - p[0]).slice(0, 3);
  $("r1").textContent = tr(`"${tokLab(TOK[S1.q])}"가 가장 많이 보는 단어: ` + top.map(([a, k]) => `"${tokLab(TOK[k])}" ${Math.round(a * 100)}%`).join(", ") + `\nGPT-2는 앞만 본다 (뒤 단어는 회색). 합은 늘 100%.`,
    `Words "${tokLab(TOK[S1.q])}" looks at most: ` + top.map(([a, k]) => `"${tokLab(TOK[k])}" ${Math.round(a * 100)}%`).join(", ") + `\nGPT-2 only looks back (later words are grey). The shares always add up to 100%.`);
}
F1.cv.addEventListener("click", ev => {
  const r = F1.cv.getBoundingClientRect(), px = ev.clientX - r.left, py = ev.clientY - r.top;
  (S1.lay || []).forEach(([x, y, wd], k) => { if (px >= x && px <= x + wd && py >= y - 12 && py <= y + 14) { S1.q = k; upd1(); } });
});
bindRange("i1l", "o1l", v => String(v), v => { S1.l = v; upd1(); });
bindRange("i1h", "o1h", v => String(v), v => { S1.h = v; upd1(); });

/* ================= 2절 / Section 2 ================= */
// Q Kᵀ → /√d → softmax → × V를 한 단계씩 (token 3개, d = 2)
// Q Kᵀ → /√d → softmax → × V one step at a time (3 tokens, d = 2)
const HD = D14.hand, S2 = { step: 1 };
const F2 = fig("cv2", w => (w >= 700 ? 250 : 520), (ctx, w, h, c) => {
  const wide = w >= 700, cw = 44, ch = 26, scaled = HD.S.map(r => r.map(v => v / Math.SQRT2));
  const items = [
    ["Q (3×2)", HD.Q, c.sig, 1, 1], ["Kᵀ (2×3)", HD.K[0].map((_, j) => HD.K.map(r => r[j])), c.vio, 1, 1],
    [tr("Q Kᵀ (3×3) 점수", "Q Kᵀ (3×3) scores"), HD.S, c.ink, 1, 1], ["÷ √2", scaled, c.ink, 2, 3],
    ["softmax (3×3)", HD.A, c.miss, 3, 3], ["V (3×2)", HD.V, c.ok, 4, 2], [tr("출력 = A V (3×2)", "output = A V (3×2)"), HD.O, c.ok, 4, 2]
  ];
  let x = 16, y = 40, rowH = 0;
  items.forEach(([lab, M, col, st], k) => {
    const W = M[0].length * cw, H = M.length * ch;
    if (x + W > w - 16) { x = 16; y += rowH + 46; rowH = 0; }
    const on = S2.step >= st;
    ctx.globalAlpha = on ? 1 : 0.25;
    txt(ctx, lab, x, y - 10, { color: col, size: 11.5, w: "700" });
    drawMat(ctx, c, M, x, y, { cw, ch, col, dp: M.flat().every(Number.isInteger) ? 0 : st === 3 ? 3 : 2, size: 11, hl: st === S2.step && on ? () => true : null });
    ctx.globalAlpha = 1;
    x += W + (wide ? 34 : 22); rowH = Math.max(rowH, H);
  });
});
function upd2() {
  paint(F2);
  const msg = {
    1: tr("1) 점수 = Q Kᵀ: query와 key의 내적. 행 i = token i의 질문이 각 token의 이름표와 얼마나 맞나. (3×2)(2×3) → (3×3)", "1) scores = Q Kᵀ: dot products of queries and keys. Row i = how well token i's question matches each token's label. (3×2)(2×3) → (3×3)"),
    2: tr("2) √d_k = √2로 나눈다: d가 크면 내적이 커져 softmax가 한 칸에 몰리므로 크기를 맞춘다.", "2) Divide by √d_k = √2: with large d the dot products grow and softmax piles onto one cell, so rescale."),
    3: tr("3) 행마다 softmax: 각 행의 합 = 1. 1행 = (0.576, 0.140, 0.284): token 1은 자기 자신을 가장 많이 본다. 3행은 점수가 같아 1/3씩.", "3) Softmax per row: every row sums to 1. Row 1 = (0.576, 0.140, 0.284): token 1 looks at itself most. Row 3 has equal scores, so 1/3 each."),
    4: tr("4) 출력 = A V: value의 가중 평균. 1행 = 0.576·(10, 0) + 0.140·(0, 10) + 0.284·(5, 5) = (7.18, 2.82).", "4) output = A V: a weighted average of the values. Row 1 = 0.576·(10, 0) + 0.140·(0, 10) + 0.284·(5, 5) = (7.18, 2.82).")
  }[S2.step];
  $("r2").textContent = msg;
}
$("b2step").addEventListener("click", () => { S2.step = S2.step >= 4 ? 1 : S2.step + 1; upd2(); });

/* ================= 3절 / Section 3 ================= */
// head 하나의 전체 attention 표 (13×13), 미리 고른 head 셋
// The full attention table of one head (13×13), with three preset heads
const S3 = { l: AT.prev[0], h: AT.prev[1] };
const F3 = fig("cv3", 440, (ctx, w, h, c) => {
  const W = attW(S3.l, S3.h), n = TOK.length, lw = 74, cell = Math.min(24, (w - lw - 30) / n), x0 = lw + 10, y0 = 84, bg = rgb(c.bg), mc = rgb(c.miss);
  panelTitle(ctx, c, tr(`${S3.l}층 head ${S3.h}: 행 = 보는 단어 (query), 열 = 보이는 단어 (key)`, `layer ${S3.l} head ${S3.h}: row = the word looking (query), column = the word looked at (key)`), 16, 18);
  for (let i = 0; i < n; i++) {
    txt(ctx, tokLab(TOK[i]), x0 - 6, y0 + i * cell + cell / 2 + 4, { color: c.ink, size: 10.5, mono: true, align: "right" });
    ctx.save(); ctx.translate(x0 + i * cell + cell / 2 + 3, y0 - 6); ctx.rotate(-Math.PI / 3); txt(ctx, tokLab(TOK[i]), 0, 0, { color: c.ink, size: 10.5, mono: true }); ctx.restore();
    for (let j = 0; j < n; j++) {
      ctx.fillStyle = j > i ? c.wash : cssRGB(mixRGB(bg, mc, Math.min(1, W[i][j]) * 0.9)); ctx.fillRect(x0 + j * cell, y0 + i * cell, cell - 1, cell - 1);
    }
  }
  txt(ctx, tr("연한 회색 칸 = 미래 (mask로 가림)", "pale grey cells = the future (hidden by the mask)"), x0, y0 + n * cell + 18, { color: c.muted, size: 10.5 });
});
function upd3() {
  paint(F3);
  const W = attW(S3.l, S3.h), first = meanOf(W.slice(1).map(r => r[0])), prev = meanOf(W.slice(1).map((r, i) => r[i]));
  $("r3").textContent = tr(`이 head가 첫 token을 보는 평균 ${Math.round(first * 100)}%, 바로 앞 token을 보는 평균 ${Math.round(prev * 100)}%.\n전체 144개 head 평균으로는 첫 token에 ${Math.round(AT.first_mean * 100)}%를 준다: 볼 곳이 없을 때 첫 token에 attention을 "버리는" 버릇 (attention sink).`,
    `This head gives the first token ${Math.round(first * 100)}% on average and the previous token ${Math.round(prev * 100)}%.\nAveraged over all 144 heads, ${Math.round(AT.first_mean * 100)}% goes to the first token: a habit of "parking" attention there when nothing else is relevant (attention sink).`);
}
bindRange("i3l", "o3l", v => String(v), v => { S3.l = v; upd3(); });
bindRange("i3h", "o3h", v => String(v), v => { S3.h = v; upd3(); });
const preset3 = (l, h) => { S3.l = l; S3.h = h; setRange("i3l", l, 0); setRange("i3h", h, 0); upd3(); };
$("b3prev").addEventListener("click", () => preset3(AT.prev[0], AT.prev[1]));
$("b3it").addEventListener("click", () => preset3(AT.it_animal[0], AT.it_animal[1]));
$("b3first").addEventListener("click", () => preset3(FIRST[0], FIRST[1]));

/* ================= 4절 / Section 4 ================= */
// 사인 위치 encoding: 자리(가로) × 차원(세로) 물결, 고른 자리의 벡터
// Sinusoidal positional encoding: waves over position (across) × dimension (down), and the chosen position's vector
const DPE = 64, NPOS = 64, S4 = { p: 10 };
const pe = (pos, i) => (i % 2 === 0 ? Math.sin(pos / Math.pow(10000, i / DPE)) : Math.cos(pos / Math.pow(10000, (i - 1) / DPE)));
const F4 = fig("cv4", w => (w >= 640 ? 300 : 360), (ctx, w, h, c) => {
  const B = { l: 70, t: 30, w: w - 88, h: h - 70 }, cwid = B.w / NPOS, chgt = B.h / DPE, bg = rgb(c.bg), pos = rgb(c.sig), neg = rgb(c.fa);
  for (let p = 0; p < NPOS; p++) for (let i = 0; i < DPE; i++) { const v = pe(p, i); ctx.fillStyle = cssRGB(mixRGB(bg, v >= 0 ? pos : neg, Math.abs(v) * 0.85)); ctx.fillRect(B.l + p * cwid, B.t + i * chgt, cwid + 0.5, chgt + 0.5); }
  ctx.strokeStyle = c.ink; ctx.lineWidth = 2; ctx.strokeRect(B.l + S4.p * cwid, B.t, cwid, B.h);
  axes(ctx, c, B, [0, NPOS], [DPE, 0], { xt: [0, 16, 32, 48, 63].map(v => [v + 0.5, String(v)]), yt: [[DPE - 0.5, "63"], [DPE - 32, "32"], [0.5, "0"]], xlabel: tr("자리 pos", "position pos"), grid: false });
  panelTitle(ctx, c, tr("PE(pos, k): 파랑 +, 빨강 −. 위 칸은 빨리, 아래 칸은 천천히 흔들린다", "PE(pos, k): blue +, red −. Top slots swing fast, bottom ones slowly"), B.l, 16);
  txt(ctx, tr("칸 k", "slot k"), 8, B.t + B.h / 2 - 16, { color: c.muted, size: 11 });
});
const F4b = fig("cv4b", 220, (ctx, w, h, c) => {
  const B = { l: 52, t: 30, w: w - 70, h: h - 70 }, A = axes(ctx, c, B, [0, 63], [-0.2, 1], { xt: [0, 16, 32, 48, 63], yt: [0, 0.5, 1], xlabel: tr("다른 자리", "other position") });
  const v0 = Array.from({ length: DPE }, (_, i) => pe(S4.p, i)), n0 = Math.hypot(...v0);
  const sin = Array.from({ length: NPOS }, (_, q) => { const v = Array.from({ length: DPE }, (_, i) => pe(q, i)); return v.reduce((s, x, i) => s + x * v0[i], 0) / (n0 * Math.hypot(...v)); });
  const lrn = G2.pos_cos[S4.p].map(v => v / 100);
  line(ctx, sin.map((v, q) => [A.X(q), A.Y(v)]), c.sig, 2); line(ctx, lrn.map((v, q) => [A.X(q), A.Y(v)]), c.miss, 2, [6, 3]);
  line(ctx, [[A.X(S4.p), B.t], [A.X(S4.p), B.t + B.h]], c.ink, 1, [3, 3]);
  panelTitle(ctx, c, tr(`자리 ${S4.p}와 다른 자리의 cosine 유사도: 파랑 = 사인 (d 64), 주황 점선 = GPT-2가 학습한 위치 embedding`, `Cosine similarity of position ${S4.p} with every other: blue = sinusoidal (d 64), orange dashed = GPT-2's learned position embedding`), B.l, 16);
});
function upd4() { paint(F4); paint(F4b); $("r4").textContent = tr(`자리 ${S4.p}: PE(${S4.p}, 0) = sin(${S4.p}) = ${fx(pe(S4.p, 0), 4)}, PE(${S4.p}, 1) = cos(${S4.p}) = ${fx(pe(S4.p, 1), 4)}. 가까운 자리일수록 벡터가 비슷하다 (아래 그림).`, `Position ${S4.p}: PE(${S4.p}, 0) = sin(${S4.p}) = ${fx(pe(S4.p, 0), 4)}, PE(${S4.p}, 1) = cos(${S4.p}) = ${fx(pe(S4.p, 1), 4)}. Nearby positions have similar vectors (lower figure).`); }
bindRange("i4p", "o4p", v => String(v), v => { S4.p = v; upd4(); });

/* ================= 6절 / Section 6 ================= */
// causal mask: 미래 칸을 −∞로 가리면 softmax가 어떻게 바뀌나 (2절 예제)
// Causal mask: how softmax changes when future cells are set to −∞ (the example of section 2)
const S6 = { m: "on" };
const softRow = r => { const m = Math.max(...r.filter(v => v > -Infinity)), e = r.map(v => (v === -Infinity ? 0 : Math.exp(v - m))), s = e.reduce((a, b) => a + b, 0); return e.map(v => v / s); };
const F6 = fig("cv6", w => (w >= 600 ? 190 : 340), (ctx, w, h, c) => {
  const sc = HD.S.map((r, i) => r.map((v, j) => (S6.m === "on" && j > i ? -Infinity : v / Math.SQRT2))), A = sc.map(softRow), cw = 54, ch = 28, wide = w >= 600;
  const x1 = 16, x2 = wide ? 16 + 3 * cw + 70 : 16, y1 = 40, y2 = wide ? 40 : 40 + 3 * ch + 50;
  txt(ctx, tr("점수 ÷ √2 (mask 적용)", "scores ÷ √2 (mask applied)"), x1, y1 - 10, { color: c.ink, size: 11.5, w: "700" });
  drawMat(ctx, c, sc.map(r => r.map(v => (v === -Infinity ? "−∞" : +v.toFixed(3)))), x1, y1, { cw, ch, dp: 3, size: 11, hl: (i, j) => (S6.m === "on" && j > i ? c.wash : null) });
  if (wide) arrowS(ctx, x1 + 3 * cw + 16, y1 + 1.5 * ch, x2 - 16, y1 + 1.5 * ch, c.muted, 1.6, 7);
  txt(ctx, "softmax", x2, y2 - 10, { color: c.miss, size: 11.5, w: "700" });
  drawMat(ctx, c, A.map(r => r.map(v => +v.toFixed(3))), x2, y2, { cw, ch, col: c.miss, dp: 3, size: 11 });
});
function upd6() { paint(F6); $("r6").textContent = S6.m === "on" ? tr("mask 켬: token 1은 자기만 (1.000), token 2는 1·2만 (0.196, 0.804) 본다. 문장을 만들 때 아직 없는 뒷말을 보면 안 되기 때문이다 (decoder, GPT).", "Mask on: token 1 sees only itself (1.000), token 2 only tokens 1–2 (0.196, 0.804). When writing a sentence it must not peek at words that do not exist yet (decoder, GPT).") : tr("mask 끔: 모두가 모두를 본다 (encoder, BERT, ViT).", "Mask off: everyone sees everyone (encoder, BERT, ViT)."); }
bindSeg("g6m", v => { S6.m = v; upd6(); });

/* ================= 7절 / Section 7 ================= */
// ViT: 28×28을 7×7 patch 16개로, CLS가 각 patch를 보는 attention
// ViT: 28×28 cut into 16 patches of 7×7; the attention CLS pays to each patch
const VT = D14.vit, VX = Uint8Array.from(atob(VT.x), ch => ch.charCodeAt(0)), S7 = { i: 3, l: 1, h: 0 };
const F7 = fig("cv7", w => (w >= 640 ? 260 : 480), (ctx, w, h, c) => {
  const wide = w >= 640, cell = 7, img = Array.from(VX.subarray(S7.i * 784, (S7.i + 1) * 784), v => v / 255), x0 = 16, y0 = 34;
  const att = S7.h === 0 ? VT.att[S7.i][S7.l].reduce((s, r) => s.map((v, k) => v + r[k] / 4), new Array(16).fill(0)) : VT.att[S7.i][S7.l][S7.h - 1], mx = Math.max(...att);
  panelTitle(ctx, c, tr(`숫자 ${VT.y[S7.i]} → patch 16개 (7×7)`, `digit ${VT.y[S7.i]} → 16 patches (7×7)`), x0, y0 - 12);
  drawImg(ctx, c, img, 28, 28, x0, y0, cell);
  ctx.strokeStyle = c.sig; ctx.lineWidth = 1.2; for (let k = 1; k < 4; k++) { line(ctx, [[x0 + k * 49, y0], [x0 + k * 49, y0 + 196]], c.sig, 1.2); line(ctx, [[x0, y0 + k * 49], [x0 + 196, y0 + k * 49]], c.sig, 1.2); }
  // 오른쪽: CLS attention을 patch 자리에 칠한 것
  // Right: the CLS attention painted onto the patch positions
  const ax = wide ? x0 + 196 + 50 : x0, ay = wide ? y0 : y0 + 196 + 50;
  panelTitle(ctx, c, tr(`CLS가 patch를 보는 비율 (${S7.l + 1}층, ${S7.h ? "head " + S7.h : "head 4개 평균"})`, `How much CLS looks at each patch (layer ${S7.l + 1}, ${S7.h ? "head " + S7.h : "mean of 4 heads"})`), ax, ay - 12);
  drawImg(ctx, c, img, 28, 28, ax, ay, cell, { frame: true });
  for (let k = 0; k < 16; k++) { const r = Math.floor(k / 4), q = k % 4; ctx.fillStyle = c.miss; ctx.globalAlpha = 0.75 * att[k] / mx; ctx.fillRect(ax + q * 49, ay + r * 49, 49, 49); ctx.globalAlpha = 1; txt(ctx, Math.round(att[k] * 100) + "%", ax + q * 49 + 24.5, ay + r * 49 + 29, { color: c.ink, size: 10.5, mono: true, align: "center", w: "600" }); }
  // 토큰 줄: CLS + patch 16개
  // Token row: CLS + 16 patches
  if (wide) {
    const tx = ax + 196 + 60, tw = (w - tx - 16) / 17;
    txt(ctx, tr("token 17개 = CLS + patch 16", "17 tokens = CLS + 16 patches"), tx, ay + 70, { color: c.ink, size: 11.5, w: "700" });
    for (let k = 0; k < 17; k++) { rrect(ctx, tx + k * tw + 1, ay + 80, tw - 2, 36, 3, k ? c.sigwash : c.okwash, k ? c.sig : c.ok, 1); if (tw > 14) txt(ctx, k ? String(k) : "C", tx + k * tw + tw / 2, ay + 102, { color: c.ink, size: 9.5, mono: true, align: "center" }); }
  }
});
function upd7() { paint(F7); $("r7").textContent = tr(`ViT의 답: ${VT.pred[S7.i]} (정답 ${VT.y[S7.i]}). ViT d 64, head 4, 2층, parameter ${commas(VT.params)}개, MNIST 시험 정확도 ${(VT.acc * 100).toFixed(1)}% (4 epoch).\n11장의 CNN은 parameter ${"11,274"}개로 98.6%: 작은 data에서는 "가까운 칸끼리"라는 가정이 있는 CNN이 유리하다.`, `ViT answer: ${VT.pred[S7.i]} (truth ${VT.y[S7.i]}). ViT d 64, 4 heads, 2 layers, ${commas(VT.params)} parameters, MNIST test accuracy ${(VT.acc * 100).toFixed(1)}% (4 epochs).\nChapter 11's CNN: 98.6% with 11,274 parameters. On small data the CNN, with its built-in "nearby cells" assumption, has the edge.`); }
$("b7next").addEventListener("click", () => { S7.i = (S7.i + 1) % 10; upd7(); });
bindSeg("g7l", v => { S7.l = +v; upd7(); });
bindSeg("g7h", v => { S7.h = +v; upd7(); });

/* ================= 8절 / Section 8 ================= */
// LLM: 다음 token 확률 (GPT-2 small). 온도는 위 50개 안에서 다시 계산
// LLM: next-token probabilities (GPT-2 small). Temperature is recomputed within the top 50
const S8 = { p: 1, t: 1 };
const F8 = fig("cv8", 270, (ctx, w, h, c) => {
  const N = G2.next[S8.p], lo = N.logits.map(v => v / S8.t), mx = Math.max(...lo), e = lo.map(v => Math.exp(v - mx)), s = e.reduce((a, b) => a + b, 0), p = e.map(v => (v / s) * N.p50);
  const B = { l: 110, t: 30, w: w - 170, h: h - 44 }, bh = B.h / 10;
  panelTitle(ctx, c, tr(`"${N.prompt}" 다음 token, 위 10개 (온도 ${fx(S8.t, 2)})`, `Next token after "${N.prompt}", top 10 (temperature ${fx(S8.t, 2)})`), 16, 16);
  for (let k = 0; k < 10; k++) {
    const y = B.t + k * bh, wd = Math.min(1, p[k] / Math.max(0.3, p[0])) * B.w;
    txt(ctx, tokLab(N.pieces[k]), B.l - 8, y + bh / 2 + 4, { color: c.ink, size: 12, mono: true, align: "right", w: "600" });
    ctx.fillStyle = k ? c.sig : c.miss; ctx.globalAlpha = 0.8; ctx.fillRect(B.l, y + 3, wd, bh - 6); ctx.globalAlpha = 1;
    txt(ctx, (p[k] * 100).toFixed(1) + "%", B.l + wd + 6, y + bh / 2 + 4, { color: c.ink, size: 11, mono: true });
  }
});
function upd8() {
  paint(F8);
  const N = G2.next[S8.p];
  $("r8").textContent = tr(`온도 1에서의 진짜 확률: ${N.pieces.slice(0, 3).map((t, k) => `"${tokLab(t)}" ${(N.p[k] * 100).toFixed(1)}%`).join(", ")}. 위 50개의 확률 합 ${(N.p50 * 100).toFixed(1)}%.\n온도를 바꾼 막대는 위 50개 안에서 다시 나눈 근사다. 온도가 낮으면 1등에 몰리고, 높으면 고르게 퍼진다.`,
    `True probabilities at temperature 1: ${N.pieces.slice(0, 3).map((t, k) => `"${tokLab(t)}" ${(N.p[k] * 100).toFixed(1)}%`).join(", ")}. The top 50 together hold ${(N.p50 * 100).toFixed(1)}%.\nWith other temperatures the bars are an approximation re-split within the top 50. Low temperature piles onto the winner; high temperature spreads out.`);
}
bindSeg("g8p", v => { S8.p = +v; upd8(); });
bindRange("i8t", "o8t", v => fx(v, 2), v => { S8.t = v; upd8(); });
// token 나누기: 영어와 한국어
// Tokenising: English and Korean
const F8b = fig("cv8b", w => (w >= 640 ? 170 : 300), (ctx, w, h, c) => {
  let y = 22;
  [["en", c.sig], ["ko", c.miss]].forEach(([k, col]) => {
    const T = G2.tokens[k];
    txt(ctx, tr(`${T.chars}글자 → token ${T.ids.length}개: "${T.text}"`, `${T.chars} characters → ${T.ids.length} tokens: "${T.text}"`), 16, y, { color: c.ink, size: 12, w: "700" }); y += 12;
    let x = 16;
    T.pieces.forEach((pc, j) => {
      const lab = pc === "�" || pc === " �" ? "▢" : tokLab(pc), wd = textW(ctx, lab, 12, "400", true) + 10;
      if (x + wd > w - 16) { x = 16; y += 30; }
      rrect(ctx, x, y, wd, 24, 4, j % 2 ? c.bg : c.wash, col, 1); txt(ctx, lab, x + wd / 2, y + 16, { color: c.ink, size: 12, mono: true, align: "center" }); x += wd + 4;
    });
    y += 50;
  });
});

// 처음 그리기
// First paint
upd1(); upd2(); upd3(); upd4(); upd6(); upd7(); upd8(); paint(F8b);
repaintAll();
