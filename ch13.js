/* 13장 그림과 조작. 한글판(ch13-rnn-KR.html)과 영어판(ch13-rnn.html)이 같이 쓴다
   Chapter 13 figures and interactions, shared by the Korean and English pages
   실험 결과와 글자 LSTM weight는 assets/data/ch13_data.js (tools/ch13_numbers.py가 만든다)
   Experiment results and the character-LSTM weights live in assets/data/ch13_data.js (made by tools/ch13_numbers.py) */
"use strict";

const D13 = CH13_DATA;
const b64bytes = s => Uint8Array.from(atob(s), ch => ch.charCodeAt(0));
function deq(d) { const q = new Int8Array(b64bytes(d.q).buffer), [r, c] = d.shape; return Array.from({ length: r }, (_, i) => Array.from({ length: c }, (_, j) => q[i * c + j] * d.s)); }
const sigm = v => 1 / (1 + Math.exp(-v));

/* ================= 1절 / Section 1 ================= */
// 순서 섞기: 값은 같고(histogram 그대로) 순서만 다르면 신호가 사라진다
// Shuffling: same values (same histogram), different order, and the signal is gone
const SER = Array.from({ length: 60 }, (_, t) => Math.sin(t * 0.32) + 0.5 * Math.sin(t * 0.11 + 1));
const S1 = { s: SER.slice(), shuf: false };
const F1 = fig("cv1", w => (w >= 600 ? 220 : 400), (ctx, w, h, c) => {
  const wide = w >= 600, B = wide ? { l: 40, t: 30, w: w * 0.62 - 50, h: h - 64 } : { l: 40, t: 30, w: w - 56, h: 170 };
  const A = axes(ctx, c, B, [0, 59], [-1.7, 1.7], { xt: [0, 20, 40, 59].map(v => [v, String(v)]), yt: [-1, 0, 1], xlabel: tr("시간 t", "time t") });
  panelTitle(ctx, c, S1.shuf ? tr("순서를 섞은 것", "order shuffled") : tr("원래 순서", "original order"), B.l, 16);
  line(ctx, S1.s.map((v, t) => [A.X(t), A.Y(v)]), S1.shuf ? c.fa : c.sig, 2); S1.s.forEach((v, t) => dot(ctx, A.X(t), A.Y(v), 2.4, S1.shuf ? c.fa : c.sig));
  // 오른쪽: 값의 histogram (섞어도 같다)
  // Right: histogram of the values (unchanged by shuffling)
  const H = wide ? { l: w * 0.62 + 40, t: 30, w: w * 0.38 - 56, h: h - 64 } : { l: 40, t: B.t + B.h + 70, w: w - 56, h: 120 }, bins = new Array(10).fill(0);
  S1.s.forEach(v => bins[clamp(Math.floor((v + 1.6) / 3.2 * 10), 0, 9)]++);
  const A2 = axes(ctx, c, H, [-1.6, 1.6], [0, Math.max(...bins) + 1], { xt: [-1, 0, 1], yt: [], xlabel: tr("값", "value") });
  panelTitle(ctx, c, tr("값의 분포 (섞어도 같다)", "distribution of values (same after shuffling)"), H.l, H.t - 14);
  bins.forEach((n, k) => { const x0 = A2.X(-1.6 + k * 0.32), x1 = A2.X(-1.6 + (k + 1) * 0.32); ctx.fillStyle = c.vio; ctx.globalAlpha = 0.6; ctx.fillRect(x0 + 1, A2.Y(n), x1 - x0 - 2, A2.Y(0) - A2.Y(n)); ctx.globalAlpha = 1; });
});
function upd1() { paint(F1); $("r1").textContent = S1.shuf ? tr("값 60개는 그대로인데 다음 값을 짐작할 수 없게 되었다. 정보는 값이 아니라 순서에 있었다.", "The same 60 values, but the next value can no longer be guessed. The information was in the order, not in the values.") : tr("다음 값은 지금까지의 흐름을 보면 짐작할 수 있다. 같은 값들의 순서만 섞어 보자.", "The next value can be guessed from the flow so far. Now shuffle the order of the same values."); }
$("b1shuf").addEventListener("click", () => { S1.s = shuffled(60, makeRng(newSeed())).map(i => SER[i]); S1.shuf = true; upd1(); });
$("b1back").addEventListener("click", () => { S1.s = SER.slice(); S1.shuf = false; upd1(); });

/* ================= 2절 / Section 2 ================= */
// 펼친 RNN: 숫자 하나짜리 hidden state h_t = tanh(w_x x_t + w_h h_{t−1})
// The unrolled RNN: a one-number hidden state h_t = tanh(w_x x_t + w_h h_{t−1})
const S2 = { x: [1, 0, 0, 1, 0, 0, 0, 0], wx: 0.5, wh: 0.8, n: 4, lay: null, timer: 0 };
const run2 = () => { const hs = []; let h = 0; S2.x.forEach(x => { h = Math.tanh(S2.wx * x + S2.wh * h); hs.push(h); }); return hs; };
const F2 = fig("cv2", w => (w >= 640 ? 330 : 420), (ctx, w, h, c) => {
  const T = S2.x.length, hs = run2(), cw = (w - 40) / T, bs = Math.min(54, cw - 14), y0 = 60;
  S2.lay = { cw, y0, bs };
  txt(ctx, tr("x (누르면 0 ↔ 1)", "x (click: 0 ↔ 1)"), 16, 18, { color: c.sig, size: 11.5, w: "700" });
  for (let t = 0; t < T; t++) {
    const cx = 20 + t * cw + cw / 2, on = t < S2.n, hc = on ? (hs[t] >= 0 ? c.sig : c.fa) : c.line;
    // 입력 칸 / input cell
    rrect(ctx, cx - 14, 26, 28, 24, 5, S2.x[t] ? c.sigwash : c.bg, c.sig, 1.2); txt(ctx, String(S2.x[t]), cx, 43, { color: c.ink, size: 13, mono: true, align: "center", w: "700" });
    arrowS(ctx, cx, 52, cx, y0 - 2, c.muted, 1.2, 5);
    // hidden 상자 / hidden box
    ctx.fillStyle = on ? cssRGB(mixRGB(rgb(c.bg), rgb(hc), Math.min(1, Math.abs(hs[t])) * 0.7)) : c.bg; ctx.fillRect(cx - bs / 2, y0, bs, bs);
    ctx.strokeStyle = t === S2.n - 1 ? c.miss : c.ink; ctx.lineWidth = t === S2.n - 1 ? 2.6 : 1.2; ctx.strokeRect(cx - bs / 2, y0, bs, bs);
    txt(ctx, `h${SUB(t + 1)}`, cx, y0 + 16, { color: c.muted, size: 10.5, align: "center" });
    if (on) txt(ctx, fx(hs[t], 3), cx, y0 + bs / 2 + 12, { color: c.ink, size: bs > 44 ? 12 : 10, mono: true, align: "center", w: "700" });
    if (t < T - 1) arrowS(ctx, cx + bs / 2 + 1, y0 + bs / 2, cx + cw - bs / 2 - 2, y0 + bs / 2, t < S2.n - 1 ? c.ok : c.line, 1.6, 6);
  }
  txt(ctx, `× w_h = ${fx(S2.wh, 2)}`, 20 + cw, y0 + bs + 16, { color: c.ok, size: 11, mono: true, align: "center" });
  // 아래: h_t 곡선
  // Below: the h_t curve
  const B = { l: 40, t: y0 + bs + 44, w: w - 56, h: h - (y0 + bs + 44) - 30 };
  const A = axes(ctx, c, B, [0.5, T + 0.5], [-1, 1], { xt: Array.from({ length: T }, (_, t) => [t + 1, String(t + 1)]), yt: [-1, 0, 1], grid: true });
  line(ctx, hs.slice(0, S2.n).map((v, t) => [A.X(t + 1), A.Y(v)]), c.miss, 2); hs.slice(0, S2.n).forEach((v, t) => dot(ctx, A.X(t + 1), A.Y(v), 3.2, c.miss));
  txt(ctx, "h_t", B.l + 6, B.t - 6, { color: c.miss, size: 11, w: "700" });
});
function upd2() {
  paint(F2);
  const hs = run2(), t = S2.n - 1, hp = t ? hs[t - 1] : 0;
  $("r2").textContent = tr(`h${SUB(t + 1)} = tanh(${fx(S2.wx, 2)} × ${S2.x[t]} + ${fx(S2.wh, 2)} × ${fx(hp, 4)}) = tanh(${fx(S2.wx * S2.x[t] + S2.wh * hp, 4)}) = ${fx(hs[t], 4)}\n같은 w_x, w_h를 모든 칸에서 쓴다. w_h가 0이면 앞을 잊고, 1에 가까우면 오래 기억한다.`,
    `h${SUB(t + 1)} = tanh(${fx(S2.wx, 2)} × ${S2.x[t]} + ${fx(S2.wh, 2)} × ${fx(hp, 4)}) = tanh(${fx(S2.wx * S2.x[t] + S2.wh * hp, 4)}) = ${fx(hs[t], 4)}\nThe same w_x and w_h are used at every step. With w_h at 0 it forgets the past; near 1 it remembers long.`);
}
F2.cv.addEventListener("click", ev => {
  const L = S2.lay; if (!L) return; const r = F2.cv.getBoundingClientRect(), px = ev.clientX - r.left, py = ev.clientY - r.top;
  if (py < 24 || py > 52) return; const t = Math.floor((px - 20) / L.cw); if (t >= 0 && t < S2.x.length) { S2.x[t] ^= 1; upd2(); }
});
bindRange("i2x", "o2x", v => fx(v, 2), v => { S2.wx = v; upd2(); });
bindRange("i2h", "o2h", v => fx(v, 2), v => { S2.wh = v; upd2(); });
$("b2step").addEventListener("click", () => { stopRun(S2); S2.n = S2.n >= S2.x.length ? 1 : S2.n + 1; upd2(); });
$("b2play").addEventListener("click", () => { S2.n = 1; upd2(); runSteps(S2, S2.x.length - 1, () => { S2.n++; upd2(); }, 450); });

/* ================= 3절 / Section 3 ================= */
// nn.RNN의 shape: 입력 (N, L, H_in) → 출력 (N, L, H), h_n (층, N, H)
// Shapes of nn.RNN: input (N, L, H_in) → output (N, L, H), h_n (layers, N, H)
const S3 = { N: 3, L: 5, Hin: 8, H: 16, nl: 2 };
const pRNN = (k, Hin, H, nl) => { let s = 0; for (let l = 0; l < nl; l++) s += k * (H * (l ? H : Hin) + H * H + 2 * H); return s; };
function box3(ctx, c, x, y, dims, labels, col, title, sc) {
  // 상자 (깊이 = 첫 차원) / a box (depth = first dimension)
  const [a, b, d] = dims, wd = Math.max(8, d * sc), ht = Math.max(8, b * sc), off = Math.min(5, 26 / a);
  for (let k = a - 1; k >= 0; k--) { ctx.fillStyle = k ? c.bg : c.wash; ctx.fillRect(x + k * off, y - k * off, wd, ht); ctx.strokeStyle = col; ctx.lineWidth = 1.2; ctx.strokeRect(x + k * off + 0.5, y - k * off + 0.5, wd, ht); }
  txt(ctx, title, x, y - a * off - 8, { color: col, size: 12, w: "700" });
  txt(ctx, `(${dims.join(", ")})`, x, y + ht + 16, { color: c.ink, size: 11.5, mono: true, w: "600" });
  txt(ctx, labels, x, y + ht + 31, { color: c.muted, size: 10.5 });
  return wd + (a - 1) * off;
}
const F3 = fig("cv3", w => (w >= 640 ? 230 : 470), (ctx, w, h, c) => {
  const { N, L, Hin, H, nl } = S3, wide = w >= 640, sc = wide ? 4.2 : 3.6;
  if (wide) {
    let x = 20; const y = 90;
    x += box3(ctx, c, x, y, [N, L, Hin], tr("batch, 길이, 입력 크기", "batch, length, input size"), c.sig, tr("입력 x", "input x"), sc) + 70;
    arrowS(ctx, x - 62, y + 20, x - 14, y + 20, c.muted, 1.4, 6); txt(ctx, `nn.RNN × ${nl}`, x - 38, y + 8, { color: c.muted, size: 10.5, align: "center" });
    x += box3(ctx, c, x, y, [N, L, H], tr("모든 칸의 마지막 층 h", "last layer's h at every step"), c.ok, tr("출력", "output"), sc) + 50;
    box3(ctx, c, x, y, [nl, N, H], tr("층마다 마지막 칸의 h", "each layer's h at the last step"), c.miss, "h_n", sc);
  } else {
    box3(ctx, c, 20, 70, [N, L, Hin], tr("batch, 길이, 입력 크기", "batch, length, input size"), c.sig, tr("입력 x", "input x"), sc);
    box3(ctx, c, 20, 220, [N, L, H], tr("모든 칸의 마지막 층 h", "last layer's h at every step"), c.ok, tr("출력", "output"), sc);
    box3(ctx, c, 20, 370, [nl, N, H], tr("층마다 마지막 칸의 h", "each layer's h at the last step"), c.miss, "h_n", sc);
  }
});
function upd3() {
  paint(F3);
  const { Hin, H, nl } = S3;
  $("r3").textContent = tr(`parameter 수 (입력 ${Hin}, hidden ${H}, ${nl}층): RNN ${commas(pRNN(1, Hin, H, nl))}, GRU ${commas(pRNN(3, Hin, H, nl))}, LSTM ${commas(pRNN(4, Hin, H, nl))}\n한 층 = H·H_in + H·H + 2H (RNN). GRU는 gate가 3묶음이라 3배, LSTM은 4묶음이라 4배. 둘째 층부터는 H_in = H.`,
    `parameters (input ${Hin}, hidden ${H}, ${nl} layer${nl > 1 ? "s" : ""}): RNN ${commas(pRNN(1, Hin, H, nl))}, GRU ${commas(pRNN(3, Hin, H, nl))}, LSTM ${commas(pRNN(4, Hin, H, nl))}\nOne layer = H·H_in + H·H + 2H (RNN). A GRU has 3 sets of gates, so 3×; an LSTM has 4, so 4×. From the second layer on, H_in = H.`);
}
[["i3N", "N"], ["i3L", "L"], ["i3i", "Hin"], ["i3H", "H"], ["i3n", "nl"]].forEach(([id, k]) => bindRange(id, id.replace(/^i/, "o"), v => String(v), v => { S3[k] = v; upd3(); }));

/* ================= 4절 / Section 4 ================= */
// sine 외삽: 200칸으로 학습, 그 뒤 150칸은 자기 예측을 다시 넣어 이어 그린다
// Sine extrapolation: trained on 200 steps; the next 150 are drawn by feeding its own predictions back
const SN = D13.sine, S4 = { H: "4" };
const F4 = fig("cv4", 270, (ctx, w, h, c) => {
  const R = SN.runs[S4.H], B = { l: 40, t: 30, w: w - 56, h: h - 70 }, A = axes(ctx, c, B, [120, 350], [-1.6, 1.6], { xt: [150, 200, 250, 300, 350], yt: [-1, 0, 1], xlabel: tr("시간 t", "time t") });
  ctx.fillStyle = c.sigwash; ctx.globalAlpha = 0.6; ctx.fillRect(A.X(120), B.t, A.X(200) - A.X(120), B.h); ctx.globalAlpha = 1;
  txt(ctx, tr("학습 구간 (0~199)", "training range (0–199)"), A.X(124), B.t + 14, { color: c.sig, size: 11, w: "700" });
  txt(ctx, tr("여기부터 자기 예측만으로", "from here on, its own predictions only"), A.X(204), B.t + 14, { color: c.miss, size: 11, w: "700" });
  line(ctx, SN.true.slice(120).map((v, i) => [A.X(120 + i), A.Y(v)]), c.muted, 1.4, [5, 4]);
  line(ctx, R.pred.map((v, i) => [A.X(200 + i), A.Y(v)]), c.miss, 2.2);
  panelTitle(ctx, c, tr(`점선 = 진짜 sin(0.3t), 주황 = hidden ${S4.H}개 RNN`, `dashed = true sin(0.3t), orange = RNN with hidden ${S4.H}`), B.l, 16);
});
function upd4() { paint(F4); const R = SN.runs[S4.H]; $("r4").textContent = tr(`hidden ${S4.H}개 (parameter ${R.params}개): 150칸 동안의 RMSE ${fx(R.rmse, 3)}\nhidden이 1개면 진동을 만들 수 없다: 흔들리려면 상태가 적어도 2개(위치와 속도처럼) 있어야 한다.`, `hidden ${S4.H} (${R.params} parameters): RMSE over the 150 steps ${fx(R.rmse, 3)}\nWith 1 hidden unit it cannot oscillate: swinging needs at least 2 states (like position and velocity).`); }
bindSeg("g4h", v => { S4.H = v; upd4(); });

/* ================= 5절 / Section 5 ================= */
// 글자 단위 LSTM: 글자 embedding의 2D 지도와 browser 안에서 글 만들기
// Character-level LSTM: a 2-D map of the character embeddings, and generating text in the browser
const CH = D13.char, VOC = CH.voc, EMB = deq(CH.emb), WIH = deq(CH.Wih), WHH = deq(CH.Whh), WO = deq(CH.Wo), HID = 64;
const S5 = { temp: 0.8, seed: 1 };
function generate(n, temp, seed) {
  const r = makeRng(seed); let h = new Array(HID).fill(0), cc = new Array(HID).fill(0), k = VOC.indexOf("l"), s = "l";
  for (let step = 0; step < n; step++) {
    const x = EMB[k], z = CH.b.map((b, j) => b + WIH[j].reduce((t, w, i) => t + w * x[i], 0) + WHH[j].reduce((t, w, i) => t + w * h[i], 0));
    // PyTorch LSTM gate 순서: i, f, g, o
    // PyTorch LSTM gate order: i, f, g, o
    cc = cc.map((v, j) => sigm(z[HID + j]) * v + sigm(z[j]) * Math.tanh(z[2 * HID + j]));
    h = cc.map((v, j) => sigm(z[3 * HID + j]) * Math.tanh(v));
    const lo = CH.bo.map((b, q) => (b + WO[q].reduce((t, w, i) => t + w * h[i], 0)) / temp), p = softmaxV(lo);
    let u = r.u(); k = p.findIndex(v => (u -= v) < 0); if (k < 0) k = p.length - 1; s += VOC[k];
  }
  return s;
}
const VOW = "aeiou";
const F5 = fig("cv5", w => (w >= 600 ? 300 : 340), (ctx, w, h, c) => {
  const P = CH.emb2d, xs = P.map(p => p[0]), ys = P.map(p => p[1]), pad = 0.15, x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const B = { l: 20, t: 30, w: w - 40, h: h - 50 }, X = v => B.l + (v - x0 + pad * (x1 - x0)) / ((1 + 2 * pad) * (x1 - x0)) * B.w, Y = v => B.t + B.h - (v - y0 + pad * (y1 - y0)) / ((1 + 2 * pad) * (y1 - y0)) * B.h;
  rrect(ctx, B.l, B.t, B.w, B.h, 6, c.bg, c.line);
  panelTitle(ctx, c, tr(`글자 ${VOC.length}개의 embedding (8차원 → 2D로 PCA). 파랑 = 모음, 회색 = 자음, 주황 = 띄어쓰기·문장부호`, `Embeddings of the ${VOC.length} characters (8-D → 2-D by PCA). Blue = vowels, grey = consonants, orange = space and punctuation`), B.l, 16);
  [...VOC].forEach((ch, k) => {
    const col = VOW.includes(ch) ? c.sig : /[a-z]/.test(ch) ? c.muted : c.miss;
    txtBg(ctx, c, ch === " " ? "␣" : ch, X(P[k][0]), Y(P[k][1]) + 5, { color: col, size: 15, w: "700", align: "center", mono: true });
  });
});
function upd5() {
  paint(F5);
  $("r5").textContent = tr(`온도 ${fx(S5.temp, 2)}로 만든 200글자 (browser 안의 LSTM, int8 weight):\n`, `200 characters generated at temperature ${fx(S5.temp, 2)} (the LSTM in the browser, int8 weights):\n`) + generate(200, S5.temp, S5.seed);
}
bindRange("i5t", "o5t", v => fx(v, 2), v => { S5.temp = v; upd5(); });
$("b5gen").addEventListener("click", () => { S5.seed++; upd5(); });
// 학습 단계별로 만든 글 (Python, 온도 0.8)
// Text generated at each training stage (Python, temperature 0.8)
$("t5").innerHTML = Object.entries(CH.snaps).map(([st, v]) => `<tr><td class="num">${st}</td><td class="num">${v.loss.toFixed(3)}</td><td><code>${v.text.slice(0, 110).replace(/&/g, "&amp;").replace(/</g, "&lt;")}…</code></td></tr>`).join("");

/* ================= 6절 / Section 6 ================= */
// LSTM gate slider: 2칸째에 +1, 14칸째에 −1을 넣었을 때 cell 상태 c_t와 출력 h_t
// LSTM gate sliders: cell state c_t and output h_t when +1 goes in at step 2 and −1 at step 14
const S6 = { f: 0.9, i: 1, o: 1 };
const G6 = t => (t === 2 ? 1 : t === 14 ? -1 : 0);
const F6 = fig("cv6", 260, (ctx, w, h, c) => {
  const T = 30, cs = [], hs = []; let cc = 0;
  for (let t = 1; t <= T; t++) { cc = S6.f * cc + S6.i * G6(t); cs.push(cc); hs.push(S6.o * Math.tanh(cc)); }
  const B = { l: 44, t: 30, w: w - 60, h: h - 70 }, A = axes(ctx, c, B, [0.5, T + 0.5], [-1.6, 1.6], { xt: [1, 5, 10, 15, 20, 25, 30], yt: [-1, 0, 1], xlabel: tr("시간 t", "time t") });
  [2, 14].forEach(t => { line(ctx, [[A.X(t), B.t], [A.X(t), B.t + B.h]], c.line, 1, [3, 3]); txt(ctx, t === 2 ? tr("+1 넣음", "+1 in") : tr("−1 넣음", "−1 in"), A.X(t) + 4, B.t + 12, { color: c.muted, size: 10.5 }); });
  line(ctx, cs.map((v, t) => [A.X(t + 1), A.Y(v)]), c.sig, 2.2); line(ctx, hs.map((v, t) => [A.X(t + 1), A.Y(v)]), c.miss, 2, [6, 3]);
  panelTitle(ctx, c, tr("파랑 = cell 상태 c_t (기억), 주황 점선 = 출력 h_t", "blue = cell state c_t (memory), orange dashed = output h_t"), B.l, 16);
});
function upd6() {
  paint(F6);
  const half = S6.f > 0 && S6.f < 1 ? Math.log(0.5) / Math.log(S6.f) : null;
  $("r6").textContent = tr(`forget ${fx(S6.f, 2)}, input ${fx(S6.i, 2)}, output ${fx(S6.o, 2)}. ` + (S6.f >= 1 ? "forget = 1: 한 번 넣은 것을 영원히 기억한다." : S6.f <= 0 ? "forget = 0: 바로 잊는다." : `기억이 반으로 줄어드는 데 ${fx(half, 1)}칸 (f^k = 0.5).`) + "\ninput은 새것을 얼마나 넣을지, output은 기억을 얼마나 내보일지. 진짜 LSTM에서는 세 gate가 숫자가 아니라 입력과 h로 그때그때 계산된다.",
    `forget ${fx(S6.f, 2)}, input ${fx(S6.i, 2)}, output ${fx(S6.o, 2)}. ` + (S6.f >= 1 ? "forget = 1: what goes in is remembered forever." : S6.f <= 0 ? "forget = 0: forgotten at once." : `Memory halves after ${fx(half, 1)} steps (f^k = 0.5).`) + "\nInput sets how much new goes in, output how much of the memory is shown. In a real LSTM the three gates are not fixed numbers but computed at every step from the input and h.");
}
bindRange("i6f", "o6f", v => fx(v, 2), v => { S6.f = v; upd6(); });
bindRange("i6i", "o6i", v => fx(v, 2), v => { S6.i = v; upd6(); });
bindRange("i6o", "o6o", v => fx(v, 2), v => { S6.o = v; upd6(); });

/* ================= 7절 / Section 7 ================= */
// 거리에 따른 gradient (학습 전), 긴 기억 실험 (학습 뒤)
// Gradient against distance (before training), and the long-memory experiment (after training)
const GR = D13.grads, MEM = D13.memory;
const F7 = fig("cv7", 270, (ctx, w, h, c) => {
  const B = { l: 58, t: 30, w: w - 76, h: h - 74 }, A = axes(ctx, c, B, [0, 99], [-26, 1], { xt: [0, 20, 40, 60, 80, 99], yt: [0, -5, -10, -15, -20, -25].map(v => [v, v ? "10" + SUP(v) : "1"]), xlabel: tr("거리 T − t (몇 칸 앞의 입력인가)", "distance T − t (how many steps back the input is)") });
  panelTitle(ctx, c, tr("마지막 출력에 대한 t번째 입력의 gradient 크기 (log, 학습 전)", "Size of the gradient of the last output w.r.t. input t (log, before training)"), B.l, 16);
  [[GR.RNN, c.fa, "RNN"], [GR.LSTM, c.sig, tr("LSTM (기본값)", "LSTM (default)")], [GR.LSTM_f3, c.ok, tr("LSTM, forget bias 3", "LSTM, forget bias 3")]].forEach(([g, col, lab], k) => {
    line(ctx, g.map((v, d) => [A.X(d), A.Y(Math.max(-26, v))]), col, 2.2);
    txtBg(ctx, c, lab, A.X(k === 2 ? 70 : 30 + 22 * k), A.Y(Math.max(-26, g[k === 2 ? 70 : 30 + 22 * k])) + (k === 2 ? -10 : 16), { color: col, size: 11.5, w: "700" });
  });
});
const F7b = fig("cv7b", 260, (ctx, w, h, c) => {
  const B = { l: 52, t: 30, w: w - 70, h: h - 74 }, n = MEM.T.length, A = axes(ctx, c, B, [-0.5, n - 0.5], [0.4, 1.02], { xt: MEM.T.map((T, k) => [k, String(T)]), yt: [[0.5, "50%"], [0.75, "75%"], [1, "100%"]], xlabel: tr("기억해야 하는 거리 T", "distance to remember, T") });
  line(ctx, [[B.l, A.Y(0.5)], [B.l + B.w, A.Y(0.5)]], c.muted, 1, [4, 4]); txt(ctx, tr("찍기 (50%)", "guessing (50%)"), B.l + B.w - 4, A.Y(0.5) - 5, { color: c.muted, size: 10.5, align: "right" });
  panelTitle(ctx, c, tr("첫 글자 기억하기: seed 3개의 시험 정확도 (점) — 2,000번 학습", "Recall the first bit: test accuracy of 3 seeds (dots) — 2,000 training steps"), B.l, 16);
  [["RNN", c.fa, -0.2], ["LSTM", c.sig, 0], ["LSTM_f3", c.ok, 0.2]].forEach(([k, col, dx]) => MEM[k].forEach((accs, i) => accs.forEach(a => dot(ctx, A.X(i + dx), A.Y(a), 4, col, c.bg))));
  [["RNN", c.fa], [tr("LSTM (forget bias 1)", "LSTM (forget bias 1)"), c.sig], ["LSTM (forget bias 3)", c.ok]].forEach(([lab, col], k) => { const y = A.Y(0.8) + k * 16; dot(ctx, B.l + 12, y, 4, col); txt(ctx, lab, B.l + 22, y + 4, { color: col, size: 11, w: "600" }); });
});

// 처음 그리기
// First paint
upd1(); upd2(); upd3(); upd4(); upd5(); upd6(); paint(F7); paint(F7b);
repaintAll();
