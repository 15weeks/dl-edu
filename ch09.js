/* 9장 그림과 조작. 한글판(ch09-metrics-weights-KR.html)과 영어판(ch09-metrics-weights.html)이 같이 쓴다
   Chapter 9 figures and interactions, shared by the Korean and English pages */
"use strict";

/* ================= 1~2절 / Sections 1-2 ================= */
// 점수 분포 두 개(음성, 양성)와 문턱. 문턱보다 크면 "양성"이라고 답한다
// Two score distributions (negative, positive) and a threshold. Scores above it are called "positive"
const R1 = makeRng(19), NEG = Array.from({ length: 500 }, () => -1 + R1.n()), POS = Array.from({ length: 500 }, () => 1 + R1.n());
const S1 = { t: 0 };
function cm(t, pos = POS, neg = NEG) { const TP = pos.filter(s => s > t).length, FN = pos.length - TP, FP = neg.filter(s => s > t).length, TN = neg.length - FP; return { TP, FN, FP, TN }; }
function metrics(m) {
  const n = m.TP + m.FN + m.FP + m.TN, acc = (m.TP + m.TN) / n, prec = m.TP + m.FP ? m.TP / (m.TP + m.FP) : 0, rec = m.TP + m.FN ? m.TP / (m.TP + m.FN) : 0;
  return { acc, prec, rec, f1: prec + rec ? 2 * prec * rec / (prec + rec) : 0 };
}
function hist(vals, lo, hi, nb) { const h = new Array(nb).fill(0); vals.forEach(v => { const k = Math.floor((v - lo) / (hi - lo) * nb); if (k >= 0 && k < nb) h[k]++; }); return h; }
function drawCM(ctx, c, m, x, y, cs) {
  const cells = [[m.TP, "TP", c.ok, tr("맞게 양성", "true positive")], [m.FN, "FN", c.fa, tr("놓침", "missed")], [m.FP, "FP", c.miss, tr("헛경보", "false alarm")], [m.TN, "TN", c.sig, tr("맞게 음성", "true negative")]];
  txt(ctx, tr("model: 양성", "model: positive"), x + cs / 2, y - 8, { color: c.muted, size: 11, align: "center" }); txt(ctx, tr("model: 음성", "model: negative"), x + cs * 1.5, y - 8, { color: c.muted, size: 11, align: "center" });
  txt(ctx, tr("정답 양성", "actual positive"), x - 8, y + cs / 2 + 4, { color: c.muted, size: 11, align: "right" }); txt(ctx, tr("정답 음성", "actual negative"), x - 8, y + cs * 1.5 + 4, { color: c.muted, size: 11, align: "right" });
  [[0, 0, 0], [1, 0, 1], [0, 1, 2], [1, 1, 3]].forEach(([i, j, k]) => {
    const [v, ab, col, nm] = cells[k], X = x + i * cs, Y = y + j * cs;
    ctx.fillStyle = col; ctx.globalAlpha = 0.18; ctx.fillRect(X + 2, Y + 2, cs - 4, cs - 4); ctx.globalAlpha = 1; ctx.strokeStyle = col; ctx.lineWidth = 1.6; ctx.strokeRect(X + 2, Y + 2, cs - 4, cs - 4);
    txt(ctx, String(v), X + cs / 2, Y + cs / 2 + 2, { color: c.ink, size: 20, w: "700", align: "center", mono: true });
    txt(ctx, `${ab} · ${nm}`, X + cs / 2, Y + cs / 2 + 20, { color: col, size: 11, w: "700", align: "center" });
  });
}
const F1 = fig("cv1", w => (w >= 680 ? 300 : 600), (ctx, w, h, c) => {
  const wide = w >= 680, B = { l: 44, t: 30, w: wide ? w * 0.55 - 60 : w - 60, h: 220 }, lo = -4.5, hi = 4.5, nb = 45, hn = hist(NEG, lo, hi, nb), hp = hist(POS, lo, hi, nb), top = Math.max(...hn, ...hp) * 1.1;
  const A = axes(ctx, c, B, [lo, hi], [0, top], { xt: [-4, -2, 0, 2, 4], yt: ticksR(0, top, 3), xlabel: tr("model의 점수", "the model's score") });
  panelTitle(ctx, c, tr("음성 500개(파랑)와 양성 500개(주황)의 점수", "Scores of 500 negatives (blue) and 500 positives (orange)"), B.l, 16);
  const bw = A.X(lo + (hi - lo) / nb) - A.X(lo);
  for (let k = 0; k < nb; k++) {
    const x = A.X(lo + k * (hi - lo) / nb), right = lo + (k + 0.5) * (hi - lo) / nb > S1.t;
    ctx.globalAlpha = right ? 0.75 : 0.3; ctx.fillStyle = c.sig; ctx.fillRect(x, A.Y(hn[k]), bw - 1, B.t + B.h - A.Y(hn[k]));
    ctx.fillStyle = c.miss; ctx.fillRect(x + bw * 0.25, A.Y(hp[k]), bw * 0.5, B.t + B.h - A.Y(hp[k])); ctx.globalAlpha = 1;
  }
  line(ctx, [[A.X(S1.t), B.t], [A.X(S1.t), B.t + B.h]], c.ink, 2.2);
  txtBg(ctx, c, tr(`문턱 ${fx(S1.t)} → 오른쪽 = "양성"`, `threshold ${fx(S1.t)} → right = "positive"`), A.X(S1.t) + 6, B.t + 14, { color: c.ink, size: 11.5, w: "700" });
  const m = cm(S1.t), cs = wide ? 92 : 84, cx = wide ? B.l + B.w + 150 : 140, cy = wide ? 62 : B.t + B.h + 80;
  drawCM(ctx, c, m, cx, cy, cs);
  txt(ctx, "confusion matrix", cx + cs, cy - 28, { color: c.ink, size: 12.5, w: "700", align: "center" });
});
const F2 = fig("cv2", 270, (ctx, w, h, c) => {
  const B = { l: 44, t: 30, w: w - 60, h: h - 74 }, ts = Array.from({ length: 121 }, (_, i) => -3 + i * 0.05), M = ts.map(t => metrics(cm(t)));
  const A = axes(ctx, c, B, [-3, 3], [0, 1.02], { xt: [-3, -2, -1, 0, 1, 2, 3], yt: [[0, "0"], [0.5, "50%"], [1, "100%"]], xlabel: tr("문턱", "threshold") });
  panelTitle(ctx, c, tr("문턱에 따른 네 가지 숫자", "Four numbers as the threshold moves"), B.l, 16);
  [["acc", c.muted, "accuracy"], ["prec", c.miss, "precision"], ["rec", c.sig, "recall"], ["f1", c.ok, "F1"]].forEach(([k, col, nm], i) => {
    line(ctx, ts.map((t, j) => [A.X(t), A.Y(M[j][k])]), col, k === "f1" ? 2.8 : 2);
    txtBg(ctx, c, nm, B.l + B.w - 4, B.t + 14 + i * 16, { color: col, size: 11.5, w: "700", align: "right" });
  });
  line(ctx, [[A.X(S1.t), B.t], [A.X(S1.t), B.t + B.h]], c.ink, 1.4, [4, 3]);
});
function upd1() {
  paint(F1); paint(F2);
  const m = cm(S1.t), r = metrics(m);
  $("r1").textContent = tr(`TP ${m.TP}, FN ${m.FN}, FP ${m.FP}, TN ${m.TN}\naccuracy = (${m.TP} + ${m.TN}) / 1000 = ${fx(r.acc, 3)}   precision = ${m.TP} / (${m.TP} + ${m.FP}) = ${fx(r.prec, 3)}   recall = ${m.TP} / (${m.TP} + ${m.FN}) = ${fx(r.rec, 3)}   F1 = ${fx(r.f1, 3)}`,
    `TP ${m.TP}, FN ${m.FN}, FP ${m.FP}, TN ${m.TN}\naccuracy = (${m.TP} + ${m.TN}) / 1000 = ${fx(r.acc, 3)}   precision = ${m.TP} / (${m.TP} + ${m.FP}) = ${fx(r.prec, 3)}   recall = ${m.TP} / (${m.TP} + ${m.FN}) = ${fx(r.rec, 3)}   F1 = ${fx(r.f1, 3)}`);
}
bindRange("i1t", "o1t", v => fx(v), v => { S1.t = v; upd1(); });

/* ================= 3절 / Section 3 ================= */
// 양성이 드물어질 때: 같은 model(문턱 0)인데 precision이 무너진다
// When positives become rare: the same model (threshold 0), yet precision collapses
const S3 = { prev: 50 };
function m3() { const np = Math.max(1, Math.round(1000 * S3.prev / 100)), nn = 1000 - np, r = makeRng(23), pos = Array.from({ length: np }, () => 1 + r.n()), neg = Array.from({ length: nn }, () => -1 + r.n()); return { m: cm(0, pos, neg), np, nn }; }
const F3 = fig("cv3", 260, (ctx, w, h, c) => {
  const { m, nn } = m3(), r = metrics(m), base = nn / 1000, B = { l: 44, t: 30, w: w - 60, h: h - 70 };
  const A = axes(ctx, c, B, [0, 5], [0, 1.05], { xt: [], yt: [[0, "0"], [0.5, "50%"], [1, "100%"]] });
  panelTitle(ctx, c, tr(`양성 ${S3.prev}% (1000개 중 ${1000 - nn}개), 문턱 0`, `positives ${S3.prev}% (${1000 - nn} of 1000), threshold 0`), B.l, 16);
  [["accuracy", r.acc, c.muted], ["precision", r.prec, c.miss], ["recall", r.rec, c.sig], ["F1", r.f1, c.ok], [tr("\"모두 음성\" model의 accuracy", "accuracy of an \"all negative\" model"), base, c.line]].forEach(([nm, v, col], i) => {
    const x = A.X(i + 0.5), bw = B.w / 5 * 0.5, y = A.Y(v);
    ctx.fillStyle = col; ctx.globalAlpha = i === 4 ? 1 : 0.8; ctx.fillRect(x - bw / 2, y, bw, B.t + B.h - y); ctx.globalAlpha = 1;
    txt(ctx, Math.round(v * 100) + "%", x, y - 5, { color: c.ink, size: 11.5, w: "700", align: "center", mono: true });
    txt(ctx, nm, x, B.t + B.h + 15, { color: c.ink, size: i === 4 ? 9.5 : 11, align: "center" });
  });
});
function upd3() {
  paint(F3);
  const { m } = m3(), r = metrics(m);
  $("r3").textContent = tr(`TP ${m.TP}, FP ${m.FP}: "양성"이라고 한 ${m.TP + m.FP}개 중 진짜 양성은 ${m.TP}개 (precision ${Math.round(r.prec * 100)}%)\n→ ${S3.prev <= 5 ? "accuracy는 높아 보이지만 경보의 대부분이 헛경보다" : "양성이 흔하면 precision도 괜찮다"}`,
    `TP ${m.TP}, FP ${m.FP}: of the ${m.TP + m.FP} called "positive", ${m.TP} really are (precision ${Math.round(r.prec * 100)}%)\n→ ${S3.prev <= 5 ? "accuracy looks high, but most alarms are false alarms" : "when positives are common, precision is fine too"}`);
}
bindRange("i3p", "o3p", v => v + "%", v => { S3.prev = v; upd3(); });

/* ================= 5절 / Section 5 ================= */
// nn.Linear(4, 3)의 weight: 행 = neuron(출력), 열 = 입력
// The weight of nn.Linear(4, 3): rows = neurons (outputs), columns = inputs
const F5 = fig("cv5", 210, (ctx, w, h, c) => {
  const r = makeRng(5), M = Array.from({ length: 3 }, () => Array.from({ length: 4 }, () => +(r.n() * 0.5).toFixed(2))), x0 = 130, y0 = 50;
  panelTitle(ctx, c, tr("nn.Linear(4, 3).weight: shape (3, 4)", "nn.Linear(4, 3).weight: shape (3, 4)"), 16, 18);
  drawMat(ctx, c, M, x0, y0, { cw: 60, ch: 32, col: c.sig, hl: (i) => (i === 1 ? c.sigwash : false) });
  ["x₁", "x₂", "x₃", "x₄"].forEach((s, j) => txt(ctx, tr(`입력 ${s}`, `input ${s}`), x0 + j * 60 + 30, y0 - 8, { color: c.muted, size: 11, align: "center" }));
  [1, 2, 3].forEach((k, i) => txt(ctx, tr(`neuron ${k}`, `neuron ${k}`), x0 - 10, y0 + i * 32 + 20, { color: i === 1 ? c.sig : c.muted, size: 11.5, align: "right", w: i === 1 ? "700" : "400" }));
  txt(ctx, tr("한 행 = neuron 하나가 입력 4개에 곱하는 weight", "one row = the 4 weights one neuron applies to its inputs"), x0, y0 + 3 * 32 + 26, { color: c.ink, size: 12 });
  txt(ctx, tr("z = W x: (3 × 4) · (4 × 1) = (3 × 1)", "z = W x: (3 × 4) · (4 × 1) = (3 × 1)"), x0, y0 + 3 * 32 + 46, { color: c.muted, size: 12, mono: true });
});

/* ================= 6절 / Section 6 ================= */
// 같은 값으로 초기화하면 숨은 neuron이 모두 똑같이 움직인다 (대칭)
// Initialising every weight to the same value makes all hidden neurons move in lockstep (symmetry)
const D6 = makeData("xor", 120, makeRng(21), 0.08);
const S6 = { nets: null, ep: 0, timer: 0 };
function reset6() {
  stopRun(S6);
  const a = new TinyNet([2, 4, 1], { hidden: "tanh", seed: 3, opt: { name: "sgd", lr: 1 } }), b = new TinyNet([2, 4, 1], { hidden: "tanh", seed: 3, opt: { name: "sgd", lr: 1 } });
  a.W = a.W.map(W => W.map(r => r.map(() => 0.5)));
  S6.nets = [a, b]; S6.ep = 0; upd6();
}
const F6a = fig("cv6a", w => (stacked(w, 240, 2, 24) ? 600 : 310), (ctx, w, h, c) => {
  const P = panels(w, h, 2, 24, 240), R = [-1.4, 1.4];
  S6.nets.forEach((net, k) => {
    const p = P[k], side = Math.min(p.w - 30, p.h - 50), B = { l: p.x + 24, t: p.y + 30, w: side, h: side };
    drawField(ctx, c, B, R, R, (x, y) => net.probs([x, y]), { alpha: 0.4 });
    const A = axes(ctx, c, B, R, R, { xt: [], yt: [], grid: false });
    clipBox(ctx, B);
    net.W[0].forEach((r_, j) => { const [a, b] = r_, cc = net.b[0][j], n2 = a * a + b * b; if (n2 < 1e-9) return; const fx0 = -cc * a / n2, fy0 = -cc * b / n2, L = 10; line(ctx, [[A.X(fx0 + b * L), A.Y(fy0 - a * L)], [A.X(fx0 - b * L), A.Y(fy0 + a * L)]], c.ink, 1.3, [5, 4]); });
    drawPoints(ctx, c, A, D6.X, D6.Y, { r: 2.8 });
    ctx.restore();
    panelTitle(ctx, c, k ? tr("무작위 초기화 (Xavier)", "random init (Xavier)") : tr("모두 0.5로 초기화", "every weight = 0.5"), B.l, p.y + 16);
    txtBg(ctx, c, tr(`정확도 ${Math.round(net.accuracy(D6.X, D6.Y) * 100)}%`, `accuracy ${Math.round(net.accuracy(D6.X, D6.Y) * 100)}%`), B.l + 6, B.t + B.h - 8, { color: c.ink, size: 12, w: "700" });
  });
});
function upd6() {
  paint(F6a);
  const W = S6.nets[0].W[0], same = W.every(r => Math.abs(r[0] - W[0][0]) < 1e-9 && Math.abs(r[1] - W[0][1]) < 1e-9);
  $("r6").textContent = tr(`epoch ${S6.ep}. 왼쪽 숨은 neuron 4개의 weight: ${W.map(r => "(" + r.map(v => fx(v, 2)).join(", ") + ")").join(" ")}\n→ ${same ? "4개가 끝까지 똑같다: 점선 4개가 한 줄로 겹친다. neuron이 4개여도 1개와 같다" : "다르다"}`,
    `epoch ${S6.ep}. Left, the 4 hidden neurons' weights: ${W.map(r => "(" + r.map(v => fx(v, 2)).join(", ") + ")").join(" ")}\n→ ${same ? "all 4 stay identical: the 4 dashed lines lie on top of each other. Four neurons act as one" : "they differ"}`);
  $("b6run").textContent = S6.timer ? tr("멈춤", "Pause") : tr("둘 다 학습", "Train both");
}
$("b6run").addEventListener("click", () => {
  if (S6.timer) { stopRun(S6); upd6(); return; }
  runSteps(S6, 100, () => { for (let i = 0; i < 10; i++) S6.nets.forEach(n => n.step(D6.X, D6.Y)); S6.ep += 10; upd6(); if (S6.ep >= 1000) return false; }, 40, upd6); upd6();
});
$("b6reset").addEventListener("click", reset6);
// 깊은 tanh network(10층, 너비 64)에서 초기화 표준편차에 따른 층별 activation 분포
// Activation distribution per layer in a deep tanh network (10 layers, width 64) for each initial SD
const S6b = { sd: 0.125 };
function acts6(sd, act = Math.tanh, width = 64, depth = 10, n = 200, seed = 31) {
  const r = makeRng(seed); let H = Array.from({ length: n }, () => Array.from({ length: width }, () => r.n())); const out = [];
  for (let l = 0; l < depth; l++) {
    const W = Array.from({ length: width }, () => Array.from({ length: width }, () => r.n() * sd));
    H = H.map(x => W.map(row => { let s = 0; for (let i = 0; i < width; i++) s += row[i] * x[i]; return act(s); }));
    out.push(H.flat());
  }
  return out;
}
const F6b = fig("cv6b", w => (w >= 640 ? 200 : 400), (ctx, w, h, c) => {
  const A6 = acts6(S6b.sd), show = [0, 2, 4, 6, 9], cols = w >= 640 ? 5 : 3, gap = 12, cw = (w - (cols - 1) * gap - 10) / cols, chh = 190;
  show.forEach((l, i) => {
    const px = 5 + (i % cols) * (cw + gap), py = Math.floor(i / cols) * chh, B = { l: px + 6, t: py + 30, w: cw - 12, h: chh - 70 }, hgm = hist(A6[l], -1, 1, 30), top = Math.max(...hgm);
    const A = axes(ctx, c, B, [-1, 1], [0, top * 1.05], { xt: [-1, 0, 1], yt: [], grid: false });
    hgm.forEach((v, k) => { const x = A.X(-1 + k * 2 / 30); ctx.fillStyle = c.sig; ctx.globalAlpha = 0.75; ctx.fillRect(x, A.Y(v), B.w / 30 - 1, B.t + B.h - A.Y(v)); ctx.globalAlpha = 1; });
    panelTitle(ctx, c, tr(`${l + 1}층`, `layer ${l + 1}`), B.l, py + 16);
    txt(ctx, `SD ${fx(sdOf(A6[l]), 3)}`, B.l + B.w, py + 16, { color: c.muted, size: 10.5, mono: true, align: "right" });
  });
});
function upd6b() {
  paint(F6b);
  const A6 = acts6(S6b.sd), s = sdOf(A6[9]), xav = Math.sqrt(2 / 128);
  $("r6b").textContent = tr(`초기화 표준편차 ${fx(S6b.sd, 3)} (Xavier라면 √(2/(64+64)) = ${fx(xav, 3)}): 10층 activation 표준편차 ${fx(s, 3)}\n→ ${s < 0.05 ? "너무 작다: 층을 지날수록 0으로 몰린다 (신호가 사라진다)" : sdOf(A6[9]) > 0.85 ? "너무 크다: 대부분 −1, 1로 붙는다 (tanh 포화, 기울기 ≈ 0)" : "알맞다: 층을 지나도 고르게 퍼져 있다"}`,
    `initial SD ${fx(S6b.sd, 3)} (Xavier would be √(2/(64+64)) = ${fx(xav, 3)}): activation SD at layer 10 = ${fx(s, 3)}\n→ ${s < 0.05 ? "too small: values collapse towards 0 layer after layer (the signal vanishes)" : s > 0.85 ? "too large: most values stick to −1 and 1 (tanh saturates, slope ≈ 0)" : "about right: still spread out evenly after many layers"}`);
}
bindRange("i6s", "o6s", v => fx(v, 3), v => { S6b.sd = v; upd6b(); });

/* ================= 7절 / Section 7 ================= */
// 20층 동안 activation 표준편차: 초기화 × activation 네 가지 (너비 64, 지금 browser에서 계산)
// Activation SD over 20 layers: four initialization × activation pairs (width 64, computed live in the browser)
const CONF7 = [["tanh + Xavier", Math.tanh, Math.sqrt(2 / 128)], ["ReLU + Xavier", v => (v > 0 ? v : 0), Math.sqrt(2 / 128)], ["ReLU + Kaiming", v => (v > 0 ? v : 0), Math.sqrt(2 / 64)], [tr("tanh + 표준편차 1", "tanh + SD 1"), Math.tanh, 1]];
let SD7 = null;
function comp7() { SD7 = CONF7.map(([, f, sd]) => acts6(sd, f, 64, 20, 150, 41).map(sdOf)); }
const F7 = fig("cv7", 300, (ctx, w, h, c) => {
  const B = { l: 52, t: 30, w: w - 66, h: h - 74 }, cols = [c.sig, c.fa, c.ok, c.miss];
  const A = axes(ctx, c, B, [1, 20], [-4, 0.5], { xt: [1, 5, 10, 15, 20], yt: [[-4, "10⁻⁴"], [-3, "0.001"], [-2, "0.01"], [-1, "0.1"], [0, "1"]], xlabel: tr("층", "layer") });
  panelTitle(ctx, c, tr("층을 지나는 동안 activation의 표준편차 (log)", "Activation SD layer by layer (log)"), B.l, 16);
  SD7.forEach((s, k) => { line(ctx, s.map((v, l) => [A.X(l + 1), A.Y(clamp(Math.log10(v), -4.5, 1))]), cols[k], 2.4); txtBg(ctx, c, CONF7[k][0], A.X(20) - 4, A.Y(clamp(Math.log10(s[19]), -4, 0.4)) + (k === 3 ? -8 : k === 2 ? 14 : -8), { color: cols[k], size: 11.5, w: "700", align: "right" }); });
});

/* ================= 8절 / Section 8 ================= */
// 학습하는 동안 층마다 weight가 얼마나 바뀌나 (처음 weight 대비). 1층을 얼리면 그 선이 0에 머문다
// How much each layer's weights change during training (relative to the start). Freeze layer 1 and its line stays at 0
const D8 = makeData("moons", 200, makeRng(41), 0.12);
const S8 = { freeze: false, net: null, W0: null, hist: [[], [], []], loss: [], ep: 0, timer: 0 };
const fro = (A, B) => Math.sqrt(A.reduce((s, r, i) => s + r.reduce((t, v, j) => t + (v - B[i][j]) ** 2, 0), 0));
const norm = A => Math.sqrt(A.reduce((s, r) => s + r.reduce((t, v) => t + v * v, 0), 0));
function reset8() { stopRun(S8); S8.net = new TinyNet([2, 16, 16, 1], { hidden: "tanh", seed: 2, opt: { name: "adam", lr: 0.01 } }); S8.W0 = S8.net.W.map(W => W.map(r => r.slice())); S8.hist = [[0], [0], [0]]; S8.loss = [S8.net.loss(D8.X, D8.Y)]; S8.ep = 0; upd8(); }
const F8 = fig("cv8", w => (stacked(w, 260, 2, 28) ? 560 : 290), (ctx, w, h, c) => {
  const P = panels(w, h, 2, 28, 260), cols = [c.sig, c.ok, c.miss];
  {
    const p = P[0], B = { l: p.x + 44, t: p.y + 30, w: p.w - 56, h: p.h - 76 }, top = Math.max(0.5, ...S8.hist.flat()) * 1.1;
    const A = axes(ctx, c, B, [0, 100], [0, top], { xt: [0, 25, 50, 75, 100], yt: ticksR(0, top, 4), xlabel: "epoch" });
    panelTitle(ctx, c, tr("‖W − W₀‖ / ‖W₀‖: 처음보다 얼마나 바뀌었나", "‖W − W₀‖ / ‖W₀‖: change from the start"), B.l, p.y + 14);
    S8.hist.forEach((H, k) => { line(ctx, H.map((v, i) => [A.X(i), A.Y(v)]), cols[k], 2.2); txtBg(ctx, c, tr(`${k + 1}층`, `layer ${k + 1}`) + (k === 0 && S8.freeze ? tr(" (얼림)", " (frozen)") : ""), A.X(Math.max(1, H.length - 1)) + 4, A.Y(H[H.length - 1]) + 4, { color: cols[k], size: 11.5, w: "700" }); });
  }
  {
    const p = P[1], B = { l: p.x + 44, t: p.y + 30, w: p.w - 56, h: p.h - 76 };
    const A = axes(ctx, c, B, [0, 100], [0, 0.75], { xt: [0, 25, 50, 75, 100], yt: [0, 0.25, 0.5, 0.75], xlabel: "epoch" });
    panelTitle(ctx, c, tr("학습 loss", "Training loss"), B.l, p.y + 14);
    line(ctx, S8.loss.map((v, i) => [A.X(i), A.Y(v)]), c.fa, 2.2);
  }
});
function upd8() {
  paint(F8);
  const L = S8.loss[S8.loss.length - 1];
  $("r8").textContent = tr(`epoch ${S8.ep}: loss ${fx(L, 3)}, 정확도 ${Math.round(S8.net.accuracy(D8.X, D8.Y) * 100)}%   1층 ${S8.freeze ? "얼림 (requires_grad = False)" : "학습"}\n층마다 바뀐 비율: ${S8.hist.map((H, k) => `${k + 1}층 ${fx(H[H.length - 1], 2)}`).join(", ")}`,
    `epoch ${S8.ep}: loss ${fx(L, 3)}, accuracy ${Math.round(S8.net.accuracy(D8.X, D8.Y) * 100)}%   layer 1 ${S8.freeze ? "frozen (requires_grad = False)" : "trainable"}\nrelative change per layer: ${S8.hist.map((H, k) => `layer ${k + 1} ${fx(H[H.length - 1], 2)}`).join(", ")}`);
  $("b8run").textContent = S8.timer ? tr("멈춤", "Pause") : tr("학습", "Train");
}
$("b8run").addEventListener("click", () => {
  if (S8.timer) { stopRun(S8); upd8(); return; }
  runSteps(S8, 50, () => {
    for (let r = 0; r < 2; r++) {
      const keep = S8.freeze ? { W: S8.net.W[0].map(x => x.slice()), b: S8.net.b[0].slice() } : null;
      const L = S8.net.epoch(D8.X, D8.Y, 20);
      if (keep) { S8.net.W[0] = keep.W; S8.net.b[0] = keep.b; }
      S8.ep++; S8.loss.push(L); S8.net.W.forEach((W, k) => S8.hist[k].push(fro(W, S8.W0[k]) / norm(S8.W0[k])));
    }
    upd8(); if (S8.ep >= 100) return false;
  }, 40, upd8); upd8();
});
$("b8reset").addEventListener("click", reset8);
bindSeg("g8f", v => { S8.freeze = v === "on"; reset8(); });

/* ================= 9절 / Section 9 ================= */
// 프로젝트 1: 덧셈기. a + b를 배운 2-16-1 network (−10~10에서 학습)
// Project 1: an adding machine. A 2-16-1 network that learned a + b (trained on −10 to 10)
const S9 = { net: null, a: 3, b: 4 };
(function () {
  const r = makeRng(9), X = [], Y = [];
  for (let i = 0; i < 400; i++) { const a = Math.round(r.u() * 20 - 10), b = Math.round(r.u() * 20 - 10); X.push([a / 10, b / 10]); Y.push([(a + b) / 20]); }
  const net = new TinyNet([2, 16, 1], { hidden: "relu", out: "linear", loss: "mse", seed: 4, init: "kaiming", opt: { name: "adam", lr: 0.01 } });
  for (let e = 0; e < 150; e++) net.epoch(X, Y, 32);
  S9.net = net;
})();
const add9 = (a, b) => S9.net.predict([a / 10, b / 10])[0] * 20;
const F9 = fig("cv9", 260, (ctx, w, h, c) => {
  const B = { l: 48, t: 30, w: w - 62, h: h - 74 };
  const A = axes(ctx, c, B, [-40, 40], [-45, 45], { xt: [-40, -20, -10, 0, 10, 20, 40], yt: [-40, -20, 0, 20, 40], xlabel: tr("정답 a + b (b = a일 때)", "true a + b (with b = a)") });
  panelTitle(ctx, c, tr("network의 답 vs 정답 (학습 범위 = 가운데 −20~20)", "Network's answer vs truth (training range = the middle, −20 to 20)"), B.l, 16);
  ctx.fillStyle = c.okwash; ctx.fillRect(A.X(-20), B.t, A.X(20) - A.X(-20), B.h);
  line(ctx, [[A.X(-40), A.Y(-40)], [A.X(40), A.Y(40)]], c.muted, 1.4, [5, 4]);
  line(ctx, Array.from({ length: 81 }, (_, i) => { const s = -40 + i; return [A.X(s), A.Y(clamp(add9(s / 2, s / 2), -45, 45))]; }), c.miss, 2.4);
  const s = S9.a + S9.b; dot(ctx, A.X(clamp(s, -40, 40)), A.Y(clamp(add9(S9.a, S9.b), -45, 45)), 6, c.sig, c.bg);
});
function upd9() {
  paint(F9);
  const p = add9(S9.a, S9.b), inr = Math.abs(S9.a) <= 10 && Math.abs(S9.b) <= 10;
  $("r9").textContent = tr(`${S9.a} + ${S9.b} = ${S9.a + S9.b},  network의 답 ${fx(p, 2)}  (오차 ${fx(p - S9.a - S9.b, 2)})\n→ ${inr ? "학습한 범위 안: 잘 맞힌다" : "학습한 범위 밖인데도 거의 맞는다: 덧셈은 직선이고 ReLU network도 직선 조각이라 그대로 이어진다"}`,
    `${S9.a} + ${S9.b} = ${S9.a + S9.b},  network says ${fx(p, 2)}  (error ${fx(p - S9.a - S9.b, 2)})\n→ ${inr ? "inside the training range: accurate" : "outside the training range yet still close: addition is a straight line and a ReLU network is made of straight pieces, so it carries on"}`);
}
bindRange("i9a", "o9a", v => String(v), v => { S9.a = v; upd9(); });
bindRange("i9b", "o9b", v => String(v), v => { S9.b = v; upd9(); });

// 처음 그리기
// First paint
upd1(); upd3(); paint(F5); reset6(); upd6b(); comp7(); paint(F7); reset8(); upd9();
repaintAll();
