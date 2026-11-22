/* 부록 A (ML 지도) 그림과 조작. 한글판(appendix-ml-map-KR.html)과 영어판(appendix-ml-map.html)이 같이 쓴다
   Appendix A (ML map) figures and interactions, shared by the Korean and English pages
   손계산 예제 숫자는 tools/appA_numbers.py로 확인했다. 그림의 model은 모두 browser에서 바로 학습한다
   The worked-example numbers were checked with tools/appA_numbers.py. Every model in the figures trains in the browser */
"use strict";

const pct = v => (100 * v).toFixed(1) + "%";
const clusterCols = c => [c.sig, c.miss, c.ok, c.vio, c.fa, c.ink];
const argmax = a => a.reduce((b, v, i) => (v > a[b] ? i : b), 0);
const dist2 = (p, q) => (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2;

// 점 하나 그리기 / draw one point
function pt(ctx, x, y, r, fill, stroke, lw = 1) {
  ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI);
  if (fill) { ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.stroke(); }
}
// 2D 연습 data: 덩어리와 반달 / 2-D practice data: blobs and moons
function makeBlobs(rng, centers, sd, nEach) {
  const X = [], Y = [];
  centers.forEach((m, k) => { for (let i = 0; i < nEach; i++) { X.push([m[0] + sd * rng.n(), m[1] + sd * rng.n()]); Y.push(k); } });
  return { X, Y };
}
function makeMoons(rng, n, noise) {
  const X = [], Y = [];
  for (let i = 0; i < n; i++) {
    const k = i % 2, t = Math.PI * rng.u(), p = k ? [1 - Math.cos(t), 0.5 - Math.sin(t)] : [Math.cos(t), Math.sin(t)];
    X.push([(p[0] - 0.5) * 1.9 + noise * rng.n(), (p[1] - 0.25) * 1.9 + noise * rng.n()]); Y.push(k);
  }
  return { X, Y };
}
const accOf = (prob, D) => D.X.reduce((s, p, i) => s + (argmax(prob(p)) === D.Y[i] ? 1 : 0), 0) / D.X.length;
// 값 하나(0~1)의 heat map / heat map of one value in 0..1
function heat(ctx, c, B, xr, yr, f, color, n = 90) {
  const off = document.createElement("canvas"); off.width = n; off.height = n;
  const ox = off.getContext("2d"), img = ox.createImageData(n, n), bg = rgb(c.bg), fg = rgb(color);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const v = f(xr[0] + (i + 0.5) / n * (xr[1] - xr[0]), yr[1] - (j + 0.5) / n * (yr[1] - yr[0]));
    const m = mixRGB(bg, fg, 0.05 + 0.6 * clamp(v, 0, 1)), q = (j * n + i) * 4;
    img.data[q] = m[0]; img.data[q + 1] = m[1]; img.data[q + 2] = m[2]; img.data[q + 3] = 255;
  }
  ox.putImageData(img, 0, 0);
  ctx.save(); ctx.imageSmoothingEnabled = true; ctx.drawImage(off, B.l, B.t, B.w, B.h); ctx.restore();
}
// 정사각형에 가까운 그림 상자 / a plot box close to square
const squareBox = (x, y, w, h, padL = 34, padB = 30) => { const s = Math.min(w - padL - 8, h - padB - 22); return { l: x + padL + Math.max(0, (w - padL - 8 - s) / 2), t: y + 20, w: s, h: s }; };

/* ================= 1절 Classic ML / Section 1 ================= */
// 같은 data에 logistic regression, k-NN, linear SVM, Gaussian Naive Bayes를 학습시키고 경계를 비교한다
// Train logistic regression, k-NN, a linear SVM and Gaussian Naive Bayes on the same data and compare their boundaries
const S1 = { model: "logit", data: "blobs", k: 7, seed: 11, fit: null };
function data1() {
  const rng = makeRng(S1.seed);
  const gen = () => (S1.data === "blobs" ? makeBlobs(rng, [[-0.9, -0.6], [0.9, 0.6]], 0.8, 50) : makeMoons(rng, 100, 0.2));
  S1.tr = gen(); S1.te = gen();
}
function fitLogit(D) {
  let w = [0, 0], b = 0; const n = D.X.length;
  for (let it = 0; it < 600; it++) {
    const g = [0, 0, 0];
    D.X.forEach((p, i) => { const e = 1 / (1 + Math.exp(-(w[0] * p[0] + w[1] * p[1] + b))) - D.Y[i]; g[0] += e * p[0]; g[1] += e * p[1]; g[2] += e; });
    w = [w[0] - 0.5 * g[0] / n, w[1] - 0.5 * g[1] / n]; b -= 0.5 * g[2] / n;
  }
  const p1 = q => 1 / (1 + Math.exp(-(w[0] * q[0] + w[1] * q[1] + b)));
  return { prob: q => [1 - p1(q), p1(q)], info: `w = (${fx(w[0], 2)}, ${fx(w[1], 2)}), b = ${fx(b, 2)}` };
}
function fitKnn(D, k) {
  const prob = q => {
    const d = D.X.map((p, i) => [dist2(p, q), D.Y[i]]).sort((a, b) => a[0] - b[0]);
    let v = 0; for (let i = 0; i < k; i++) v += d[i][1];
    return [1 - v / k, v / k];
  };
  return { prob, info: tr(`가장 가까운 ${k}개의 다수결`, `majority vote of the ${k} nearest points`) };
}
function fitSvm(D) {
  const y = D.Y.map(v => (v ? 1 : -1)), n = D.X.length, lam = 0.02; let w = [0, 0], b = 0;
  for (let it = 0; it < 3000; it++) {
    const g = [lam * w[0], lam * w[1], 0];
    D.X.forEach((p, i) => { if (y[i] * (w[0] * p[0] + w[1] * p[1] + b) < 1) { g[0] -= y[i] * p[0] / n; g[1] -= y[i] * p[1] / n; g[2] -= y[i] / n; } });
    w = [w[0] - 0.05 * g[0], w[1] - 0.05 * g[1]]; b -= 0.05 * g[2];
  }
  const f = q => w[0] * q[0] + w[1] * q[1] + b, sv = D.X.filter((p, i) => y[i] * f(p) <= 1.02);
  const p1 = q => 1 / (1 + Math.exp(-2 * f(q)));
  return { prob: q => [1 - p1(q), p1(q)], w, b, sv, info: tr(`margin 폭 2/‖w‖ = ${fx(2 / Math.hypot(w[0], w[1]), 2)}, support vector ${sv.length}개`, `margin width 2/‖w‖ = ${fx(2 / Math.hypot(w[0], w[1]), 2)}, ${sv.length} support vectors`) };
}
function fitNb(D) {
  const st = [0, 1].map(k => {
    const P = D.X.filter((_, i) => D.Y[i] === k), m = [0, 1].map(j => meanOf(P.map(p => p[j])));
    return { m, v: [0, 1].map(j => meanOf(P.map(p => (p[j] - m[j]) ** 2)) + 1e-3), prior: P.length / D.X.length };
  });
  const prob = q => {
    const l = st.map(s => Math.log(s.prior) + [0, 1].reduce((a, j) => a - 0.5 * Math.log(2 * Math.PI * s.v[j]) - (q[j] - s.m[j]) ** 2 / (2 * s.v[j]), 0));
    const mx = Math.max(...l), e = l.map(v => Math.exp(v - mx));
    return [e[0] / (e[0] + e[1]), e[1] / (e[0] + e[1])];
  };
  return { prob, info: tr(`축마다 따로 Gaussian: 파랑 평균 (${fx(st[0].m[0], 2)}, ${fx(st[0].m[1], 2)}), 주황 평균 (${fx(st[1].m[0], 2)}, ${fx(st[1].m[1], 2)})`, `a separate Gaussian per axis: blue mean (${fx(st[0].m[0], 2)}, ${fx(st[0].m[1], 2)}), orange mean (${fx(st[1].m[0], 2)}, ${fx(st[1].m[1], 2)})`) };
}
const XR1 = [-3.4, 3.4], YR1 = [-2.6, 2.6];
const F1 = fig("cv1", w => Math.round(clamp(w * 0.62, 300, 420)), (ctx, w, h, c) => {
  if (!S1.fit) return;
  const B = { l: 40, t: 18, w: w - 52, h: h - 52 };
  drawField(ctx, c, B, XR1, YR1, (x, y) => S1.fit.prob([x, y]), { cell: 5 });
  const A = axes(ctx, c, B, XR1, YR1, { xt: [-2, 0, 2], yt: [-2, 0, 2], grid: false, xlabel: "x₁", ylabel: "x₂" });
  if (S1.model === "svm") {
    const { w: wv, b } = S1.fit; clipBox(ctx, B);
    [-1, 0, 1].forEach(s => {
      const pts = Math.abs(wv[1]) > Math.abs(wv[0]) ? [XR1[0], XR1[1]].map(x => [A.X(x), A.Y((s - b - wv[0] * x) / wv[1])]) : [YR1[0], YR1[1]].map(y => [A.X((s - b - wv[1] * y) / wv[0]), A.Y(y)]);
      line(ctx, pts, c.ink, s ? 1.2 : 2, s ? [5, 4] : null);
    });
    ctx.restore();
    S1.fit.sv.forEach(p => pt(ctx, A.X(p[0]), A.Y(p[1]), 7, null, c.ink, 1.4));
  }
  drawPoints(ctx, c, A, S1.tr.X, S1.tr.Y, { r: 3.4 });
});
function fit1() {
  const D = S1.tr;
  S1.fit = S1.model === "logit" ? fitLogit(D) : S1.model === "knn" ? fitKnn(D, S1.k) : S1.model === "svm" ? fitSvm(D) : fitNb(D);
  const name = { logit: "logistic regression", knn: "k-NN", svm: "linear SVM", nb: "Gaussian Naive Bayes" }[S1.model];
  $("r1").textContent = `${name}: ${tr("학습", "train")} ${pct(accOf(S1.fit.prob, D))}, ${tr("시험", "test")} ${pct(accOf(S1.fit.prob, S1.te))}\n${S1.fit.info}`;
  paint(F1);
}
bindSeg("g1m", v => { S1.model = v; fit1(); });
bindSeg("g1d", v => { S1.data = v; data1(); fit1(); });
bindRange("i1k", "o1k", v => String(v), v => { S1.k = v; if (S1.model === "knn") fit1(); });
$("b1new").addEventListener("click", () => { S1.seed = newSeed(); data1(); fit1(); });

/* ================= 2절 Trees·Ensembles / Section 2 ================= */
// (가) decision tree 하나와 random forest의 경계 / (a) boundaries of one decision tree and a random forest
function buildTree(X, Y, idx, depth, maxD, rng, oneFeat) {
  const n = idx.length, c1 = idx.reduce((s, i) => s + Y[i], 0), p = c1 / n;
  if (depth >= maxD || c1 === 0 || c1 === n || n < 4) return { leaf: true, p };
  let best = null;
  const feats = oneFeat ? [rng.u() < 0.5 ? 0 : 1] : [0, 1];
  for (const f of feats) {
    const s = idx.slice().sort((a, b) => X[a][f] - X[b][f]); let l1 = 0;
    for (let k = 1; k < n; k++) {
      l1 += Y[s[k - 1]];
      const v0 = X[s[k - 1]][f], v1 = X[s[k]][f]; if (v1 === v0) continue;
      const nl = k, nr = n - k, r1 = c1 - l1, pl = l1 / nl, pr = r1 / nr;
      const g = nl * (1 - pl * pl - (1 - pl) ** 2) + nr * (1 - pr * pr - (1 - pr) ** 2);
      if (!best || g < best.g) best = { g, f, t: (v0 + v1) / 2, k, s };
    }
  }
  if (!best) return { leaf: true, p };
  return { f: best.f, t: best.t, l: buildTree(X, Y, best.s.slice(0, best.k), depth + 1, maxD, rng, oneFeat), r: buildTree(X, Y, best.s.slice(best.k), depth + 1, maxD, rng, oneFeat) };
}
const treeP = (nd, q) => { while (!nd.leaf) nd = q[nd.f] < nd.t ? nd.l : nd.r; return nd.p; };
const S2 = { model: "tree", depth: 3, trees: 20, seed: 21 };
function data2() { const rng = makeRng(S2.seed); S2.tr = makeMoons(rng, 160, 0.3); S2.te = makeMoons(rng, 160, 0.3); }
function fit2() {
  const D = S2.tr, n = D.X.length, rng = makeRng(S2.seed + 5), all = Array.from({ length: n }, (_, i) => i);
  if (S2.model === "tree") { const T = buildTree(D.X, D.Y, all, 0, S2.depth, rng, false); S2.prob = q => { const p = treeP(T, q); return [1 - p, p]; }; }
  else {
    const T = Array.from({ length: S2.trees }, () => buildTree(D.X, D.Y, all.map(() => Math.floor(rng.u() * n)), 0, S2.depth, rng, true));
    S2.prob = q => { const p = meanOf(T.map(t => treeP(t, q))); return [1 - p, p]; };
  }
  const name = S2.model === "tree" ? tr(`decision tree (깊이 ${S2.depth})`, `decision tree (depth ${S2.depth})`) : tr(`random forest (tree ${S2.trees}개, 깊이 ${S2.depth})`, `random forest (${S2.trees} trees, depth ${S2.depth})`);
  $("r2a").textContent = `${name}: ${tr("학습", "train")} ${pct(accOf(S2.prob, S2.tr))}, ${tr("시험", "test")} ${pct(accOf(S2.prob, S2.te))}`;
  paint(F2a);
}
const XR2 = [-3.4, 3.4], YR2 = [-2.4, 2.4];
const F2a = fig("cv2a", w => Math.round(clamp(w * 0.6, 280, 400)), (ctx, w, h, c) => {
  if (!S2.prob) return;
  const B = { l: 40, t: 18, w: w - 52, h: h - 52 };
  drawField(ctx, c, B, XR2, YR2, (x, y) => S2.prob([x, y]), { cell: 4 });
  const A = axes(ctx, c, B, XR2, YR2, { xt: [-2, 0, 2], yt: [-2, 0, 2], grid: false, xlabel: "x₁", ylabel: "x₂" });
  drawPoints(ctx, c, A, S2.tr.X, S2.tr.Y, { r: 3.2 });
});
bindSeg("g2m", v => { S2.model = v; fit2(); });
bindRange("i2d", "o2d", v => String(v), v => { S2.depth = v; fit2(); });
bindRange("i2n", "o2n", v => String(v), v => { S2.trees = v; if (S2.model === "forest") fit2(); });
$("b2new").addEventListener("click", () => { S2.seed = newSeed(); data2(); fit2(); });

// (나) gradient boosting: 남은 오차(residual)에 stump를 하나씩 더한다 / (b) gradient boosting: add one stump at a time to the residual
const GB = (() => { const rng = makeRng(77), x = Array.from({ length: 40 }, (_, i) => 6 * i / 39); return { x, y: x.map(v => Math.sin(1.2 * v) + 0.6 + 0.25 * rng.n()) }; })();
const S2b = { eta: 0.3, m: 10, F: [], mse: [] };
function fitStump(x, r) {
  let best = null;
  for (let k = 1; k < x.length; k++) {
    const L = r.slice(0, k), R = r.slice(k), ml = meanOf(L), mr = meanOf(R);
    const e = L.reduce((s, v) => s + (v - ml) ** 2, 0) + R.reduce((s, v) => s + (v - mr) ** 2, 0);
    if (!best || e < best.e) best = { e, t: (x[k - 1] + x[k]) / 2, ml, mr };
  }
  return best;
}
function boost2() {
  const { x, y } = GB, f0 = meanOf(y); let F = x.map(() => f0); const stumps = [];
  S2b.mse = [meanOf(y.map((v, i) => (v - F[i]) ** 2))];
  for (let m = 0; m < 100; m++) {
    const s = fitStump(x, y.map((v, i) => v - F[i])); stumps.push(s);
    F = F.map((v, i) => v + S2b.eta * (x[i] < s.t ? s.ml : s.mr));
    S2b.mse.push(meanOf(y.map((v, i) => (v - F[i]) ** 2)));
  }
  S2b.model = (q, m) => stumps.slice(0, m).reduce((s, st) => s + S2b.eta * (q < st.t ? st.ml : st.mr), f0);
}
const F2b = fig("cv2b", w => (w >= 620 ? 300 : 560), (ctx, w, h, c) => {
  if (!S2b.model) return;
  const P = panels(w, h, 2, 22, 300), { x, y } = GB, m = S2b.m;
  const B1 = { l: P[0].x + 36, t: P[0].y + 24, w: P[0].w - 46, h: P[0].h - 58 };
  const A = axes(ctx, c, B1, [0, 6], [-1, 2.4], { xt: [0, 2, 4, 6], yt: [-1, 0, 1, 2], xlabel: "x", title: tr(`round ${m}의 model`, `model after round ${m}`) });
  x.forEach((v, i) => { const f = S2b.model(v, m); line(ctx, [[A.X(v), A.Y(y[i])], [A.X(v), A.Y(f)]], c.fa, 1); });
  const pts = []; for (let i = 0; i <= 300; i++) { const q = 6 * i / 300; pts.push([A.X(q), A.Y(S2b.model(q, m))]); }
  line(ctx, pts, c.sig, 2.2);
  x.forEach((v, i) => pt(ctx, A.X(v), A.Y(y[i]), 3, c.ink));
  const B2 = { l: P[1].x + 44, t: P[1].y + 24, w: P[1].w - 54, h: P[1].h - 58 }, ymax = Math.max(...S2b.mse) * 1.05;
  const A2 = axes(ctx, c, B2, [0, 100], [0, ymax], { xt: [0, 25, 50, 75, 100], yt: [0, ymax / 2, ymax].map(v => [v, fx(v, 2)]), xlabel: "round", title: tr("학습 MSE", "training MSE") });
  line(ctx, S2b.mse.map((v, i) => [A2.X(i), A2.Y(v)]), c.miss, 2);
  pt(ctx, A2.X(m), A2.Y(S2b.mse[m]), 4.5, c.fa);
});
function show2b() { $("r2b").textContent = tr(`η ${S2b.eta}, round ${S2b.m}: 학습 MSE ${fx(S2b.mse[S2b.m], 3)} (round 0 = 평균 하나: ${fx(S2b.mse[0], 3)}). 빨간 선 = 아직 남은 오차, 다음 stump가 이것을 배운다.`, `η ${S2b.eta}, round ${S2b.m}: training MSE ${fx(S2b.mse[S2b.m], 3)} (round 0 = one mean: ${fx(S2b.mse[0], 3)}). Red lines = the error still left, which the next stump learns.`); paint(F2b); }
bindRange("i2r", "o2r", v => String(v), v => { S2b.m = v; show2b(); });
bindSeg("g2e", v => { S2b.eta = +v; boost2(); show2b(); });

/* ================= 3절 Clustering / Section 3 ================= */
// (가) k-means, k-means++, DBSCAN / (a) k-means, k-means++ and DBSCAN
const S3 = { mode: "kmeans", data: "blobs", k: 4, eps: 0.35, seed: 31, timer: 0 };
function data3() {
  const rng = makeRng(S3.seed);
  S3.X = (S3.data === "blobs" ? makeBlobs(rng, [[-1.6, 1], [1.4, 1.2], [-1, -1.3], [1.5, -1.1]], 0.42, 40) : makeMoons(rng, 200, 0.12)).X;
  reset3();
}
function reset3() {
  stopRun(S3); S3.it = 0; S3.A = null; S3.done = false;
  if (S3.mode === "dbscan") { dbscan3(); return show3(); }
  const rng = makeRng(S3.seed + 101), X = S3.X;
  if (S3.mode === "kmeans") S3.C = shuffled(X.length, rng).slice(0, S3.k).map(i => X[i].slice());
  else {
    // k-means++: 다음 중심은 가장 가까운 중심까지 거리²에 비례하는 확률로 뽑는다
    // k-means++: pick each next centre with probability proportional to the squared distance to the nearest centre
    S3.C = [X[Math.floor(rng.u() * X.length)].slice()];
    while (S3.C.length < S3.k) {
      const d = X.map(p => Math.min(...S3.C.map(q => dist2(p, q)))), tot = d.reduce((s, v) => s + v, 0);
      let u = rng.u() * tot, i = 0; while (i < X.length - 1 && (u -= d[i]) > 0) i++;
      S3.C.push(X[i].slice());
    }
  }
  show3();
}
const assign3 = () => S3.X.map(p => argmax(S3.C.map(q => -dist2(p, q))));
const J3 = () => S3.X.reduce((s, p, i) => s + dist2(p, S3.C[S3.A[i]]), 0);
function step3() {
  if (S3.mode === "dbscan") return false;
  if (!S3.A) { S3.A = assign3(); S3.it = 1; show3(); return true; }
  S3.C = S3.C.map((q, k) => { const P = S3.X.filter((_, i) => S3.A[i] === k); return P.length ? [meanOf(P.map(p => p[0])), meanOf(P.map(p => p[1]))] : q; });
  const A2 = assign3(), same = A2.every((v, i) => v === S3.A[i]);
  S3.A = A2; S3.it++; S3.done = same; show3();
  return !same;
}
function dbscan3() {
  const X = S3.X, n = X.length, e2 = S3.eps ** 2, nb = X.map(p => X.reduce((a, q, j) => (dist2(p, q) <= e2 ? (a.push(j), a) : a), []));
  const core = nb.map(a => a.length >= 5), lab = new Array(n).fill(-1); let k = 0;
  for (let i = 0; i < n; i++) {
    if (!core[i] || lab[i] >= 0) continue;
    const st = [i]; lab[i] = k;
    while (st.length) { const j = st.pop(); if (!core[j]) continue; nb[j].forEach(m => { if (lab[m] < 0) { lab[m] = k; st.push(m); } }); }
    k++;
  }
  S3.lab = lab; S3.core = core; S3.nk = k;
}
const XR3 = [-3.4, 3.4], YR3 = [-2.6, 2.6];
const F3a = fig("cv3a", w => Math.round(clamp(w * 0.6, 280, 400)), (ctx, w, h, c) => {
  if (!S3.X) return;
  const B = { l: 40, t: 18, w: w - 52, h: h - 52 }, cols = clusterCols(c);
  const A = axes(ctx, c, B, XR3, YR3, { xt: [-2, 0, 2], yt: [-2, 0, 2], grid: false, xlabel: "x₁", ylabel: "x₂" });
  if (S3.mode === "dbscan") {
    const i0 = S3.core.indexOf(true);
    if (i0 >= 0) { ctx.save(); ctx.setLineDash([4, 4]); pt(ctx, A.X(S3.X[i0][0]), A.Y(S3.X[i0][1]), S3.eps / (XR3[1] - XR3[0]) * B.w, null, c.muted, 1.2); ctx.restore(); }
    S3.X.forEach((p, i) => { const L = S3.lab[i]; if (L < 0) pt(ctx, A.X(p[0]), A.Y(p[1]), 2.4, c.muted); else pt(ctx, A.X(p[0]), A.Y(p[1]), S3.core[i] ? 3.8 : 2.8, S3.core[i] ? cols[L % 6] : null, cols[L % 6], 1.4); });
    return;
  }
  S3.X.forEach((p, i) => pt(ctx, A.X(p[0]), A.Y(p[1]), 3.2, S3.A ? cols[S3.A[i] % 6] : c.muted));
  S3.C.forEach((q, k) => { const x = A.X(q[0]), y = A.Y(q[1]); pt(ctx, x, y, 9, c.bg, c.ink, 2); line(ctx, [[x - 5, y - 5], [x + 5, y + 5]], cols[k % 6], 3); line(ctx, [[x - 5, y + 5], [x + 5, y - 5]], cols[k % 6], 3); });
});
function show3() {
  let s;
  if (S3.mode === "dbscan") s = tr(`DBSCAN (ε ${fx(S3.eps, 2)}, minPts 5): cluster ${S3.nk}개, noise ${S3.lab.filter(v => v < 0).length}개 (회색). 꽉 찬 점 = core, 빈 점 = border. 점선 원 = 반지름 ε`, `DBSCAN (ε ${fx(S3.eps, 2)}, minPts 5): ${S3.nk} clusters, ${S3.lab.filter(v => v < 0).length} noise points (grey). Filled = core, hollow = border. Dashed circle = radius ε`);
  else if (!S3.A) s = tr(`${S3.mode === "kmeans" ? "k-means (무작위 시작)" : "k-means++ 시작"}: 중심 ${S3.k}개를 놓았다. "한 step"을 누르면 점을 가장 가까운 중심에 배정한다.`, `${S3.mode === "kmeans" ? "k-means (random start)" : "k-means++ start"}: ${S3.k} centres placed. Press "One step" to assign each point to its nearest centre.`);
  else s = tr(`step ${S3.it}: J = ${fx(J3(), 2)}${S3.done ? " · 배정이 더 안 바뀐다 (수렴)" : ""}`, `step ${S3.it}: J = ${fx(J3(), 2)}${S3.done ? " · assignments no longer change (converged)" : ""}`);
  $("r3a").textContent = s; paint(F3a);
}
bindSeg("g3m", v => { S3.mode = v; reset3(); });
bindSeg("g3d", v => { S3.data = v; data3(); });
bindRange("i3k", "o3k", v => String(v), v => { S3.k = v; if (S3.mode !== "dbscan") reset3(); });
bindRange("i3e", "o3e", v => fx(v, 2), v => { S3.eps = v; if (S3.mode === "dbscan") { dbscan3(); show3(); } });
$("b3step").addEventListener("click", () => { stopRun(S3); step3(); });
$("b3run").addEventListener("click", () => runSteps(S3, 30, step3, 350));
$("b3reset").addEventListener("click", () => { S3.seed = newSeed(); data3(); });

// (나) hierarchical clustering과 dendrogram / (b) hierarchical clustering and the dendrogram
const H3 = (() => { const rng = makeRng(5); return makeBlobs(rng, [[-1.5, 1.1], [1.4, 0.9], [0.1, -1.4]], 0.45, 4).X; })();
const S3b = { link: "single", cut: 1 };
function linkage3() {
  const n = H3.length; let cl = H3.map((_, i) => ({ id: i, m: [i] })); const merges = []; let next = n;
  const d = (A, B) => { const v = []; A.m.forEach(i => B.m.forEach(j => v.push(Math.sqrt(dist2(H3[i], H3[j]))))); return S3b.link === "single" ? Math.min(...v) : S3b.link === "complete" ? Math.max(...v) : meanOf(v); };
  while (cl.length > 1) {
    let best = null;
    for (let a = 0; a < cl.length; a++) for (let b = a + 1; b < cl.length; b++) { const v = d(cl[a], cl[b]); if (!best || v < best.v) best = { a, b, v }; }
    const A = cl[best.a], Bc = cl[best.b], C = { id: next++, m: A.m.concat(Bc.m), l: A, r: Bc, h: best.v };
    merges.push(C); cl = cl.filter((_, i) => i !== best.a && i !== best.b).concat([C]);
  }
  S3b.root = cl[0]; S3b.merges = merges; S3b.hmax = merges[merges.length - 1].h;
}
function cutLabels3() {
  const lab = new Array(H3.length).fill(0); let k = 0;
  const walk = nd => { if (nd.h !== undefined && nd.h > S3b.cut) { walk(nd.l); walk(nd.r); } else { nd.m.forEach(i => { lab[i] = k; }); k++; } };
  walk(S3b.root); return { lab, k };
}
const F3b = fig("cv3b", w => (w >= 620 ? 300 : 560), (ctx, w, h, c) => {
  if (!S3b.root) return;
  const P = panels(w, h, 2, 22, 300), cols = clusterCols(c), { lab, k } = cutLabels3();
  const B1 = squareBox(P[0].x, P[0].y, P[0].w, P[0].h);
  const A = axes(ctx, c, B1, [-3, 3], [-3, 3], { xt: [-2, 0, 2], yt: [-2, 0, 2], grid: false, title: tr(`cluster ${k}개`, `${k} clusters`) });
  H3.forEach((p, i) => { pt(ctx, A.X(p[0]), A.Y(p[1]), 5, cols[lab[i] % 6]); txt(ctx, String(i + 1), A.X(p[0]) + 7, A.Y(p[1]) - 6, { color: c.muted, size: 10 }); });
  // dendrogram: 잎을 가로로 늘어놓고 합칠 때의 거리를 높이로 / leaves along x, merge distance as height
  const order = []; const lv = nd => { if (nd.h === undefined) order.push(nd.id); else { lv(nd.l); lv(nd.r); } }; lv(S3b.root);
  const B2 = { l: P[1].x + 40, t: P[1].y + 24, w: P[1].w - 50, h: P[1].h - 58 }, top = S3b.hmax * 1.08;
  const A2 = axes(ctx, c, B2, [-0.5, order.length - 0.5], [0, top], { xt: order.map((id, i) => [i, String(id + 1)]), yt: [0, top / 2, top].map(v => [v, fx(v, 1)]), grid: false, title: tr("dendrogram (높이 = 합친 거리)", "dendrogram (height = merge distance)") });
  const xy = nd => { if (nd.h === undefined) return [A2.X(order.indexOf(nd.id)), A2.Y(0)]; const a = xy(nd.l), b = xy(nd.r), y = A2.Y(nd.h); line(ctx, [[a[0], a[1]], [a[0], y], [b[0], y], [b[0], b[1]]], nd.h > S3b.cut ? c.muted : cols[lab[nd.m[0]] % 6], 1.6); return [(a[0] + b[0]) / 2, y]; };
  xy(S3b.root);
  line(ctx, [[B2.l, A2.Y(S3b.cut)], [B2.l + B2.w, A2.Y(S3b.cut)]], c.fa, 1.6, [6, 4]);
  txt(ctx, tr("자르는 높이", "cut height"), B2.l + B2.w, A2.Y(S3b.cut) - 5, { color: c.fa, size: 10.5, align: "right" });
});
function show3b() { const { k } = cutLabels3(); $("r3b").textContent = tr(`${S3b.link} linkage, 높이 ${fx(S3b.cut, 2)}에서 자르면 cluster ${k}개. 맨 위 합침 높이 ${fx(S3b.hmax, 2)}.`, `${S3b.link} linkage, cutting at height ${fx(S3b.cut, 2)} gives ${k} clusters. Top merge height ${fx(S3b.hmax, 2)}.`); paint(F3b); }
bindRange("i3h", "o3h", v => fx(v, 2), v => { S3b.cut = v; show3b(); });
bindSeg("g3l", v => { S3b.link = v; linkage3(); show3b(); });

/* ================= 4절 압축과 이상 탐지 / Section 4 ================= */
// (가) PCA: 어느 방향으로 눌러야 분산이 가장 많이 남나 / (a) PCA: which direction keeps the most variance
const P4 = (() => { const rng = makeRng(41), X = []; for (let i = 0; i < 150; i++) { const a = Math.sqrt(3) * rng.n(), b = rng.n(); X.push([(a + b) / Math.SQRT2, (a - b) / Math.SQRT2]); } return X; })();
const S4 = { ang: 0 };
const cov4 = (() => { const mx = meanOf(P4.map(p => p[0])), my = meanOf(P4.map(p => p[1])); const sxx = meanOf(P4.map(p => (p[0] - mx) ** 2)), syy = meanOf(P4.map(p => (p[1] - my) ** 2)), sxy = meanOf(P4.map(p => (p[0] - mx) * (p[1] - my))); return { mx, my, sxx, syy, sxy, best: (0.5 * Math.atan2(2 * sxy, sxx - syy) * 180 / Math.PI + 180) % 180 }; })();
const varAlong = deg => { const t = deg * Math.PI / 180, u = [Math.cos(t), Math.sin(t)]; return u[0] * u[0] * cov4.sxx + 2 * u[0] * u[1] * cov4.sxy + u[1] * u[1] * cov4.syy; };
const F4a = fig("cv4a", w => Math.round(clamp(w * 0.58, 280, 380)), (ctx, w, h, c) => {
  const B = squareBox(0, 0, w, h, 40, 34), A = axes(ctx, c, B, [-5, 5], [-5, 5], { xt: [-4, 0, 4], yt: [-4, 0, 4], grid: false, xlabel: "x₁", ylabel: "x₂" });
  const t = S4.ang * Math.PI / 180, u = [Math.cos(t), Math.sin(t)], { mx, my } = cov4;
  clipBox(ctx, B);
  line(ctx, [[A.X(mx - 7 * u[0]), A.Y(my - 7 * u[1])], [A.X(mx + 7 * u[0]), A.Y(my + 7 * u[1])]], c.sig, 2);
  P4.forEach(p => { const s = (p[0] - mx) * u[0] + (p[1] - my) * u[1], f = [mx + s * u[0], my + s * u[1]]; line(ctx, [[A.X(p[0]), A.Y(p[1])], [A.X(f[0]), A.Y(f[1])]], c.line, 0.8); });
  P4.forEach(p => { const s = (p[0] - mx) * u[0] + (p[1] - my) * u[1]; pt(ctx, A.X(p[0]), A.Y(p[1]), 2.6, c.muted); pt(ctx, A.X(mx + s * u[0]), A.Y(my + s * u[1]), 2.2, c.sig); });
  ctx.restore();
  const share = varAlong(S4.ang) / (cov4.sxx + cov4.syy), bx = B.l + 8, by = B.t + 8;
  ctx.fillStyle = c.wash; ctx.fillRect(bx, by, 120, 12); ctx.fillStyle = c.sig; ctx.fillRect(bx, by, 120 * share, 12);
  txt(ctx, tr(`남은 분산 ${pct(share)}`, `variance kept ${pct(share)}`), bx, by + 26, { color: c.ink, size: 11, w: "600" });
});
function show4a() { $("r4a").textContent = tr(`방향 ${S4.ang}°: 이 선 위로 눌렀을 때 분산 ${fx(varAlong(S4.ang), 2)} (전체 ${fx(cov4.sxx + cov4.syy, 2)}의 ${pct(varAlong(S4.ang) / (cov4.sxx + cov4.syy))}). PC1 = ${fx(cov4.best, 1)}°에서 최대.`, `Direction ${S4.ang}°: variance after projecting onto this line ${fx(varAlong(S4.ang), 2)} (${pct(varAlong(S4.ang) / (cov4.sxx + cov4.syy))} of the total ${fx(cov4.sxx + cov4.syy, 2)}). Largest at PC1 = ${fx(cov4.best, 1)}°.`); paint(F4a); }
const i4a = bindRange("i4a", "o4a", v => v + "°", v => { S4.ang = v; show4a(); });
$("b4best").addEventListener("click", () => { S4.ang = Math.round(cov4.best); i4a.value = S4.ang; $("o4a").textContent = S4.ang + "°"; show4a(); });

// (나) 10차원 data를 PCA와 t-SNE로 2D에 그리기 / (b) 10-D data drawn in 2-D by PCA and by t-SNE
const T4 = (() => {
  const rng = makeRng(43), K = 5, n = 24, d = 10, C = Array.from({ length: K }, () => Array.from({ length: d }, () => 2.2 * rng.n())), X = [], Y = [];
  C.forEach((m, k) => { for (let i = 0; i < n; i++) { X.push(m.map(v => v + rng.n())); Y.push(k); } });
  const mu = Array.from({ length: d }, (_, j) => meanOf(X.map(x => x[j]))), Z = X.map(x => x.map((v, j) => v - mu[j]));
  const S = Array.from({ length: d }, (_, a) => Array.from({ length: d }, (_, b) => meanOf(Z.map(z => z[a] * z[b]))));
  const V = []; let Sd = S.map(r => r.slice());
  for (let c = 0; c < 2; c++) { let v = Array.from({ length: d }, (_, j) => (j === c ? 1 : 0.3)); for (let it = 0; it < 200; it++) { const u = matVec(Sd, v), nn = Math.hypot(...u); v = u.map(x => x / nn); } const lam = matVec(Sd, v).reduce((s, x, j) => s + x * v[j], 0); V.push(v); Sd = Sd.map((r, a) => r.map((x, b) => x - lam * v[a] * v[b])); }
  return { X, Y, n: X.length, pca: Z.map(z => V.map(v => v.reduce((s, x, j) => s + x * z[j], 0))) };
})();
const S4b = { perp: 15, it: 0, timer: 0 };
function initTsne() {
  stopRun(S4b);
  const { X, n } = T4, D = X.map(a => X.map(b => a.reduce((s, v, j) => s + (v - b[j]) ** 2, 0))), P = X.map(() => new Array(n).fill(0)), H = Math.log(S4b.perp);
  for (let i = 0; i < n; i++) {
    let lo = 0, hi = Infinity, beta = 1;
    for (let t = 0; t < 60; t++) {
      let sum = 0, sdp = 0;
      for (let j = 0; j < n; j++) if (j !== i) { const p = Math.exp(-D[i][j] * beta); P[i][j] = p; sum += p; sdp += D[i][j] * p; }
      const ent = Math.log(sum) + beta * sdp / sum;
      for (let j = 0; j < n; j++) P[i][j] /= sum;
      if (Math.abs(ent - H) < 1e-5) break;
      if (ent > H) { lo = beta; beta = hi === Infinity ? beta * 2 : (beta + hi) / 2; } else { hi = beta; beta = (beta + lo) / 2; }
    }
  }
  S4b.P = P.map((r, i) => r.map((v, j) => Math.max((v + P[j][i]) / (2 * n), 1e-12)));
  const rng = makeRng(47); S4b.Yt = Array.from({ length: n }, () => [1e-2 * rng.n(), 1e-2 * rng.n()]); S4b.V = S4b.Yt.map(() => [0, 0]); S4b.it = 0; S4b.kl = NaN;
  show4b();
}
function tsneIter() {
  const { n } = T4, Yt = S4b.Yt, ex = S4b.it < 100 ? 4 : 1, mom = S4b.it < 100 ? 0.5 : 0.8, num = Yt.map(() => new Array(n).fill(0)); let Z = 0;
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const v = 1 / (1 + dist2(Yt[i], Yt[j])); num[i][j] = num[j][i] = v; Z += 2 * v; }
  let kl = 0;
  for (let i = 0; i < n; i++) {
    const g = [0, 0];
    for (let j = 0; j < n; j++) if (j !== i) { const q = num[i][j] / Z, m = 4 * (ex * S4b.P[i][j] - q) * num[i][j]; g[0] += m * (Yt[i][0] - Yt[j][0]); g[1] += m * (Yt[i][1] - Yt[j][1]); kl += S4b.P[i][j] * Math.log(S4b.P[i][j] / Math.max(q, 1e-12)); }
    S4b.V[i] = [mom * S4b.V[i][0] - 100 * g[0], mom * S4b.V[i][1] - 100 * g[1]];
  }
  Yt.forEach((y, i) => { y[0] += S4b.V[i][0]; y[1] += S4b.V[i][1]; });
  S4b.it++; S4b.kl = kl;
}
const F4b = fig("cv4b", w => (w >= 620 ? 300 : 580), (ctx, w, h, c) => {
  if (!S4b.Yt) return;
  const P = panels(w, h, 2, 22, 300), cols = clusterCols(c);
  [[T4.pca, tr("PCA (선형, 분산 큰 두 방향)", "PCA (linear, two largest-variance directions)")], [S4b.Yt, tr(`t-SNE (iteration ${S4b.it})`, `t-SNE (iteration ${S4b.it})`)]].forEach(([Y2, title], k) => {
    const xs = Y2.map(p => p[0]), ys = Y2.map(p => p[1]), cx = (Math.max(...xs) + Math.min(...xs)) / 2, cy = (Math.max(...ys) + Math.min(...ys)) / 2;
    const r = Math.max(1e-6, Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) * 0.58;
    const B = squareBox(P[k].x, P[k].y, P[k].w, P[k].h, 12, 10);
    ctx.strokeStyle = c.line; ctx.strokeRect(B.l, B.t, B.w, B.h); panelTitle(ctx, c, title, B.l, B.t - 6);
    Y2.forEach((p, i) => pt(ctx, B.l + (p[0] - cx + r) / (2 * r) * B.w, B.t + (cy + r - p[1]) / (2 * r) * B.h, 3.2, cols[T4.Y[i] % 6]));
  });
});
function show4b() { $("r4b").textContent = tr(`10차원 점 ${T4.n}개 (cluster 5개, 색 = 정답 무리). perplexity ${S4b.perp}, iteration ${S4b.it}${isFinite(S4b.kl) ? `, KL(P‖Q) = ${fx(S4b.kl, 3)}` : ""}`, `${T4.n} points in 10-D (5 clusters, colour = true group). Perplexity ${S4b.perp}, iteration ${S4b.it}${isFinite(S4b.kl) ? `, KL(P‖Q) = ${fx(S4b.kl, 3)}` : ""}`); paint(F4b); }
$("b4run").addEventListener("click", () => runSteps(S4b, 50, () => { for (let k = 0; k < 10; k++) tsneIter(); show4b(); return S4b.it < 500; }, 40));
$("b4reset").addEventListener("click", initTsne);
bindSeg("g4p", v => { S4b.perp = +v; initTsne(); });

// (다) isolation forest: 무작위로 자르면 이상한 점은 빨리 혼자가 된다 / (c) isolation forest: random cuts isolate odd points quickly
const I4 = (() => { const rng = makeRng(49), X = []; for (let i = 0; i < 100; i++) X.push([0.6 * rng.n(), 0.6 * rng.n()]); X.push([2.2, 1.8], [-2.3, 1.4], [1.9, -2.1], [-0.3, 2.5]); return X; })();
const cOf = m => (m <= 1 ? 0 : 2 * (Math.log(m - 1) + 0.5772156649) - 2 * (m - 1) / m);
function iTree(idx, depth, rng, lim) {
  if (idx.length <= 1 || depth >= lim) return { size: idx.length };
  const f = rng.u() < 0.5 ? 0 : 1, v = idx.map(i => I4[i][f]), lo = Math.min(...v), hi = Math.max(...v);
  if (hi === lo) return { size: idx.length };
  const t = lo + rng.u() * (hi - lo);
  return { f, t, l: iTree(idx.filter(i => I4[i][f] < t), depth + 1, rng, lim), r: iTree(idx.filter(i => I4[i][f] >= t), depth + 1, rng, lim) };
}
const pathLen = (nd, q, d = 0) => (nd.f === undefined ? d + cOf(nd.size) : pathLen(q[nd.f] < nd.t ? nd.l : nd.r, q, d + 1));
const IF4 = (() => { const rng = makeRng(53); return Array.from({ length: 60 }, () => iTree(shuffled(I4.length, rng).slice(0, 64), 0, rng, 8)); })();
const score4 = q => Math.pow(2, -meanOf(IF4.map(t => pathLen(t, q))) / cOf(64));
const S4c = { sel: 101, seed: 59 };
function splits4() {
  // 고른 점 하나가 혼자 남을 때까지의 자르기 (모든 점으로 만든 tree 하나) / cuts until the chosen point is alone (one tree on all points)
  const rng = makeRng(S4c.seed), q = I4[S4c.sel], cuts = []; let idx = I4.map((_, i) => i), box = [-3, 3, -3, 3];
  while (idx.length > 1 && cuts.length < 40) {
    const f = rng.u() < 0.5 ? 0 : 1, v = idx.map(i => I4[i][f]), lo = Math.min(...v), hi = Math.max(...v); if (hi === lo) break;
    const t = lo + rng.u() * (hi - lo); cuts.push({ f, t, box: box.slice() });
    if (q[f] < t) { idx = idx.filter(i => I4[i][f] < t); box[f === 0 ? 1 : 3] = t; } else { idx = idx.filter(i => I4[i][f] >= t); box[f === 0 ? 0 : 2] = t; }
  }
  return cuts;
}
const F4c = fig("cv4c", w => Math.round(clamp(w * 0.58, 280, 380)), (ctx, w, h, c) => {
  const B = squareBox(0, 0, w, h, 40, 34);
  drawField(ctx, c, B, [-3, 3], [-3, 3], (x, y) => { const s = score4([x, y]), p = clamp((s - 0.5) * 2.5 + 0.5, 0, 1); return [1 - p, p]; }, { cell: 6, alpha: 0.5 });
  const A = axes(ctx, c, B, [-3, 3], [-3, 3], { xt: [-2, 0, 2], yt: [-2, 0, 2], grid: false });
  splits4().forEach(cu => { const b = cu.box; if (cu.f === 0) line(ctx, [[A.X(cu.t), A.Y(b[2])], [A.X(cu.t), A.Y(b[3])]], c.ink, 1); else line(ctx, [[A.X(b[0]), A.Y(cu.t)], [A.X(b[1]), A.Y(cu.t)]], c.ink, 1); });
  I4.forEach((p, i) => pt(ctx, A.X(p[0]), A.Y(p[1]), i === S4c.sel ? 6 : 2.8, i === S4c.sel ? c.fa : c.ink, i === S4c.sel ? c.bg : null, 1.5));
  S4c.A = A;
});
function show4c() { const n = splits4().length, q = I4[S4c.sel], E = meanOf(IF4.map(t => pathLen(t, q))); $("r4c").textContent = tr(`고른 점 (${fx(q[0], 1)}, ${fx(q[1], 1)}): 이 tree에서 ${n}번 만에 혼자. tree 60개 평균 path 길이 E[h] = ${fx(E, 2)}, score s = ${fx(score4(q), 3)} (0.5 근처 = 보통, 1에 가까울수록 이상)`, `Chosen point (${fx(q[0], 1)}, ${fx(q[1], 1)}): alone after ${n} cuts in this tree. Mean path length over 60 trees E[h] = ${fx(E, 2)}, score s = ${fx(score4(q), 3)} (near 0.5 = ordinary, closer to 1 = anomalous)`); paint(F4c); }
dragOn(F4c, (x, y) => { if (!S4c.A) return null; let best = -1, bd = 400; I4.forEach((p, i) => { const d = (S4c.A.X(p[0]) - x) ** 2 + (S4c.A.Y(p[1]) - y) ** 2; if (d < bd) { bd = d; best = i; } }); return best >= 0 ? best : null; }, i => { if (i !== S4c.sel) { S4c.sel = i; show4c(); } });
$("b4tree").addEventListener("click", () => { S4c.seed = newSeed(); show4c(); });
$("b4pick").addEventListener("click", () => { S4c.sel = S4c.sel >= 100 ? 0 : 101; show4c(); });

/* ================= 5절 강화학습 / Section 5 ================= */
// (가) cliff walking: Q-learning과 SARSA / (a) cliff walking: Q-learning and SARSA
const RW = 4, CW = 12, START = [3, 0], GOAL = [3, 11], MOVES = [[-1, 0], [0, 1], [1, 0], [0, -1]];
const isCliff = (r, cc) => r === 3 && cc > 0 && cc < 11;
const S5 = { show: "q", eps: 0.1, seed: 61 };
function reset5() { const z = () => Array.from({ length: RW * CW }, () => [0, 0, 0, 0]); S5.agents = { q: { Q: z(), ret: [] }, s: { Q: z(), ret: [] } }; S5.rng = makeRng(S5.seed); show5(); }
const greedy5 = (Q, s) => { const v = Q[s], m = Math.max(...v), b = v.map((x, i) => (x === m ? i : -1)).filter(i => i >= 0); return b[Math.floor(S5.rng.u() * b.length)]; };
const pick5 = (Q, s) => (S5.rng.u() < S5.eps ? Math.floor(S5.rng.u() * 4) : greedy5(Q, s));
function move5(r, cc, a) {
  const nr = clamp(r + MOVES[a][0], 0, RW - 1), nc = clamp(cc + MOVES[a][1], 0, CW - 1);
  if (isCliff(nr, nc)) return { r: START[0], c: START[1], rew: -100 };
  return { r: nr, c: nc, rew: -1 };
}
function episode5(ag, sarsa) {
  let r = START[0], cc = START[1], s = r * CW + cc, a = pick5(ag.Q, s), G = 0;
  for (let t = 0; t < 1000; t++) {
    const m = move5(r, cc, a), s2 = m.r * CW + m.c, end = m.r === GOAL[0] && m.c === GOAL[1]; G += m.rew;
    const a2 = pick5(ag.Q, s2), target = end ? m.rew : m.rew + (sarsa ? ag.Q[s2][a2] : Math.max(...ag.Q[s2]));
    ag.Q[s][a] += 0.5 * (target - ag.Q[s][a]);
    if (end) break;
    r = m.r; cc = m.c; s = s2; a = a2;
  }
  ag.ret.push(G);
}
function path5(Q) {
  let r = START[0], cc = START[1]; const p = [[r, cc]];
  for (let t = 0; t < 48; t++) { const v = Q[r * CW + cc], a = argmax(v), m = move5(r, cc, a); if (m.rew === -100) { p.push([3, 1]); break; } r = m.r; cc = m.c; p.push([r, cc]); if (r === GOAL[0] && cc === GOAL[1]) break; }
  return p;
}
const F5a = fig("cv5a", w => (w >= 560 ? Math.round(w * 0.33) + 170 : Math.round(w * 0.36) + 170), (ctx, w, h, c) => {
  if (!S5.agents) return;
  const ag = S5.agents[S5.show], Q = ag.Q, cs = Math.min((w - 20) / CW, 52), gx = (w - cs * CW) / 2, gy = 8;
  const vals = Q.map((v, s) => (isCliff(Math.floor(s / CW), s % CW) ? NaN : Math.max(...v))).filter(isFinite), vmin = Math.min(...vals, -1), vmax = Math.max(...vals, 0);
  const bg = rgb(c.bg), sig = rgb(c.sig);
  for (let r = 0; r < RW; r++) for (let cc = 0; cc < CW; cc++) {
    const s = r * CW + cc, x = gx + cc * cs, y = gy + r * cs;
    if (isCliff(r, cc)) ctx.fillStyle = c.fa;
    else { const t = vmax > vmin ? (Math.max(...Q[s]) - vmin) / (vmax - vmin) : 0, m = mixRGB(bg, sig, 0.08 + 0.5 * t); ctx.fillStyle = `rgb(${m[0]},${m[1]},${m[2]})`; }
    ctx.fillRect(x, y, cs - 1, cs - 1);
    if (!isCliff(r, cc) && !(r === GOAL[0] && cc === GOAL[1]) && Q[s].some(v => v !== 0)) { const a = argmax(Q[s]), cx = x + cs / 2, cy = y + cs / 2; arrowS(ctx, cx - MOVES[a][1] * cs * 0.22, cy - MOVES[a][0] * cs * 0.22, cx + MOVES[a][1] * cs * 0.22, cy + MOVES[a][0] * cs * 0.22, c.muted, 1.2, 4); }
  }
  txt(ctx, "S", gx + cs / 2, gy + 3 * cs + cs / 2 + 4, { color: c.ink, w: "700", align: "center", size: 13 });
  txt(ctx, "G", gx + 11 * cs + cs / 2, gy + 3 * cs + cs / 2 + 4, { color: c.ink, w: "700", align: "center", size: 13 });
  txt(ctx, tr("절벽 (−100, 처음으로)", "cliff (−100, back to start)"), gx + 6 * cs, gy + 3 * cs + cs / 2 + 4, { color: c.bg, w: "600", align: "center", size: 11 });
  if (ag.ret.length) line(ctx, path5(Q).map(([r, cc]) => [gx + cc * cs + cs / 2, gy + r * cs + cs / 2]), S5.show === "q" ? c.sig : c.miss, 3.5);
  // 아래: episode마다 보상 합 (20개 이동 평균) / below: return per episode (20-episode moving average)
  const B = { l: 46, t: gy + RW * cs + 26, w: w - 58, h: h - (gy + RW * cs + 26) - 34 }, N = Math.max(100, S5.agents.q.ret.length);
  const A = axes(ctx, c, B, [0, N], [-120, 0], { xt: [0, N / 2, N].map(v => [v, String(Math.round(v))]), yt: [-100, -50, 0], xlabel: "episode", title: tr("episode마다 보상 합 (20개 평균)", "return per episode (20-episode average)") });
  [["q", c.sig], ["s", c.miss]].forEach(([k, col]) => { const R = S5.agents[k].ret; if (R.length < 2) return; const p = R.map((_, i) => { const a = R.slice(Math.max(0, i - 19), i + 1); return [A.X(i + 1), A.Y(Math.max(-120, meanOf(a)))]; }); line(ctx, p, col, 1.8); });
  txt(ctx, "Q-learning", B.l + 6, B.t + 12, { color: c.sig, size: 10.5, w: "600" }); txt(ctx, "SARSA", B.l + 86, B.t + 12, { color: c.miss, size: 10.5, w: "600" });
});
function show5() {
  const ag = S5.agents[S5.show], n = ag.ret.length, last = ag.ret.slice(-50), p = path5(ag.Q), edge = p.filter(([r]) => r === 2).length;
  const name = S5.show === "q" ? "Q-learning" : "SARSA";
  $("r5a").textContent = n ? tr(`${name}, episode ${n}: 최근 50개 평균 보상 ${fx(meanOf(last), 1)}. 지금 정책대로 가면 ${p.length - 1}걸음 (절벽 바로 위 줄을 ${edge}칸 지난다)`, `${name}, ${n} episodes: mean return of the last 50 = ${fx(meanOf(last), 1)}. Following the current policy takes ${p.length - 1} steps (${edge} cells along the row just above the cliff)`) : tr("아직 학습 전. '50 episode 학습'을 누른다.", "Not trained yet. Press 'Train 50 episodes'.");
  paint(F5a);
}
$("b5train").addEventListener("click", () => { for (let e = 0; e < 50; e++) { episode5(S5.agents.q, false); episode5(S5.agents.s, true); } show5(); });
$("b5reset").addEventListener("click", () => { S5.seed = newSeed(); reset5(); });
bindSeg("g5m", v => { S5.show = v; show5(); });
bindRange("i5e", "o5e", v => fx(v, 2), v => { S5.eps = v; });

// (나) 팔 3개 bandit의 policy gradient, baseline(critic) 있음·없음 / (b) policy gradient on a 3-armed bandit, with and without a baseline (critic)
const ARMS = [0.2, 0.5, 0.8];
const S5b = { base: "none", seed: 71 };
function reset5b() { S5b.th = [0, 0, 0]; S5b.V = 0; S5b.hist = [[1 / 3, 1 / 3, 1 / 3]]; S5b.rew = []; S5b.rng = makeRng(S5b.seed); show5b(); }
const softmax3 = th => { const m = Math.max(...th), e = th.map(v => Math.exp(v - m)), z = e[0] + e[1] + e[2]; return e.map(v => v / z); };
function pull5b(n) {
  for (let k = 0; k < n; k++) {
    const pi = softmax3(S5b.th); let u = S5b.rng.u(), a = 0; while (a < 2 && (u -= pi[a]) > 0) a++;
    const r = S5b.rng.u() < ARMS[a] ? 1 : 0, adv = S5b.base === "critic" ? r - S5b.V : r;
    S5b.th = S5b.th.map((t, i) => t + 0.1 * adv * ((i === a ? 1 : 0) - pi[i]));
    S5b.V += 0.05 * (r - S5b.V); S5b.rew.push(r); S5b.hist.push(softmax3(S5b.th));
  }
}
const F5b = fig("cv5b", w => Math.round(clamp(w * 0.4, 220, 300)), (ctx, w, h, c) => {
  if (!S5b.hist) return;
  const H = S5b.hist, N = Math.max(200, H.length - 1), B = { l: 44, t: 22, w: w - 56, h: h - 56 }, cols = [c.sig, c.miss, c.ok];
  const A = axes(ctx, c, B, [0, N], [0, 1], { xt: [0, N / 2, N].map(v => [v, String(Math.round(v))]), yt: [0, 0.5, 1], xlabel: tr("당긴 횟수", "pulls"), title: tr("정책 π: 팔마다 고를 확률 (쌓은 그림)", "policy π: probability of each arm (stacked)") });
  for (let a = 2; a >= 0; a--) {
    ctx.beginPath(); ctx.moveTo(A.X(0), A.Y(0));
    H.forEach((p, i) => ctx.lineTo(A.X(i), A.Y(p.slice(0, a + 1).reduce((s, v) => s + v, 0))));
    ctx.lineTo(A.X(H.length - 1), A.Y(0)); ctx.closePath(); ctx.fillStyle = cols[a]; ctx.globalAlpha = 0.55; ctx.fill(); ctx.globalAlpha = 1;
  }
  ARMS.forEach((m, a) => txt(ctx, tr(`팔 ${a + 1} (평균 보상 ${m})`, `arm ${a + 1} (mean reward ${m})`), B.l + 8 + a * 150, B.t + 14, { color: cols[a], size: 10.5, w: "700" }));
});
function show5b() { const pi = S5b.hist[S5b.hist.length - 1], n = S5b.rew.length; $("r5b").textContent = tr(`${n}번 당김, π = (${pi.map(v => fx(v, 2)).join(", ")}), 최근 100번 평균 보상 ${n ? fx(meanOf(S5b.rew.slice(-100)), 2) : "-"} (최대 0.8)`, `${n} pulls, π = (${pi.map(v => fx(v, 2)).join(", ")}), mean reward of the last 100 = ${n ? fx(meanOf(S5b.rew.slice(-100)), 2) : "-"} (best possible 0.8)`); paint(F5b); }
$("b5pull").addEventListener("click", () => { pull5b(100); show5b(); });
$("b5breset").addEventListener("click", () => { S5b.seed = newSeed(); reset5b(); });
bindSeg("g5b", v => { S5b.base = v; reset5b(); });

/* ================= 6절 Genetic algorithm / Section 6 ================= */
// 봉우리가 셋인 지형에서 가장 높은 곳 찾기 / find the highest point on a landscape with three peaks
const fit6 = (x, y) => 1.0 * Math.exp(-((x - 1.2) ** 2 + (y - 0.8) ** 2) / 0.5) + 0.7 * Math.exp(-((x + 1) ** 2 + (y + 0.9) ** 2) / 0.5) + 0.55 * Math.exp(-((x + 1.2) ** 2 + (y - 1.3) ** 2) / 0.4);
const S6 = { mut: 0.25, seed: 81, timer: 0 };
function reset6() {
  stopRun(S6); S6.rng = makeRng(S6.seed);
  // 처음 무리 20개를 지형 전체에 고르게 뿌린다 / spread the first 20 candidates evenly over the whole landscape
  S6.pop = Array.from({ length: 20 }, () => [-2.5 + 5 * S6.rng.u(), -2.5 + 5 * S6.rng.u()]); S6.gen = 0; S6.best = []; S6.mean = [];
  record6(); show6();
}
function record6() { const f = S6.pop.map(p => fit6(p[0], p[1])); S6.best.push(Math.max(...f)); S6.mean.push(meanOf(f)); }
function gen6() {
  const rng = S6.rng, P = S6.pop, f = P.map(p => fit6(p[0], p[1])), order = f.map((v, i) => i).sort((a, b) => f[b] - f[a]);
  const tour = () => { let b = Math.floor(rng.u() * P.length); for (let k = 0; k < 2; k++) { const j = Math.floor(rng.u() * P.length); if (f[j] > f[b]) b = j; } return P[b]; };
  const next = [P[order[0]].slice(), P[order[1]].slice()];
  while (next.length < P.length) {
    const a = tour(), b = tour(), u = -0.25 + 1.5 * rng.u(), child = [a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1])];
    next.push(child.map(v => clamp(rng.u() < 0.5 ? v + S6.mut * rng.n() : v, -2.5, 2.5)));
  }
  S6.pop = next; S6.gen++; record6();
}
const F6 = fig("cv6", w => (w >= 620 ? 320 : 600), (ctx, w, h, c) => {
  if (!S6.pop) return;
  const P = panels(w, h, 2, 22, 300), B = squareBox(P[0].x, P[0].y, P[0].w, P[0].h, 30, 26);
  heat(ctx, c, B, [-2.5, 2.5], [-2.5, 2.5], (x, y) => fit6(x, y), c.ok);
  const A = axes(ctx, c, B, [-2.5, 2.5], [-2.5, 2.5], { xt: [-2, 0, 2], yt: [-2, 0, 2], grid: false, title: tr(`세대 ${S6.gen}`, `generation ${S6.gen}`) });
  pt(ctx, A.X(1.2), A.Y(0.8), 8, null, c.ink, 1.2);
  const f = S6.pop.map(p => fit6(p[0], p[1])), bi = argmax(f);
  S6.pop.forEach((p, i) => pt(ctx, A.X(p[0]), A.Y(p[1]), i === bi ? 5 : 3.4, i === bi ? c.fa : c.vio, c.bg, 1));
  const B2 = { l: P[1].x + 40, t: P[1].y + 24, w: P[1].w - 50, h: P[1].h - 58 }, G = Math.max(30, S6.gen);
  const A2 = axes(ctx, c, B2, [0, G], [0, 1.1], { xt: [0, G / 2, G].map(v => [v, String(Math.round(v))]), yt: [0, 0.5, 1], xlabel: tr("세대", "generation"), title: tr("가장 좋은 점수와 평균 점수", "best and mean fitness") });
  line(ctx, S6.best.map((v, i) => [A2.X(i), A2.Y(v)]), c.fa, 2); line(ctx, S6.mean.map((v, i) => [A2.X(i), A2.Y(v)]), c.vio, 1.6, [5, 3]);
  txt(ctx, tr("최고", "best"), B2.l + 6, B2.t + 12, { color: c.fa, size: 10.5, w: "600" }); txt(ctx, tr("평균", "mean"), B2.l + 50, B2.t + 12, { color: c.vio, size: 10.5, w: "600" });
});
function show6() { const f = S6.pop.map(p => fit6(p[0], p[1])), b = S6.pop[argmax(f)]; $("r6").textContent = tr(`세대 ${S6.gen}: 최고 점수 ${fx(Math.max(...f), 3)} at (${fx(b[0], 2)}, ${fx(b[1], 2)}), 평균 ${fx(meanOf(f), 3)}. 가장 높은 봉우리 = 1.000 at (1.2, 0.8)`, `Generation ${S6.gen}: best fitness ${fx(Math.max(...f), 3)} at (${fx(b[0], 2)}, ${fx(b[1], 2)}), mean ${fx(meanOf(f), 3)}. Highest peak = 1.000 at (1.2, 0.8)`); paint(F6); }
$("b6step").addEventListener("click", () => { stopRun(S6); gen6(); show6(); });
$("b6run").addEventListener("click", () => runSteps(S6, 30, () => { gen6(); show6(); }, 160));
$("b6reset").addEventListener("click", () => { S6.seed = newSeed(); reset6(); });
bindRange("i6m", "o6m", v => fx(v, 2), v => { S6.mut = v; });

/* ================= 시작 / Start ================= */
data1(); fit1(); data2(); fit2(); boost2(); show2b(); data3(); linkage3(); S3b.cut = +$("i3h").value; show3b();
show4a(); initTsne(); show4c(); reset5(); reset5b(); reset6();
