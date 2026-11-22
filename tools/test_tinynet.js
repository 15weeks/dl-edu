// TinyNet의 backprop gradient를 수치 미분과 비교한다 (dropout 포함)
// Compare TinyNet backprop gradients with numerical derivatives (dropout included)
// 실행 / run: node tools/test_tinynet.js
function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
global.makeRng = seed => { const u = mulberry32(seed); let sp = null; return { u, n() { if (sp !== null) { const s = sp; sp = null; return s; } let a, b, s; do { a = 2 * u() - 1; b = 2 * u() - 1; s = a * a + b * b; } while (s >= 1 || s === 0); const m = Math.sqrt(-2 * Math.log(s) / s); sp = b * m; return a * m; } }; };
const { TinyNet, makeData } = require("../assets/dlfig.js");
const cases = [
  { sizes: [2, 3, 1], hidden: "tanh", out: "sigmoid", loss: "bce" },
  { sizes: [2, 4, 3], hidden: "relu", out: "softmax", loss: "ce" },
  { sizes: [2, 3, 1], hidden: "sigmoid", out: "linear", loss: "mse" },
  { sizes: [2, 3, 2], hidden: "tanh", out: "sigmoid", loss: "mse" },
  { sizes: [2, 3, 3], hidden: "leaky", out: "softmax", loss: "mse" },
  { sizes: [3, 4, 4, 2], hidden: "gelu", out: "softmax", loss: "ce" },
  { sizes: [2, 5, 1], hidden: "elu", out: "sigmoid", loss: "bce", drop: 0.4 }
];
let worst = 0;
for (const cs of cases) {
  const net = new TinyNet(cs.sizes, { hidden: cs.hidden, out: cs.out, loss: cs.loss, seed: 7, opt: { drop: cs.drop || 0 } });
  const rng = makeRng(3), X = Array.from({ length: 5 }, () => Array.from({ length: cs.sizes[0] }, () => rng.n())), K = cs.sizes[cs.sizes.length - 1];
  const Y = X.map((_, i) => (K === 1 ? i % 2 : i % K));
  // dropout mask를 고정: 매 forward마다 같은 난수열
  // Fix the dropout mask: the same random sequence on every forward pass
  const seq = [0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.5, 0.05, 0.95];
  let k = 0; net.rng = { u: () => seq[k++ % seq.length], n: () => 0 };
  const lossAt = () => { k = 0; let tot = 0; X.forEach((x, s) => { tot += net.lossOne(net.forward(x, true).as[net.W.length], net.target(Y[s])); }); return tot / X.length; };
  k = 0; const { gW } = net.grads(X, Y, true);
  let err = 0;
  net.W.forEach((W, l) => W.forEach((row, j) => row.forEach((w, i) => {
    const h = 1e-6; row[i] = w + h; const lp = lossAt(); row[i] = w - h; const lm = lossAt(); row[i] = w;
    err = Math.max(err, Math.abs((lp - lm) / (2 * h) - gW[l][j][i]));
  })));
  worst = Math.max(worst, err);
  console.log(JSON.stringify(cs), "max |grad error| =", err.toExponential(2));
}
// 학습이 실제로 되는지: moons를 Adam으로
// Does it actually learn: moons with Adam
const d = makeData("moons", 200, makeRng(5), 0.08), net = new TinyNet([2, 12, 12, 1], { hidden: "tanh", seed: 2, opt: { name: "adam", lr: 0.02 } });
for (let e = 0; e < 150; e++) net.epoch(d.X, d.Y, 20);
console.log("moons accuracy after 150 epochs (Adam):", net.accuracy(d.X, d.Y).toFixed(3));
for (const opt of ["sgd", "momentum", "rmsprop", "adam"]) {
  const n2 = new TinyNet([2, 8, 3], { hidden: "relu", seed: 4, opt: { name: opt, lr: opt === "sgd" ? 0.3 : opt === "momentum" ? 0.05 : 0.01 } }), d3 = makeData("three", 150, makeRng(6), 0.12);
  for (let e = 0; e < 60; e++) n2.epoch(d3.X, d3.Y, 15);
  console.log("three-class", opt, "accuracy", n2.accuracy(d3.X, d3.Y).toFixed(3));
}
console.log(worst < 1e-5 ? "GRADIENT CHECK OK" : "GRADIENT CHECK FAILED");
