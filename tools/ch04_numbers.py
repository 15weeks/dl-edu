# 4장 예제 숫자와 미리 돌린 실험 (PyTorch). 결과를 assets/data/ch04_data.js로 쓴다
# Chapter 4 example numbers and precomputed experiments (PyTorch). Writes the results to assets/data/ch04_data.js
# 실행 / run: python tools/ch04_numbers.py [iris.csv 경로/path]
import sys, json, math, pathlib
import numpy as np
import torch
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from dlrng import Rng, make_data

ROOT = pathlib.Path(__file__).resolve().parent.parent
torch.set_num_threads(4)
out = {}

# 1절: 회귀. browser와 같은 data (seed 4, 60개): x = n(), e = n(), y = m x + s e
# Section 1: regression. Same data as the browser (seed 4, 60 points): x = n(), e = n(), y = m x + s e
r = Rng(4); xs = []; es = []
for _ in range(60):
    xs.append(r.n()); es.append(r.n())
xs, es = np.array(xs), np.array(es)
for m in (1.0, 0.0):
    y = m * xs + 0.5 * es
    w, b = np.polyfit(xs, y, 1)
    yh = w * xs + b
    mse = np.mean((yh - y) ** 2); rr = np.corrcoef(yh, y)[0, 1] if m else np.corrcoef(xs, y)[0, 1]
    print(f"s1 m={m}: w={w:.3f} b={b:.3f} MSE={mse:.3f} r={rr:.3f}")

# 2~3절: qwerties. 두 무리 중심 (1, 1)과 (1 + d, 1), 퍼짐 1, 200개 (browser seed 31)
# Sections 2-3: qwerties. Group centres (1, 1) and (1 + d, 1), spread 1, 200 points (browser seed 31)
def qwerties(d, seed=31, n=200):
    r = Rng(seed); X, Y = [], []
    for i in range(n):
        k = i % 2; X.append([(1 + d if k else 1) + r.n(), 1 + r.n()]); Y.append(k)
    return np.array(X), np.array(Y)
X, Y = qwerties(4)
Xt, Yt = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)[:, None]
lrs = np.logspace(-3, np.log10(3), 28)
acc = []
for lr in lrs:
    row = []
    for seed in range(5):
        torch.manual_seed(seed)
        net = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
        opt = torch.optim.SGD(net.parameters(), lr=float(lr))
        for _ in range(1000):
            opt.zero_grad(); loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xt), Yt); loss.backward(); opt.step()
        with torch.no_grad():
            p = net(Xt); a = ((p > 0).float() == Yt).float().mean().item() if torch.isfinite(p).all() else 0.5
        row.append(round(a, 3))
    acc.append(row)
out["lrExp"] = {"lrs": [round(float(v), 5) for v in lrs], "acc": acc}
good = [i for i, row in enumerate(acc) if np.mean(row) > 0.9]
print("s3 lr range with mean acc > 0.9:", (round(lrs[good[0]], 4), round(lrs[good[-1]], 4)) if good else None)
print("s3 mean acc per lr:", [round(float(np.mean(r_)), 2) for r_ in acc])

# 4절: 선형 층 두 개 = 하나
# Section 4: two linear layers = one
W1 = np.array([[1, 0.5], [-0.5, 1]]); b1 = np.array([0.2, 0.0]); W2 = np.array([[0.8, -0.6], [0.6, 0.8]]); b2 = np.array([0.0, 0.1])
print("s4 W2 W1 =", (W2 @ W1).tolist(), " W2 b1 + b2 =", (W2 @ b1 + b2).tolist())
x0 = np.array([1.0, 2.0]); print("s4 check x=(1,2):", W2 @ (W1 @ x0 + b1) + b2, (W2 @ W1) @ x0 + (W2 @ b1 + b2))

# 5절: iris (petal length, petal width)
# Section 5: iris (petal length, petal width)
iris_path = sys.argv[1] if len(sys.argv) > 1 else None
if iris_path:
    rows = [l.strip().split(",") for l in open(iris_path).read().strip().split("\n")[1:]]
    names = ["setosa", "versicolor", "virginica"]
    iris = [[float(r_[2]), float(r_[3]), names.index(r_[4])] for r_ in rows]
    out["iris"] = iris
    A = np.array(iris); Xi = A[:, :2]; Yi = A[:, 2].astype(int)
    mu, sd = Xi.mean(0), Xi.std(0)
    out["irisNorm"] = {"mu": mu.round(4).tolist(), "sd": sd.round(4).tolist()}
    Xn = torch.tensor((Xi - mu) / sd, dtype=torch.float32); Yn = torch.tensor(Yi)
    accs = []
    for seed in range(5):
        torch.manual_seed(seed)
        net = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 3))
        opt = torch.optim.Adam(net.parameters(), lr=0.02)
        for _ in range(300):
            opt.zero_grad(); torch.nn.functional.cross_entropy(net(Xn), Yn).backward(); opt.step()
        accs.append((net(Xn).argmax(1) == Yn).float().mean().item())
    print("s5 iris 2-8-3 accuracy over 5 seeds:", [round(a, 3) for a in accs])
    print("s5 iris means (pl, pw) per class:", [A[Yi == k, :2].mean(0).round(2).tolist() for k in range(3)])

# 6절: parameter 수와 depth × breadth 실험 (spiral, 300개, 70% 학습 / 30% 시험)
# Section 6: parameter counts and the depth × breadth experiment (spiral, 300 points, 70% train / 30% test)
def n_params(sizes): return sum(sizes[i] * sizes[i + 1] + sizes[i + 1] for i in range(len(sizes) - 1))
print("s6 params 2-16-16-3 =", n_params([2, 16, 16, 3]), " 2-2-1 =", n_params([2, 2, 1]))
Xs, Ys = make_data("spiral", 300, Rng(77), 0.06)
Xs, Ys = np.array(Xs), np.array(Ys)
perm = np.random.RandomState(0).permutation(300); tr, te = perm[:210], perm[210:]
out["spiral"] = {"X": Xs.round(4).tolist(), "Y": Ys.tolist(), "test": sorted(te.tolist())}
depths, widths = [1, 2, 3, 4, 5], [2, 4, 8, 16, 32]
grid = []
Xtr, Ytr = torch.tensor(Xs[tr], dtype=torch.float32), torch.tensor(Ys[tr], dtype=torch.float32)[:, None]
Xte, Yte = torch.tensor(Xs[te], dtype=torch.float32), torch.tensor(Ys[te], dtype=torch.float32)[:, None]
for d in depths:
    row = []
    for wd in widths:
        accs = []
        for seed in range(3):
            torch.manual_seed(seed)
            layers, n_in = [], 2
            for _ in range(d):
                layers += [torch.nn.Linear(n_in, wd), torch.nn.ReLU()]; n_in = wd
            net = torch.nn.Sequential(*layers, torch.nn.Linear(n_in, 1))
            opt = torch.optim.Adam(net.parameters(), lr=0.01)
            for _ in range(1500):
                opt.zero_grad(); torch.nn.functional.binary_cross_entropy_with_logits(net(Xtr), Ytr).backward(); opt.step()
            with torch.no_grad():
                accs.append(((net(Xte) > 0).float() == Yte).float().mean().item())
        row.append({"acc": round(float(np.mean(accs)), 3), "params": n_params([2] + [wd] * d + [1])})
    grid.append(row)
    print("s6 depth", d, [c_["acc"] for c_ in row])
out["depthBreadth"] = {"depths": depths, "widths": widths, "grid": grid}

# 7절: Sequential과 class가 같은 parameter 수
# Section 7: Sequential and a class give the same parameter count
seq = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.ReLU(), torch.nn.Linear(16, 16), torch.nn.ReLU(), torch.nn.Linear(16, 3))
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 16); self.fc2 = torch.nn.Linear(16, 16); self.out = torch.nn.Linear(16, 3)
    def forward(self, x):
        x = torch.relu(self.fc1(x)); x = torch.relu(self.fc2(x)); return self.out(x)
print("s7 params Sequential / class:", sum(p.numel() for p in seq.parameters()), sum(p.numel() for p in Net().parameters()))

(ROOT / "assets" / "data").mkdir(exist_ok=True)
js = ("/* 4장 data와 미리 돌린 실험 결과. tools/ch04_numbers.py가 만든다 (손으로 고치지 말 것)\n"
      "   Chapter 4 data and precomputed experiment results. Generated by tools/ch04_numbers.py (do not edit by hand) */\n"
      "const CH04_DATA = " + json.dumps(out, separators=(",", ":")) + ";\n")
(ROOT / "assets" / "data" / "ch04_data.js").write_text(js, encoding="utf-8")
print("wrote assets/data/ch04_data.js", len(js), "bytes")
