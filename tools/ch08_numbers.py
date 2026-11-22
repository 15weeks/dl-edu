# 8장: MNIST로 FFN을 학습하고 page에 넣을 weight·실험 결과를 assets/data/ch08_data.js로 쓴다 (PyTorch, CPU)
# Chapter 8: train FFNs on MNIST and write the weights and experiment results for the page to assets/data/ch08_data.js (PyTorch, CPU)
# 실행 / run: python tools/ch08_numbers.py <torchvision data 폴더/folder>
# data: MNIST (LeCun, Cortes, Burges), torchvision으로 내려받는다 / downloaded with torchvision
import sys, json, base64, pathlib
import numpy as np
import torch, torchvision

ROOT = pathlib.Path(__file__).resolve().parent.parent
torch.set_num_threads(4)
root = sys.argv[1]
tr = torchvision.datasets.MNIST(root, train=True, download=False); te = torchvision.datasets.MNIST(root, train=False, download=False)
Xtr28 = tr.data.numpy(); Ytr = tr.targets.numpy(); Xte28 = te.data.numpy(); Yte = te.targets.numpy()
pool = lambda X: X.reshape(len(X), 14, 2, 14, 2).mean((2, 4)).reshape(len(X), 196) / 255.0   # 28×28 → 14×14, 0~1
Xtr, Xte = pool(Xtr28.astype(np.float32)), pool(Xte28.astype(np.float32))
T = lambda a: torch.tensor(a, dtype=torch.float32)

def make(width, seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(196, width), torch.nn.ReLU(), torch.nn.Linear(width, 10))

def train(net, X, Y, steps=None, epochs=None, lr=1e-3, bs=64, seed=0, log=None):
    g = torch.Generator().manual_seed(seed); opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xt, Yt = T(X), torch.tensor(Y); n = len(X); s = 0; ep = 0
    while True:
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; opt.zero_grad(); torch.nn.functional.cross_entropy(net(Xt[idx]), Yt[idx]).backward(); opt.step(); s += 1
            if steps and s >= steps: return
        ep += 1
        if log is not None: log(ep)
        if epochs and ep >= epochs: return

def acc(net, X, Y):
    with torch.no_grad(): return float((net(T(X)).argmax(1).numpy() == Y).mean())

out = {}
# 2~3절: 196-64-10, 5 epoch. 학습 전후 weight 분포
# Sections 2-3: 196-64-10, 5 epochs. Weight distribution before and after training
net = make(64, 0); W0 = net[0].weight.detach().numpy().copy()
train(net, Xtr, Ytr, epochs=5)
a = acc(net, Xte, Yte); print("s2 196-64-10 test accuracy", round(a, 4), " params", sum(p.numel() for p in net.parameters()))
W1 = net[0].weight.detach().numpy(); b1 = net[0].bias.detach().numpy(); W2 = net[2].weight.detach().numpy(); b2 = net[2].bias.detach().numpy()
def q8(W):
    s = float(np.abs(W).max()) / 127; q = np.clip(np.round(W / s), -127, 127).astype(np.int8)
    return {"s": s, "shape": list(W.shape), "q": base64.b64encode(q.tobytes()).decode()}
out["net"] = {"W1": q8(W1), "b1": b1.round(5).tolist(), "W2": q8(W2), "b2": b2.round(5).tolist(), "acc": round(a, 4)}
# int8로 줄인 weight의 정확도 (page에서 쓰는 것)
# Accuracy with the int8 weights (what the page uses)
def deq(d): return (np.frombuffer(base64.b64decode(d["q"]), dtype=np.int8).astype(np.float32) * d["s"]).reshape(d["shape"])
h = np.maximum(Xte @ deq(out["net"]["W1"]).T + b1, 0); p = h @ deq(out["net"]["W2"]).T + b2
print("s2 int8 weights test accuracy", round(float((p.argmax(1) == Yte).mean()), 4))
bins = np.linspace(-0.4, 0.4, 41)
out["hist"] = {"bins": bins.round(3).tolist(), "before": np.histogram(W0, bins)[0].tolist(), "after": np.histogram(W1, bins)[0].tolist(),
               "sd_before": round(float(W0.std()), 4), "sd_after": round(float(W1.std()), 4)}
print("s3 layer-1 weight SD before", round(float(W0.std()), 4), "after", round(float(W1.std()), 4))
norms = np.linalg.norm(W2, axis=0); top = np.argsort(-norms)[:8]
out["templates"] = [W1[j].round(3).tolist() for j in top]

# 4절: 섞은 pixel과 민 숫자
# Section 4: scrambled pixels and shifted digits
perm = np.random.RandomState(1).permutation(196)
nets = make(64, 0); train(nets, Xtr[:, perm], Ytr, epochs=5)
out["scramble"] = {"perm": perm.tolist(), "acc": round(acc(nets, Xte[:, perm], Yte), 4)}
print("s4 scrambled-pixel FFN accuracy", out["scramble"]["acc"])
shift_acc = []
for s in range(0, 7):
    Xs = np.zeros_like(Xte28); Xs[:, :, s:] = Xte28[:, :, :28 - s] if s else Xte28
    shift_acc.append(round(acc(net, pool(Xs.astype(np.float32)), Yte), 4))
out["shift"] = shift_acc; print("s4 accuracy when shifted right by 0..6 pixels:", shift_acc)

# 6절: data 크기 × network 너비
# Section 6: data size × network width
sizes, widths = [300, 1000, 3000, 10000, 60000], [8, 64, 256]
grid = []
for wd in widths:
    row = []
    for n in sizes:
        nn_ = make(wd, 1); train(nn_, Xtr[:n], Ytr[:n], steps=3000, seed=1); row.append(round(acc(nn_, Xte, Yte), 4))
    grid.append(row); print("s6 width", wd, row)
out["sizeGrid"] = {"sizes": sizes, "widths": widths, "acc": grid}

# 8절: 가장 좋은 model 남기기. 학습 1000개, dev 5000개, 196-256-10, 40 epoch
# Section 8: keep the best model. 1000 training, 5000 dev, 196-256-10, 40 epochs
netc = make(256, 2); logs = []
Xd, Yd = Xtr[50000:55000], Ytr[50000:55000]
train(netc, Xtr[:1000], Ytr[:1000], epochs=40, seed=2, log=lambda ep: logs.append([ep, round(acc(netc, Xtr[:1000], Ytr[:1000]), 4), round(acc(netc, Xd, Yd), 4)]))
best = max(logs, key=lambda r: r[2]); out["ckpt"] = {"log": logs, "best": best}
print("s8 best dev epoch", best, " last", logs[-1])

# 그림에 쓸 시험 숫자 (class마다 3개, 28×28 uint8)
# Test digits for the figures (3 per class, 28×28 uint8)
idx = []
for k in range(10): idx += list(np.where(Yte == k)[0][:3])
out["samples"] = {"y": Yte[idx].tolist(), "x": base64.b64encode(Xte28[idx].astype(np.uint8).tobytes()).decode()}
(ROOT / "assets" / "data").mkdir(exist_ok=True)
js = ("/* 8장 MNIST FFN의 weight(int8)와 실험 결과. tools/ch08_numbers.py가 만든다 (손으로 고치지 말 것)\n"
      "   Chapter 8 MNIST FFN weights (int8) and experiment results. Made by tools/ch08_numbers.py (do not edit by hand)\n"
      "   data: MNIST (Y. LeCun, C. Cortes, C. J. C. Burges) */\n"
      "const CH08_DATA = " + json.dumps(out, separators=(",", ":")) + ";\n")
(ROOT / "assets" / "data" / "ch08_data.js").write_text(js, encoding="utf-8")
print("wrote assets/data/ch08_data.js", len(js), "bytes")
