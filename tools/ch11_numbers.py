# 11장: MNIST CNN, FFN과의 이동 비교, Gaussian 폭 분류, FMNIST CNN을 학습하고 page용 data를 assets/data/ch11_data.js로 쓴다 (PyTorch, CPU)
# Chapter 11: train an MNIST CNN, compare it with an FFN under shifts, classify Gaussian widths, train an FMNIST CNN, and write the page data to assets/data/ch11_data.js (PyTorch, CPU)
# 실행 / run: python tools/ch11_numbers.py <torchvision data 폴더/folder>
import sys, json, base64, pathlib
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parent.parent
torch.set_num_threads(4)
def load(cls):
    tr, te = cls(sys.argv[1], train=True), cls(sys.argv[1], train=False)
    f = lambda d: torch.tensor(d.data.numpy().astype(np.float32) / 255.0).unsqueeze(1)
    return f(tr), tr.targets.clone(), f(te), te.targets.clone()

def train(net, X, Y, epochs, lr=1e-3, seed=0, bs=64):
    g = torch.Generator().manual_seed(seed); opt = torch.optim.Adam(net.parameters(), lr=lr)
    for ep in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), bs):
            b = perm[i:i + bs]; opt.zero_grad(); F.cross_entropy(net(X[b]), Y[b]).backward(); opt.step()
@torch.no_grad()
def acc(net, X, Y): return float(sum((net(X[i:i + 2000]).argmax(1) == Y[i:i + 2000]).sum() for i in range(0, len(X), 2000)) / len(X))
nparams = net_ = lambda net: sum(p.numel() for p in net.parameters())

def q8(W):
    W = W.detach().numpy(); s = float(np.abs(W).max()) / 127 or 1.0; q = np.clip(np.round(W / s), -127, 127).astype(np.int8)
    return {"s": s, "shape": list(W.shape), "q": base64.b64encode(q.tobytes()).decode()}, torch.tensor(q.astype(np.float32) * s)
b64u8 = lambda a: base64.b64encode((a * 255).round().clip(0, 255).astype(np.uint8).tobytes()).decode()
out = {}

# 0) 손 계산 예제 확인 (page의 calc 상자)
# 0) Check the hand-worked examples (the page's calc boxes)
X1 = torch.zeros(1, 1, 7, 7); X1[..., 2:5, 2:5] = 1
Kv = torch.tensor([[-1., 0, 1]] * 3).reshape(1, 1, 3, 3)
print("s1 conv 7×7 square * vertical edge:\n", F.conv2d(X1, Kv)[0, 0].int().tolist())
T = F.conv_transpose2d(torch.tensor([[1., 2], [3, 4]]).reshape(1, 1, 2, 2), torch.ones(1, 1, 3, 3), stride=2)
print("s4 transpose conv stride 2:\n", T[0, 0].int().tolist())
P = torch.tensor([[1., 3, 2, 0], [4, 2, 1, 1], [0, 1, 5, 2], [2, 0, 3, 4]]).reshape(1, 1, 4, 4)
print("s5 max pool", F.max_pool2d(P, 2)[0, 0].tolist(), "mean pool", F.avg_pool2d(P, 2)[0, 0].tolist())
print("s5 stride-2 conv 8→8, 3×3 params", sum(p.numel() for p in nn.Conv2d(8, 8, 3, stride=2, padding=1).parameters()))

# 1) MNIST CNN: conv(1→8, 5×5, pad 2) → ReLU → maxpool 2 → conv(8→16, 5×5, pad 2) → ReLU → maxpool 2 → Linear(784, 10)
# 1) MNIST CNN: conv(1→8, 5×5, pad 2) → ReLU → maxpool 2 → conv(8→16, 5×5, pad 2) → ReLU → maxpool 2 → Linear(784, 10)
Xtr, Ytr, Xte, Yte = load(torchvision.datasets.MNIST)
torch.manual_seed(0)
cnn = nn.Sequential(nn.Conv2d(1, 8, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(8, 16, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(784, 10))
train(cnn, Xtr, Ytr, 3)
a_cnn = acc(cnn, Xte, Yte); print("s7 MNIST CNN params", nparams(cnn), "test acc", round(a_cnn, 4))
# int8로 줄인 weight로도 같은지 확인 (page는 int8 weight로 계산한다)
# Check that int8 weights give the same result (the page computes with int8 weights)
packed = []
for m in cnn:
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        d, Wq = q8(m.weight); m.weight.data = Wq; packed.append({"W": d, "b": m.bias.detach().numpy().round(5).tolist()})
a_q = acc(cnn, Xte, Yte); print("s7 MNIST CNN int8 test acc", round(a_q, 4))
out["cnn"] = {"layers": packed, "acc": round(a_q, 4), "params": nparams(cnn)}

# FFN 784-64-10과 비교: 왼쪽·오른쪽으로 0~6칸 옮긴 시험 그림의 정확도
# Compare with an FFN 784-64-10: accuracy on test images shifted 0 to 6 pixels left or right
torch.manual_seed(0)
ffn = nn.Sequential(nn.Flatten(), nn.Linear(784, 64), nn.ReLU(), nn.Linear(64, 10))
train(ffn, Xtr, Ytr, 3)
print("s7 FFN params", nparams(ffn), "test acc", round(acc(ffn, Xte, Yte), 4))
shifts = list(range(0, 7)); sc, sf = [], []
for s in shifts:
    ac, af = [], []
    for sgn in ((1, -1) if s else (1,)):
        # 옮기고, 반대편으로 넘어온 열은 0으로
        # Shift, and zero the columns that wrapped round to the other side
        Xs = torch.roll(Xte, s * sgn, dims=3)
        if s: (Xs[..., :s] if sgn > 0 else Xs[..., -s:]).zero_()
        ac.append(acc(cnn, Xs, Yte)); af.append(acc(ffn, Xs, Yte))
    sc.append(round(float(np.mean(ac)), 4)); sf.append(round(float(np.mean(af)), 4))
print("s7 shift", shifts, "\n   CNN", sc, "\n   FFN", sf)
out["shift"] = {"s": shifts, "cnn": sc, "ffn": sf, "params": [nparams(cnn), nparams(ffn)]}
idx = []
for k in range(10): idx += list(np.where(Yte.numpy() == k)[0][:2])
out["mnist"] = {"y": Yte[idx].tolist(), "x": b64u8(Xte[idx, 0].numpy())}

# 2) Gaussian 폭 분류: 32×32 그림, 좁은 것(σ 1.5~2.5) vs 넓은 것(σ 3~4), 잡음 표준편차 0.3
# 2) Gaussian width classification: 32×32 images, narrow (σ 1.5–2.5) vs wide (σ 3–4), noise SD 0.3
def gauss(n, seed):
    r = np.random.RandomState(seed); y = r.randint(0, 2, n); sig = np.where(y == 0, r.uniform(1.5, 2.5, n), r.uniform(3, 4, n))
    cx, cy = r.uniform(8, 24, n), r.uniform(8, 24, n); g = np.arange(32)
    X = np.exp(-((g[None, None, :] - cx[:, None, None]) ** 2 + (g[None, :, None] - cy[:, None, None]) ** 2) / (2 * sig[:, None, None] ** 2))
    X = X + 0.3 * r.randn(n, 32, 32)
    return torch.tensor(X, dtype=torch.float32).unsqueeze(1), torch.tensor(y)
Gtr, Gytr = gauss(4000, 1); Gte, Gyte = gauss(1000, 2)
torch.manual_seed(1)
gcnn = nn.Sequential(nn.Conv2d(1, 6, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2), nn.Conv2d(6, 4, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2), nn.Flatten(), nn.Linear(256, 2))
train(gcnn, Gtr, Gytr, 10, seed=1)
torch.manual_seed(1); gffn = nn.Sequential(nn.Flatten(), nn.Linear(1024, 8), nn.ReLU(), nn.Linear(8, 2)); train(gffn, Gtr, Gytr, 10, seed=1)
out["gauss"] = {"cnn": round(acc(gcnn, Gte, Gyte), 4), "ffn": round(acc(gffn, Gte, Gyte), 4), "params": [nparams(gcnn), nparams(gffn)]}
print("s7 Gaussian widths: CNN", out["gauss"]["cnn"], "params", nparams(gcnn), "| FFN", out["gauss"]["ffn"], "params", nparams(gffn))

# 3) FMNIST CNN: conv(1→16, 3×3, pad 1) → pool → conv(16→32, 3×3, pad 1) → pool → 1568 → 64 → 10
# 3) FMNIST CNN: conv(1→16, 3×3, pad 1) → pool → conv(16→32, 3×3, pad 1) → pool → 1568 → 64 → 10
Ftr, Fytr, Fte, Fyte = load(torchvision.datasets.FashionMNIST)
torch.manual_seed(2)
fcnn = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(1568, 64), nn.ReLU(), nn.Linear(64, 10))
train(fcnn, Ftr, Fytr, 5, seed=2)
with torch.no_grad(): pred = torch.cat([fcnn(Fte[i:i + 2000]).argmax(1) for i in range(0, len(Fte), 2000)])
cm = np.zeros((10, 10), int)
for t, p in zip(Fyte.numpy(), pred.numpy()): cm[t, p] += 1
fa = float((pred == Fyte).float().mean()); print("s9 FMNIST CNN params", nparams(fcnn), "test acc", round(fa, 4))
print("s9 per-class recall", (cm.diagonal() / cm.sum(1)).round(3).tolist())
fidx = [int(np.where(Fyte.numpy() == k)[0][0]) for k in range(10)]
out["fmnist"] = {"acc": round(fa, 4), "params": nparams(fcnn), "cm": cm.tolist(), "x": b64u8(Fte[fidx, 0].numpy())}
js = ("/* 11장 MNIST CNN weight(int8), 이동 비교, Gaussian, FMNIST 결과. tools/ch11_numbers.py가 만든다 (손으로 고치지 말 것)\n"
      "   Chapter 11 MNIST CNN weights (int8), shift comparison, Gaussian and FMNIST results. Made by tools/ch11_numbers.py (do not edit by hand) */\n"
      "const CH11_DATA = " + json.dumps(out, separators=(",", ":")) + ";\n")
(ROOT / "assets" / "data" / "ch11_data.js").write_text(js, encoding="utf-8")
print("wrote assets/data/ch11_data.js", len(js), "bytes")
