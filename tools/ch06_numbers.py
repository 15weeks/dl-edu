# 6장 예제 숫자 (PyTorch로 확인)와 wine quality data를 assets/data/ch06_data.js로
# Chapter 6 example numbers (checked in PyTorch) and the wine quality data written to assets/data/ch06_data.js
# 실행 / run: python tools/ch06_numbers.py winequality-red.csv
# data: P. Cortez et al., "Modeling wine preferences by data mining from physicochemical properties", Decision Support Systems 47(4), 2009. UCI Machine Learning Repository (CC BY 4.0)
import sys, json, math, pathlib
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sig = lambda z: 1 / (1 + math.exp(-z))

# 1절: activation 값과 기울기
# Section 1: activation values and slopes
for name, f in [("sigmoid", torch.sigmoid), ("tanh", torch.tanh), ("relu", torch.relu), ("gelu", torch.nn.functional.gelu)]:
    z = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True); y = f(z); y.sum().backward()
    print(f"s1 {name}: f(-2,0,2) = {y.detach().numpy().round(4)}  f' = {z.grad.numpy().round(4)}")

# 3절: 정답 y = 1인데 p = 0.01 (자신 있게 틀림). z = logit(p)
# Section 3: answer y = 1 but p = 0.01 (confidently wrong). z = logit(p)
for p in (0.01, 0.5, 0.9):
    z = torch.tensor(math.log(p / (1 - p)), requires_grad=True)
    mse = (torch.sigmoid(z) - 1) ** 2; mse.backward(); gm = z.grad.item()
    z2 = torch.tensor(math.log(p / (1 - p)), requires_grad=True)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(z2, torch.tensor(1.0)); bce.backward(); gb = z2.grad.item()
    print(f"s3 p={p}: MSE={mse.item():.4f} dMSE/dz={gm:.4f}   BCE={bce.item():.4f} dBCE/dz={gb:.4f}")

# 4절: 같은 계산의 PyTorch 이름들
# Section 4: PyTorch names for the same calculation
torch.set_printoptions(precision=4)
z = torch.tensor([[2.0, 1.0, 0.0]]); t = torch.tensor([0])
print("s4 CrossEntropyLoss(logits) =", round(torch.nn.CrossEntropyLoss()(z, t).item(), 4),
      " NLLLoss(log_softmax) =", round(torch.nn.NLLLoss()(torch.log_softmax(z, 1), t).item(), 4))
zb = torch.tensor([0.5]); yb = torch.tensor([1.0])
print("s4 BCEWithLogitsLoss =", round(torch.nn.BCEWithLogitsLoss()(zb, yb).item(), 4), " BCELoss(sigmoid) =", round(torch.nn.BCELoss()(torch.sigmoid(zb), yb).item(), 4))

# 6절: batch normalization 예제
# Section 6: batch normalization example
xb = torch.tensor([[2.0], [4.0], [6.0], [8.0]])
bn = torch.nn.BatchNorm1d(1, affine=False); bn.train(); out = bn(xb)
print("s6 BN([2,4,6,8]) =", out.squeeze().detach().numpy().round(3), " running mean after one batch =", round(bn.running_mean.item(), 3), " running var =", round(bn.running_var.item(), 3))

# 7절: wine quality. quality > 5 → 1 (좋음)
# Section 7: wine quality. quality > 5 → 1 (good)
path = sys.argv[1]
rows = [l.strip().split(";") for l in open(path).read().strip().split("\n")]
head = [h.strip('"') for h in rows[0]]; A = np.array([[float(v) for v in r] for r in rows[1:]])
X, q = A[:, :11], A[:, 11]; Y = (q > 5).astype(int)
print("s7 shape", X.shape, " good fraction", Y.mean().round(3))
print("s7 feature min/max:"); [print(f"   {head[i]:<22} {X[:, i].min():8.4f} {X[:, i].max():8.4f}  mean {X[:, i].mean():8.4f} sd {X[:, i].std():8.4f}") for i in range(11)]
perm = np.random.RandomState(0).permutation(len(X)); te, tr = perm[:320], perm[320:]
def run(norm, seed):
    torch.manual_seed(seed)
    mu, sd = X[tr].mean(0), X[tr].std(0)
    f = (lambda Z: (Z - mu) / sd) if norm else (lambda Z: Z)
    Xt, Yt = torch.tensor(f(X[tr]), dtype=torch.float32), torch.tensor(Y[tr], dtype=torch.float32)[:, None]
    Xe, Ye = torch.tensor(f(X[te]), dtype=torch.float32), torch.tensor(Y[te], dtype=torch.float32)[:, None]
    net = torch.nn.Sequential(torch.nn.Linear(11, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    opt = torch.optim.SGD(net.parameters(), lr=0.01)
    for ep in range(60):
        for i in range(0, len(Xt), 32):
            opt.zero_grad(); torch.nn.functional.binary_cross_entropy_with_logits(net(Xt[i:i + 32]), Yt[i:i + 32]).backward(); opt.step()
    with torch.no_grad():
        return ((net(Xe) > 0).float() == Ye).float().mean().item()
for norm in (False, True):
    print(f"s7 SGD lr 0.01, 60 epochs, z-score={norm}: test acc over 5 seeds", [round(run(norm, s), 3) for s in range(5)])
data = {"names": head[:11], "X": X.round(4).tolist(), "Y": Y.tolist(), "test": sorted(te.tolist())}
(ROOT / "assets" / "data").mkdir(exist_ok=True)
js = ("/* 6장 wine quality data (빨간 포도주 1599개, 특성 11개, quality > 5 → 1). tools/ch06_numbers.py가 만든다\n"
      "   Chapter 6 wine quality data (1599 red wines, 11 features, quality > 5 → 1). Made by tools/ch06_numbers.py\n"
      "   출처 / source: P. Cortez et al., Decision Support Systems 47(4), 2009; UCI Machine Learning Repository (CC BY 4.0) */\n"
      "const CH06_WINE = " + json.dumps(data, separators=(",", ":")) + ";\n")
(ROOT / "assets" / "data" / "ch06_data.js").write_text(js, encoding="utf-8")
print("wrote assets/data/ch06_data.js", len(js), "bytes")
