# 9장 예제 숫자: 초기화별 신호 크기(PyTorch), 심장병 예측과 빠진 값 채우기 프로젝트
# Chapter 9 example numbers: signal size per initialization (PyTorch), heart-disease and missing-value projects
# 실행 / run: python tools/ch09_numbers.py heart.csv winequality-red.csv
# data: Cleveland heart disease (Janosi, Steinbrunn, Pfisterer, Detrano, UCI, CC BY 4.0); wine quality (Cortez et al. 2009, UCI, CC BY 4.0)
import sys, math
import numpy as np
import torch

torch.manual_seed(0)
# 5~7절: 20층에서 activation 표준편차 (입력 표준편차 1, 너비 256)
# Sections 5-7: activation SD over 20 layers (input SD 1, width 256)
x = torch.randn(1000, 256)
for name, act, init in [("tanh + Xavier", torch.tanh, "xavier"), ("ReLU + Xavier", torch.relu, "xavier"), ("ReLU + Kaiming", torch.relu, "kaiming"), ("tanh, SD 1", torch.tanh, "one")]:
    h = x; sds = []
    for _ in range(20):
        W = torch.empty(256, 256)
        if init == "xavier": torch.nn.init.xavier_normal_(W)
        elif init == "kaiming": torch.nn.init.kaiming_normal_(W, nonlinearity="relu")
        else: torch.nn.init.normal_(W, std=1.0)
        h = act(h @ W.T); sds.append(h.std().item())
    print(f"s7 {name:<16} SD after layer 1, 5, 10, 20:", [round(sds[i], 4) for i in (0, 4, 9, 19)])
print("s5 nn.Linear(4, 3).weight.shape =", tuple(torch.nn.Linear(4, 3).weight.shape))

# 9절-1: 심장병 예측 (13 특성, 진단 > 0 → 1)
# Section 9-1: heart disease (13 features, diagnosis > 0 → 1)
rows = [l.strip().split(",") for l in open(sys.argv[1]) if l.strip() and "?" not in l]
A = np.array([[float(v) for v in r] for r in rows]); X, Y = A[:, :13], (A[:, 13] > 0).astype(np.float32)
print("s9 heart: rows", len(A), " positive fraction", Y.mean().round(3))
accs = []
for seed in range(10):
    rs = np.random.RandomState(seed); p = rs.permutation(len(X)); te, tr = p[:60], p[60:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Xt, Xe = torch.tensor((X[tr] - mu) / sd, dtype=torch.float32), torch.tensor((X[te] - mu) / sd, dtype=torch.float32)
    torch.manual_seed(seed); net = torch.nn.Sequential(torch.nn.Linear(13, 16), torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(16, 1))
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-3)
    for _ in range(200):
        net.train(); opt.zero_grad(); torch.nn.functional.binary_cross_entropy_with_logits(net(Xt)[:, 0], torch.tensor(Y[tr])).backward(); opt.step()
    net.eval()
    with torch.no_grad(): accs.append(float(((net(Xe)[:, 0] > 0).numpy() == Y[te]).mean()))
print("s9 heart test accuracy (60 held out, 10 splits): mean", round(float(np.mean(accs)), 3), " sd", round(float(np.std(accs)), 3))

# 9절-2: 빠진 값 채우기: wine의 residual sugar를 나머지 10개로 예측
# Section 9-2: missing values: predict wine residual sugar from the other 10 features
rows = [l.strip().split(";") for l in open(sys.argv[2]).read().strip().split("\n")[1:]]
W = np.array([[float(v) for v in r] for r in rows])[:, :11]
tgt = 3; feats = [i for i in range(11) if i != tgt]
rs = np.random.RandomState(0); p = rs.permutation(len(W)); te, tr = p[:320], p[320:]
mu, sd = W[tr].mean(0), W[tr].std(0)
Z = (W - mu) / sd
Xt, Yt = torch.tensor(Z[tr][:, feats], dtype=torch.float32), torch.tensor(Z[tr][:, tgt], dtype=torch.float32)
Xe, Ye = torch.tensor(Z[te][:, feats], dtype=torch.float32), Z[te][:, tgt]
torch.manual_seed(0); net = torch.nn.Sequential(torch.nn.Linear(10, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)); opt = torch.optim.Adam(net.parameters(), lr=0.005)
for _ in range(400):
    opt.zero_grad(); torch.nn.functional.mse_loss(net(Xt)[:, 0], Yt).backward(); opt.step()
with torch.no_grad(): pr = net(Xe)[:, 0].numpy()
r = np.corrcoef(pr, Ye)[0, 1]; mean_r = 0.0
print("s9 wine residual sugar imputation: test r =", round(float(r), 3), " (filling with the mean gives r = 0)")
print("s9 typical error in g/L:", round(float(np.sqrt(np.mean((pr - Ye) ** 2)) * sd[tgt]), 3), " vs filling with the mean:", round(float(np.std(Ye) * sd[tgt]), 3))
