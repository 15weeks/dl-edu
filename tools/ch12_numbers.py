# 12장 (1): MNIST ↔ FMNIST transfer (층별 freeze), autoencoder pretraining, 유명 CNN의 parameter 수, Gram matrix 예제 (PyTorch, CPU)
# Chapter 12 (1): MNIST ↔ FMNIST transfer (per-layer freezing), autoencoder pretraining, parameter counts of famous CNNs, the Gram-matrix example (PyTorch, CPU)
# 결과는 assets/data/ch12_data.js의 "transfer", "aepre", "famous" 항목에 쓴다 (다른 항목은 ch12_style.py, ch12_cifar.py가 쓴다)
# Results go to the "transfer", "aepre" and "famous" keys of assets/data/ch12_data.js (the other keys come from ch12_style.py and ch12_cifar.py)
# 실행 / run: python tools/ch12_numbers.py <torchvision data 폴더/folder>
import sys, json, pathlib
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ch12_numbers_io import save_keys, OUT
torch.set_num_threads(4)

def load(cls):
    tr, te = cls(sys.argv[1], train=True), cls(sys.argv[1], train=False)
    f = lambda d: torch.tensor(d.data.numpy().astype(np.float32) / 255.0).unsqueeze(1)
    return f(tr), tr.targets.clone(), f(te), te.targets.clone()
def body(): return nn.Sequential(nn.Conv2d(1, 8, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(8, 16, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten())
def train(net, X, Y, epochs, bs=64, seed=0, lr=1e-3, loss="ce"):
    g = torch.Generator().manual_seed(seed); opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr)
    for ep in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), bs):
            b = perm[i:i + bs]; opt.zero_grad()
            (F.cross_entropy(net(X[b]), Y[b]) if loss == "ce" else F.mse_loss(net(X[b]), X[b])).backward(); opt.step()
@torch.no_grad()
def acc(net, X, Y): return float(sum((net(X[i:i + 2500]).argmax(1) == Y[i:i + 2500]).sum() for i in range(0, len(X), 2500)) / len(X))
def subset(Y, n, seed):
    r = np.random.RandomState(seed); return torch.tensor(np.concatenate([r.choice(np.where(Y.numpy() == k)[0], n // 10, replace=False) for k in range(10)]))

# 0) Gram matrix 손 계산 예제
# 0) Hand-worked Gram-matrix example
Fm = torch.tensor([[1., 0, 1, 0], [1, 1, 0, 0], [0, 1, 0, 1]])
print("s6 Gram F Fᵀ =", (Fm @ Fm.T).int().tolist())

# 1) 유명한 CNN의 parameter 수 (weight 없이 구조만 만든다)
# 1) Parameter counts of famous CNNs (structure only, no weights)
famous = {}
for name, fn in (("AlexNet", torchvision.models.alexnet), ("VGG-16", torchvision.models.vgg16), ("ResNet-18", torchvision.models.resnet18)):
    m = fn(weights=None); famous[name] = {"params": sum(p.numel() for p in m.parameters()), "conv": sum(isinstance(x, nn.Conv2d) for x in m.modules()), "fc": sum(isinstance(x, nn.Linear) for x in m.modules())}
    print("s3", name, famous[name])

# 2) MNIST ↔ FMNIST transfer: 출발 data 전부로 학습 → 도착 data 500장 (class마다 50장)으로 다시 학습
# 2) MNIST ↔ FMNIST transfer: train on all of the source → retrain on 500 target images (50 per class)
D = {"mnist": load(torchvision.datasets.MNIST), "fmnist": load(torchvision.datasets.FashionMNIST)}
transfer = {}
for src, dst in (("mnist", "fmnist"), ("fmnist", "mnist")):
    torch.manual_seed(0); pre = nn.Sequential(body(), nn.Linear(784, 10)); train(pre, D[src][0], D[src][1], 3)
    print(f"s2 source {src} test acc", round(acc(pre, D[src][2], D[src][3]), 4))
    Xd, Yd, Xt, Yt = D[dst]; res = {}
    for init in ("pre", "rand"):
        for f1 in (0, 1):
            for f2 in (0, 1):
                accs = []
                for seed in range(3):
                    torch.manual_seed(100 + seed); net = nn.Sequential(body(), nn.Linear(784, 10))
                    if init == "pre": net[0].load_state_dict(pre[0].state_dict())
                    for p in net[0][0].parameters(): p.requires_grad = not f1
                    for p in net[0][3].parameters(): p.requires_grad = not f2
                    idx = subset(Yd, 500, seed); train(net, Xd[idx], Yd[idx], 40, bs=50, seed=seed)
                    accs.append(acc(net, Xt, Yt))
                res[f"{init}{f1}{f2}"] = [round(float(np.mean(accs)), 4), round(float(np.std(accs)), 4)]
                print(f"s2 {src}→{dst} init {init} freeze1 {f1} freeze2 {f2}: {res[f'{init}{f1}{f2}']}")
    torch.manual_seed(0); full = nn.Sequential(body(), nn.Linear(784, 10)); train(full, Xd, Yd, 3)
    res["full"] = round(acc(full, Xt, Yt), 4); print(f"s2 {dst} trained on all 60,000:", res["full"])
    transfer[f"{src}_{dst}"] = res

# 3) autoencoder pretraining: FMNIST 60,000장을 label 없이 복원 학습 → encoder + 새 head를 500장으로
# 3) Autoencoder pretraining: learn to rebuild all 60,000 FMNIST images without labels → encoder + new head on 500 images
Xd, Yd, Xt, Yt = D["fmnist"]
torch.manual_seed(0)
enc = body()[:-1]
dec = nn.Sequential(nn.ConvTranspose2d(16, 8, 2, stride=2), nn.ReLU(), nn.ConvTranspose2d(8, 1, 2, stride=2), nn.Sigmoid())
ae = nn.Sequential(enc, dec); train(ae, Xd, Yd, 3, loss="mse")
with torch.no_grad(): print("s5 conv AE test MSE", round(float(F.mse_loss(ae(Xt), Xt)), 5))
aepre = {}
for mode in ("rand", "ae_frozen", "ae_tune"):
    accs = []
    for seed in range(3):
        torch.manual_seed(100 + seed); net = nn.Sequential(body(), nn.Linear(784, 10))
        if mode != "rand": net[0][:-1].load_state_dict(enc.state_dict())
        if mode == "ae_frozen":
            for p in net[0].parameters(): p.requires_grad = False
        idx = subset(Yd, 500, seed); train(net, Xd[idx], Yd[idx], 40, bs=50, seed=seed); accs.append(acc(net, Xt, Yt))
    aepre[mode] = [round(float(np.mean(accs)), 4), round(float(np.std(accs)), 4)]
    print("s5 FMNIST 500 labels,", mode, aepre[mode])
save_keys({"famous": famous, "transfer": transfer, "aepre": aepre})
print("wrote", OUT)
