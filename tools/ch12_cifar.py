# 12장 (3): CIFAR10 5,000장으로 transfer: 작은 CNN을 처음부터 vs ImageNet으로 학습한 ResNet-18, VGG-16 (PyTorch, MPS)
# Chapter 12 (3): transfer with 5,000 CIFAR10 images: a small CNN from scratch vs ResNet-18 and VGG-16 trained on ImageNet (PyTorch, MPS)
# 결과는 assets/data/ch12_data.js의 "cifar" 항목에 쓴다
# Results go to the "cifar" key of assets/data/ch12_data.js
# 실행 / run: python tools/ch12_cifar.py <cifar-10-batches-py가 있는 폴더/folder holding cifar-10-batches-py>
import sys, pathlib, time
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ch12_numbers_io import save_keys

dev = "mps" if torch.backends.mps.is_available() else "cpu"
tr = torchvision.datasets.CIFAR10(sys.argv[1], train=True, download=False); te = torchvision.datasets.CIFAR10(sys.argv[1], train=False, download=False)
def pick(ds, n, seed):
    y = np.array(ds.targets); r = np.random.RandomState(seed)
    idx = np.concatenate([r.choice(np.where(y == k)[0], n // 10, replace=False) for k in range(10)])
    return torch.tensor(ds.data[idx]).permute(0, 3, 1, 2).float() / 255, torch.tensor(y[idx])
Xtr, Ytr = pick(tr, 5000, 0); Xte, Yte = pick(te, 2000, 1)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
out = {"n_train": 5000, "n_test": 2000}

def fit(net, X, Y, epochs, lr, bs=100, seed=0, prep=lambda x: x):
    g = torch.Generator().manual_seed(seed); opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr); net.train()
    for ep in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), bs):
            b = perm[i:i + bs]; opt.zero_grad(); F.cross_entropy(net(prep(X[b]).to(dev)), Y[b].to(dev)).backward(); opt.step()
@torch.no_grad()
def acc(net, X, Y, prep=lambda x: x, bs=200):
    net.eval(); return float(sum((net(prep(X[i:i + bs]).to(dev)).argmax(1).cpu() == Y[i:i + bs]).sum() for i in range(0, len(X), bs)) / len(X))

# 1) 작은 CNN을 처음부터 (11장 6절의 "같은 설계, CIFAR10": 136,874개)
# 1) A small CNN from scratch (chapter 11 section 6, "same design, CIFAR10": 136,874 parameters)
torch.manual_seed(0)
small = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                      nn.Flatten(), nn.Linear(2048, 64), nn.ReLU(), nn.Linear(64, 10)).to(dev)
t0 = time.time(); fit(small, Xtr, Ytr, 30, 1e-3)
out["scratch"] = {"acc": round(acc(small, Xte, Yte), 4), "trained": sum(p.numel() for p in small.parameters())}; print("s4 small CNN from scratch", out["scratch"], f"{time.time() - t0:.0f} s")

# ImageNet 모양으로: 224×224로 키우고 ImageNet 평균·표준편차로 정규화
# ImageNet style: enlarge to 224×224 and normalise with the ImageNet mean and SD
up = lambda s: (lambda x: (F.interpolate(x, size=s, mode="bilinear", align_corners=False) - MEAN) / STD)
@torch.no_grad()
def features(body, X, s, bs=100):
    body.eval(); return torch.cat([body(up(s)(X[i:i + bs]).to(dev)).flatten(1).cpu() for i in range(0, len(X), bs)])
def probe(Ftr, Fte, seed=0):
    torch.manual_seed(seed); head = nn.Linear(Ftr.shape[1], 10).to(dev); fit(head, Ftr, Ytr, 60, 1e-3); return acc(head, Fte, Yte), sum(p.numel() for p in head.parameters())

# 2) ResNet-18: 몸통을 얼리고 마지막 fc만 새로 (linear probe)
# 2) ResNet-18: freeze the body, train only a new final fc (linear probe)
rn = torchvision.models.resnet18(weights="IMAGENET1K_V1").to(dev); body = nn.Sequential(*list(rn.children())[:-1])
t0 = time.time(); a, n = probe(features(body, Xtr, 224), features(body, Xte, 224))
out["resnet_frozen"] = {"acc": round(a, 4), "trained": n}; print("s4 ResNet-18 frozen", out["resnet_frozen"], f"{time.time() - t0:.0f} s")
# 3) ResNet-18 전체를 조금씩 (fine-tune), 계산을 줄이려고 112×112
# 3) Fine-tune all of ResNet-18, at 112×112 to save compute
torch.manual_seed(0); rn = torchvision.models.resnet18(weights="IMAGENET1K_V1"); rn.fc = nn.Linear(512, 10); rn = rn.to(dev)
t0 = time.time(); fit(rn, Xtr, Ytr, 4, 1e-4, bs=50, prep=up(112))
out["resnet_tune"] = {"acc": round(acc(rn, Xte, Yte, prep=up(112)), 4), "trained": sum(p.numel() for p in rn.parameters())}; print("s4 ResNet-18 fine-tuned", out["resnet_tune"], f"{time.time() - t0:.0f} s")
# 4) VGG-16: 몸통과 classifier 앞 두 층을 얼리고 마지막 Linear(4096 → 10)만 새로
# 4) VGG-16: freeze the body and the first two classifier layers; train only a new last Linear(4096 → 10)
vg = torchvision.models.vgg16(weights="IMAGENET1K_V1").to(dev)
vbody = nn.Sequential(vg.features, vg.avgpool, nn.Flatten(), *list(vg.classifier.children())[:-1])
t0 = time.time(); a, n = probe(features(vbody, Xtr, 224), features(vbody, Xte, 224))
out["vgg_frozen"] = {"acc": round(a, 4), "trained": n}; print("s4 VGG-16 frozen", out["vgg_frozen"], f"{time.time() - t0:.0f} s")
save_keys({"cifar": out}); print("wrote cifar results")
