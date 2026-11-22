# 15장 (1): 1D GAN 학습 과정, MNIST linear GAN, MNIST·FMNIST·CIFAR10 CNN GAN (PyTorch, MPS가 있으면 MPS)
# Chapter 15 (1): 1-D GAN training snapshots, an MNIST linear GAN, CNN GANs on MNIST, FMNIST and CIFAR10 (PyTorch, MPS when available)
# 결과는 assets/data/ch15_data.js의 "gan1d", "gans" 항목 (diffusion 항목은 ch15_diffusion.py가 쓴다)
# Results go to the "gan1d" and "gans" keys of assets/data/ch15_data.js (the diffusion keys come from ch15_diffusion.py)
# 실행 / run: python tools/ch15_gan.py <torchvision data 폴더/folder>
import sys, io, base64, pathlib, time
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from datajs import save_keys
MAKERS = ["tools/ch15_gan.py", "tools/ch15_diffusion.py"]
dev = "mps" if torch.backends.mps.is_available() else "cpu"
torch.set_num_threads(4)
bce = nn.BCEWithLogitsLoss()

# 1) 1D GAN: data 분포 두 가지 (봉우리 하나, 봉우리 둘), 100번마다 G의 histogram과 D(x)를 남긴다
# 1) 1-D GAN: two data distributions (one peak, two peaks); every 100 steps keep G's histogram and D(x)
def gan1d(sample_data, seed=0, steps=3000):
    torch.manual_seed(seed)
    G = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 1))
    D = nn.Sequential(nn.Linear(1, 32), nn.LeakyReLU(0.2), nn.Linear(32, 32), nn.LeakyReLU(0.2), nn.Linear(32, 1))
    oG, oD = torch.optim.Adam(G.parameters(), 1e-3, betas=(0.5, 0.999)), torch.optim.Adam(D.parameters(), 1e-3, betas=(0.5, 0.999))
    edges = torch.linspace(-5, 5, 51); grid = torch.linspace(-5, 5, 101).view(-1, 1); zfix = torch.randn(4000, 1); snaps = []
    for st in range(steps + 1):
        if st % 100 == 0:
            with torch.no_grad():
                g = G(zfix).squeeze(1); h = torch.histc(g.clamp(-5, 4.999), 50, -5, 5) / (len(g) * 0.2)
                snaps.append({"step": st, "hist": h.numpy().round(3).tolist(), "d": torch.sigmoid(D(grid)).squeeze(1).numpy().round(3).tolist(),
                              "mean": round(float(g.mean()), 3), "sd": round(float(g.std()), 3)})
        if st == steps: break
        x = sample_data(256); z = torch.randn(256, 1)
        oD.zero_grad(); (bce(D(x), torch.ones(256, 1)) + bce(D(G(z).detach()), torch.zeros(256, 1))).backward(); oD.step()
        oG.zero_grad(); bce(D(G(torch.randn(256, 1))), torch.ones(256, 1)).backward(); oG.step()
    return snaps
one = lambda n: 2 + 0.5 * torch.randn(n, 1)
two = lambda n: torch.where(torch.rand(n, 1) < 0.5, -2.0, 2.0) + 0.4 * torch.randn(n, 1)
g1 = {"one": gan1d(one), "two": gan1d(two)}
for k, s in g1.items(): print(f"s2 1-D GAN ({k} peak{'s' if k == 'two' else ''}): final G mean {s[-1]['mean']}, sd {s[-1]['sd']}")

# 2) 그림 GAN. 고정한 z 64개로 8×8 격자 PNG를 epoch마다 남긴다
# 2) Image GANs. Keep an 8×8 grid PNG from 64 fixed z at chosen epochs
def grid_png(x):
    x = ((x.clamp(-1, 1) + 1) / 2).cpu(); n, C, H, W = x.shape; g = x.view(8, 8, C, H, W).permute(0, 3, 1, 4, 2).reshape(8 * H, 8 * W, C).numpy()
    b = io.BytesIO(); Image.fromarray((g[..., 0] if C == 1 else g).__mul__(255).round().astype(np.uint8)).save(b, "PNG", optimize=True); return base64.b64encode(b.getvalue()).decode()
def load(cls):
    d = cls(sys.argv[1], train=True, download=False)
    x = torch.tensor(d.data if isinstance(d.data, np.ndarray) else d.data.numpy()).float() / 127.5 - 1
    return x.unsqueeze(1) if x.dim() == 3 else x.permute(0, 3, 1, 2)
def train_gan(G, D, X, epochs, keep, zdim, lr=2e-4, bs=128, seed=0):
    torch.manual_seed(seed); G, D = G.to(dev), D.to(dev); zfix = torch.randn(64, zdim, device=dev)
    oG, oD = torch.optim.Adam(G.parameters(), lr, betas=(0.5, 0.999)), torch.optim.Adam(D.parameters(), lr, betas=(0.5, 0.999)); out, t0 = {}, time.time()
    for ep in range(1, epochs + 1):
        perm = torch.randperm(len(X))
        for i in range(0, len(X) - bs + 1, bs):
            x = X[perm[i:i + bs]].to(dev); z = torch.randn(bs, zdim, device=dev); ones, zeros = torch.ones(bs, 1, device=dev), torch.zeros(bs, 1, device=dev)
            oD.zero_grad(); (bce(D(x), ones) + bce(D(G(z).detach()), zeros)).backward(); oD.step()
            oG.zero_grad(); bce(D(G(torch.randn(bs, zdim, device=dev))), ones).backward(); oG.step()
        if ep in keep:
            G.eval()
            with torch.no_grad(): out[str(ep)] = grid_png(G(zfix))
            G.train()
    print(f"   {epochs} epochs in {time.time() - t0:.0f} s, G params {sum(p.numel() for p in G.parameters())}, D params {sum(p.numel() for p in D.parameters())}")
    return out, sum(p.numel() for p in G.parameters()), sum(p.numel() for p in D.parameters())
class Reshape(nn.Module):
    def __init__(s, *shape): super().__init__(); s.shape = shape
    def forward(s, x): return x.view(x.shape[0], *s.shape)
def lin_G(z=64): return nn.Sequential(nn.Linear(z, 256), nn.LeakyReLU(0.2), nn.Linear(256, 512), nn.LeakyReLU(0.2), nn.Linear(512, 784), nn.Tanh(), Reshape(1, 28, 28))
def lin_D(): return nn.Sequential(nn.Flatten(), nn.Linear(784, 512), nn.LeakyReLU(0.2), nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Linear(256, 1))
def cnn_G(z=64): return nn.Sequential(nn.Linear(z, 128 * 7 * 7), nn.BatchNorm1d(128 * 7 * 7), nn.ReLU(), Reshape(128, 7, 7),
                                      nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh())
def cnn_D(): return nn.Sequential(nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2), nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2), nn.Flatten(), nn.Linear(128 * 7 * 7, 1))
def cifar_G(z=100): return nn.Sequential(Reshape(z, 1, 1), nn.ConvTranspose2d(z, 256, 4, 1, 0), nn.BatchNorm2d(256), nn.ReLU(), nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
                                          nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh())
def cifar_D(): return nn.Sequential(nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2), nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
                                    nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2), nn.Flatten(), nn.Linear(256 * 4 * 4, 1))
gans = {}
M, Fm, C = load(torchvision.datasets.MNIST), load(torchvision.datasets.FashionMNIST), load(torchvision.datasets.CIFAR10)
print("s3 MNIST linear GAN"); gans["mnist_lin"] = train_gan(lin_G(), lin_D(), M, 20, (1, 5, 20), 64)
print("s3 MNIST CNN GAN"); gans["mnist_cnn"] = train_gan(cnn_G(), cnn_D(), M, 10, (1, 3, 10), 64)
print("s3 FMNIST CNN GAN"); gans["fmnist_cnn"] = train_gan(cnn_G(), cnn_D(), Fm, 10, (1, 3, 10), 64)
print("s3 CIFAR10 CNN GAN"); gans["cifar_cnn"] = train_gan(cifar_G(), cifar_D(), C, 15, (1, 5, 15), 100)
gans = {k: {"png": v[0], "G": v[1], "D": v[2]} for k, v in gans.items()}
p = save_keys(15, {"gan1d": g1, "gans": gans}, MAKERS); print("wrote", p)
