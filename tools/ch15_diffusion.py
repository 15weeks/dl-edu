# 15장 (2): noise schedule, 2D 점 구름 diffusion, MNIST 조건부 DDPM과 classifier-free guidance (PyTorch, MPS가 있으면 MPS)
# Chapter 15 (2): the noise schedule, 2-D point-cloud diffusion, a class-conditional MNIST DDPM with classifier-free guidance (PyTorch, MPS when available)
# 결과는 assets/data/ch15_data.js의 "sched", "diff2d", "ddpm" 항목
# Results go to the "sched", "diff2d" and "ddpm" keys of assets/data/ch15_data.js
# 실행 / run: python tools/ch15_diffusion.py <torchvision data 폴더/folder>
import sys, io, base64, math, pathlib, time
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

# 1) noise schedule: T = 1000, β가 1e-4에서 0.02까지 곧게 (DDPM 논문)
# 1) Noise schedule: T = 1000, β linear from 1e-4 to 0.02 (the DDPM paper)
T = 1000; beta = torch.linspace(1e-4, 0.02, T); alpha = 1 - beta; abar = torch.cumprod(alpha, 0)
sched = {t: round(float(abar[t - 1]), 5) for t in (1, 100, 250, 500, 750, 1000)}; print("s5 ᾱ_t:", sched)

def tsin(t, dim):
    # 시간 t를 사인·코사인 벡터로 (14장 위치 encoding과 같은 식)
    # Time t as a sine/cosine vector (the same formula as chapter 14's positional encoding)
    f = torch.exp(-math.log(10000) * torch.arange(dim // 2, device=t.device) / (dim // 2)); a = t.float()[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], 1)

# 2) 2D 점 구름: swiss roll 모양, T2 = 100, ε를 맞히는 MLP
# 2) 2-D point cloud: a swiss-roll shape, T2 = 100, an MLP that predicts ε
def roll(n, g):
    t = 1.5 * math.pi * (1 + 2 * torch.rand(n, generator=g)); x = torch.stack([t * t.cos(), t * t.sin()], 1) / 10
    return x + 0.03 * torch.randn(n, 2, generator=g)
T2 = 100; b2 = torch.linspace(1e-4, 0.1, T2); a2 = 1 - b2; ab2 = torch.cumprod(a2, 0)
class EpsMLP(nn.Module):
    def __init__(s): super().__init__(); s.net = nn.Sequential(nn.Linear(2 + 16, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2))
    def forward(s, x, t): return s.net(torch.cat([x, tsin(t, 16)], 1))
g = torch.Generator().manual_seed(0); torch.manual_seed(0); net2 = EpsMLP(); opt = torch.optim.Adam(net2.parameters(), 1e-3)
for it in range(6000):
    x0 = roll(512, g); t = torch.randint(0, T2, (512,), generator=g); e = torch.randn(512, 2, generator=g)
    xt = ab2[t].sqrt()[:, None] * x0 + (1 - ab2[t]).sqrt()[:, None] * e
    opt.zero_grad(); loss = F.mse_loss(net2(xt, t), e); loss.backward(); opt.step()
print("s3 2-D diffusion final training loss", round(float(loss), 4))
keep = (0, 10, 20, 40, 60, 80, 100); g = torch.Generator().manual_seed(7)
x0 = roll(800, g); e = torch.randn(800, 2, generator=g)
fwd = {str(k): (x0 if k == 0 else ab2[k - 1].sqrt() * x0 + (1 - ab2[k - 1]).sqrt() * e).numpy().round(3).tolist() for k in keep}
x = torch.randn(800, 2, generator=g); rev = {"100": x.numpy().round(3).tolist()}
with torch.no_grad():
    for k in range(T2, 0, -1):
        t = torch.full((800,), k - 1); eh = net2(x, t)
        x = (x - b2[k - 1] / (1 - ab2[k - 1]).sqrt() * eh) / a2[k - 1].sqrt()
        if k > 1: x = x + b2[k - 1].sqrt() * torch.randn(800, 2, generator=g)
        if (k - 1) in keep: rev[str(k - 1)] = x.numpy().round(3).tolist()
diff2d = {"T": T2, "keep": list(keep), "fwd": fwd, "rev": rev}

# 3) MNIST 조건부 DDPM: 작은 U-Net, 10%는 class를 지워 학습 → classifier-free guidance
# 3) Class-conditional MNIST DDPM: a small U-Net; the class is dropped 10% of the time → classifier-free guidance
class Block(nn.Module):
    def __init__(s, i, o, e=128):
        super().__init__(); s.c1 = nn.Conv2d(i, o, 3, padding=1); s.n1 = nn.GroupNorm(8, o); s.c2 = nn.Conv2d(o, o, 3, padding=1); s.n2 = nn.GroupNorm(8, o)
        s.emb = nn.Linear(e, o); s.skip = nn.Conv2d(i, o, 1) if i != o else nn.Identity()
    def forward(s, x, e):
        h = F.silu(s.n1(s.c1(x))) + s.emb(e)[:, :, None, None]; h = F.silu(s.n2(s.c2(h))); return h + s.skip(x)
class UNet(nn.Module):
    def __init__(s):
        super().__init__(); s.tm = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 128)); s.cls = nn.Embedding(11, 128)
        s.inp = nn.Conv2d(1, 32, 3, padding=1); s.b1 = Block(32, 32); s.b2 = Block(32, 64); s.b3 = Block(64, 128); s.b4 = Block(192, 64); s.b5 = Block(96, 32); s.out = nn.Conv2d(32, 1, 3, padding=1)
    def forward(s, x, t, c):
        e = s.tm(tsin(t, 64)) + s.cls(c); h1 = s.b1(s.inp(x), e); h2 = s.b2(F.avg_pool2d(h1, 2), e); m = s.b3(F.avg_pool2d(h2, 2), e)
        u = s.b4(torch.cat([F.interpolate(m, scale_factor=2), h2], 1), e); u = s.b5(torch.cat([F.interpolate(u, scale_factor=2), h1], 1), e); return s.out(u)
tr = torchvision.datasets.MNIST(sys.argv[1], train=True); X = (tr.data.float() / 127.5 - 1).unsqueeze(1); Y = tr.targets.clone()
torch.manual_seed(0); unet = UNet().to(dev); opt = torch.optim.Adam(unet.parameters(), 5e-4); B, A, AB = beta.to(dev), alpha.to(dev), abar.to(dev)
print("s6 U-Net parameters", sum(p.numel() for p in unet.parameters())); t0 = time.time()
for ep in range(12):
    perm = torch.randperm(len(X))
    for i in range(0, len(X) - 127, 128):
        x0 = X[perm[i:i + 128]].to(dev); c = Y[perm[i:i + 128]].to(dev); c = torch.where(torch.rand(128, device=dev) < 0.1, torch.full_like(c, 10), c)
        t = torch.randint(0, T, (128,), device=dev); e = torch.randn_like(x0); xt = AB[t].sqrt()[:, None, None, None] * x0 + (1 - AB[t]).sqrt()[:, None, None, None] * e
        opt.zero_grad(); loss = F.mse_loss(unet(xt, t, c), e); loss.backward(); opt.step()
    print(f"s6 epoch {ep + 1}: loss {float(loss):.4f}  ({time.time() - t0:.0f} s)")
unet.eval()
@torch.no_grad()
def sample(c, w, seed, keep=()):
    # DDPM 거꾸로 1000걸음. w = guidance 세기: ε = (1 + w) ε(c) − w ε(빈 class)
    # DDPM reverse, 1000 steps. w = guidance scale: ε = (1 + w) ε(c) − w ε(no class)
    g = torch.Generator(device="cpu").manual_seed(seed); n = len(c); x = torch.randn(n, 1, 28, 28, generator=g).to(dev); shots = {}
    null = torch.full_like(c, 10)
    for k in range(T, 0, -1):
        t = torch.full((n,), k - 1, device=dev)
        if w == 0 and (c == 10).all(): eh = unet(x, t, c)
        else: ec, eu = unet(torch.cat([x, x]), torch.cat([t, t]), torch.cat([c, null])).chunk(2); eh = (1 + w) * ec - w * eu
        x = (x - B[k - 1] / (1 - AB[k - 1]).sqrt() * eh) / A[k - 1].sqrt()
        if k > 1: x = x + B[k - 1].sqrt() * torch.randn(x.shape, generator=g).to(dev)
        if (k - 1) in keep: shots[str(k - 1)] = x.clamp(-1, 1).cpu()
    return x.clamp(-1, 1).cpu(), shots
def png(x, rows, cols):
    x = ((x + 1) / 2)[:, 0]; H = x.shape[1]; g = x.view(rows, cols, H, H).permute(0, 2, 1, 3).reshape(rows * H, cols * H).numpy()
    b = io.BytesIO(); Image.fromarray((g * 255).round().astype(np.uint8)).save(b, "PNG", optimize=True); return base64.b64encode(b.getvalue()).decode()
steps_keep = (1000, 800, 600, 400, 300, 200, 100, 50, 0)
_, shots = sample(torch.full((16,), 10, device=dev), 0.0, 1, keep=tuple(k for k in steps_keep if k < 1000))
g0 = torch.Generator().manual_seed(1); shots["1000"] = torch.randn(16, 1, 28, 28, generator=g0).clamp(-1, 1)
uncond = {k: png(v, 4, 4) for k, v in shots.items()}
# guidance 세기에 따른 조건부 sample과, 작은 CNN 분류기로 잰 "요청한 숫자로 읽히는 비율"
# Conditional samples by guidance scale, and the share read as the requested digit by a small CNN classifier
torch.manual_seed(0); clf = nn.Sequential(nn.Conv2d(1, 8, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(8, 16, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(784, 10)).to(dev)
oc = torch.optim.Adam(clf.parameters(), 1e-3)
for ep in range(2):
    perm = torch.randperm(len(X))
    for i in range(0, len(X) - 127, 128):
        oc.zero_grad(); F.cross_entropy(clf(X[perm[i:i + 128]].to(dev)), Y[perm[i:i + 128]].to(dev)).backward(); oc.step()
te = torchvision.datasets.MNIST(sys.argv[1], train=False); Xt = (te.data.float() / 127.5 - 1).unsqueeze(1)
with torch.no_grad(): cacc = float((clf(Xt.to(dev)).argmax(1).cpu() == te.targets).float().mean())
print("s8 judge CNN test accuracy", round(cacc, 4))
guid = {"w": [0, 1, 3], "png": [], "hit": [], "judge": round(cacc, 4)}
for w in guid["w"]:
    c = torch.arange(10).repeat_interleave(50).to(dev); xs, _ = sample(c, float(w), 2)
    with torch.no_grad(): hit = float((clf(xs.to(dev)).argmax(1) == c).float().mean())
    pick = torch.cat([xs[k * 50:k * 50 + 4] for k in range(10)]); guid["png"].append(png(pick, 10, 4)); guid["hit"].append(round(hit, 4))
    print(f"s8 guidance w={w}: {hit:.3f} of 500 samples read as the requested digit")
# forward process 그림에 쓸 진짜 시험 숫자 4장 (28×28, uint8)
# Four real test digits for the forward-process figure (28×28, uint8)
real = base64.b64encode(te.data[[3, 2, 1, 18]].numpy().astype(np.uint8).tobytes()).decode()
p = save_keys(15, {"sched": {str(k): v for k, v in sched.items()}, "diff2d": diff2d, "real": real,
                   "ddpm": {"params": sum(p.numel() for p in unet.parameters()), "uncond": uncond, "guid": guid}}, MAKERS); print("wrote", p)
