# 10장: MNIST(14×14) autoencoder들을 학습하고 page에 넣을 weight와 결과를 assets/data/ch10_data.js로 쓴다 (PyTorch, CPU)
# Chapter 10: train MNIST (14×14) autoencoders and write the weights and results for the page to assets/data/ch10_data.js (PyTorch, CPU)
# 실행 / run: python tools/ch10_numbers.py <torchvision data 폴더/folder>
import sys, json, base64, pathlib
import numpy as np
import torch, torchvision

ROOT = pathlib.Path(__file__).resolve().parent.parent
torch.set_num_threads(4)
tr = torchvision.datasets.MNIST(sys.argv[1], train=True); te = torchvision.datasets.MNIST(sys.argv[1], train=False)
pool = lambda X: X.reshape(len(X), 14, 2, 14, 2).mean((2, 4)).reshape(len(X), 196) / 255.0
Xtr = torch.tensor(pool(tr.data.numpy().astype(np.float32))); Xte = torch.tensor(pool(te.data.numpy().astype(np.float32))); Yte = te.targets.numpy()

def mlp(sizes, out_sig=True):
    L = []
    for i in range(len(sizes) - 1):
        L.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2: L.append(torch.nn.ReLU())
    if out_sig: L.append(torch.nn.Sigmoid())
    return torch.nn.Sequential(*L)

def corrupt(x, g, noise=True, occl=True):
    x = x.clone()
    if noise:
        s = torch.rand(len(x), 1, generator=g) * 0.6; x = (x + s * torch.randn(x.shape, generator=g)).clamp(0, 1)
    if occl:
        m = torch.ones(len(x), 14, 14)
        r0 = torch.randint(0, 11, (len(x),), generator=g); h = torch.randint(2, 6, (len(x),), generator=g); use = torch.rand(len(x), generator=g) < 0.5
        for i in range(len(x)):
            if use[i]: m[i, r0[i]:r0[i] + h[i], :] = 0
        x = x * m.reshape(len(x), 196)
    return x

def train(enc, dec, epochs=8, lr=2e-3, denoise=False, seed=0, tied=None):
    g = torch.Generator().manual_seed(seed)
    params = list(enc.parameters()) + (list(dec.parameters()) if dec is not None else [])
    opt = torch.optim.Adam(params, lr=lr)
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), 128):
            xb = Xtr[perm[i:i + 128]]; xin = corrupt(xb, g) if denoise else xb
            opt.zero_grad(); out = tied(xin) if tied else dec(enc(xin)); torch.nn.functional.mse_loss(out, xb).backward(); opt.step()

def q8(W):
    W = W.detach().numpy(); s = float(np.abs(W).max()) / 127 or 1.0; q = np.clip(np.round(W / s), -127, 127).astype(np.int8)
    return {"s": s, "shape": list(W.shape), "q": base64.b64encode(q.tobytes()).decode()}
def pack(seq): return [{"W": q8(m.weight), "b": m.bias.detach().numpy().round(5).tolist()} for m in seq if isinstance(m, torch.nn.Linear)]
mse = lambda a, b: float(((a - b) ** 2).mean())
out = {}
torch.manual_seed(0)
# 1·2·4절: denoising AE 196-64-16-64-196 (잡음과 가림을 넣어 학습)
# Sections 1, 2, 4: denoising AE 196-64-16-64-196 (trained with noise and occlusion)
enc, dec = mlp([196, 64, 16], out_sig=False), mlp([16, 64, 196])
enc = torch.nn.Sequential(*list(enc), torch.nn.ReLU())
train(enc, dec, epochs=10, denoise=True)
with torch.no_grad():
    rec = dec(enc(Xte)); print("s1 denoising AE clean test MSE", round(mse(rec, Xte), 5))
    g = torch.Generator().manual_seed(5)
    for s in (0.2, 0.4, 0.6):
        noisy = (Xte + s * torch.randn(Xte.shape, generator=g)).clamp(0, 1)
        print(f"s2 noise {s}: noisy-vs-clean MSE {mse(noisy, Xte):.4f} → denoised {mse(dec(enc(noisy)), Xte):.4f}")
    m = torch.ones(196); m[5 * 14:9 * 14] = 0; occ = Xte * m
    print(f"s4 rows 5-8 hidden: occluded MSE {mse(occ, Xte):.4f} → restored {mse(dec(enc(occ)), Xte):.4f}")
out["dae"] = {"enc": pack(enc), "dec": pack(dec), "mse": round(mse(rec, Xte), 5)}
print("s1 parameters", sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in dec.parameters()))

# 3절: bottleneck 크기 1~64
# Section 3: bottleneck size 1 to 64
sizes = [1, 2, 4, 8, 16, 32, 64]; errs = []; recs = []; idx3 = [3, 2, 1]
for k in sizes:
    torch.manual_seed(1); e_ = torch.nn.Sequential(torch.nn.Linear(196, 128), torch.nn.ReLU(), torch.nn.Linear(128, k)); d_ = mlp([k, 128, 196])
    train(e_, d_, epochs=6, seed=1)
    with torch.no_grad(): r_ = d_(e_(Xte)); errs.append(round(mse(r_, Xte), 5)); recs.append((r_[idx3].numpy() * 255).round().astype(np.uint8))
    print("s3 bottleneck", k, "test MSE", errs[-1])
out["bottleneck"] = {"sizes": sizes, "mse": errs, "orig": base64.b64encode((Xte[idx3].numpy() * 255).round().astype(np.uint8).tobytes()).decode(),
                     "recs": [base64.b64encode(r.tobytes()).decode() for r in recs]}

# 5·7절: latent 2개짜리 AE. decoder weight + 시험 1000개의 code
# Sections 5 and 7: an AE with a 2-number latent. Decoder weights + codes of 1000 test images
torch.manual_seed(2); e2 = torch.nn.Sequential(torch.nn.Linear(196, 128), torch.nn.ReLU(), torch.nn.Linear(128, 2)); d2 = mlp([2, 128, 196])
train(e2, d2, epochs=12, seed=2)
with torch.no_grad():
    Z = e2(Xte).numpy(); print("s5 2-D latent AE test MSE", round(mse(d2(e2(Xte)), Xte), 5))
sel = np.random.RandomState(0).choice(len(Xte), 1000, replace=False)
out["latent"] = {"dec": pack(d2), "z": Z[sel].round(3).tolist(), "y": Yte[sel].tolist(), "range": [float(Z[sel, 0].min()), float(Z[sel, 0].max()), float(Z[sel, 1].min()), float(Z[sel, 1].max())]}
print("s5 latent range", [round(v, 2) for v in out["latent"]["range"]])

# 6절: tied weights (196-32, decoder = encoderᵀ) vs 따로
# Section 6: tied weights (196-32, decoder = encoderᵀ) vs separate
torch.manual_seed(3)
class Tied(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.W = torch.nn.Parameter(torch.randn(32, 196) * 0.05); s.b1 = torch.nn.Parameter(torch.zeros(32)); s.b2 = torch.nn.Parameter(torch.zeros(196))
    def forward(s, x): return torch.sigmoid(torch.relu(x @ s.W.T + s.b1) @ s.W + s.b2)
tied = Tied(); train(tied, None, epochs=8, seed=3, tied=tied)
torch.manual_seed(3); eu = torch.nn.Sequential(torch.nn.Linear(196, 32), torch.nn.ReLU()); du = mlp([32, 196]); train(eu, du, epochs=8, seed=3)
with torch.no_grad():
    t_mse, u_mse = mse(tied(Xte), Xte), mse(du(eu(Xte)), Xte)
pt, pu = sum(p.numel() for p in tied.parameters()), sum(p.numel() for p in eu.parameters()) + sum(p.numel() for p in du.parameters())
print(f"s6 tied: {pt} params, MSE {t_mse:.5f}   untied: {pu} params, MSE {u_mse:.5f}")
out["tied"] = {"params": [pt, pu], "mse": [round(t_mse, 5), round(u_mse, 5)]}

# 그림에 쓸 시험 숫자 40장 (14×14, uint8)과 그 2D code
# 40 test digits for the figures (14×14, uint8) and their 2-D codes
idx = []
for k in range(10): idx += list(np.where(Yte == k)[0][:4])
out["samples"] = {"y": Yte[idx].tolist(), "x": base64.b64encode((Xte[idx].numpy() * 255).round().astype(np.uint8).tobytes()).decode(), "z": Z[idx].round(3).tolist()}
js = ("/* 10장 MNIST autoencoder weight(int8)와 결과. tools/ch10_numbers.py가 만든다 (손으로 고치지 말 것)\n"
      "   Chapter 10 MNIST autoencoder weights (int8) and results. Made by tools/ch10_numbers.py (do not edit by hand) */\n"
      "const CH10_DATA = " + json.dumps(out, separators=(",", ":")) + ";\n")
(ROOT / "assets" / "data" / "ch10_data.js").write_text(js, encoding="utf-8")
print("wrote assets/data/ch10_data.js", len(js), "bytes")
