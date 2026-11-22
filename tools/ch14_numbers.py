# 14장 (1): attention 손 계산, Transformer block parameter, MNIST로 작은 ViT 학습과 CLS attention (PyTorch, CPU)
# Chapter 14 (1): attention by hand, Transformer-block parameters, a small ViT trained on MNIST and its CLS attention (PyTorch, CPU)
# 결과는 assets/data/ch14_data.js의 "hand", "block", "vit" 항목 ("gpt2" 항목은 ch14_gpt2.py가 쓴다)
# Results go to the "hand", "block" and "vit" keys of assets/data/ch14_data.js (the "gpt2" key comes from ch14_gpt2.py)
# 실행 / run: python tools/ch14_numbers.py <torchvision data 폴더/folder>
import sys, base64, pathlib, math
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from datajs import save_keys
torch.set_num_threads(4)
MAKERS = ["tools/ch14_numbers.py", "tools/ch14_gpt2.py"]

# 1) 손 계산: token 3개, d_k = 2
# 1) By hand: 3 tokens, d_k = 2
Q = torch.tensor([[1., 0], [0, 1], [1, 1]]); K = torch.tensor([[2., 0], [0, 2], [1, 1]]); V = torch.tensor([[10., 0], [0, 10], [5, 5]])
S = Q @ K.T; Ss = S / math.sqrt(2); A = F.softmax(Ss, -1); O = A @ V
print("s2 QKᵀ =", S.tolist()); print("s2 /√2 =", Ss.numpy().round(3).tolist()); print("s2 softmax =", A.numpy().round(3).tolist()); print("s2 output =", O.numpy().round(2).tolist())
hand = {"Q": Q.tolist(), "K": K.tolist(), "V": V.tolist(), "S": S.tolist(), "A": A.numpy().round(4).tolist(), "O": O.numpy().round(3).tolist()}

# 2) Transformer block 하나의 parameter (d 64, head 4, FFN 256)
# 2) Parameters of one Transformer block (d 64, 4 heads, FFN 256)
blk = nn.TransformerEncoderLayer(64, 4, 256, batch_first=True)
block = {n: p.numel() for n, p in blk.named_parameters()}; block_total = sum(block.values()); print("s5 block parameters", block_total, block)

# 3) 작은 ViT: 28×28을 7×7 patch 16개로, d 64, head 4, 2층, CLS token
# 3) A small ViT: 28×28 cut into 16 patches of 7×7, d 64, 4 heads, 2 layers, a CLS token
tr = torchvision.datasets.MNIST(sys.argv[1], train=True); te = torchvision.datasets.MNIST(sys.argv[1], train=False)
f = lambda d: torch.tensor(d.data.numpy().astype(np.float32) / 255.0).unsqueeze(1)
Xtr, Ytr, Xte, Yte = f(tr), tr.targets, f(te), te.targets
class ViT(nn.Module):
    def __init__(s, d=64, h=4, L=2, P=7):
        super().__init__(); s.P = P; s.emb = nn.Linear(P * P, d); s.cls = nn.Parameter(torch.zeros(1, 1, d)); s.pos = nn.Parameter(torch.randn(1, 17, d) * 0.02)
        s.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d, h, 4 * d, batch_first=True, norm_first=True, dropout=0.0) for _ in range(L)])
        s.norm = nn.LayerNorm(d); s.head = nn.Linear(d, 10)
    def patches(s, x):
        # (N, 1, 28, 28) → (N, 16, 49): 왼쪽 위부터 줄 순서로
        # (N, 1, 28, 28) → (N, 16, 49): row by row from the top left
        return x.unfold(2, s.P, s.P).unfold(3, s.P, s.P).reshape(x.shape[0], -1, s.P * s.P)
    def forward(s, x, keep=False):
        z = torch.cat([s.cls.expand(x.shape[0], -1, -1), s.emb(s.patches(x))], 1) + s.pos; atts = []
        for b in s.blocks:
            if keep:
                y = b.norm1(z); a, w = b.self_attn(y, y, y, need_weights=True, average_attn_weights=False); atts.append(w)
            z = b(z)
        return (s.head(s.norm(z[:, 0])), atts) if keep else s.head(s.norm(z[:, 0]))
torch.manual_seed(0); vit = ViT(); opt = torch.optim.Adam(vit.parameters(), lr=1e-3); g = torch.Generator().manual_seed(0)
for ep in range(4):
    perm = torch.randperm(len(Xtr), generator=g)
    for i in range(0, len(Xtr), 128):
        b = perm[i:i + 128]; opt.zero_grad(); F.cross_entropy(vit(Xtr[b]), Ytr[b]).backward(); opt.step()
    vit.eval()
    with torch.no_grad(): acc = float(sum((vit(Xte[i:i + 2000]).argmax(1) == Yte[i:i + 2000]).sum() for i in range(0, 10000, 2000)) / 10000)
    vit.train(); print(f"s7 ViT epoch {ep + 1}: test acc {acc:.4f}")
vit.eval(); np_ = sum(p.numel() for p in vit.parameters()); print("s7 ViT parameters", np_)
idx = [int(np.where(Yte.numpy() == k)[0][1]) for k in range(10)]
with torch.no_grad():
    logits, atts = vit(Xte[idx], keep=True)
# CLS(0번 token)가 patch 16개를 보는 attention: 층마다 head 4개
# Attention from CLS (token 0) to the 16 patches: 4 heads per layer
cls_att = [[a[k, :, 0, 1:].numpy().round(4).tolist() for a in atts] for k in range(10)]
vit_out = {"acc": round(acc, 4), "params": np_, "y": Yte[idx].tolist(), "pred": logits.argmax(1).tolist(),
           "x": base64.b64encode((Xte[idx, 0].numpy() * 255).round().astype(np.uint8).tobytes()).decode(), "att": cls_att}
p = save_keys(14, {"hand": hand, "block": {"total": block_total, "parts": block}, "vit": vit_out}, MAKERS); print("wrote", p)
