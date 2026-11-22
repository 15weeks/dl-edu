# 13장: RNN 손 계산, PyTorch shape, 주기 3 무늬, sine 외삽, 글자 단위 LSTM (Lorem ipsum), 긴 기억 실험 (PyTorch, CPU)
# Chapter 13: RNN hand calculation, PyTorch shapes, a period-3 pattern, sine extrapolation, a character-level LSTM (Lorem ipsum), a long-memory experiment (PyTorch, CPU)
# 결과는 assets/data/ch13_data.js
# Results go to assets/data/ch13_data.js
# 실행 / run: python tools/ch13_numbers.py
import json, base64, pathlib, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parent.parent
torch.set_num_threads(4)
out = {}
nparams = lambda m: sum(p.numel() for p in m.parameters())
def q8(W):
    W = W.detach().numpy(); s = float(np.abs(W).max()) / 127 or 1.0; q = np.clip(np.round(W / s), -127, 127).astype(np.int8)
    return {"s": s, "shape": list(W.shape), "q": base64.b64encode(q.tobytes()).decode()}

# 1) 손 계산: 숫자 하나짜리 RNN, w_x 0.5, w_h 0.8, b 0, x = (1, 0, 0, 1)
# 1) By hand: a one-number RNN, w_x 0.5, w_h 0.8, b 0, x = (1, 0, 0, 1)
h, hs = 0.0, []
for x in (1, 0, 0, 1): h = math.tanh(0.5 * x + 0.8 * h); hs.append(round(h, 4))
print("s2 scalar RNN h1..h4", hs)

# 2) PyTorch shape와 parameter 수
# 2) PyTorch shapes and parameter counts
rnn = nn.RNN(input_size=8, hidden_size=16, num_layers=2, batch_first=True)
y, hn = rnn(torch.zeros(3, 5, 8))
print("s3 nn.RNN(8, 16, num_layers=2, batch_first=True): input (3, 5, 8) → output", tuple(y.shape), "h_n", tuple(hn.shape))
for n, p in rnn.named_parameters(): print("   ", n, tuple(p.shape))
out["params"] = {k: nparams(c(8, 64, batch_first=True)) for k, c in (("RNN", nn.RNN), ("GRU", nn.GRU), ("LSTM", nn.LSTM))}
print("s3/s6 parameters with input 8, hidden 64:", out["params"])

# 3) 주기 3 무늬 1, 0, 0, 1, 0, 0, …: 바로 앞 값 하나만 보는 FFN vs RNN
# 3) Period-3 pattern 1, 0, 0, 1, 0, 0, …: an FFN that sees only the previous value vs an RNN
seq = torch.tensor([1., 0, 0] * 40)
X, Y = seq[:-1].view(1, -1, 1), seq[1:].view(1, -1, 1)
torch.manual_seed(0)
class Next(nn.Module):
    def __init__(s, cell, H): super().__init__(); s.r = cell(1, H, batch_first=True); s.o = nn.Linear(H, 1)
    def forward(s, x): return s.o(s.r(x)[0])
net = Next(nn.RNN, 4); opt = torch.optim.Adam(net.parameters(), lr=0.02)
for it in range(400): opt.zero_grad(); F.binary_cross_entropy_with_logits(net(X), Y).backward(); opt.step()
with torch.no_grad(): a_rnn = float(((net(X) > 0).float() == Y)[:, 3:].float().mean())
ff = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1)); opt = torch.optim.Adam(ff.parameters(), lr=0.02)
for it in range(400): opt.zero_grad(); F.binary_cross_entropy_with_logits(ff(X[0]), Y[0]).backward(); opt.step()
with torch.no_grad(): a_ff = float(((ff(X[0]) > 0).float() == Y[0])[3:].float().mean())
out["period3"] = {"rnn": round(a_rnn, 4), "ffn": round(a_ff, 4)}; print("s4 period-3 next-value accuracy: RNN", a_rnn, "FFN(previous value only)", round(a_ff, 4))

# 4) sine 외삽: sin(0.3 t) 200칸으로 다음 값 예측을 학습 → 끝에서부터 자기 예측을 다시 넣어 150칸
# 4) Sine extrapolation: learn next-value prediction on 200 steps of sin(0.3 t) → from the end, feed its own predictions back for 150 steps
t = torch.arange(0, 350, dtype=torch.float32); wave = torch.sin(0.3 * t)
Xs, Ys = wave[:199].view(1, -1, 1), wave[1:200].view(1, -1, 1)
sine = {"true": wave.numpy().round(4).tolist(), "train_len": 200, "runs": {}}
for H in (1, 2, 4, 16):
    torch.manual_seed(1); m = Next(nn.RNN, H); opt = torch.optim.Adam(m.parameters(), lr=0.01)
    for it in range(1500): opt.zero_grad(); F.mse_loss(m(Xs), Ys).backward(); opt.step()
    with torch.no_grad():
        o, hcur = m.r(Xs); last = m.o(o[:, -1:]); preds = []
        for k in range(150):
            preds.append(float(last)); o, hcur = m.r(last.view(1, 1, 1), hcur); last = m.o(o)
    err = float(np.sqrt(np.mean((np.array(preds) - wave[200:350].numpy()) ** 2)))
    sine["runs"][str(H)] = {"pred": np.round(preds, 4).tolist(), "rmse": round(err, 4), "params": nparams(m)}
    print(f"s4 sine hidden {H}: free-run RMSE over 150 steps {err:.4f}, params {nparams(m)}")
out["sine"] = sine

# 5) 글자 단위 LSTM: Cicero 원문(Lorem ipsum의 출처, 공유 저작물) 일부로 다음 글자 예측
# 5) Character-level LSTM: next-character prediction on part of Cicero's text (the source of Lorem ipsum, public domain)
TEXT = ("lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        "ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. "
        "duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. "
        "excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. "
        "sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, "
        "eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. "
        "nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. "
        "neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem. "
        "ut enim ad minima veniam, quis nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi consequatur. "
        "quis autem vel eum iure reprehenderit qui in ea voluptate velit esse quam nihil molestiae consequatur, vel illum qui dolorem eum fugiat quo voluptas nulla pariatur. "
        "at vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, "
        "similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga. et harum quidem rerum facilis est et expedita distinctio. "
        "nam libero tempore, cum soluta nobis est eligendi optio cumque nihil impedit quo minus id quod maxime placeat facere possimus, omnis voluptas assumenda est, omnis dolor repellendus. ")
VOC = sorted(set(TEXT)); ix = {ch: i for i, ch in enumerate(VOC)}; data = torch.tensor([ix[ch] for ch in TEXT])
class CharLSTM(nn.Module):
    def __init__(s, V, E=8, H=64): super().__init__(); s.emb = nn.Embedding(V, E); s.lstm = nn.LSTM(E, H, batch_first=True); s.out = nn.Linear(H, V)
    def forward(s, x, hc=None): o, hc = s.lstm(s.emb(x), hc); return s.out(o), hc
@torch.no_grad()
def sample(m, n=160, temp=0.8, seed=0, start="l"):
    g = torch.Generator().manual_seed(seed); x = torch.tensor([[ix[start]]]); hc = None; s = start
    for _ in range(n):
        lo, hc = m(x, hc); p = F.softmax(lo[0, -1] / temp, -1); k = int(torch.multinomial(p, 1, generator=g)); s += VOC[k]; x = torch.tensor([[k]])
    return s
torch.manual_seed(2); cm = CharLSTM(len(VOC)); opt = torch.optim.Adam(cm.parameters(), lr=0.01); g = torch.Generator().manual_seed(2)
snaps, L = {}, 64
for it in range(1, 1501):
    st = torch.randint(0, len(data) - L - 1, (32,), generator=g); xb = torch.stack([data[s:s + L] for s in st]); yb = torch.stack([data[s + 1:s + L + 1] for s in st])
    opt.zero_grad(); lo, _ = cm(xb); loss = F.cross_entropy(lo.reshape(-1, len(VOC)), yb.reshape(-1)); loss.backward(); nn.utils.clip_grad_norm_(cm.parameters(), 1.0); opt.step()
    if it in (1, 100, 300, 1500): snaps[str(it)] = {"loss": round(float(loss), 3), "text": sample(cm)}; print(f"s5 step {it} loss {float(loss):.3f}: {snaps[str(it)]['text'][:90]}")
E = cm.emb.weight.detach().numpy(); Ec = E - E.mean(0); U, S, Vt = np.linalg.svd(Ec, full_matrices=False); P = Ec @ Vt[:2].T
lw = cm.lstm
out["char"] = {"voc": "".join(VOC), "n_text": len(TEXT), "snaps": snaps, "emb2d": P.round(3).tolist(), "params": nparams(cm),
               "emb": q8(cm.emb.weight), "Wih": q8(lw.weight_ih_l0), "Whh": q8(lw.weight_hh_l0), "b": (lw.bias_ih_l0 + lw.bias_hh_l0).detach().numpy().round(5).tolist(),
               "Wo": q8(cm.out.weight), "bo": cm.out.bias.detach().numpy().round(5).tolist()}
print("s5 char LSTM params", nparams(cm), "vocab", len(VOC), "text length", len(TEXT))

# 6) 거리에 따른 gradient: ‖∂(h_T의 합)/∂x_t‖, T = 100, hidden 32, 무작위 입력 20개 평균 (학습 전, PyTorch 기본 초기화)
# 6) Gradient against distance: ‖∂(sum of h_T)/∂x_t‖, T = 100, hidden 32, mean of 20 random inputs (before training, PyTorch default init)
def grad_curve(cell, fbias=None, T=100, H=32, reps=20):
    torch.manual_seed(5); r = cell(1, H, batch_first=True)
    if fbias is not None:
        for n, p in r.named_parameters():
            if "bias" in n: p.data[H:2 * H] = fbias / 2      # bias_ih + bias_hh = fbias
    x = torch.randn(reps, T, 1, requires_grad=True); r(x)[0][:, -1].sum().backward()
    return np.log10(x.grad.abs().squeeze(-1).mean(0).numpy() + 1e-30)[::-1].round(3).tolist()   # [0] = 마지막 칸 / the last step
grads = {"RNN": grad_curve(nn.RNN), "LSTM": grad_curve(nn.LSTM), "LSTM_f3": grad_curve(nn.LSTM, 3.0)}
for k, v in grads.items(): print(f"s7 log10 gradient, {k}: distance 0 {v[0]:.2f}, 20 {v[20]:.2f}, 50 {v[50]:.2f}, 99 {v[99]:.2f}")
out["grads"] = grads

# 7) 긴 기억: 0/1 무작위 sequence의 첫 글자를 T칸 뒤에 맞히기 (찍으면 50%). 같은 학습 2,000번, seed 3개 중 성공(90% 넘음) 수
# 7) Long memory: recall the first bit of a random 0/1 sequence after T steps (guessing = 50%). Same 2,000 training steps; how many of 3 seeds succeed (above 90%)
class Recall(nn.Module):
    def __init__(s, cell, H=16, fb=1.0):
        super().__init__(); s.r = cell(1, H, batch_first=True); s.o = nn.Linear(H, 1)
        if cell is nn.LSTM:
            for n, p in s.r.named_parameters():
                if "bias_ih" in n: p.data[H:2 * H] = fb       # forget gate bias (1이 흔한 설정 / 1 is common practice)
    def forward(s, x): return s.o(s.r(x)[0][:, -1]).squeeze(-1)
mem = {"T": [5, 10, 20, 40, 80], "RNN": [], "LSTM": [], "LSTM_f3": []}
for T in mem["T"]:
    for name, cell, fb in (("RNN", nn.RNN, 0), ("LSTM", nn.LSTM, 1.0), ("LSTM_f3", nn.LSTM, 3.0)):
        accs = []
        for seed in (0, 1, 2):
            torch.manual_seed(seed); m = Recall(cell, fb=fb); opt = torch.optim.Adam(m.parameters(), lr=0.01); g = torch.Generator().manual_seed(seed)
            for it in range(2000):
                xb = torch.randint(0, 2, (64, T, 1), generator=g).float(); opt.zero_grad()
                F.binary_cross_entropy_with_logits(m(xb), xb[:, 0, 0]).backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            xt = torch.randint(0, 2, (1000, T, 1), generator=torch.Generator().manual_seed(99)).float()
            with torch.no_grad(): accs.append(float(((m(xt) > 0).float() == xt[:, 0, 0]).float().mean()))
        mem[name].append([round(a, 3) for a in accs])
    print(f"s7 recall after T={T}: RNN {mem['RNN'][-1]}, LSTM {mem['LSTM'][-1]}, LSTM forget bias 3 {mem['LSTM_f3'][-1]}")
out["memory"] = mem
js = ("/* 13장 결과와 글자 LSTM weight(int8). tools/ch13_numbers.py가 만든다 (손으로 고치지 말 것)\n"
      "   Chapter 13 results and character-LSTM weights (int8). Made by tools/ch13_numbers.py (do not edit by hand) */\n"
      "const CH13_DATA = " + json.dumps(out, separators=(",", ":")) + ";\n")
(ROOT / "assets" / "data" / "ch13_data.js").write_text(js, encoding="utf-8"); print("wrote assets/data/ch13_data.js", len(js), "bytes")
