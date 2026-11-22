# 14장 (2): GPT-2 small(OpenAI, MIT license)의 token, attention, 다음 token 확률, 위치 embedding
# Chapter 14 (2): tokens, attention, next-token probabilities and position embeddings of GPT-2 small (OpenAI, MIT license)
# 결과는 assets/data/ch14_data.js의 "gpt2" 항목
# Results go to the "gpt2" key of assets/data/ch14_data.js
# 실행 / run: transformers가 있는 python으로 / with a python that has transformers:  python tools/ch14_gpt2.py
import sys, pathlib
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from datajs import save_keys
tok = GPT2TokenizerFast.from_pretrained("gpt2"); m = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()
cfg = m.config; out = {"params": sum(p.numel() for p in m.parameters()), "layers": cfg.n_layer, "heads": cfg.n_head, "d": cfg.n_embd, "vocab": cfg.vocab_size, "ctx": cfg.n_positions}
print("s8 GPT-2 small", out)
piece = lambda i: tok.decode([i])

# 1) token 나누기: 영어 문장과 한국어 문장
# 1) Tokenising: an English and a Korean sentence
toks = {}
for key, s in (("en", "The animal didn't cross the street because it was too tired."), ("ko", "나는 커피를 마신다.")):
    ids = tok(s)["input_ids"]; toks[key] = {"text": s, "ids": ids, "pieces": [piece(i) for i in ids], "chars": len(s)}
    print(f"s8 {key}: {len(s)} characters → {len(ids)} tokens", [piece(i) for i in ids])
out["tokens"] = toks

# 2) 영어 문장의 attention: 12층 × head 12개, 모두 저장
# 2) Attention for the English sentence: 12 layers × 12 heads, all stored
ids = torch.tensor([toks["en"]["ids"]])
with torch.no_grad(): o = m(ids, output_attentions=True)
att = torch.stack(o.attentions)[:, 0]                 # (층 layer, head, L, L)
L = att.shape[-1]; P = toks["en"]["pieces"]; it, animal = P.index(" it"), P.index(" animal")
prev = torch.stack([att[:, :, i, i - 1] for i in range(1, L)], -1).mean(-1)
first = att[:, :, 1:, 0].mean(-1)
li, hi = divmod(int(att[:, :, it, animal].argmax()), 12); lp, hp = divmod(int(prev.argmax()), 12)
print(f"s1 'it'→'animal' strongest: layer {li + 1}, head {hi + 1}: {float(att[li, hi, it, animal]):.3f}")
print(f"s3 previous-token head: layer {lp + 1}, head {hp + 1}: mean {float(prev[lp, hp]):.3f}")
print(f"s3 mean attention to the first token over all heads: {float(first.mean()):.3f}")
out["att"] = {"L": L, "w": (att.numpy() * 1000).round().astype(int).tolist(), "it_animal": [li + 1, hi + 1, round(float(att[li, hi, it, animal]), 3)],
              "prev": [lp + 1, hp + 1, round(float(prev[lp, hp]), 3)], "first_mean": round(float(first.mean()), 3)}

# 3) 다음 token 확률: prompt 4개, 위 50개의 logit (온도 slider용)과 진짜 확률
# 3) Next-token probabilities: 4 prompts, the top 50 logits (for the temperature slider) and their true probabilities
nxt = []
for s in ("The capital of France is", "I drank a cup of", "The cat sat on the", "1, 2, 3, 4,"):
    ids = torch.tensor([tok(s)["input_ids"]])
    with torch.no_grad(): lo = m(ids).logits[0, -1]
    p = F.softmax(lo, -1); top = torch.topk(lo, 50)
    nxt.append({"prompt": s, "pieces": [piece(int(i)) for i in top.indices], "logits": top.values.numpy().round(3).tolist(), "p": p[top.indices].numpy().round(5).tolist(), "p50": round(float(p[top.indices].sum()), 4)})
    print(f"s8 '{s}' →", [(nxt[-1]['pieces'][k], round(nxt[-1]['p'][k], 3)) for k in range(5)], "top-50 mass", nxt[-1]["p50"])
out["next"] = nxt

# 4) 가장 그럴듯한 것만 이어 붙이기 (greedy) 20 token
# 4) Keep appending the single most likely token (greedy), 20 tokens
ids = tok("Deep learning is", return_tensors="pt")["input_ids"]
with torch.no_grad(): g = m.generate(ids, max_new_tokens=20, do_sample=False, pad_token_id=tok.eos_token_id)
out["greedy"] = {"prompt": "Deep learning is", "text": tok.decode(g[0])}; print("s8 greedy:", out["greedy"]["text"])

# 5) 학습된 위치 embedding(wpe) 처음 64자리끼리의 cosine 유사도
# 5) Cosine similarity between the first 64 learned position embeddings (wpe)
W = F.normalize(m.transformer.wpe.weight[:64].detach(), dim=1); C = (W @ W.T).numpy()
out["pos_cos"] = (C * 100).round().astype(int).tolist(); print("s4 learned position cos: pos 10 vs 11", round(float(C[10, 11]), 3), "vs 40", round(float(C[10, 40]), 3))
p = save_keys(14, {"gpt2": out}, ["tools/ch14_numbers.py", "tools/ch14_gpt2.py"]); print("wrote", p)
