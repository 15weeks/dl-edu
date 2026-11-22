# 1장(들어가기)의 예제 숫자와 code 조각을 계산·확인한다
# Compute and check the worked-example numbers and code snippets of chapter 1 (introduction)
# 실행 / run: python tools/ch01_numbers.py
import math
import numpy as np
from scipy import stats
import torch

print("== 1.1 tensor 크기 / tensor sizes ==")
print("batch (32, 3, 224, 224) =", 32 * 3 * 224 * 224, "numbers")

print("== 1.2 transpose ==")
A = np.array([[1, 2, 3], [4, 5, 6]])
print(A.shape, "->", A.T.shape, A.T.tolist())

print("== 1.3 dot product ==")
a, b = np.array([2, 1]), np.array([1, 3])
dot = a @ b
cos = dot / (np.linalg.norm(a) * np.linalg.norm(b))
print("a.b =", dot, "|a| =", round(np.linalg.norm(a), 3), "|b| =", round(np.linalg.norm(b), 3), "cos =", round(cos, 4), "angle =", round(math.degrees(math.acos(cos)), 1))
w, x, bias = np.array([0.5, -1, 2]), np.array([1, 2, 3]), 0.5
print("neuron z =", w @ x + bias)
print("projection length of b on a =", round(dot / np.linalg.norm(a), 3))

print("== 1.4 matrix multiplication ==")
B = np.array([[1, 0], [0, 1], [1, 1]])
print("A @ B =", (A @ B).tolist(), "shape", (A @ B).shape)

print("== 1.5 softmax ==")
def softmax(z, T=1.0):
    e = np.exp(np.array(z, float) / T)
    return e / e.sum()
z = [2, 1, 0]
print("e^z =", np.round(np.exp(z), 3), "sum =", round(np.exp(z).sum(), 3))
for T in (1, 0.5, 2):
    print("T", T, np.round(softmax(z, T), 3))
print("torch softmax:", torch.softmax(torch.tensor([2.0, 1.0, 0.0]), dim=0).numpy().round(3))

print("== 1.6 log ==")
print("ln(0.9*0.8) =", round(math.log(0.72), 4), " ln0.9 + ln0.8 =", round(math.log(0.9), 4), "+", round(math.log(0.8), 4), "=", round(math.log(0.9) + math.log(0.8), 4))
print("0.01**200 =", 0.01 ** 200, " sum of logs =", round(200 * math.log(0.01), 1))

print("== 1.7 entropy, cross-entropy (nats) ==")
H = lambda p: -(p * math.log(p) + (1 - p) * math.log(1 - p))
for p in (0.5, 0.9, 0.99):
    print(f"H({p}) = {H(p):.3f} nats = {H(p) / math.log(2):.3f} bits")
for pred in ([0.2, 0.7, 0.1], [0.6, 0.3, 0.1]):
    print("pred", pred, "CE = -ln", pred[1], "=", round(-math.log(pred[1]), 3))
print("torch CE check:", float(torch.nn.functional.nll_loss(torch.log(torch.tensor([[0.2, 0.7, 0.1]])), torch.tensor([1]))))

print("== 1.8 argmax ==")
v = np.array([2.0, 3.5, 1.2])
print("max", v.max(), "argmax", v.argmax(), "min", v.min(), "argmin", v.argmin())

print("== 1.9 mean, variance ==")
d = np.array([2, 4, 4, 4, 5, 5, 7, 9])
print("mean", d.mean(), "sum sq dev", ((d - d.mean()) ** 2).sum(), "var(pop)", d.var(), "sd", d.std(), "var(sample)", round(d.var(ddof=1), 3), "sd(sample)", round(d.std(ddof=1), 3))

print("== 1.10 sampling ==")
print("SE for sd 10, n 10:", round(10 / math.sqrt(10), 2), " n 100:", 10 / math.sqrt(100))
np.random.seed(42); s1 = np.random.rand(3)
np.random.seed(42); s2 = np.random.rand(3)
print("seed 42 twice equal:", np.allclose(s1, s2), np.round(s1, 4))

print("== 1.11 t-test (Welch) ==")
A_ = np.array([0.81, 0.83, 0.80, 0.82, 0.84]); B_ = np.array([0.84, 0.86, 0.83, 0.85, 0.87])
r = stats.ttest_ind(B_, A_, equal_var=False)
se = math.sqrt(A_.var(ddof=1) / 5 + B_.var(ddof=1) / 5)
print("means", A_.mean(), B_.mean(), "sd", round(A_.std(ddof=1), 4), "SE diff", round(se, 4), "t", round(r.statistic, 3), "p", round(r.pvalue, 4), "df", r.df if hasattr(r, "df") else "")

print("== 1.12 NumPy / PyTorch ==")
print("broadcast (3,1)+(1,4) ->", (np.zeros((3, 1)) + np.zeros((1, 4))).shape)
xt = torch.tensor(3.0, requires_grad=True); y = xt ** 2; y.backward()
print("d(x^2)/dx at 3 =", xt.grad.item())
t = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
print("tensor shape", tuple(t.shape), "t.T shape", tuple(t.T.shape), "matmul", (t @ torch.tensor([[1., 0.], [0., 1.], [1., 1.]])).tolist())
