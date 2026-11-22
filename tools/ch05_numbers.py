# 5장 예제 숫자: 다항식 회귀의 overfitting, k-fold, ridge, L1/L2 해 경로 (ch05.js와 같은 계산)
# Chapter 5 example numbers: polynomial overfitting, k-fold, ridge, L1/L2 solution paths (same maths as ch05.js)
# 실행 / run: python tools/ch05_numbers.py
import sys, math, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from dlrng import Rng, shuffled

def make(seed, n):
    r = Rng(seed); X, Y = [], []
    for _ in range(n):
        x = r.u() * 2 - 1; X.append(x); Y.append(0.8 * math.sin(math.pi * x) + 0.25 * r.n())
    return np.array(X), np.array(Y)
XT, YT = make(8, 15); XV, YV = make(9, 200)

def legendre(x, d):
    P = [np.ones_like(x), x]
    for k in range(1, d):
        P.append(((2 * k + 1) * x * P[k] - k * P[k - 1]) / (k + 1))
    return np.stack(P[:d + 1], 1)

def fit(x, y, d, lam):
    F = legendre(x, d); A = F.T @ F; I = np.eye(d + 1); I[0, 0] = 0
    return np.linalg.solve(A + lam * I + 1e-10 * np.eye(d + 1), F.T @ y)

mse = lambda c, x, y: float(np.mean((legendre(x, len(c) - 1) @ c - y) ** 2))
print("s1 degree: train MSE / test MSE")
for d in range(15):
    c = fit(XT, YT, d, 0)
    print(f"   d={d:2d}  {mse(c, XT, YT):.4f} / {mse(c, XV, YV):.4f}")

# 3절: k-fold (k = 5), fold = 섞은 순서의 자리 % k
# Section 3: k-fold (k = 5), fold = position in the shuffled order % k
order = shuffled(15, Rng(5))
def cv(d, k):
    errs = []
    for f in range(k):
        val = [order[i] for i in range(15) if i % k == f]; tr = [i for i in range(15) if i not in val]
        c = fit(XT[tr], YT[tr], d, 0); errs.append(mse(c, XT[val], YT[val]))
    return float(np.mean(errs))
for k in (3, 5):
    cvs = [cv(d, k) for d in range(11)]
    print(f"s3 k={k} CV MSE by degree:", [round(v, 3) for v in cvs], " best d =", int(np.argmin(cvs)))
print("s3 shuffled order:", order)

# 5절: degree 14 + ridge
# Section 5: degree 14 + ridge
for lam in (0, 1e-4, 1e-3, 1e-2, 1e-1, 1):
    c = fit(XT, YT, 14, lam)
    print(f"s5 lambda={lam:g}: train {mse(c, XT, YT):.4f}  test {mse(c, XV, YV):.4f}  |c|max {np.abs(c).max():.2f}")

# 8절: L1, L2 해 경로. L(w) = ½(w − w*)ᵀA(w − w*)
# Section 8: L1 and L2 solution paths. L(w) = ½(w − w*)ᵀA(w − w*)
A = np.array([[1.0, 0.4], [0.4, 0.5]]); ws = np.array([2.0, 0.6])
def l2(lam): return np.linalg.solve(A + lam * np.eye(2), A @ ws)
def l1(lam):
    w = ws.copy(); L = np.linalg.eigvalsh(A).max(); t = 1 / L
    for _ in range(20000):
        g = A @ (w - ws); v = w - t * g; w = np.sign(v) * np.maximum(np.abs(v) - t * lam, 0)
    return w
for lam in (0.2, 0.5, 1.0):
    print(f"s8 lambda={lam}: L2 w = {l2(lam).round(3)}  L1 w = {l1(lam).round(3)}")
lo, hi = 0, 2
for _ in range(50):
    m = (lo + hi) / 2
    if abs(l1(m)[1]) < 1e-9: hi = m
    else: lo = m
print("s8 L1 makes w2 exactly 0 from lambda ≈", round(hi, 3))
