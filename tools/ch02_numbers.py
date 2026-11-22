# 2장 예제 숫자 확인 (NumPy, PyTorch autograd)
# Check the worked-example numbers of chapter 2 (NumPy, PyTorch autograd)
# 실행 / run: python tools/ch02_numbers.py
import numpy as np
import torch


def show(name, v):
    print(f"{name:<42} {v}")


# 1절: 직선 y = w x를 점 4개에 맞추기, MSE loss와 기울기
# Section 1: fit y = w x to four points, MSE loss and its slope
x = np.array([1, 2, 3, 4.0])
y = np.array([2.1, 3.9, 6.2, 7.8])
L = lambda w: np.mean((w * x - y) ** 2)
dL = lambda w: np.mean(2 * (w * x - y) * x)
show("s1 L(1)", L(1.0))
show("s1 dL/dw at w=1", dL(1.0))
show("s1 w after one step (eta 0.03)", 1 - 0.03 * dL(1.0))
show("s1 L after one step", L(1 - 0.03 * dL(1.0)))
show("s1 best w = sum(xy)/sum(x^2)", (x * y).sum() / (x * x).sum())
show("s1 L(best w)", L((x * y).sum() / (x * x).sum()))
w = 1.0
for _ in range(10):
    w -= 0.03 * dL(w)
show("s1 w after 10 steps", w)
wt = torch.tensor(1.0, requires_grad=True)
torch.mean((wt * torch.tensor(x) - torch.tensor(y)) ** 2).backward()
show("s1 torch grad at w=1", wt.grad.item())

# 2절: x^3의 x = 2 기울기
# Section 2: slope of x^3 at x = 2
show("s2 (2.001^3 - 8)/0.001", (2.001 ** 3 - 8) / 0.001)
show("s2 3*2^2", 3 * 2 ** 2)

# 3절: f = x^2 - 4x + 5의 최솟값
# Section 3: minimum of f = x^2 - 4x + 5
f3 = lambda v: v * v - 4 * v + 5
show("s3 f'(x)=0 at x", 2.0)
show("s3 f(2)", f3(2.0))
show("s3 f(1.9), f(2.1)", (f3(1.9), f3(2.1)))

# 4절: product rule y = x^2 (x + 1), chain rule y = (3x + 1)^2
# Section 4: product rule y = x^2 (x + 1), chain rule y = (3x + 1)^2
xt = torch.tensor(2.0, requires_grad=True)
(xt ** 2 * (xt + 1)).backward()
show("s4 product rule at x=2 (torch)", xt.grad.item())
xt = torch.tensor(1.0, requires_grad=True)
((3 * xt + 1) ** 2).backward()
show("s4 chain rule at x=1 (torch)", xt.grad.item())
u, du, dx = 4.0, 3 * 0.1, 0.1
show("s4 chain: dx 0.1 -> du, dy(exact)", (du, (4.3) ** 2 - 16))

# 5절: f = (x - 2)^2 + 1, x = 5, eta = 0.1
# Section 5: f = (x - 2)^2 + 1, x = 5, eta = 0.1
f5 = lambda v: (v - 2) ** 2 + 1
d5 = lambda v: 2 * (v - 2)
xs = [5.0]
for _ in range(5):
    xs.append(xs[-1] - 0.1 * d5(xs[-1]))
show("s5 path", np.round(xs, 4))
show("s5 f(5), f(4.4)", (f5(5.0), f5(4.4)))

# 6절: 거리에 (1 - 2 eta)가 곱해진다
# Section 6: the distance is multiplied by (1 - 2 eta)
for eta in (0.02, 0.1, 0.2, 0.9, 1.05):
    v = 5.0
    for _ in range(10):
        v -= eta * d5(v)
    show(f"s6 eta {eta}: factor, x after 10 steps", (round(1 - 2 * eta, 3), round(v, 4)))
# 점점 줄어드는 learning rate eta_t = eta0 / (1 + t/5)
# Decaying learning rate eta_t = eta0 / (1 + t/5)
v = 5.0
for t in range(30):
    v -= 1.05 / (1 + t / 5) * d5(v)
show("s6 decay eta0 1.05: x after 30 steps", round(v, 4))

# 7절: f = x^4/4 - x^2 + x/4의 바닥 두 개
# Section 7: the two valleys of f = x^4/4 - x^2 + x/4
r = np.sort(np.roots([1, 0, -2, 0.25]).real)
f7 = lambda v: v ** 4 / 4 - v ** 2 + v / 4
show("s7 f'=0 at", np.round(r, 4))
show("s7 f at those", np.round(f7(r), 4))


def run7(v, eta=0.05, n=400):
    for _ in range(n):
        v -= eta * (v ** 3 - 2 * v + 0.25)
    return v


show("s7 start -0.5 / 0.5 / 2.0 ends at", tuple(round(run7(s), 3) for s in (-0.5, 0.5, 2.0)))
# saddle: f = x^2 + y^4/4 - y^2/2
gx = lambda a, b: 2 * a
gy = lambda a, b: b ** 3 - b
for y0 in (0.0, 0.001):
    a, b, n = 1.5, y0, 0
    for n in range(1, 400):
        a, b = a - 0.1 * gx(a, b), b - 0.1 * gy(a, b)
    show(f"s7 saddle start (1.5, {y0}) after 400 steps", (round(a, 4), round(b, 4)))

# 8절: f = x^2 + 3 y^2, (2, 1)
# Section 8: f = x^2 + 3 y^2 at (2, 1)
xt = torch.tensor([2.0, 1.0], requires_grad=True)
(xt[0] ** 2 + 3 * xt[1] ** 2).backward()
g = xt.grad.numpy()
show("s8 gradient (torch)", g)
show("s8 one step eta 0.1", np.array([2.0, 1.0]) - 0.1 * g)
show("s8 f before, after", (2 ** 2 + 3, 1.6 ** 2 + 3 * 0.4 ** 2))

# 9절: 층 10개를 지나며 곱해지는 gradient
# Section 9: the gradient multiplied through 10 layers
for a in (0.25, 0.9, 1.0, 1.5):
    show(f"s9 {a}^10", a ** 10)
