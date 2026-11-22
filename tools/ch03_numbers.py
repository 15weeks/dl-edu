# 3장 예제 숫자 확인: 2-2-1 network의 forward, loss, backprop, 한 걸음 (PyTorch autograd와 비교)
# Check the chapter 3 example numbers: forward, loss, backprop and one step of a 2-2-1 network (compared with PyTorch autograd)
# 실행 / run: python tools/ch03_numbers.py
import numpy as np
import torch

np.set_printoptions(precision=4, suppress=True)
sig = lambda z: 1 / (1 + np.exp(-z))

# 2절: neuron 하나
# Section 2: one neuron
x, w, b = np.array([1.0, 2.0]), np.array([0.5, -0.25]), 0.5
z = w @ x + b
print("s2 z =", z, " sigmoid(z) =", round(sig(z), 4), " step =", int(z > 0))

# 4~7절: 2-2-1 network
# Sections 4-7: the 2-2-1 network
x = np.array([1.0, 0.5]); y = 1.0
W1 = np.array([[0.4, -0.6], [0.7, 0.2]]); b1 = np.array([0.1, -0.3])
W2 = np.array([[0.9, -0.5]]); b2 = np.array([0.2])
z1 = W1 @ x + b1; h = sig(z1); z2 = W2 @ h + b2; yh = sig(z2)
L = -(y * np.log(yh) + (1 - y) * np.log(1 - yh))
print("s4 z1 =", z1, " h =", h, " z2 =", z2, " yhat =", yh)
print("s5 loss (BCE) =", L)
d2 = yh - y
gW2 = np.outer(d2, h); gb2 = d2
d1 = (W2.T @ d2) * h * (1 - h)
gW1 = np.outer(d1, x); gb1 = d1
print("s6 delta2 =", d2, " dL/dW2 =", gW2, " dL/db2 =", gb2)
print("s6 W2^T delta2 =", W2.T @ d2, " sigma'(z1) =", h * (1 - h))
print("s6 delta1 =", d1, " dL/dW1 =", gW1, " dL/db1 =", gb1)
eta = 0.5
W1n, b1n, W2n, b2n = W1 - eta * gW1, b1 - eta * gb1, W2 - eta * gW2, b2 - eta * gb2
yh2 = sig(W2n @ sig(W1n @ x + b1n) + b2n)
print("s7 after one step (eta 0.5): W2 =", W2n, " b2 =", b2n, " W1 =", W1n, " b1 =", b1n)
print("s7 new yhat =", yh2, " new loss =", -np.log(yh2))

# 8절: PyTorch로 같은 계산
# Section 8: the same calculation in PyTorch
net = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Sigmoid(), torch.nn.Linear(2, 1), torch.nn.Sigmoid())
with torch.no_grad():
    net[0].weight.copy_(torch.tensor(W1)); net[0].bias.copy_(torch.tensor(b1))
    net[2].weight.copy_(torch.tensor(W2)); net[2].bias.copy_(torch.tensor(b2))
xt = torch.tensor([[1.0, 0.5]]); yt = torch.tensor([[1.0]])
out = net(xt); loss = torch.nn.BCELoss()(out, yt); loss.backward()
print("s8 torch yhat =", round(out.item(), 4), " loss =", round(loss.item(), 4))
print("s8 torch W2.grad =", net[2].weight.grad.numpy().round(4), " W1.grad =", net[0].weight.grad.numpy().round(4))
print("s8 parameter count 2-2-1 =", sum(p.numel() for p in net.parameters()))
