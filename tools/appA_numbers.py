# 부록 A (ML 지도)의 손계산 예제 숫자를 확인한다 (NumPy만 사용)
# Check the worked-example numbers of appendix A (ML map), NumPy only
# 실행 / run: python3 tools/appA_numbers.py
import numpy as np

np.set_printoptions(precision=4, suppress=True)
sig = lambda z: 1 / (1 + np.exp(-z))


def head(s):
    print("\n== " + s)


# 1절: Classic ML
# Section 1: classic ML
head("1 Naive Bayes spam")
p_spam, p_free_spam, p_free_ham = 0.4, 0.5, 0.05
num, den = p_spam * p_free_spam, p_spam * p_free_spam + (1 - p_spam) * p_free_ham
print("spam part", num, "ham part", (1 - p_spam) * p_free_ham, "P(spam|free) =", round(num / den, 4))
head("1 logistic and SVM")
print("sigma(1.2) =", round(sig(1.2), 4))
w, b = np.array([1.0, 1.0]), -3.0
print("SVM w.x+b at (2,2) =", w @ np.array([2, 2]) + b, " margin width 2/|w| =", round(2 / np.linalg.norm(w), 4))

# 2절: tree와 ensemble
# Section 2: trees and ensembles
head("2 Gini split")
gini = lambda p: 1 - np.sum(np.square(p))
g0, gl, gr = gini([0.5, 0.5]), gini([0.8, 0.2]), gini([0.2, 0.8])
print("parent", g0, "left", round(gl, 4), "right", round(gr, 4), "weighted", round(0.5 * gl + 0.5 * gr, 4), "gain", round(g0 - (0.5 * gl + 0.5 * gr), 4))
head("2 AdaBoost")
eps = 0.2
alpha = 0.5 * np.log((1 - eps) / eps)
wrong, right = eps * np.exp(alpha), (1 - eps) * np.exp(-alpha)
print("alpha", round(alpha, 4), "e^alpha", round(np.exp(alpha), 4), "share of mistakes after reweighting", round(wrong / (wrong + right), 4))
head("2 gradient boosting")
y = np.array([3.0, 5.0, 10.0]); F0 = np.full(3, y.mean()); r0 = y - F0
h1 = np.array([-2.0, -2.0, 4.0]); F1 = F0 + 0.5 * h1
print("F0", F0, "residual", r0, "MSE", round(np.mean(r0 ** 2), 4))
print("F1", F1, "residual", y - F1, "MSE", round(np.mean((y - F1) ** 2), 4))

# 3절: clustering
# Section 3: clustering
head("3 k-means in 1-D")
x = np.array([1, 2, 3, 9, 10, 11], float); c = np.array([1.0, 4.0])
for step in range(4):
    a = np.argmin(np.abs(x[:, None] - c[None, :]), axis=1)
    J_before = np.sum((x - c[a]) ** 2)
    c_new = np.array([x[a == k].mean() for k in range(2)])
    J_after = np.sum((x - c_new[a]) ** 2)
    print(f"step {step + 1}: assign {a.tolist()} J(old centres) {J_before:.4f} -> centres {c_new.tolist()} J {J_after:.4f}")
    if np.allclose(c_new, c):
        break
    c = c_new

# 4절: 압축과 이상 탐지
# Section 4: compression and anomaly detection
head("4 PCA")
S = np.array([[2.0, 1.0], [1.0, 2.0]]); lam, V = np.linalg.eigh(S)
print("eigenvalues", lam[::-1], "first PC", V[:, ::-1][:, 0], "explained", round(lam.max() / lam.sum(), 4))
head("4 isolation forest")
n = 256; H = np.log(n - 1) + 0.5772156649; cn = 2 * H - 2 * (n - 1) / n
print("c(256) =", round(cn, 4), " s(h=3) =", round(2 ** (-3 / cn), 4), " s(h=10) =", round(2 ** (-10 / cn), 4))

# 5절: 강화학습
# Section 5: reinforcement learning
head("5 Q-learning and SARSA update")
Q, alpha_, r, gamma = 0.0, 0.5, -1.0, 0.9
print("Q-learning (max Q' = 10):", Q + alpha_ * (r + gamma * 10 - Q), " SARSA (Q' of taken action = 2):", round(Q + alpha_ * (r + gamma * 2 - Q), 4))
head("5 policy gradient on a 3-armed bandit")
theta = np.zeros(3); pi = np.exp(theta) / np.exp(theta).sum(); eta, G, a = 0.1, 1.0, 0
grad = -pi.copy(); grad[a] += 1
theta2 = theta + eta * G * grad; pi2 = np.exp(theta2) / np.exp(theta2).sum()
print("pi", pi.round(4), "theta after", theta2.round(4), "pi after", pi2.round(4))
head("5 actor-critic advantage")
print("A = r + gamma V(s') - V(s) =", round(1.0 + 0.9 * 5.0 - 4.0, 4))

# 6절: genetic algorithm
# Section 6: genetic algorithm
head("6 GA crossover and mutation")
p1, p2 = "11010010", "01101101"
c1, c2 = p1[:3] + p2[3:], p2[:3] + p1[3:]
print("parents", p1, p1.count("1"), p2, p2.count("1"), "children", c1, c1.count("1"), c2, c2.count("1"))
print("mutation 1/8 per bit: expected flips", 8 * (1 / 8), " P(no flip) =", round((7 / 8) ** 8, 4))
fit = np.array([4, 5, 3, 6], float)
print("roulette selection probabilities", (fit / fit.sum()).round(4))
