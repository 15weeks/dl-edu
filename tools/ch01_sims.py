# 1장 11절 t-test 시뮬레이션 숫자와 12절 broadcasting error 문구 확인
# Check chapter 1 section 11 t-test simulation numbers and the section 12 broadcasting error text
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)

def sim(d, s, n, reps=20000):
    # Welch t-test를 reps번 반복해 t의 분포와 p < 0.05 비율을 구한다
    # Repeat a Welch t-test reps times; return t percentiles and the fraction with p < 0.05
    A = 0.82 + s * rng.standard_normal((reps, n))
    B = 0.82 + d + s * rng.standard_normal((reps, n))
    r = stats.ttest_ind(B, A, axis=1, equal_var=False)
    return np.percentile(r.statistic, [10, 25, 50, 75, 90]).round(2), (r.pvalue < 0.05).mean().round(3)

for d, s, n in [(0.03, 0.016, 5), (0.03, 0.04, 5), (0.0, 0.016, 5), (0.03, 0.016, 10), (0.03, 0.04, 20)]:
    print(d, s, n, *sim(d, s, n))
print("t crit df 8:", round(stats.t.ppf(0.975, 8), 3))
try:
    np.zeros((2, 3)) + np.zeros((3, 2))
except ValueError as e:
    print("numpy error:", e)
