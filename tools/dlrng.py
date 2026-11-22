# assets/dl.js의 makeRng(mulberry32 + Marsaglia polar)와 assets/dlfig.js의 makeData를 Python으로 옮긴 것
# Python port of makeRng (mulberry32 + Marsaglia polar) in assets/dl.js and makeData in assets/dlfig.js
# browser와 똑같은 난수·data를 만들어 예제 숫자를 미리 계산할 때 쓴다
# Produces exactly the same random numbers and data as the browser, for precomputing example numbers
import math

M32 = 0xFFFFFFFF


def _imul(a, b):
    return (a * b) & M32


class Rng:
    def __init__(self, seed):
        self.a = seed & M32
        self.spare = None

    def u(self):
        self.a = (self.a + 0x6D2B79F5) & M32
        a = self.a
        t = _imul(a ^ (a >> 15), 1 | a)
        t = ((t + _imul(t ^ (t >> 7), 61 | t)) & M32) ^ t
        return ((t ^ (t >> 14)) & M32) / 4294967296

    def n(self):
        if self.spare is not None:
            s, self.spare = self.spare, None
            return s
        while True:
            a, b = 2 * self.u() - 1, 2 * self.u() - 1
            s = a * a + b * b
            if 0 < s < 1:
                break
        m = math.sqrt(-2 * math.log(s) / s)
        self.spare = b * m
        return a * m


def make_data(kind, n, rng, noise=0.1):
    X, Y = [], []
    g = lambda: rng.n() * noise
    for i in range(n):
        if kind == "blobs":
            k = i % 2; x = (0.6 if k else -0.6) + g() * 3; y = (0.35 if k else -0.35) + g() * 3
        elif kind == "xor":
            k = i % 2; sx = 1 if rng.u() < 0.5 else -1; sy = -sx if k else sx
            x = sx * (0.25 + 0.7 * rng.u()) + g(); y = sy * (0.25 + 0.7 * rng.u()) + g()
        elif kind == "circles":
            k = i % 2; a = rng.u() * 2 * math.pi; rr = (1.0 if k else 0.42) + g(); x = rr * math.cos(a); y = rr * math.sin(a)
        elif kind == "moons":
            k = i % 2; a = rng.u() * math.pi
            x = 1 - math.cos(a) - 0.5 if k else math.cos(a) - 0.5
            y = 0.25 - math.sin(a) + 0.25 if k else math.sin(a) - 0.25
            x *= 0.85; y *= 0.85; x += g(); y += g()
        elif kind == "spiral":
            k = i % 2; t = 0.25 + rng.u() * 0.75; a = t * 3.6 * math.pi + k * math.pi
            x = t * math.cos(a) * 1.1 + g() * 0.6; y = t * math.sin(a) * 1.1 + g() * 0.6
        elif kind == "three":
            k = i % 3; cx = [-0.7, 0.7, 0][k]; cy = [-0.45, -0.45, 0.65][k]; x = cx + g() * 3; y = cy + g() * 3
        X.append([x, y]); Y.append(k)
    return X, Y


def shuffled(n, rng):
    # dlfig.js의 shuffled와 같은 Fisher-Yates 섞기
    # Fisher-Yates shuffle, identical to shuffled() in dlfig.js
    a = list(range(n))
    for i in range(n - 1, 0, -1):
        j = int(rng.u() * (i + 1))
        a[i], a[j] = a[j], a[i]
    return a
