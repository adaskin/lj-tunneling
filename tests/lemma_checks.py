#!/usr/bin/env python3
"""
lemma_checks.py -- numerical checks of the two cut lemmas used in the paper's Sec. 2.

Lemma 1 (pivot).  Every tree tensor network of node degree <= 3 on N particles contains a bond
    whose two sides hold >= ceil(N/3) particles each.   Checked on random full binary trees
    (uniform, balanced, caterpillar = MPS, and skewed splits), N = 4..60.
Lemma 2 (cut size).  The entropy of a pure state averaged over all size-a subsets, s(a), is
    symmetric (s(a) = s(N-a)), concave in a (strong subadditivity) and obeys s(a) <= a s(1) (Han).
    Checked on random pure states with controlled entanglement.
RUN   python lemma_checks.py        (about 20 s)
"""
import itertools, math, random
import numpy as np


def S_of(psi, A, N, d):
    A = list(A)
    B = [i for i in range(N) if i not in A]
    M = np.transpose(psi, A + B).reshape(d ** len(A), -1)
    sv = np.linalg.svd(M, compute_uv=False) ** 2
    sv = sv[sv > 1e-14]
    return float(-(sv * np.log2(sv)).sum())


def lemma2():
    rng = np.random.default_rng(1)
    conc, han, sym = 1e9, 1e9, 0.0
    for N, d in ((5, 3), (6, 2), (7, 2), (4, 4)):
        for _ in range(20):
            prod = np.ones(1)
            for _i in range(N):
                v = rng.standard_normal(d) + 1j * rng.standard_normal(d)
                prod = np.multiply.outer(prod, v / np.linalg.norm(v))
            noise = rng.standard_normal((d,) * N) + 1j * rng.standard_normal((d,) * N)
            psi = prod.reshape((d,) * N) + rng.uniform(0.05, 0.6) * 0.2 * np.sqrt(d ** N) \
                * noise / np.linalg.norm(noise)
            psi /= np.linalg.norm(psi)
            sb = [0.0] + [np.mean([S_of(psi, A, N, d) for A in itertools.combinations(range(N), a)])
                          for a in range(1, N)] + [0.0]
            for a in range(1, N):
                conc = min(conc, 2 * sb[a] - sb[a - 1] - sb[a + 1])
                han = min(han, a * sb[1] - sb[a])
                sym = max(sym, abs(sb[a] - sb[N - a]))
    print(f"Lemma 2: min second difference {conc:.2e} (>= 0 : concave), "
          f"min a*s(1)-s(a) {han:.2e} (>= 0), max asymmetry {sym:.1e}")
    return conc >= -1e-12 and han >= -1e-12


def bond_sides(n, mode):
    """smaller-side sizes of all internal bonds of a random full binary tree with n leaves"""
    sizes = []

    def build(m):
        if m == 1:
            return 1
        if mode == 'uniform':
            k = random.randint(1, m - 1)
        elif mode == 'balanced':
            k = m // 2
        elif mode == 'caterpillar':
            k = 1
        else:
            k = max(1, min(m - 1, int(random.betavariate(0.3, 0.3) * m)))
        l, r = build(k), build(m - k)
        sizes.extend((l, r))
        return l + r
    build(n)
    return [min(s, n - s) for s in sizes if 1 < s < n]


def lemma1():
    random.seed(3)
    bad = tot = 0
    for N in range(4, 61):
        for mode in ('uniform', 'balanced', 'caterpillar', 'skewed'):
            for _ in range(200):
                best = max(bond_sides(N, mode), default=0)
                tot += 1
                bad += best < math.ceil(N / 3)
    print(f"Lemma 1: {tot} random binary trees, N = 4..60: violations = {bad}")
    return bad == 0


if __name__ == '__main__':
    ok = lemma1() & lemma2()
    print('LEMMA CHECKS ' + ('PASSED' if ok else 'FAILED'))
