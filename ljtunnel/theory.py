"""
ljtunnel.theory -- the bounds of the paper as code (no simulation).

Everything here is arithmetic on the entropy of a cut.  Entropies are in bits.

CUTS
    A cut is a subset A of the N particles, |A| = a; S_A(t) is the entropy of the
    reduced state; sigma(A) = max_t S_A(t).  An MPS in the particle order
    0..N-1 has bonds b = 1..N-1 = the prefix cuts A = {0..b-1}; a tensor
    network of any other layout has a different set of cuts.

    pivot_sizes(N)       sizes (a, N-a) that EVERY tree tensor network of node degree
                         <= 3 must contain (centroid lemma): N/3 <= a <= N/2.
    chord_lower(a, N, Sc) concavity of the subset-averaged entropy: a cut of size a
                         carries at least 2a/N of the central value Sc
    han_upper(N, S1)     the central cut carries at most floor(N/2) times the
                         one-particle entropy S1 (subadditivity)
    hilbert_cap(a,N,d)   min(a, N-a) log2 d

GROWTH LAWS FOR THE CENTRAL CUT  S_N = S(floor(N/2) | N)
    law_linear(S_ref, s)   S_N = S_ref + s (N - N_ref)            [constant growth s]
    law_pairs(S0, beta)    S_N = S0 + beta floor(N/2) ceil(N/2)   [crossing pairs]
    law_power(S_ref, p)    S_N = S_ref (N/N_ref)^p                [p=1/2 area, 2 volume]
    The three are the same function family seen from different physics; data decide.

COSTS (MPS, bond dimension chi = 2^S at every bond: conservative)
    memory  16 N d chi^2        flops  sweeps * N * 2 (d chi)^3
    wall = first N whose cost exceeds the budget.
"""
import math


# ------------------------------------------------------------------ cut lemmas
def pivot_sizes(N):
    """Sizes of the smaller side guaranteed by the centroid lemma: ceil(N/3)..floor(N/2)."""
    return math.ceil(N / 3), N // 2


def chord_lower(a, N, S_central):
    """Lower bound on the (subset-averaged) entropy of a size-a cut from the central
    value: concavity + S(0) = 0 + symmetry => s(a) >= a/floor(N/2) * s(floor(N/2))."""
    a = min(a, N - a)
    return a / (N // 2) * S_central


def han_upper(N, S_one):
    """Central cut <= floor(N/2) * (one-particle entropy)."""
    return (N // 2) * S_one


def hilbert_cap(a, N, d):
    return min(a, N - a) * math.log2(d)


# ------------------------------------------------------------------ growth laws
def law_linear(S_ref, s, N_ref=4):
    return lambda N: S_ref + s * (N - N_ref)


def pairs(N):
    """Number of particle pairs split by a central cut: floor(N/2) ceil(N/2)."""
    return (N // 2) * ((N + 1) // 2)


def law_pairs(S0, beta):
    return lambda N: S0 + beta * pairs(N)


def law_power(S_ref, p, N_ref=4):
    return lambda N: S_ref * (N / N_ref) ** p


def fit_pairs(data):
    """Least-squares (S0, beta) for S = S0 + beta*pairs(N) from {N: S}."""
    xs = [pairs(N) for N in data]
    ys = list(data.values())
    n = len(xs)
    xm, ym = sum(xs) / n, sum(ys) / n
    sxx = sum((x - xm) ** 2 for x in xs)
    beta = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sxx
    return ym - beta * xm, beta


def fit_linear(data):
    """Least-squares (S0 at N=0, s) for S = S0 + s N from {N: S}."""
    xs, ys = list(data), list(data.values())
    n = len(xs)
    xm, ym = sum(xs) / n, sum(ys) / n
    s = (sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
         / sum((x - xm) ** 2 for x in xs))
    return ym - s * xm, s


# ------------------------------------------------------------------ MPS costs
def mem_bytes(N, d, chi):
    return 16.0 * N * d * chi ** 2


def sweep_flops(N, d, chi, sweeps):
    return sweeps * N * 2.0 * (d * chi) ** 3


def S_budget_mem(N, d, B_mem):
    """Largest central entropy (bits) whose MPS fits in memory."""
    return 0.5 * math.log2(B_mem / (16.0 * N * d))


def S_budget_flop(N, d, B_flop, sweeps):
    """Largest central entropy (bits) whose sweeps fit in the flop budget."""
    return math.log2((B_flop / (2.0 * sweeps * N)) ** (1.0 / 3.0) / d)


def S_budget(N, d, B_mem, B_flop, sweeps):
    return min(S_budget_mem(N, d, B_mem), S_budget_flop(N, d, B_flop, sweeps))


def wall(Sfun, d, B_mem, B_flop, sweeps, N_from=2, nmax=400, which='both'):
    """First failing N for the entropy law Sfun(N) (a callable returning bits)."""
    for N in range(N_from, nmax):
        S = Sfun(N)
        okm = S <= S_budget_mem(N, d, B_mem)
        okf = S <= S_budget_flop(N, d, B_flop, sweeps)
        ok = {'mem': okm, 'flop': okf, 'both': okm and okf}[which]
        if not ok:
            return N
    return nmax


def s_crit(d, S_ref, N_ref, N_hi, B_flop, sweeps):
    """Growth below which the flop budget still holds at N_hi (constant-growth law)."""
    return ((math.log2(B_flop / (2.0 * sweeps * d ** 3)) - math.log2(N_hi) - 3 * S_ref)
            / (3.0 * (N_hi - N_ref)))


# ------------------------------------------------------------------ doublet leakage
def leak_bound(ov_sub):
    """|<O>_psi - <O>_P| <= (2 sqrt(l) + l) ||O||,  l = 1 - ov_sub."""
    l = max(0.0, 1.0 - ov_sub)
    return 2.0 * math.sqrt(l) + l
