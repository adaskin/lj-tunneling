"""ljtunnel.graph -- cuts of a weighted graph, and what they say about layouts.

The correlation graph of an N-particle state has an edge weight w_ij (the pairwise mutual
information, maximised over time).  The graph cut of a subset A is

        C(A) = 1/2 * sum_{i in A, j not in A} w_ij .

Enumerating every cut costs 2^(N-1) evaluations of a quadratic form -- a few milliseconds up to
N = 20 -- so it is NOT what limits the particle number of a simulation: the weights w_ij have to
be measured on the state, and the state has k^N amplitudes.  Given the weights, everything here
is cheap.

    all_cuts(W)            C(A) for every bipartition (A contains particle 0)
    prefix_cuts(W, order)  C of the prefix cuts of a particle order (the bonds of an MPS)
    best_order(W)          the order minimising the largest prefix cut (exhaustive, N <= 9)
    bisection_tree(W)      recursive minimum-balanced-cut layout (exact cut at every level)
"""
import itertools
import math

import numpy as np


def bit_matrix(N, chunk=None):
    """All masks containing particle 0 (excluding the full set) and their 0/1 membership matrix."""
    masks = np.arange(1, 1 << N, 2, dtype=np.int64)
    masks = masks[masks != (1 << N) - 1]
    bits = ((masks[:, None] >> np.arange(N)) & 1).astype(float)
    return masks, bits


def all_cuts(W):
    """(masks, C) with C[m] = 1/2 sum_{i in A, j not in A} W_ij for every bipartition."""
    W = np.array(W, float)
    np.fill_diagonal(W, 0.0)
    N = len(W)
    masks, X = bit_matrix(N)
    s = W.sum(1)
    C = 0.5 * (X @ s - np.einsum('mi,ij,mj->m', X, W, X, optimize=True))
    return masks, C


def cut_value(W, A):
    W = np.array(W, float)
    N = len(W)
    A = list(A)
    B = [j for j in range(N) if j not in A]
    return 0.5 * float(W[np.ix_(A, B)].sum())


def prefix_cuts(W, order):
    return [cut_value(W, order[:b]) for b in range(1, len(order))]


def small_sizes(masks, N):
    pc = np.array([bin(int(m)).count('1') for m in masks])
    return np.minimum(pc, N - pc)


def size_extremes(W):
    """For every smaller-side size a: (min C, max C) over the cuts of that size."""
    N = len(W)
    masks, C = all_cuts(W)
    sz = small_sizes(masks, N)
    return {int(a): (float(C[sz == a].min()), float(C[sz == a].max())) for a in range(1, N // 2 + 1)}


def best_order(W, max_N=9):
    """Particle order minimising max_b C(prefix_b), by exhaustive search (N <= max_N)."""
    N = len(W)
    if N > max_N:
        raise ValueError(f"exhaustive order search is limited to N <= {max_N}")
    best, arg = np.inf, None
    for p in itertools.permutations(range(N)):
        if p[0] > p[-1]:
            continue
        v = max(prefix_cuts(W, p))
        if v < best - 1e-12:
            best, arg = v, p
    return list(arg), float(best)


def bisection_tree(W, min_part=None):
    """Recursive bisection into the most balanced minimum cut at every level.

    Returns dict(order, tree_cost, mps_cost, sets): `order` is the leaf order of the recursion tree,
    `tree_cost` the largest C(set) over the sets of the tree (the cost of the tree tensor network with
    that layout), `mps_cost` the largest prefix cut of `order` (the same layout used as an MPS)."""
    W = np.array(W, float)
    N = len(W)
    sets = []

    def split(idx):
        n = len(idx)
        if n <= 1:
            return list(idx)
        sub = W[np.ix_(idx, idx)]
        masks, C = all_cuts(sub) if n > 1 else (None, None)
        sz = small_sizes(masks, n)
        h = n // 2
        sel = np.where(sz == h)[0]
        j = sel[np.argmin(C[sel])]
        m = int(masks[j])
        A = [idx[i] for i in range(n) if (m >> i) & 1]
        B = [idx[i] for i in range(n) if not (m >> i) & 1]
        sets.append(tuple(A))
        sets.append(tuple(B))
        return split(A) + split(B)

    order = split(list(range(N)))
    tree_cost = max(cut_value(W, A) for A in sets) if sets else 0.0
    return dict(order=order, tree_cost=float(tree_cost),
                mps_cost=float(max(prefix_cuts(W, order))), sets=sets)


def sharing_weights(N, S1):
    """Uniform weights w = 2 S1 / (N - 1): every particle shares its one-particle entropy equally
    among its N - 1 partners.  Then C(A) = a (N - a) / (N - 1) * S1 for |A| = a."""
    W = np.full((N, N), 2.0 * S1 / (N - 1))
    np.fill_diagonal(W, 0.0)
    return W
