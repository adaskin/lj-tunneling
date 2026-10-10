"""ljtunnel.entanglement -- entropies of the bipartitions of a pure N-particle state.

A *cut* is a subset A of the particles; for a pure state S(A) = S(A^c).  Cuts are stored as
bitmasks (bit i set = particle i in A) in the canonical representative that CONTAINS particle 0,
so the 2^(N-1) - 1 distinct bipartitions are enumerated exactly once.  The entropy is always
computed on the smaller side, whose reduced density matrix has dimension k^min(|A|, N-|A|).

Entropies are in bits.  For the contiguous bonds b = 1..N-1 (A = {0..b-1}, the cuts of a matrix
product state in the particle order) we also keep the Renyi-2 entropy and the numerical Schmidt
rank chi(eps) at discarded weight eps = 1e-2, 1e-3, 1e-6.
"""
import itertools

import numpy as np

from .log import log

EPS_RANK = (1e-2, 1e-3, 1e-6)


# ------------------------------------------------------------------ bitmask helpers
def mask_of(A):
    return sum(1 << int(i) for i in A)


def members(mask, N):
    return tuple(i for i in range(N) if (mask >> i) & 1)


def canon_mask(mask, N):
    """The representative of {A, A^c} that contains particle 0."""
    return mask if (mask & 1) else ((1 << N) - 1) ^ mask


def popcount(mask):
    return bin(mask).count('1')


# ------------------------------------------------------------------ one cut
def reduced_eigs(psi, X, k):
    """Eigenvalues of the reduced density matrix of the particles X (tuple) of psi (shape (k,)*N)."""
    N = psi.ndim
    other = [i for i in range(N) if i not in X]
    Psi = np.transpose(psi, list(X) + other).reshape(k ** len(X), -1)
    rho = Psi @ Psi.conj().T
    rho /= np.trace(rho).real
    return np.linalg.eigvalsh(rho)


def spectrum_stats(ev, eps=EPS_RANK):
    """(S_vN, S_2, ranks) from the eigenvalues of a reduced density matrix, entropies in bits.

    chi(eps) = smallest r with sum_{j>r} lambda_j <= eps: the bond dimension an MPS truncated at
    error eps needs.  The exact rank is >= 2^S_vN; chi(eps) can be much smaller.  S_vN >= S_2."""
    ev = np.sort(np.clip(ev, 0.0, None))[::-1]
    nz = ev[ev > 1e-12]
    S = float(-np.sum(nz * np.log(nz)) / np.log(2))
    S2 = float(-np.log2(np.sum(ev ** 2)))
    tail = np.cumsum(ev[::-1])[::-1]
    ranks = [int(np.argmax(tail <= e)) if tail[-1] <= e else len(ev) for e in eps]
    return S, S2, ranks


def entropy_bits(psi, A, k):
    """Entropy (bits) of the cut A (any subset), computed on its smaller side."""
    N = psi.ndim
    A = tuple(sorted(A))
    if len(A) > N - len(A):
        A = tuple(i for i in range(N) if i not in A)
    return spectrum_stats(reduced_eigs(psi, A, k))[0]


# ------------------------------------------------------------------ the set of cuts of a run
class CutPlan:
    """Which cuts are measured, and how.

    mode   'all'          every bipartition
           'bonds+pairs'  singles, pairs and the contiguous bonds
           'bonds'        singles and the contiguous bonds
    Singles and contiguous bonds are always included (they give the one-particle entropies and the
    matrix-product-state bonds).  A cut whose smaller side has dimension k^a above max_cut_dim is
    skipped (NaN in the series).  Cuts with a smaller side of >= 3 particles that are not contiguous
    bonds are evaluated every big_cut_stride-th snapshot only.
    """

    def __init__(self, N, k, mode='all', max_cut_dim=3000, big_cut_stride=1):
        self.N, self.k, self.mode = N, k, mode
        want = {canon_mask(1 << i, N) for i in range(N)}
        want |= {(1 << b) - 1 for b in range(1, N)}                     # prefix cuts contain particle 0
        if mode in ('bonds+pairs', 'all') and N > 2:       # for N = 2 the pair is the whole (pure) state
            want |= {canon_mask((1 << i) | (1 << j), N) for i, j in itertools.combinations(range(N), 2)}
        if mode == 'all':
            want |= {m for m in range(1, 1 << N) if (m & 1) and m != (1 << N) - 1}

        def small(m):
            A = members(m, N)
            return A if len(A) <= N - len(A) else tuple(i for i in range(N) if i not in A)

        self.masks = np.array(sorted(want, key=lambda m: (len(small(m)), m)), dtype=np.int64)
        self.small = [small(int(m)) for m in self.masks]
        self.dim = np.array([k ** len(X) for X in self.small], dtype=float)
        pref = {(1 << b) - 1: b for b in range(1, N)}
        self.bond_b = np.arange(1, N)
        index = {int(m): j for j, m in enumerate(self.masks)}
        self.bond_idx = np.array([index[(1 << b) - 1] for b in range(1, N)])
        small_ok = np.array([len(X) <= 2 for X in self.small])
        self.skip = (self.dim > max_cut_dim) & ~small_ok
        self.big = np.array([(len(X) >= 3) and (int(m) not in pref) for X, m in zip(self.small, self.masks)])
        self.stride = max(1, int(big_cut_stride))
        self.index = index
        # a contiguous bond is what a matrix-product state in the particle order pays: say so if it is not measured
        self.skipped_bonds = [int(b) for b, j in zip(self.bond_b, self.bond_idx) if self.skip[j]]
        if self.skipped_bonds:
            log(f"WARNING: contiguous bond(s) {self.skipped_bonds} of N={N}, k={k} are NOT measured (smaller side above "
                f"max_cut_dim = {max_cut_dim}); S_nat / S_cen need them -- raise [protocol] max_cut_dim")

    @property
    def n_cuts(self):
        return len(self.masks)

    def n_skipped(self):
        return int(self.skip.sum())

    def measure(self, psi, snap_index=0):
        """Entropies of all cuts of psi.  Returns (S[n_cuts], bond[N-1, 4]) with bond columns
        S_2, chi(1e-2), chi(1e-3), chi(1e-6).  Cuts not measured at this snapshot are NaN."""
        S = np.full(self.n_cuts, np.nan)
        bond = np.full((self.N - 1, 4), np.nan)
        which_bond = {int(j): b for b, j in enumerate(self.bond_idx)}
        for j in range(self.n_cuts):
            if self.skip[j]:
                continue
            if self.big[j] and (snap_index % self.stride):
                continue
            ev = reduced_eigs(psi, self.small[j], self.k)
            s, s2, ranks = spectrum_stats(ev)
            S[j] = s
            b = which_bond.get(j)
            if b is not None:
                bond[b] = [s2, *ranks]
        return S, bond
