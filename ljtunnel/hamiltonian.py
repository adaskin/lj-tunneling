"""ljtunnel.hamiltonian -- the N-particle Hamiltonian in the product basis of one-body modes,
applied matrix-free (never as a k^N x k^N matrix)."""
import itertools

import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator

from .model import modes, project_pair


class ModeN:
    """H_N and the observable O_N = sum_i Theta(x_i) in the k^N product basis of one-body modes at
    tilt c (particle i carries the tilt offset i*C0)."""

    def __init__(self, model, c, k, N):
        g, C0 = model.g, model.mp['C0']
        self.k, self.N, self.c = k, N, c
        es, Ps = [], []
        for i in range(N):
            e, P = modes(g, model.mp, c, i * C0, k)
            es.append(e)
            Ps.append(P)
        self.P = Ps
        E0 = 0.0
        for i, e in enumerate(es):
            sh = [1] * N
            sh[i] = k
            E0 = E0 + e.reshape(sh)
        self.E0 = E0 + np.zeros((k,) * N)
        self.pairs = {(i, j): project_pair(g['VP'], Ps[i], Ps[j])
                      for i, j in itertools.combinations(range(N), 2)}
        DR = g['DR']
        self.O1 = [P.T @ (DR[:, None] * P) for P in Ps]     # Theta(x_i) in the mode basis
        self.dim = k ** N

    # -- the operator --------------------------------------------------------------------
    def apply_t(self, psi):
        """H psi for a state of shape (k,)*N."""
        k, N = self.k, self.N
        out = self.E0 * psi
        for (i, j), V in self.pairs.items():
            t = np.moveaxis(psi, (i, j), (0, 1)).reshape(k * k, -1)
            r = (V @ t).reshape((k, k) + (k,) * (N - 2))
            out = out + np.moveaxis(r, (0, 1), (i, j))
        return out

    def matvec(self, v):
        return self.apply_t(np.asarray(v).reshape((self.k,) * self.N)).reshape(-1)

    def one_body(self, ops, psi):
        out = 0.0
        for i, O in enumerate(ops):
            out = out + np.moveaxis(np.tensordot(O, psi, axes=(1, i)), 0, i)
        return out

    def O_apply(self, psi):
        return self.one_body(self.O1, psi)

    def O_particle(self, psi):
        """<Theta(x_i)> for every particle i."""
        out = []
        for i, O in enumerate(self.O1):
            out.append(float(np.real(np.vdot(psi, np.moveaxis(np.tensordot(O, psi, axes=(1, i)), 0, i)))))
        return out

    def linop(self, extra=None):
        if extra is None:
            return LinearOperator((self.dim, self.dim), matvec=self.matvec, dtype=float)

        def mv(v):
            x = np.asarray(v).reshape((self.k,) * self.N)
            return (self.apply_t(x) + self.one_body(extra, x)).reshape(-1)
        return LinearOperator((self.dim, self.dim), matvec=mv, dtype=float)

    def lowest(self, n=8, v0=None, extra=None, tol=1e-9):
        """The n lowest levels (optionally of H + extra one-body operators)."""
        if n >= self.dim:
            w, vec = np.linalg.eigh(dense_H(self))
            return w[:n], vec[:, :n]
        ev, vc = eigsh(self.linop(extra), k=n, which='SA', tol=tol, v0=v0, ncv=max(3 * n, 30))
        o = np.argsort(ev)
        return ev[o], vc[:, o]

    def O_matrix(self, vc):
        sh = (self.k,) * self.N
        Ov = np.stack([self.O_apply(vc[:, j].reshape(sh)).reshape(-1) for j in range(vc.shape[1])], 1)
        return vc.T @ Ov

    def penalty_ops(self, sides, W):
        """One-body penalty that confines particle i to the valley sides[i] (+1 right, -1 left)."""
        I = np.eye(self.k)
        return [W * (I - D) if s > 0 else W * D for D, s in zip(self.O1, sides)]


def dense_H(M):
    """Explicit k^N x k^N Hamiltonian (reference for the tests; small k, N only)."""
    k, N = M.k, M.N
    I = np.eye(k)
    H = np.diag(M.E0.reshape(-1)).astype(float)
    for (i, j), V in M.pairs.items():
        T = V.reshape(k, k, k, k)
        letters = 'abcdefgh'[:N]
        up = letters.upper()
        lhs = ''.join(letters[x] for x in range(N))
        rhs = ''.join(up[x] for x in range(N))
        spec, ops = [letters[i] + letters[j] + up[i] + up[j]], [T]
        for x in range(N):
            if x not in (i, j):
                spec.append(letters[x] + up[x])
                ops.append(I)
        full = np.einsum(','.join(spec) + '->' + lhs + rhs, *ops)
        H = H + full.reshape(k ** N, k ** N)
    return H
