"""ljtunnel.fullgrid -- N = 2 on the full position grid (no mode truncation): the reference run.

Two distinguishable particles, each on the nx*ny grid (1200 sites for the production grid), so the
state has 1.44e6 amplitudes.  H is applied matrix-free as   H psi = H1a psi + psi H1b^T + V * psi   on the
(n1, n1) array psi (the pair potential is diagonal in position).  What the paper takes from this
run: the resonance tilt c*, the doublet splitting dE_2 and the switching time t_0 = pi / dE_2.
"""
import time

import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator

from .log import log
from .model import Model, h1


class FullGrid2:
    """Two particles on the full grid at tilt c (particle b carries the offset C0)."""

    def __init__(self, model, c):
        g, mp = model.g, model.mp
        self.n1 = g['NB1']
        self.H1a = h1(g, mp, c, 0.0).tocsr()
        self.H1b = h1(g, mp, c, mp['C0']).tocsr()
        self.VP = g['VP']
        self.DR = g['DR']
        self.c = c
        self.dim = self.n1 ** 2

    def apply(self, psi):
        return self.H1a @ psi + (self.H1b @ psi.T).T + self.VP * psi

    def matvec(self, v):
        return self.apply(np.asarray(v).reshape(self.n1, self.n1)).reshape(-1)

    def linop(self):
        return LinearOperator((self.dim, self.dim), matvec=self.matvec, dtype=float)

    def O_diag(self):
        return (self.DR[:, None] + self.DR[None, :]).reshape(-1)

    def lowest(self, n=6, v0=None, tol=1e-9):
        ev, vc = eigsh(self.linop(), k=n, which='SA', tol=tol, v0=v0, ncv=max(3 * n, 24))
        o = np.argsort(ev)
        return ev[o], vc[:, o]


def resonance(model, n_levels=6, c_lo=0.2, c_hi=0.9, c_tol=1e-4, mO_tol=2e-3):
    """Tilt at which <O>_gs = 1/2, by secant/bisection on the ground state; returns the full record."""
    t0 = time.time()
    v0 = None
    hist = []

    def at(c):
        nonlocal v0
        H = FullGrid2(model, c)
        ev, vc = H.lowest(n_levels, v0=v0)
        v0 = vc[:, 0]
        Od = H.O_diag()
        mO = float(np.sum(vc[:, 0] ** 2 * Od))
        log(f"   [fullgrid] c={c:.5f}  <O>_gs={mO:.4f}  dE={ev[1] - ev[0]:.6f}  ({time.time() - t0:.0f}s)")
        hist.append((c, mO))
        return mO, ev, vc, H

    lo, hi = c_lo, c_hi
    mlo, *_ = at(lo)
    mhi, *_ = at(hi)
    if not (min(mlo, mhi) < 0.5 < max(mlo, mhi)):
        raise RuntimeError(f"<O>_gs = 1/2 not bracketed on [{c_lo}, {c_hi}]: {mlo:.3f}, {mhi:.3f}")
    c = lo + (0.5 - mlo) / (mhi - mlo) * (hi - lo)
    for _ in range(12):
        m, ev, vc, H = at(c)
        if abs(m - 0.5) < mO_tol or (hi - lo) < c_tol:
            break
        if (m - 0.5) * (mlo - 0.5) > 0:
            lo, mlo = c, m
        else:
            hi, mhi = c, m
        c = c + (0.5 - m) / (mhi - mlo) * (hi - lo) * 0.5 + 0.0
        c = min(max(c, lo + 0.05 * (hi - lo)), hi - 0.05 * (hi - lo))
    Od = H.O_diag()
    On = np.array([[float(np.sum(vc[:, i] * vc[:, j] * Od)) for j in range(vc.shape[1])]
                   for i in range(vc.shape[1])])
    return dict(c=float(c), ev=ev, On=On, mO=float(m), dE=float(ev[1] - ev[0]), history=hist,
                wall_s=time.time() - t0, vc0=vc[:, 0], vc1=vc[:, 1])


def reference(model, c, n_levels=6):
    """Doublet at a given tilt: levels, <O> matrix of the lowest levels, dE = E1 - E0, t_0 = pi/dE."""
    t0 = time.time()
    H = FullGrid2(model, c)
    ev, vc = H.lowest(n_levels)
    Od = H.O_diag()
    On = np.array([[float(np.sum(vc[:, i] * vc[:, j] * Od)) for j in range(vc.shape[1])]
                   for i in range(vc.shape[1])])
    dE = float(ev[1] - ev[0])
    log(f"[fullgrid N=2] c={c:.4f}  E0={ev[0]:.6f}  dE={dE:.6f}  t0=pi/dE={np.pi / dE:.4f}  "
        f"<O>_gs={On[0, 0]:.4f}  ({time.time() - t0:.0f}s)")
    return dict(c=float(c), ev=ev, On=On, dE=dE, t0=float(np.pi / dE), mO=float(On[0, 0]),
                wall_s=time.time() - t0)
