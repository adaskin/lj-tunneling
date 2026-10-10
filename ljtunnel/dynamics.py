"""ljtunnel.dynamics -- real-time propagation by Chebyshev expansion of exp(-iH dt)."""
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.special import jv


def spectral_bounds(M):
    """Lower and upper bound of the spectrum of H (Lanczos, padded)."""
    lo = eigsh(M.linop(), k=1, which='SA', tol=1e-4)[0][0]
    hi = eigsh(M.linop(), k=1, which='LA', tol=1e-4)[0][0]
    return lo - 1e-3, hi + 1e-3


def cheb_nterms(z):
    return int(z + 12 * max(1.0, z) ** (1 / 3) + 20)


def cheb_step(M, psi, dt, lo, hi):
    """psi(t+dt) = exp(-i H dt) psi(t); Chebyshev series of Bessel functions."""
    a, b = 0.5 * (hi + lo), 0.5 * (hi - lo)

    def Hs(x):
        return (M.apply_t(x) - a * x) / b

    z = b * dt
    nterms = cheb_nterms(z)
    t0 = psi.astype(complex)
    t1 = Hs(t0)
    out = jv(0, z) * t0 + 2 * (-1j) * jv(1, z) * t1
    for n in range(2, nterms):
        t2 = 2 * Hs(t1) - t0
        out = out + 2 * (-1j) ** n * jv(n, z) * t2
        t0, t1 = t1, t2
    return np.exp(-1j * a * dt) * out


def snapshots(M, psi0, times, bounds=None):
    """Generator over the snapshot times: yields (i, t_i, psi(t_i)); i = 0 is psi0 itself."""
    lo, hi = bounds if bounds is not None else spectral_bounds(M)
    psi = psi0.astype(complex)
    yield 0, float(times[0]), psi
    for i in range(1, len(times)):
        psi = cheb_step(M, psi, times[i] - times[i - 1], lo, hi)
        yield i, float(times[i]), psi


def matvec_count(bounds, T, dt):
    """Number of H applications of a run of window T with snapshot spacing dt."""
    lo, hi = bounds
    z = 0.5 * (hi - lo) * dt
    return int(round(T / dt)) * cheb_nterms(z)
