"""ljtunnel.cost -- what a simulation costs before you start it, as a function of (N, k, T).

The state of N particles in a k-mode basis has k^N amplitudes; the bipartition entropies need the
reduced density matrix of the smaller side, of dimension k^min(a, N-a).  This table is the reason
the particle number of an exact simulation stops near N = 7 at k = 12, however few cuts one
evaluates: it is the state, not the 2^(N-1) cuts, that does not fit.

flops per H application   4 C(N,2) k^(N+2)        (real pair matrices on a complex state)
Chebyshev propagation     ~ b*T matvecs, b = half-width of the spectrum of H
                          (measured 15 / 27 / 47 at N = 3 / 4 / 5 for k = 12; beyond N = 5 it is
                          EXTRAPOLATED linearly in C(N,2) from the last two measured points)
"""
import math
import time

import numpy as np

from .dynamics import cheb_nterms


def calibrate(model, N=5, k=12, c=2.9, n=3):
    """Measured seconds per H application and the flop rate, on this machine."""
    from .hamiltonian import ModeN
    M = ModeN(model, c, k, N)
    x = (np.random.default_rng(0).standard_normal((k,) * N)).astype(complex)
    M.apply_t(x)
    t = time.time()
    for _ in range(n):
        M.apply_t(x)
    sec = (time.time() - t) / n
    flops = 4.0 * math.comb(N, 2) * k ** (N + 2)
    return dict(N=N, k=k, sec=sec, flops=flops, rate=flops / sec)


def half_width(N, measured=None):
    """Half-width b of the spectrum of H.  `measured` = {N: b}; beyond the largest measured N the
    value is extrapolated linearly in C(N,2) from the last two points."""
    measured = measured or {3: 15.1, 4: 26.9, 5: 46.5}
    if N in measured:
        return measured[N], False
    Ns = sorted(measured)
    n1, n2 = Ns[-2], Ns[-1]
    slope = (measured[n2] - measured[n1]) / (math.comb(n2, 2) - math.comb(n1, 2))
    return measured[n2] + slope * (math.comb(N, 2) - math.comb(n2, 2)), N > n2


def table(cfg, k=None, N_list=range(2, 13), T=None, dt=None, rate=3.0e10, n_vectors=6, measured_b=None):
    """Rows of the cost table.  `rate` is the sustained flop rate (use calibrate() to measure it)."""
    k = k or cfg['basis']['k']
    pr = cfg['protocol']
    T = pr['T'] if T is None else T
    dt = pr['dt_snap'] if dt is None else dt
    rows = []
    for N in N_list:
        amp = float(k) ** N
        b, extrap = half_width(N, measured_b)
        flops = 4.0 * math.comb(N, 2) * float(k) ** (N + 2)
        nmv = (T / dt) * cheb_nterms(b * dt) if N >= 3 else (T / dt) * cheb_nterms(5.0 * dt)
        a_big = N // 2
        rho_gb = 16.0 * float(k) ** (2 * a_big) / 1e9
        rows.append(dict(N=N, amplitudes=amp, gb_vector=16.0 * amp / 1e9, gb_run=n_vectors * 16.0 * amp / 1e9,
                         flops_matvec=flops, sec_matvec=flops / rate, b=b, b_extrapolated=extrap,
                         matvecs=nmv, hours=nmv * flops / rate / 3600.0, central_dim=float(k) ** a_big,
                         rho_gb=rho_gb, n_cuts=2 ** (N - 1) - 1))
    return rows


def format_table(rows):
    out = [f"{'N':>3} {'amplitudes':>11} {'GB/vector':>10} {'GB/run':>9} {'s/matvec':>9} {'matvecs':>9} "
           f"{'hours':>9} {'central rho':>12} {'cuts':>6}"]
    for r in rows:
        out.append(f"{r['N']:3d} {r['amplitudes']:11.2e} {r['gb_vector']:10.3g} {r['gb_run']:9.3g} "
                   f"{r['sec_matvec']:9.3g} {r['matvecs']:9.2e} {r['hours']:9.3g} "
                   f"{r['central_dim']:7.0f}^2 ({r['rho_gb']:.3g} GB) {r['n_cuts']:6d}"
                   + ('  *' if r['b_extrapolated'] else ''))
    out.append("(* spectral half-width extrapolated)")
    return "\n".join(out)
