"""ljtunnel.resonance -- finding the tilt c* at which two occupation isomers are degenerate, and
the two-level (doublet) structure of the low-lying spectrum there.

At the resonance tilt the ground state has <O> close to 1/2.  A *doublet* is a pair of low levels
whose <O> differ by about one particle and which are strongly mixed by O (acceptance gates in
config [doublet]).  A clean two-level doublet is not required for tunnelling dynamics; the weights
of the start on the low levels (lowW, deepW) are the validity check when the gates fail.
"""
import numpy as np

from .hamiltonian import ModeN
from .log import log


def doublet_params(cfg, N):
    """[doublet] gates with the optional per-N override table [doublet.N<N>]."""
    dg = {k_: v for k_, v in cfg['doublet'].items() if not isinstance(v, dict)}
    dg.update(cfg['doublet'].get(f'N{N}', {}))
    return dg


def pick_doublet(ev, Od, Om, do_min, do_max, mix_min):
    """The lowest-splitting pair (i, j) among the candidate levels that passes the gates, or None."""
    n = len(Od)
    best = None
    for i in range(n):
        for j in range(i + 1, n):
            dO = Od[i] - Od[j]
            do = float(np.sqrt(dO ** 2 + 4 * Om[i, j] ** 2))
            s2 = 2 * abs(Om[i, j]) / do if do > 1e-12 else 0.0
            d = ev[j] - ev[i]
            if do_min <= do <= do_max and s2 >= mix_min and d > 1e-8:
                if best is None or d < best[0]:
                    best = (d, i, j, do, s2)
    return best


def doublet_at(model, c, k, N, v0=None):
    dg = doublet_params(model.cfg, N)
    M = ModeN(model, c, k, N)
    ev, vc = M.lowest(dg['n_low'], v0=v0)
    On = M.O_matrix(vc)
    res = pick_doublet(ev, np.diag(On).real, np.abs(On), dg['do_min'], dg['do_max'], dg['mix_min'])
    return M, ev, vc, On, res


def locate(model, N, k, tilt='auto', c_guess=None):
    """Find the resonance for (N, k).

    tilt   'auto'   measured value from [tilts] if there is one, else scan
           'scan'   always scan
           float    use this tilt
    c_guess  skip the coarse scan and refine around this tilt.
    Returns dict(M, ev, vc, On, res, c, mO, how) or None if no resonance was found.
    """
    cfg = model.cfg
    c_fixed = None
    if isinstance(tilt, (int, float)):
        c_fixed = float(tilt)
    elif tilt == 'auto':
        c_fixed = model.tilt(N, k)
    sc = dict(cfg['scan'])
    sc.update(cfg['scan'].get(f'N{N}', {}))
    if c_fixed is not None:
        M, ev, vc, On, res = doublet_at(model, c_fixed, k, N)
        return dict(M=M, ev=ev, vc=vc, On=On, res=res, c=c_fixed, mO=float(On[0, 0].real),
                    how='fixed')
    c0 = c_guess
    if c0 is None:
        v0, prev, crossing = None, None, None
        for c in np.arange(sc['cmin'], sc['cmax'] + 0.5 * sc['cstep'], sc['cstep']):
            M = ModeN(model, c, k, N)
            ev, vc = M.lowest(6, v0=v0)
            v0 = vc[:, 0]
            Od0 = M.O_matrix(vc)[0, 0].real
            log(f"   [scan N={N} k={k}] c={c:5.2f} dE={ev[1] - ev[0]:.5f} <O>={Od0:.3f}")
            if 0.2 < Od0 < 0.8:
                c0 = c
                break
            if crossing is None and prev is not None and prev[1] > 0.5 >= Od0:
                crossing = 0.5 * (prev[0] + c)          # <O> jumped over the whole band between two grid points
            prev = (c, Od0)
        if c0 is None and crossing is not None:
            c0 = crossing
            log(f"   [scan N={N} k={k}] no tilt with 0.2 < <O> < 0.8 on the grid; refining around the <O> = 1/2 crossing at c={c0:.3f}")
        if c0 is None:
            return None
    best, nod, v0 = None, None, None
    hw, npts = sc['refine_halfwidth'], sc['refine_points']
    for c in np.linspace(max(c0 - hw, 0), c0 + hw, npts):
        M, ev, vc, On, res = doublet_at(model, c, k, N, v0)
        v0 = vc[:, 0]
        Od0 = On[0, 0].real
        log(f"   [refine N={N} k={k}] c={c:5.2f} <O>={Od0:.3f} "
            + (f"dE={res[0]:.5f}" if res else "no doublet"))
        if res is not None and (best is None or abs(Od0 - 0.5) < abs(best[4] - 0.5)):
            best = (M, ev, vc, On, Od0, res, c)
        if nod is None or abs(Od0 - 0.5) < abs(nod[4] - 0.5):
            nod = (M, ev, vc, On, Od0, res, c)
    best = best or nod
    if best is None:
        return None
    M, ev, vc, On, Od0, res, c = best
    return dict(M=M, ev=ev, vc=vc, On=On, res=res, c=c, mO=float(Od0), how='scan')


def exact_tilt(model, N, k, c_lo, c_hi, tol=2e-4, max_iter=40):
    """The tilt in [c_lo, c_hi] at which the exact ground state has <O> = 1/2, by bisection (<O>_gs falls with c).  `locate` picks a
    point of a grid instead; this is the point itself (used for the small-k validation cases of data/mps)."""
    def occ(c):
        o = float(doublet_at(model, c, k, N)[3][0, 0].real)
        log(f"   [exact tilt N={N} k={k}] c={c:.5f} <O>_gs={o:.5f}")
        return o
    if not (occ(c_lo) > 0.5 > occ(c_hi)):
        raise ValueError(f"<O>_gs = 1/2 is not bracketed by [{c_lo}, {c_hi}] at N={N}, k={k}")
    for _ in range(max_iter):
        c = 0.5 * (c_lo + c_hi)
        c_lo, c_hi = (c, c_hi) if occ(c) > 0.5 else (c_lo, c)
        if c_hi - c_lo < tol:
            break
    return round(0.5 * (c_lo + c_hi), 4)
