"""ljtunnel.extrapolation -- carrying the measured entropies to larger N under STATED scenarios.

Model.  The central-cut entropy is the weighted cut of the correlation graph; if every particle
shares its one-particle entropy S1(N) equally among its N-1 partners,

        S_cen(N) = b (N-b) / (N-1) * S1(N),         b = floor(N/2)           ("sharing law").

The only inputs are the measured S1(N) up to the largest simulated N.  S1(N) is bounded (by log2 of the
local dimension) and the data show it saturating, so three scenarios are carried forward:

  sat    S1(N) = S_inf - (S_inf - S1(N0)) exp(-(N - N0)/tau), (S_inf, tau) fitted to the data
  pess   S1 keeps growing at the mean increment of the last two intervals
         (pessimistic for the classical side: the central cut then grows like N^2)
  opt    S1 is already saturated at its largest measured value
         (optimistic for the classical side)

Each is turned into a matrix-product-state wall with the cost model of ljtunnel.walls.
Nothing here is a measurement beyond the largest simulated N; it is the stated model.
"""
import math

import numpy as np
from scipy.optimize import least_squares

from . import theory as th


def sharing(N, S1):
    b = N // 2
    return b * (N - b) / (N - 1) * S1


def fit_sat(Ns, S1s):
    """S1(N) = S_inf - (S_inf - S1(N0)) exp(-(N - N0)/tau) through the first point."""
    N0, S0 = Ns[0], S1s[0]

    def res(p):
        Sinf, tau = p
        return [Sinf - (Sinf - S0) * math.exp(-(n - N0) / tau) - s for n, s in zip(Ns[1:], S1s[1:])]
    sol = least_squares(res, x0=[S1s[-1] + 0.3, 1.5], bounds=([S1s[-1], 0.2], [20, 50]))
    return float(sol.x[0]), float(sol.x[1])


def scenarios(cfg, Ns, S1v, Scen, chi_last=None):
    """Walls and curves under the three scenarios.

    Ns, S1v, Scen   particle numbers, mean one-particle entropies and central-bond entropies measured
    chi_last        (chi(1e-3), S_cen) at the largest N: if given, the numerical-rank ratio at that N is
                    carried as a constant shift of S (a stated assumption, not a fit)."""
    B, tp, ed, ex = cfg['budgets'], cfg['tn_wall'], cfg['ed_wall'], cfg['extrapolation']
    Bm, Bf, sweeps = B['classical_mem'], B['classical_flop'], tp['sweeps']
    d_grid, d_mode = tp['d_grid'], tp['d_mode']
    M = cfg['grid']['nx'] * cfg['grid']['ny']
    Sinf, tau = fit_sat(Ns, S1v)
    inc = (S1v[-1] - S1v[-3]) / 2.0 if len(Ns) >= 3 else (S1v[-1] - S1v[-2])
    N0, S0 = Ns[0], S1v[0]

    def s1_sat(N, Si=Sinf, ta=tau):
        return Si - (Si - S0) * math.exp(-(N - N0) / ta)

    scen = {'sat': lambda N: sharing(N, s1_sat(N)),
            'opt': lambda N: sharing(N, S1v[-1] if N > Ns[-1] else s1_sat(N)),
            'pess': lambda N: sharing(N, S1v[-1] + inc * (N - Ns[-1]) if N > Ns[-1] else s1_sat(N))}
    NN = list(range(2, ex['N_scan_max'] + 1))
    curves = {k_: [f(N) for N in NN] for k_, f in scen.items()}
    walls = {f"{name}_d{d}_{which}": th.wall(f, d, Bm, Bf, sweeps, N_from=Ns[-1] + 1, nmax=tp['nmax'], which=which)
             for name, f in scen.items() for d in (d_grid, d_mode) for which in ('flop', 'mem', 'both')}
    chi_shift, walls_chi = 0.0, {}
    if chi_last and chi_last[0]:
        chi_shift = math.log2(chi_last[0] / 2.0 ** chi_last[1])
        walls_chi = {f"{name}_d{d}_both": th.wall(lambda N, f=f: f(N) + chi_shift, d, Bm, Bf, sweeps,
                                                  N_from=Ns[-1] + 1, nmax=tp['nmax'], which='both')
                     for name, f in scen.items() for d in (d_grid, d_mode)}
    bud = {f"d{d}": [th.S_budget(N, d, Bm, Bf, sweeps) for N in NN] for d in (d_grid, d_mode)}
    Nref, Sref = Ns[-1], Scen[-1]
    scrit = {f"d{d}_N{Nhi}": th.s_crit(d, Sref, Nref, Nhi, Bf, sweeps)
             for d in (d_grid, d_mode) for Nhi in ex['N_hi_list']}
    slope = {k_: (curves[k_][60] - curves[k_][40]) / 20.0 for k_ in curves}
    nfit = ex['pair_fit_max']
    pf_all = th.fit_pairs(dict(zip(Ns, Scen)))
    sub = {n: s for n, s in zip(Ns, Scen) if n <= nfit}
    pf4, lf4 = th.fit_pairs(sub), th.fit_linear(sub)
    probe = (5, 6, 8, 12)
    return dict(
        Ns_measured=list(Ns), S_cen_measured=list(Scen), S1_measured=list(S1v),
        fit=dict(S_inf=Sinf, tau=tau, S1_last=S1v[-1], inc=inc), NN=NN, curves=curves,
        model_at_measured={n: sharing(n, s) for n, s in zip(Ns, S1v)},
        walls=walls, walls_chi=walls_chi, chi_shift=chi_shift, S_budget=bud, s_crit=scrit, slope=slope,
        pair_fit=dict(S0=pf_all[0], beta=pf_all[1]), pair_fit4=dict(S0=pf4[0], beta=pf4[1]),
        pair_pred={n: pf4[0] + pf4[1] * th.pairs(n) for n in probe},
        lin_fit4=dict(S0=lf4[0], s=lf4[1]), lin_pred={n: lf4[0] + lf4[1] * n for n in probe},
        share_pred={k_: {n: curves[k_][n - 2] for n in (6, 8, 12, 16)} for k_ in curves},
        pair_curve=[pf_all[0] + pf_all[1] * th.pairs(N) for N in NN],
        pair_curve4=[pf4[0] + pf4[1] * th.pairs(N) for N in NN],
        lin_curve4=[lf4[0] + lf4[1] * N for N in NN],
        budgets=dict(Bmem=Bm, Bflop=Bf, sweeps=sweeps, d_grid=d_grid, d_mode=d_mode, v=ed['vectors']))
