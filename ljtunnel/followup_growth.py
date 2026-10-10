"""ljtunnel.followup_growth -- Stage 5a of followup/TASK_followup_rival_and_mps.md: the shape of the entropy growth S_cen(N), N = 2..8, at k = 6.

N = 2..5 come from exact bond-only runs at k = 6 (T = 200; the window T <= 60 is cut afterwards, as everywhere in the repository), N = 6, 7, 8 from
matrix-product-state runs (T = 60).  The three growth laws of the paper (pair, linear, sharing) are fitted on N <= 5 and compared with N = 6, 7, 8.
Definitions and the consistency rule are in data/followup/PREREG.md.  Everything below the entry points reads saved files only.
"""
import glob
import os

import numpy as np

from . import analysis, extrapolation as ex, theory as th
from .config import ROOT
from .store import load_run

OUTDIR = os.path.join(ROOT, 'data', 'followup', 'stage5a')
T_MPS = 60.0


def runs_in(directory, pattern):
    return [load_run(p) for p in sorted(glob.glob(os.path.join(directory, pattern)))]


def window_row(run, T_win):
    """S_cen, S1mean, ranks of one run (exact or MPS) for the window [0, T_win]."""
    s = analysis.bond_summary(run, T_win)
    m = run['meta']
    row = dict(N=s['N'], k=s['k'], c=s['c'], T=s['T'], S_cen=s['S_cen'], S1mean=s['S1mean'], S_nat=s['S_nat'], tS_cen=s['tS_cen'],
               chi3_cen=s['chi3_cen'], chi6_cen=s['chi6_cen'], bonds=s['bonds'], wall_s=float(m.get('wall_s', np.nan)))
    if 'mps' in m:
        row.update(kind='mps', chi_max=m['mps']['chi_max'], chi_reached=m['mps']['chi_reached'], trunc_err=m['mps']['trunc_err'],
                   drift_E=m['mps']['drift_E'])
        ok = analysis.window(run, T_win)
        row['chi_kept_cen'] = int(run['chi'][ok][:, s['jc']].max())
    else:
        row['kind'] = 'exact'
    return row


def growth_rows(exact_dir=os.path.join(OUTDIR, 'exact_k6'), mps_dir=os.path.join(OUTDIR, 'mps'), T_win=T_MPS):
    """Rows N = 2..8: exact runs (N <= 5) and MPS runs (N >= 6), all at the window T_win, plus the exact rows at the full window."""
    rows = {}
    for r in runs_in(exact_dir, 'N*_k6_*.npz'):
        rows[r['meta']['N']] = dict(window_row(r, T_win), full=window_row(r, None))
    for r in runs_in(mps_dir, 'N*_k6_mps_*.npz'):
        N = r['meta']['N']
        if N not in rows:
            rows[N] = window_row(r, T_win)
    return [rows[N] for N in sorted(rows)]


def laws(rows, fit_max=5):
    """Fit pair, linear and sharing(S1 scenarios) on N <= fit_max; predictions for every N; the sharing law is also evaluated at the
    MEASURED S1mean of each N (tests the sharing structure itself, not the S1 extrapolation)."""
    fit = [r for r in rows if r['N'] <= fit_max]
    Ns = [r['N'] for r in fit]
    Sc = [r['S_cen'] for r in fit]
    S1 = [r['S1mean'] for r in fit]
    S0p, beta = th.fit_pairs(dict(zip(Ns, Sc)))
    S0l, s = th.fit_linear(dict(zip(Ns, Sc)))
    Sinf, tau = ex.fit_sat(Ns, S1)
    inc = (S1[-1] - S1[-3]) / 2.0 if len(Ns) >= 3 else S1[-1] - S1[-2]
    N0, S10 = Ns[0], S1[0]

    def s1_sat(N):
        return Sinf - (Sinf - S10) * np.exp(-(N - N0) / tau)
    out = dict(fit_N=Ns, pair=dict(S0=S0p, beta=beta), linear=dict(S0=S0l, s=s), sat=dict(S_inf=Sinf, tau=tau), pess_inc=inc, rows=[])
    for r in rows:
        N = r['N']
        pred = {'pair': S0p + beta * th.pairs(N), 'linear': S0l + s * N, 'sharing_measuredS1': ex.sharing(N, r['S1mean']),
                'sharing_sat': ex.sharing(N, s1_sat(N)),
                'sharing_opt': ex.sharing(N, S1[-1] if N > Ns[-1] else s1_sat(N)),
                'sharing_pess': ex.sharing(N, S1[-1] + inc * (N - Ns[-1]) if N > Ns[-1] else s1_sat(N))}
        out['rows'].append(dict(N=N, S_cen=r['S_cen'], S1mean=r['S1mean'], pred=pred,
                                rel_err={k_: v / r['S_cen'] - 1 for k_, v in pred.items()},
                                consistent={k_: bool(abs(v / r['S_cen'] - 1) <= 0.15) for k_, v in pred.items()}, tested=bool(N > fit_max)))
    return out
