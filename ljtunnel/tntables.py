"""ljtunnel.tntables -- the numbers behind the tensor-network section, as flat rows (csv + npz) that a figure script can read.

    mps_validation      every matrix-product-state run next to the exact run of the same (N, k, tilt, start), same window
    mps_series          central-bond entropy, <O>, bond dimension against time for those pairs (long format)
    window_convergence  S_cen, S1, ranks, time of the maximum against the window T (k = 12, N = 2..5)
    rank_vs_entropy     chi(1e-2, 1e-3, 1e-6) against 2^S for every stored exact run
    wall_sensitivity    the tensor-network walls when S1(5) is moved, and with the uniform N = 5 run
    channel_ranks       interaction channels R of the MPO and the pair-matrix error they cause, against the tolerance
    step_cost           measured seconds of a TDVP step against chi, next to the counting model (needs TeNPy, ~5 min)

Only `channel_ranks` and `step_cost` need TeNPy (through `ljtunnel.mps`); the others read stored runs.  Every function returns a list
of flat dicts (`step_cost`: also the machine record); `write_csv` writes them.  Nothing here is a simulation except `step_cost`.
"""
import csv
import itertools
import os
import platform
import time

import numpy as np

from . import analysis, store
from . import extrapolation as ex
from . import theory as th
from .model import modes, project_pair


# ------------------------------------------------------------------ helpers
def write_csv(path, rows, columns=None):
    """Rows (flat dicts) to a csv with a header; missing values are empty."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    columns = columns or list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in rows:
            w.writerow(['' if r.get(c) is None else (f"{r[c]:.10g}" if isinstance(r[c], float) else r[c]) for c in columns])
    return path


def _runs(directory, pattern='*.npz'):
    return [store.load_run(p) for p in store.list_runs(str(directory), pattern)]


def _first_minimum(run):
    """Time of the first local minimum of <O>(t) after the start (the paper's fallback for the observation time)."""
    O, t = run['O'], run['t']
    i = next((i for i in range(1, len(O) - 1) if O[i] < O[i - 1] and O[i] <= O[i + 1]), None)
    return None if i is None else float(t[i])


def _central_series(run, bc):
    return analysis.S_of(run, tuple(range(bc)))


# ------------------------------------------------------------------ MPS against the exact runs
def mps_validation(mps_dir, ref_dirs):
    """(rows, series): one row per MPS run that has an exact run of the same (N, k, tilt), start and a window at least as long;
    both are summarised over the MPS run's window.  `max_dS` = largest difference over every cut measured in both and every
    snapshot, `max_dO` the same for <O>(t).  series: long-format time series of the central bond."""
    refs = [e for d in ref_dirs for e in _runs(d) if e['meta']['start'] == 'interacting']
    rows, series = [], []
    for r in _runs(mps_dir):
        m, mm = r['meta'], r['meta']['mps']
        match = [e for e in refs if (e['meta']['N'], e['meta']['k']) == (m['N'], m['k'])
                 and abs(e['meta']['c'] - m['c']) < 1e-6 and e['t'].max() >= r['t'].max() - 1e-9]
        if not match:
            continue
        e = match[0]
        T = float(r['t'].max())
        sm, se = analysis.bond_summary(r, T), analysis.bond_summary(e, T)
        nt = len(r['t'])
        ecol = {int(x): j for j, x in enumerate(e['masks'])}
        dS = [np.nanmax(np.abs(r['S'][:, j] - e['S'][:nt, ecol[int(x)]])) for j, x in enumerate(r['masks']) if int(x) in ecol]
        ok = analysis.window(r, T)
        case = f"N{m['N']}_k{m['k']}"
        rows.append(dict(
            case=case, N=m['N'], k=m['k'], c=m['c'], T=T, chi_max=mm['chi_max'], eps=mm['eps'], dt_tdvp=mm['dt'],
            channel_R=mm['channel_R'], channel_err=mm['channel_err'], start_variance=mm.get('start_variance'),
            S_cen_mps=sm['S_cen'], S_cen_exact=se['S_cen'], S_nat_mps=sm['S_nat'], S_nat_exact=se['S_nat'],
            tS_cen_mps=sm['tS_cen'], tS_cen_exact=se['tS_cen'],
            chi2_mps=sm['chi2_cen'], chi2_exact=se['chi2_cen'], chi3_mps=sm['chi3_cen'], chi3_exact=se['chi3_cen'],
            chi6_mps=sm['chi6_cen'], chi6_exact=se['chi6_cen'],
            chi_kept=int(r['chi'][ok].max()), trunc_err=float(r['trunc_err'][ok].max()),
            energy_drift=mm['drift_E'], max_dS=float(max(dS)), max_dO=float(np.abs(r['O'] - e['O'][:nt]).max()),
            n_cuts_compared=len(dS), wall_mps_s=m.get('wall_s'), wall_exact_s=e['meta'].get('wall_s'),
            file_mps=os.path.basename(r['path']), file_exact=os.path.basename(e['path'])))
        for i, t in enumerate(r['t']):
            series.append(dict(case=case, t=float(t), S_cen_mps=float(_central_series(r, sm['bond_central'])[i]),
                               S_cen_exact=float(_central_series(e, se['bond_central'])[i]), O_mps=float(r['O'][i]),
                               O_exact=float(e['O'][i]), chi_cen_mps=int(r['chi'][i, sm['jc']]),
                               trunc_err_mps=float(r['trunc_err'][i])))
    return rows, series


# ------------------------------------------------------------------ the window and the ranks
def window_convergence(sources, windows=range(10, 201, 10)):
    """sources = [(label, path), ...]; one row per (source, window T <= the run's length)."""
    rows = []
    for label, path in sources:
        r = store.load_run(path)
        t_min = _first_minimum(r)
        for T in windows:
            if T > r['t'].max() + 1e-9:
                continue
            s, g = analysis.bond_summary(r, T), analysis.graph_record(r, T, exhaustive_max_N=0)
            rows.append(dict(source=label, N=s['N'], k=s['k'], c=s['c'], protocol=r['meta']['protocol']['kind'],
                             dt_snap=r['meta']['protocol']['dt_snap'], T=float(T), S_cen=s['S_cen'], tS_cen=s['tS_cen'],
                             S_nat=s['S_nat'], S1mean=g['S1mean'], chi2=s['chi2_cen'], chi3=s['chi3_cen'], chi6=s['chi6_cen'],
                             n_edge=g['n_edge'], t_first_O_min=t_min, run_length=float(r['t'].max()),
                             file=os.path.basename(path)))
    return rows


def rank_vs_entropy(paths, windows=(60, 120, 200)):
    """chi(eps) of the central bond against 2^S_cen, one row per (run, window); `cap` = k^min(b, N-b) is the largest rank."""
    rows = []
    for path in paths:
        r = store.load_run(path)
        for T in windows:
            if T > r['t'].max() + 1e-9:
                continue
            s = analysis.bond_summary(r, T)
            if s['chi3_cen'] is None:
                continue
            b, N, k = s['bond_central'], s['N'], s['k']
            cap = k ** min(b, N - b)
            rows.append(dict(N=N, k=k, start=r['meta']['start'], c=s['c'], protocol=r['meta']['protocol']['kind'], T=float(T),
                             S_cen=s['S_cen'], two_pow_S=2 ** s['S_cen'], chi2=s['chi2_cen'], chi3=s['chi3_cen'],
                             chi6=s['chi6_cen'], ratio3=s['chi3_cen'] / 2 ** s['S_cen'],
                             ratio6=(s['chi6_cen'] / 2 ** s['S_cen']) if s['chi6_cen'] else None, cap=cap,
                             chi6_over_cap=(s['chi6_cen'] / cap) if s['chi6_cen'] else None, file=os.path.basename(path)))
    return rows


# ------------------------------------------------------------------ the walls
def wall_sensitivity(cfg, larger_N, probe_run=None, shifts=None):
    """The three-scenario walls (first failing N, grid d_grid / modes d_mode, both budgets) when the last measured point is moved
    by dS1 in S1 and in S_cen, and, if `probe_run` is given (the uniform N=5 run), with its N=5 values at windows 120 and 200."""
    Ns, S1, Sc = list(larger_N['Ns_measured']), list(larger_N['S1_measured']), list(larger_N['S_cen_measured'])
    d_grid, d_mode = cfg['tn_wall']['d_grid'], cfg['tn_wall']['d_mode']
    shifts = np.round(np.arange(-0.30, 0.3001, 0.05), 3) + 0.0 if shifts is None else shifts

    def row(label, S1b, Scb, chi_last=None):
        return row_from(label, ex.scenarios(cfg, Ns, S1b, Scb, chi_last=chi_last), S1b, Scb)

    def row_from(label, o, S1b, Scb):
        w, wc = o['walls'], o['walls_chi']
        out = dict(label=label, S1_last=S1b[-1], S_cen_last=Scb[-1], S_inf=o['fit']['S_inf'], tau=o['fit']['tau'],
                   slope_sat=o['slope']['sat'], slope_opt=o['slope']['opt'], slope_pess=o['slope']['pess'])
        for name in ('sat', 'opt', 'pess'):
            out[f'wall_{name}_grid'], out[f'wall_{name}_mode'] = w[f'{name}_d{d_grid}_both'], w[f'{name}_d{d_mode}_both']
            out[f'wallchi_{name}_grid'] = wc.get(f'{name}_d{d_grid}_both')
            out[f'wallchi_{name}_mode'] = wc.get(f'{name}_d{d_mode}_both')
        return out

    rows = []
    for dS in shifts:
        S1b, Scb = list(S1), list(Sc)
        S1b[-1] += dS
        Scb[-1] += dS
        r = row(f'shift {dS:+.2f}', S1b, Scb)
        r['dS1'] = float(dS)
        rows.append(r)
    if probe_run is not None:
        Np = int(probe_run['meta']['N'])
        i = Ns.index(Np)                              # the probe replaces its own particle number, and the data stop there
        for T in (120, 200):
            s, g = analysis.bond_summary(probe_run, T), analysis.graph_record(probe_run, T, exhaustive_max_N=0)
            o = ex.scenarios(cfg, Ns[:i + 1], S1[:i] + [g['S1mean']], Sc[:i] + [s['S_cen']], chi_last=(s['chi3_cen'], s['S_cen']))
            r = row_from(f'uniform N={Np}, T<={T}', o, S1[:i] + [g['S1mean']], Sc[:i] + [s['S_cen']])
            r['dS1'] = g['S1mean'] - S1[i]
            rows.append(r)
    return rows


# ------------------------------------------------------------------ the interaction MPO (needs TeNPy only for the import)
def channel_ranks(model, cases=((4, 6, 2.455), (4, 12, 2.26), (6, 6, 3.8), (6, 12, 3.6), (4, 8, 2.35)),
                  tols=(1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)):
    """R = number of interaction channels of the MPO (bond dimension R + 2) and the worst operator-norm error of any projected
    pair matrix, against the tolerance `channel_tol`; for (N, k, c).  The tilts are only representative."""
    from . import mps
    g, mp = model.g, model.mp
    rows = []
    for N, k, c in cases:
        Ps = [modes(g, mp, c, i * mp['C0'], k)[1] for i in range(N)]
        exact = {(i, j): project_pair(g['VP'], Ps[i], Ps[j]) for i, j in itertools.combinations(range(N), 2)}
        for tol in tols:
            e, phi = mps.channels(g['VP'], Ps, tol)
            F = [np.einsum('ap,as,aq->spq', P, phi, P, optimize=True) for P in Ps]
            err = max(float(np.linalg.norm(V - np.einsum('s,spP,sqQ->pqPQ', e, F[i], F[j], optimize=True).reshape(k * k, k * k), 2))
                      for (i, j), V in exact.items())
            rows.append(dict(N=N, k=k, c=c, tol=tol, R=len(e), D_mpo=len(e) + 2, single_site_span=k * (k + 1) // 2, max_pair_error=err,
                             max_pair_norm=max(float(np.linalg.norm(V, 2)) for V in exact.values())))
    return rows


# ------------------------------------------------------------------ the measured cost of a TDVP step
def _zgemm_rate(n=1536, reps=3):
    a = np.random.default_rng(0).standard_normal((n, n)) + 1j * np.random.default_rng(1).standard_normal((n, n))
    a @ a
    t = time.perf_counter()
    for _ in range(reps):
        a @ a
    return 8.0 * n ** 3 * reps / (time.perf_counter() - t)


def step_cost(model, cases=((4, 12, 2.26, (16, 32, 48, 64)), (4, 6, 2.455, (16, 32))), dt=0.5, warm=8, reps=2):
    """Seconds per two-site TDVP step (TeNPy, dense MPO) against the bond-dimension cap, after `warm` steps from the product start
    have grown the entanglement.  Per case also the number and time of the two-site and one-site H_eff matvecs, and the
    counting model's operation count 2 N (d chi)^3 (`theory.sweep_flops`, one sweep) converted to seconds at this machine's
    complex matrix-multiplication rate.  Returns (rows, machine)."""
    import tenpy
    from tenpy.algorithms import tdvp
    from tenpy.algorithms.mps_common import OneSiteH, TwoSiteH
    from . import mps
    rate = _zgemm_rate()
    machine = dict(cpu_count=os.cpu_count(), platform=platform.platform(), python=platform.python_version(), numpy=np.__version__,
                   tenpy=tenpy.__version__, zgemm_flops_per_s=rate, date=time.strftime('%Y-%m-%d'))
    acc = {'two': [0, 0.0], 'one': [0, 0.0]}

    def timed(cls, key):
        orig = cls.matvec

        def wrapped(self, theta):
            t = time.perf_counter()
            out = orig(self, theta)
            acc[key][0] += 1
            acc[key][1] += time.perf_counter() - t
            return out
        cls.matvec = wrapped
        return orig

    rows = []
    for N, k, c, chis in cases:
        prob = mps.MPSProblem(model, c, k, N, check=False)
        H = prob.hamiltonian()
        for chi in chis:
            psi = prob.product_mps(mps.start_sides(N))
            opts = dict(preserve_norm=True, trunc_params=dict(chi_max=chi, trunc_cut=1e-4, svd_min=1e-12))
            eng = tdvp.TwoSiteTDVPEngine(psi, H, opts)
            eng.run_evolution(warm, dt)
            for key in acc:
                acc[key] = [0, 0.0]
            o2, o1 = timed(TwoSiteH, 'two'), timed(OneSiteH, 'one')
            try:
                t = time.perf_counter()
                for _ in range(reps):
                    eng.run_evolution(1, dt)
                sec = (time.perf_counter() - t) / reps
            finally:
                TwoSiteH.matvec, OneSiteH.matvec = o2, o1
            chi_r = int(max(psi.chi))
            flops = th.sweep_flops(N, k, chi_r, 1)
            rows.append(dict(N=N, k=k, c=c, R=prob.R, D_mpo=prob.R + 2, chi_cap=chi, chi_reached=chi_r, dt=dt, sec_per_step=sec,
                             matvec2_per_step=acc['two'][0] / reps, sec_matvec2_each=acc['two'][1] / max(acc['two'][0], 1),
                             matvec1_per_step=acc['one'][0] / reps, sec_matvec1_each=acc['one'][1] / max(acc['one'][0], 1),
                             frac_in_matvec=(acc['two'][1] + acc['one'][1]) / reps / sec, model_flops=flops,
                             model_sec_at_zgemm_rate=flops / rate, ratio_measured_to_model=sec / (flops / rate)))
    return rows, machine
