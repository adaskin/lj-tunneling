"""run.py -- the run file.  Every simulation or analysis is one function below; `main()` at the bottom
calls the ones you want: comment lines in or out, then run the file.

Works three ways, the same code:
  * Spyder / VS Code / PyCharm: the `# %%` markers are cells; run a cell, or the whole file.
  * IPython / Jupyter:   %run run.py     or     from run import *   then call a function
                         (notebooks/run.ipynb is this file split at the `# %%` markers)
  * terminal:            python run.py

Every function returns what it computed and writes .npz files under data/ (arrays + a JSON record
of the parameters); logs go to data/logs/.  All constants are in config.toml.  The measurement
protocol (window T, snapshot spacing, start state, which cuts) is [protocol] in config.toml and is
the same for every N; a run with another T or dt is marked 'custom' and not mixed with the others.
"""
# %% setup ------------------------------------------------------------------------------------
import os
import sys
import time
from pathlib import Path

import numpy as np

if '__file__' in globals():
    ROOT = Path(__file__).resolve().parent
else:                                   # notebook / interactive window: the repo root is the cwd or its parent
    ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT))

import ljtunnel as lj
from ljtunnel import (analysis, cost, experiments, extrapolation, fullgrid, graph, montecarlo, onebody,
                      quantum, records, store, tntables, walls)

CFG = lj.load_config(ROOT / 'config.toml')
DATA = ROOT / 'data'
RUNS, DERIVED, STUDIES = DATA / 'runs', DATA / 'derived', DATA / 'studies'
CONTROLS = DATA / 'controls'          # the off-resonance control (never in data/runs: records.pick has no tilt filter)
PREVIOUS = ROOT / 'previous-data'     # everything made before the Theta(x = 0) rerun (theta_barrier = 0)
PARAMETER_MAP = STUDIES / 'parameter_map' / 'tier1_parameter_map.txt'     # the Appendix-A ledger (new convention)
_MODEL = {}


def model(sigma_smear=None):
    """The grid and the config (built once per session)."""
    key = sigma_smear
    if key not in _MODEL:
        _MODEL[key] = lj.Model(CFG, sigma_smear=sigma_smear)
    return _MODEL[key]


# %% 1. simulations: one function each; results go to data/runs/ ---------------------------------
def run_dynamics(N=4, k=None, starts=('interacting',), tilt='auto', c_guess=None, cuts=None,
                 big_cut_stride=None, force=False, checkpoint=False):
    """Resonance -> start state(s) -> real-time series of <O>(t) and of the entropy of every cut.

    Writes data/runs/N{N}_k{k}_{int|prod}_c{c}_T{T}_dt{dt}.npz per start.
      starts  ('interacting',) is the correlated start; add 'product' for the control start
      tilt    'auto' (measured value in [tilts], else scan) | 'scan' | a float
      c_guess refine around this tilt instead of scanning (N >= 5: much cheaper)
      cuts    None = config ('all' bipartitions) | 'bonds+pairs' | 'bonds'
      big_cut_stride  evaluate the cuts with >= 3 particles on a side every n-th snapshot (N >= 6)
      checkpoint  (float tilt only) write the start states and the series so far to data/ckpt/ every few snapshots; a rerun of the
                  same call continues from there (hours-long runs survive a restart or a power cut)
    The window T and spacing dt are the uniform protocol of config.toml; do not pass them here."""
    return experiments.dynamics_run(model(), N, k, starts=list(starts), tilt=tilt, c_guess=c_guess,
                                    cuts=cuts, big_cut_stride=big_cut_stride, force=force, checkpoint=checkpoint)


def run_dynamics_batch(Ns=(2, 3, 4), ks=(12,), starts=('interacting',), **kw):
    """run_dynamics over a list of particle numbers and mode counts."""
    out = []
    for k in ks:
        for N in Ns:
            out += run_dynamics(N=N, k=k, starts=starts, **kw)
    return out


def run_off_resonance(N=5, k=12, c=None, starts=('interacting',), force=False):
    """The off-resonance control: the same run at a tilt where the ground state is localized (c from [control]).
    Compares the entanglement that is present without tunneling (paper, Sec. on the correlated start).
    Writes data/controls/ (not data/runs: a run at another tilt there would displace the resonant point of the same (N, k) in
    records.pick); run_records adds it to the graph table, where """
    c = CFG['control'][f'N{N}_off_c'] if c is None else c
    return experiments.dynamics_run(model(), N, k, starts=list(starts), tilt=float(c), outdir=str(CONTROLS), force=force)


def run_mps(N=4, k=None, tilt='auto', c_guess=None, T=None, dt_snap=None, chi_max=None, eps=None, pairs=False,
            start_key=None, save=True, dt=None):
    """Real-time matrix-product-state series (TeNPy: DMRG start, TDVP) of the same model and the same correlated start
    as run_dynamics, without storing the k^N amplitudes: the way to N = 6, 7, 8 ... (needs  pip install physics-tenpy).

    Writes data/mps/N{N}_k{k}_mps_c{c}_T{T}_dt{dt}_chi{chi_max}_eps{eps}.npz; protocol kind 'mps' (never mixed with the exact runs).
      tilt       'auto' (measured value in [tilts], else DMRG search around c_guess) | a float
      T, dt_snap default to the protocol of config.toml; a shorter T costs proportionally less
      chi_max    bond-dimension cap and eps = discarded weight per truncation (config [mps]); chi(1e-6) can only be read below eps
      pairs      also measure every pair (mutual information -> edge weights), ~ N^2 more measurements
      start_key  valley assignment of the start (default: particle 0 on the right, the rest on the left)
      dt         TDVP step (default [mps] dt; the runs of data/mps were made with 0.5)
    Cost per bond update ~ R chi^3 k^2 (R ~ 60-170 interaction channels at k = 12); run_mps_summary() compares with the exact runs."""
    from ljtunnel import mps
    from ljtunnel.log import tee
    k = k or CFG['basis']['k']
    with tee(str(DATA / 'logs' / f"N{N}_k{k}_mps_{time.strftime('%Y%m%d_%H%M%S')}.log")):
        c = float(tilt) if isinstance(tilt, (int, float)) else model().tilt(N, k)
        if c is None:
            known = {n_: model().tilt(n_, k) for n_ in range(2, N) if model().tilt(n_, k) is not None}
            if c_guess is None and not known:
                raise ValueError(f"no tilt known for k={k}: pass tilt=<float> or c_guess=<float>")
            c_guess = c_guess if c_guess is not None else known[max(known)] + 0.75 * (N - max(known))
            c = mps.locate_tilt(model(), N, k, c_guess - 0.4, c_guess + 0.4)
            print(f"N={N} k={k}: resonance tilt c* = {c:.4f} (DMRG, <O>_gs = 1/2); put it into [tilts] to reuse")
        prob = mps.MPSProblem(model(), c, k, N, channel_tol=CFG['mps']['channel_tol'])
        run = mps.run_series(prob, sides=mps.start_sides(N, start_key), T=T, dt_snap=dt_snap, dt=dt, chi_max=chi_max, eps=eps,
                             pairs=pairs, label=f"N{N}k{k}")
        if save:
            print('wrote', os.path.relpath(mps.save(run), ROOT))
    return run


def run_mps_check(N=4, k=6, tilt=None, T=40.0, chi_max=100, eps=1e-8, pairs=True, dt=None):
    """The validation behind data/mps/: the exact run (bonds and pairs, protocol custom, written to data/mps/ed_ref/) and the MPS run
    of the same (N, k, tilt, start, window T), then the largest difference over every cut measured by both and over <O>(t).
      tilt   a float | None = [tilts] if there is one, else the exact-code resonance scan (`run_mps_case` bisects <O>_gs = 1/2)
      pairs  also measure the pair cuts in the MPS run (the exact reference always has them)
      dt     TDVP step (default [mps] dt; the stored validation runs use 0.5)
    The exact run is the expensive part from N = 6 (k = 6) or N = 5 (k = 12)."""
    c = float(tilt) if tilt is not None else model().tilt(N, k)
    if c is None:
        from ljtunnel.resonance import locate
        c = float(locate(model(), N, k, tilt='scan')['c'])
    paths = experiments.dynamics_run(model(), N, k, starts=['interacting'], tilt=c, T=T, dt_snap=1.0,
                                     cuts='bonds+pairs', outdir=str(DATA / 'mps' / 'ed_ref'), force=True)
    ed = store.load_run(paths[0])
    run = run_mps(N=N, k=k, tilt=c, T=T, dt_snap=1.0, chi_max=chi_max, eps=eps, pairs=pairs, dt=dt,
                  start_key=ed['meta']['diag']['start_key'])
    col = {int(m): j for j, m in enumerate(ed['masks'])}
    dS = max(np.nanmax(np.abs(run['S'][:, j] - ed['S'][:, col[int(m)]])) for j, m in enumerate(run['masks']))
    dO = float(np.abs(run['O'] - ed['O']).max())
    print(f"N={N} k={k} c={c:.4f} T={T:g}: max |dS| over every cut {dS:.2e} bit, max |d<O>| {dO:.2e}")
    return dict(dS=float(dS), dO=dO, mps=run, exact=ed)


def run_sampling_study(Ns=(3, 4), k=12, dts=None, force=False):
    """The snapshot spacing at the production window: same state, same T, different dt_snap.
    Writes data/studies/dt_scan/*.npz (protocol kind 'custom'); the maxima over snapshots depend on dt.
    force=True overwrites an existing file (otherwise a second file with a time stamp is written next to it)."""
    dts = CFG['studies']['dt_scan'] if dts is None else dts
    out = []
    for N in Ns:
        for dt in dts:
            out += experiments.dynamics_run(model(), N, k, starts=['interacting'], dt_snap=float(dt),
                                            outdir=str(STUDIES / 'dt_scan'), force=force)
    return out


def run_fullgrid_N2(c=None, save=True):
    """N = 2 on the full 40x30 grid per particle (no mode truncation): dE_2 and t_0 = pi/dE_2 (~1 min)."""
    fg = CFG['fullgrid']
    r = fullgrid.reference(model(fg['sigma_smear']), fg['c_ref'] if c is None else c, fg['n_levels'])
    if save:
        store.save_result(DERIVED / 'fullgrid_N2.npz', meta=dict(c=r['c'], dE=r['dE'], t0=r['t0'], mO=r['mO'],
                                                                   fullgrid=fg), ev=r['ev'], On=r['On'])
    return r


def run_fullgrid_resonance():
    """Search the tilt where <O>_gs = 1/2 on the full grid (several eigensolves; ~10 min)."""
    fg = CFG['fullgrid']
    return fullgrid.resonance(model(fg['sigma_smear']), fg['n_levels'], fg['c_lo'], fg['c_hi'], fg['c_tol'])


# %% 2. analysis of saved runs (no simulation) ---------------------------------------------------
def load_runs(pattern='*.npz', directory=RUNS, kind='uniform'):
    """All saved runs of one protocol kind ('uniform' | 'custom' | None for every kind)."""
    runs = [store.load_run(p) for p in store.list_runs(str(directory), pattern)]
    return [r for r in runs if kind is None or r['meta']['protocol']['kind'] == kind]


def run_summary(T_win=None, directory=RUNS):
    """One line per saved run: central bond, worst/best balanced cut, doublet data, for ONE window."""
    rows = []
    print(f"{'N':>2} {'k':>3} {'start':>11} {'c':>6} {'T':>5} {'S_cen':>7} {'S_nat':>7} {'S_perp':>7} {'S_max':>7} "
          f"{'chi3':>5} {'S1mean':>7} {'<O> range':>12}")
    for r in sorted(load_runs(directory=directory), key=lambda r: (r['meta']['k'], r['meta']['N'], r['meta']['start'])):
        s = analysis.summarize(r, T_win)
        g = analysis.graph_record(r, T_win)
        rows.append(s)
        print(f"{s['N']:2d} {s['k']:3d} {r['meta']['start']:>11} {s['c']:6.3f} {s['T']:5.0f} {s['S_cen']:7.4f} "
              f"{g['S_nat']:7.4f} {g['S_perp']:7.4f} {g['S_max']:7.4f} {s['chi3']:5d} {s['S1mean']:7.4f} "
              f"[{s['Omin']:.2f},{s['Omax']:.2f}]")
    return rows


def run_mps_summary(T_win=None, directory=DATA / 'mps'):
    """One line per saved MPS run: the cost of the matrix-product state (central bond, worst bond, Schmidt ranks that the
    truncated run needed, bond dimension reached, truncation error) next to the exact run of the same (N, k, tilt) where one
    exists (data/runs or data/mps/ed_ref), over the same window.  A rank is only a measurement where kept < chi_max."""
    from ljtunnel import mps
    exact = [r for d in (RUNS, directory / 'ed_ref') for r in load_runs(directory=d, kind=None)
             if r['meta']['start'] == 'interacting']
    print(f"{'N':>2} {'k':>3} {'c':>6} {'T':>5} {'chi_max':>7} {'eps':>7} | {'S_cen':>7} {'S_nat':>7} {'chi3':>5} {'chi6':>5} "
          f"{'kept':>5} {'trunc':>8} | {'exact S_cen':>11} {'chi3':>5} {'chi6':>5}")
    rows = []
    for p in store.list_runs(str(directory)):
        r = store.load_run(p)
        s = mps.summarize(r, T_win)
        m = r['meta']['mps']
        e = [x for x in exact if (x['meta']['N'], x['meta']['k']) == (s['N'], s['k']) and abs(x['meta']['c'] - s['c']) < 1e-6
             and x['t'].max() >= s['T'] - 1e-9]
        es = analysis.summarize(e[0], s['T']) if e else None
        rows.append(dict(s, exact=es))
        print(f"{s['N']:2d} {s['k']:3d} {s['c']:6.3f} {s['T']:5.0f} {m['chi_max']:7d} {m['eps']:7.0e} | {s['S_cen']:7.4f} "
              f"{s['S_nat']:7.4f} {s['chi3_cen']:5d} {s['chi6_cen']:5d} {s['chi_kept']:5d} {s['trunc_err']:8.1e} | "
              + (f"{es['S_cen']:11.4f} {es.get('chi3', 0):5d} {es.get('chi6', 0):5d}" if es else f"{'-':>11} {'-':>5} {'-':>5}"))
    return rows


def run_mps_tables(step_cost=False):
    """The tables behind the tensor-network section, from the stored runs -> data/derived/tn_tables.npz and one csv per table in
    data/derived/tn_tables/ (see the README there): mps_validation, mps_series, window_convergence, rank_vs_entropy,
    wall_sensitivity, channel_ranks and, with step_cost=True (needs TeNPy, ~5 min of an IDLE machine: it is a timing
    measurement), step_cost.  With step_cost=False the step_cost table already stored in tn_tables.npz is kept (the paper needs it).
    Reads data/derived/larger_N.npz: run_extrapolation first."""
    import glob
    uniform5 = RUNS / f"N5_k12_int_c{model().tilt(5, 12):.3f}_T200_dt1.npz"
    ed_ref = DATA / 'mps' / 'ed_ref'
    records.check_convention(store.list_runs(str(DATA / 'mps')) + store.list_runs(str(ed_ref)) + store.list_runs(str(RUNS)),
                             theta_barrier(), 'runs behind the tensor-network tables')
    tabs = {}
    tabs['mps_validation'], series = tntables.mps_validation(DATA / 'mps', [ed_ref, RUNS])
    tabs['mps_series'] = series
    src = [(f'N{N}', p) for N in (2, 3, 4) for p in glob.glob(str(RUNS / f'N{N}_k12_int_*_T200_dt1.npz'))]
    src += [('N5', str(uniform5))]
    tabs['window_convergence'] = tntables.window_convergence(src)
    paths = store.list_runs(str(RUNS)) + store.list_runs(str(ed_ref))
    tabs['rank_vs_entropy'] = tntables.rank_vs_entropy(paths, windows=(30, 40, 60, 120, 200))
    tabs['wall_sensitivity'] = tntables.wall_sensitivity(CFG, store.load_object(DERIVED / 'larger_N.npz'), store.load_run(str(uniform5)))
    machine = None
    try:
        tabs['channel_ranks'] = tntables.channel_ranks(model())
        if step_cost:
            tabs['step_cost'], machine = tntables.step_cost(model())
    except ImportError as e:
        print('skipped (needs TeNPy):', e)
    if 'step_cost' not in tabs and (DERIVED / 'tn_tables.npz').exists():      # keep the stored timing measurement
        old = store.load_object(DERIVED / 'tn_tables.npz')
        if old.get('tables', {}).get('step_cost'):
            tabs['step_cost'], machine = old['tables']['step_cost'], old.get('machine')
            print('step_cost: kept the stored measurement (machine:', (machine or {}).get('platform', '?'), ')')
    store.save_object(DERIVED / 'tn_tables.npz', dict(tables=tabs, machine=machine))
    for name, rows in tabs.items():
        print(f"{name:20s} {len(rows):4d} rows -> {os.path.relpath(tntables.write_csv(DERIVED / 'tn_tables' / (name + '.csv'), rows), ROOT)}")
    return tabs


def run_window_scan(N=4, k=12, start='interacting', windows=(30, 60, 90, 120, 160, 200), directory=RUNS):
    """How the maxima depend on the window, from ONE saved series (no re-simulation)."""
    r = [r for r in load_runs(directory=directory)
         if r['meta']['N'] == N and r['meta']['k'] == k and r['meta']['start'] == start][0]
    print(f"N={N} k={k} {start}: window  S_cen   S_nat   S_perp  S_max   t(S_cen)")
    out = []
    for Tw in windows:
        s, g = analysis.summarize(r, Tw), analysis.graph_record(r, Tw)
        out.append((Tw, s['S_cen'], g['S_nat'], g['S_perp'], g['S_max'], s['tS_cen']))
        print(f"          {Tw:6.0f}  {s['S_cen']:6.3f}  {g['S_nat']:6.3f}  {g['S_perp']:6.3f}  {g['S_max']:6.3f}  {s['tS_cen']:6.1f}")
    return out


def run_cut_table(N=4, k=12, start='interacting', T_win=None, directory=RUNS):
    """Every bipartition of a saved run: exact sigma(A) = max_t S_A(t) against the graph cut C(A)."""
    r = [r for r in load_runs(directory=directory)
         if r['meta']['N'] == N and r['meta']['k'] == k and r['meta']['start'] == start][0]
    sig, tat = analysis.maxima(r, T_win)
    w = analysis.edge_weights(r, T_win)
    rows = []
    for j in np.argsort(-np.nan_to_num(sig, nan=-1)):
        A = lj.entanglement.members(int(r['masks'][j]), N)
        rows.append((A, float(sig[j]), float(tat[j]), graph.cut_value(w, A)))
    print(f"N={N} k={k} {start}: {len(rows)} bipartitions   (A, sigma(A), t_max, C(A))")
    for A, s, t, c in rows:
        print(f"  {str(A):20s} {s:7.4f}  t={t:6.1f}  C={c:7.4f}  diff={s - c:+.3f}")
    return rows


def run_graph_cuts(N_max=12, S1=2.0, directory=RUNS):
    """Graph cuts at ANY N, given edge weights: time the enumeration of all 2^(N-1)-1 cuts of the sharing-law
    graph (uniform weights, S1 = the one-particle entropy) up to N_max.  The weights themselves come from
    a simulation (analysis.edge_weights); only for N <= 7 does the state fit (see run_cost_table)."""
    out = []
    for N in range(3, N_max + 1):
        W = graph.sharing_weights(N, S1)
        t = time.time()
        masks, C = graph.all_cuts(W)
        dt = time.time() - t
        a = np.array([bin(int(m)).count('1') for m in masks])
        a = np.minimum(a, N - a)
        out.append((N, len(masks), dt, float(C[a == N // 2].min())))
        print(f"N={N:2d}: {len(masks):6d} cuts in {dt * 1e3:7.2f} ms   balanced cut {C[a == N // 2].min():.3f} bits")
    return out


def theta_barrier():
    """The weight of the grid column x = 0 in Theta(x) that this config simulates ([model] theta_barrier)."""
    return float(CFG['model'].get('theta_barrier', 0.5))


def _check_run_tilts(directory=RUNS):
    """data/runs holds the resonant point of every (N, k): a uniform run at another tilt would displace it in records.pick."""
    bad = []
    for p in store.list_runs(str(directory)):
        m = store.load_run(p)['meta']
        t = model().tilt(m['N'], m['k'])
        if m['protocol']['kind'] == 'uniform' and (t is None or abs(m['c'] - t) > 1e-6):
            bad.append(os.path.basename(p))
    if bad:
        raise ValueError(f"runs without a [tilts] entry or at another tilt than it in {directory}: {', '.join(bad)}.  records.pick would use them instead of the "
                         f"resonant run; move them to data/controls/ (run_off_resonance does) or data/studies/.")


def run_records(T_win=None):
    """Turn the saved runs into the ledger and graph tables of the paper: data/derived/ledger.npz and graph.npz.  T_win cuts the uniform runs.
    Refuses runs made with another Theta(x = 0) convention than the config ([model] theta_barrier) and runs at an off-table tilt."""
    _check_run_tilts()
    ledger, graph_rec = records.collect(str(RUNS), T_win=T_win, theta_barrier=theta_barrier(), controls_dir=str(CONTROLS))
    store.save_object(DERIVED / 'ledger.npz', dict(rows=ledger, T_win=T_win))
    store.save_object(DERIVED / 'graph.npz', dict(rows=graph_rec, T_win=T_win))
    n_u = sum(1 for r in ledger if r['protocol'] == 'uniform')
    print(f"ledger: {n_u} uniform records; graph: {len(graph_rec)} runs")
    return ledger, graph_rec


def run_protocol_tables():
    """Sampling sensitivity from the saved series: window scan (one series, cut afterwards) and spacing scan."""
    wins = CFG['studies']['windows']
    out = dict(window={}, spacing=records.spacing_scan(store.list_runs(str(STUDIES / 'dt_scan'))))
    for r in load_runs(directory=RUNS):
        if r['meta']['start'] == 'interacting' and r['meta']['k'] == CFG['basis']['k']:
            out['window'][int(r['meta']['N'])] = records.window_scan(r, wins)
    store.save_object(DERIVED / 'protocol_tables.npz', {'window': {str(k): v for k, v in out['window'].items()},
                                                          'spacing': out['spacing']})
    for N, rows in out['window'].items():
        print(f"N={N}: " + "  ".join(f"T<={r['T']:.0f}: S_cen {r['S_cen']:.4f} S_max {r['S_max']:.4f}" for r in rows))
    for r in out['spacing']:
        print(f"N={r['N']} dt={r['dt']:<4g} S_cen {r['S_cen']:.4f} S_one {r['S_one']:.4f}")
    return out


def run_parameter_choice():
    """Why these model parameters: the earlier one-input-at-a-time study, parsed into data/derived/parameter_choice.npz.

    (a) one particle, barrier A swept: action S0, trap mass mu, splitting dE (converged grid), switching time pi/dE;
    (b) N = 2 exact diagonalization, one input changed at a time: tilt, splitting, overlap of the start state with the
        two lowest levels (need >= overlap_min) and the shift of the splitting when the core cap is doubled
        (need < cap_max).  The pass/fail verdict is recomputed here and compared with the stored gates.
    """
    from ljtunnel import parameter_study
    pm = parameter_study.parameter_map(str(PARAMETER_MAP))
    lim = dict(overlap_min=0.9, cap_max=0.1)
    for r in pm['sensitivity']:
        r['pass_overlap'] = bool(r['ov_sub'] >= lim['overlap_min'])
        r['pass_cap'] = bool(r['cap_shift'] < lim['cap_max'])
        g = r['gates']
        assert g['H2a_static'] == r['pass_overlap'] and g['cap'] == r['pass_cap'], r['label']
    for r in pm['onebody']:       # closed forms for the table's first columns
        assert abs(r['S0'] - 4 / 3 * (2 * r['A']) ** 0.5) < 1e-3 and abs(r['mut'] - (8 * r['A']) ** 0.5) < 1e-3
    pm['limits'] = lim
    pm['source'] = 'data/studies/parameter_map/tier1_parameter_map.txt'
    store.save_object(DERIVED / 'parameter_choice.npz', pm)
    for r in pm['sensitivity']:
        print(f"{r['label']:<24} c*={r['c_star']:.3f} dE={r['dE']:.4f} overlap={r['ov_sub']:.3f} cap={r['cap_shift']:.3f}"
              f"  {'pass' if r['pass_overlap'] and r['pass_cap'] else 'FAIL'}")
    return pm


def run_cosine_check(directory=None):
    """The splitting-based rival: how much of <O>(t) does ONE cosine (frequency = the doublet splitting) explain?
    One row per correlated-start run of the uniform protocol; writes data/derived/cosine_check.npz."""
    rows = []
    for r in load_runs(directory=directory or RUNS):
        m = r['meta']
        if m['start'] != 'interacting':
            continue
        c = analysis.cosine_residual(r)
        rows.append(dict(N=int(m['N']), k=int(m['k']), **c))
        print(f"N={m['N']} k={m['k']:<3} dE={c['dE']:.4f}  residual/std: at dE {c['fixed']:.2f}, best cosine {c['free']:.2f} (w={c['w_free']:.4f})")
    store.save_object(DERIVED / 'cosine_check.npz', {'rows': rows})
    return rows


def run_parameter_scan(N=3, k=12, As=(4.0, 6.0, 8.0, 10.0, 12.0)):
    """Is the barrier chosen at N = 2 still the right one at larger N?  Static scan (no propagation): for each A the
    resonance tilt is found again (fine refinement, 0.01), the correlated and the product start are prepared and the
    weight of each in the doublet is recorded.  N = 3, k = 12 takes about a minute.  Writes data/derived/parameter_scan_N{N}.npz."""
    import copy
    rows = []
    for A in As:
        cfg = copy.deepcopy(CFG)
        cfg['model']['A'] = float(A)
        cfg['scan'].update(refine_halfwidth=0.12, refine_points=25)
        m = lj.Model(cfg)
        loc = lj.resonance.locate(m, N, k, tilt='scan')
        if loc is None:
            rows.append(dict(N=N, k=k, A=float(A), found=False))
            continue
        st = lj.starts.prepare(m, loc, N, k, cfg['protocol']['W_penalty'])
        d = st['diag']
        rows.append(dict(N=N, k=k, A=float(A), found=True, c=float(loc['c']), mO=float(loc['mO']), dE=float(d.get('dE', float('nan'))),
                         doublet_ok=bool(d['doublet']), levels=d['doublet_levels'], low_levels=[round(x - d['levels'][0], 4) for x in d['levels'][:8]], ov_int=float(d['ov_interacting']), ov_prod=float(d['ov_product']),
                         ))
        print(f"N={N} k={k} A={A:g}: c*={loc['c']:.2f} <O>_gs={loc['mO']:.3f} dE={rows[-1]['dE']:.4f} overlap {rows[-1]['ov_int']:.3f} (product {rows[-1]['ov_prod']:.3f}) doublet {rows[-1]['levels']} ok={rows[-1]['doublet_ok']} levels-E0 {rows[-1]['low_levels']}")
    store.save_object(DERIVED / f'parameter_scan_N{N}.npz', {'rows': rows})
    return rows


def run_extrapolation():
    """The sharing law carried to larger N, the entropy budgets and the first failing N of each classical route.
    Reads data/derived/ledger.npz + graph.npz (run_records first); writes data/derived/larger_N.npz."""
    led = store.load_object(DERIVED / 'ledger.npz')['rows']
    gr = [g for g in store.load_object(DERIVED / 'graph.npz')['rows'] if g['role'] == 'primary' and g['k'] == CFG['basis']['k']]
    S1 = {}
    for g in gr:
        t = model().tilt(g['N'], CFG['basis']['k'])
        if t is None:
            raise KeyError(f"a k = {CFG['basis']['k']} run exists for N = {g['N']} but config.toml [tilts] has no entry for it")
        if abs(g['c'] - t) < 1e-6:
            S1[g['N']] = g['S1mean']
    pts = {N: records.pick(led, N) for N in sorted(S1)}
    Ns = [N for N in sorted(S1) if pts[N]]
    last = pts[Ns[-1]]
    out = extrapolation.scenarios(CFG, Ns, [S1[N] for N in Ns], [pts[N]['S_cen'] for N in Ns],
                                  chi_last=(last.get('chi3'), last['S_cen']))
    if len(Ns) >= 4 and Ns[-1] in (6, 8, 12, 16):         # the sharing-law value at the last point as it was predicted from the points below it
        prev = extrapolation.scenarios(CFG, Ns[:-1], [S1[N] for N in Ns[:-1]], [pts[N]['S_cen'] for N in Ns[:-1]])
        out['share_pred_prev'] = dict(N=Ns[-1], values={sc: prev['share_pred'][sc][Ns[-1]] for sc in prev['share_pred']})
    out['ed'] = walls.ed_walls(CFG)
    store.save_object(DERIVED / 'larger_N.npz', out)
    print('measured N', Ns, ' S_cen', [round(pts[N]['S_cen'], 3) for N in Ns])
    print('first failing N (tensor network, grid d=%d):' % CFG['tn_wall']['d_grid'],
          {k: v for k, v in out['walls'].items() if k.endswith(f"d{CFG['tn_wall']['d_grid']}_both")})
    return out


def run_paper(compile_pdf=False):
    """Rebuild everything the paper reads from the saved runs: tables, extrapolation, quantum counts, numbers.tex, figures."""
    import subprocess
    run_records()
    run_protocol_tables()
    run_extrapolation()
    run_quantum()
    run_cosine_check()
    for script in ('make_numbers.py', 'figs.py'):
        subprocess.run([sys.executable, '-I', str(ROOT / 'paper' / 'scripts' / script)], check=True, cwd=ROOT / 'paper')
    if compile_pdf:
        for cmd in (['pdflatex'], ['bibtex'], ['pdflatex'], ['pdflatex']):
            subprocess.run(cmd + ['-interaction=nonstopmode', 'main'] if cmd[0] == 'pdflatex' else cmd + ['main'],
                           cwd=ROOT / 'paper', stdout=subprocess.DEVNULL)


# %% 3. counting models: pure arithmetic, seconds ----------------------------------------------------
def run_cost_table(k=None, N_list=range(2, 13), calibrate=True):
    """What a simulation costs before you start it: amplitudes, memory, matvecs, hours, cut sizes."""
    rate = cost.calibrate(model())['rate'] if calibrate else 3.0e10
    rows = cost.table(CFG, k=k, N_list=N_list, rate=rate)
    print(cost.format_table(rows))
    return rows


def run_walls():
    """Exact-diagonalisation counting and the tensor-network entropy budgets."""
    r = walls.run(CFG)
    print('ED first failing N:', r['ed'])
    return r


def run_quantum():
    """Qubit and T-gate counts, and the largest trap-units mass whose count fits the budget, per N."""
    meta, rows = quantum.edge_table(CFG)
    store.save_object(DERIVED / 'quantum.npz', dict(meta=meta, rows=rows))
    for r in rows:
        if r['N'] in (4, 8, 16, 24, 40, 88):
            print(f"N={r['N']:3d} Q={r['Q']:4d} T_qub={r['T_qub_mu0']:.2e} edge {r['edge_qub']:.2f} "
                  f"(complete {r['edge_qub_complete']:.2f})")
    return meta, rows


def run_montecarlo():
    """Sign-problem severity Gamma_inf(M) of absolute-value path sampling, UV exponent, dense cross-checks."""
    r = montecarlo.run(CFG)
    store.save_result(DERIVED / 'montecarlo.npz', meta={k: v for k, v in r.items() if np.isscalar(v)},
                      **{k: v for k, v in r.items() if not np.isscalar(v)})
    print(f"p_UV = {r['p_UV']:.4f}   Gamma_inf(production grid) = {r['Gamma_grid']:.1f}")
    return r


def run_onebody_checks():
    """One-particle problems with known answers: exact vs instanton, real-time readout, box lemma, grid order."""
    r = onebody.run(CFG)
    store.save_result(DERIVED / 'onebody.npz', meta={k: v for k, v in r.items() if np.isscalar(v)},
                      **{k: v for k, v in r.items() if not np.isscalar(v)})
    print({k: (round(v, 6) if np.isscalar(v) else '...') for k, v in r.items()})
    return r


# %% 4. tests (kept separate: kron / eig / propagator against dense references) ---------------------
def run_tests(slow=False):
    sys.path.insert(0, str(ROOT / 'tests'))
    import test_numerics
    return test_numerics.main(['--slow'] if slow else [])


# %% 5. the full rerun after a change of convention: `rerun_all.sh` drives everything below ----------------------
# One entry per group of files of data/runs: (N, k, starts, BLAS threads, estimated wall hours).  The hours are from the cost audit
# (one core of the build container; the mini-PC is 1.1-1.5x faster); only their ORDER matters: the scheduler starts the longest
# job first.  The tilts are the frozen picks of [tilts] (never a scan).  N = 5, k = 20 and N = 6, k = 12 are new in this version.
_I, _PI = ('interacting',), ('product', 'interacting')
RERUN_RUNS = [(5, 20, _I, 2, 11.5), (6, 12, _I, 2, 11.7), (5, 16, _I, 2, 2.6), (5, 12, _I, 1, 0.42), (4, 20, _I, 1, 0.27),
              (3, 32, _I, 1, 0.10), (4, 16, _I, 1, 0.08), (4, 12, _PI, 1, 0.07), (3, 28, _I, 1, 0.04), (3, 24, _I, 1, 0.02),
              (3, 20, _I, 1, 0.01), (3, 16, _I, 1, 0.005), (3, 12, _PI, 1, 0.005), (2, 20, _I, 1, 0.002), (2, 16, _I, 1, 0.002),
              (2, 12, _PI, 1, 0.002)]
# the validation of data/mps: exact run + MPS run of one case (T, chi_max, eps as of the stored runs; TDVP step 0.5)
MPS_CASES = {'N3_k6': dict(N=3, k=6, T=60.0, chi_max=120, eps=1e-8, pairs=True, threads=1, hours=0.01),
             'N4_k6': dict(N=4, k=6, T=60.0, chi_max=120, eps=1e-8, pairs=True, threads=1, hours=0.05),
             'N5_k6': dict(N=5, k=6, T=40.0, chi_max=100, eps=1e-8, pairs=True, threads=2, hours=0.25),
             'N4_k12': dict(N=4, k=12, T=40.0, chi_max=64, eps=1e-6, pairs=False, threads=2, hours=1.0)}
MPS_DT = 0.5
# <O>_gs at the config tilt with the weight-1/2 convention (REVIEW.md E3; computed independently of the runs): regression anchors
EXPECT_MO = {(2, 12): 0.5699, (2, 16): 0.4894, (2, 20): 0.4224, (3, 12): 0.5201, (3, 16): 0.5841, (3, 20): 0.5735, (3, 24): 0.4191,
             (3, 28): 0.5087, (3, 32): 0.5787, (4, 12): 0.5484, (4, 16): 0.4630, (4, 20): 0.5215, (5, 12): 0.4792}


def run_production(N, k, starts=('interacting',), force=True):
    """One production run of data/runs at the tilt of [tilts] (never a scan), overwriting an existing file."""
    c = model().tilt(N, k)
    if c is None:
        raise KeyError(f"no tilt for N={N}, k={k} in [tilts]: add it (a scan costs hours at N >= 5)")
    return run_dynamics(N=N, k=k, starts=starts, tilt=float(c), force=force, checkpoint=True)


def run_mps_case(name):
    """One validation case of data/mps (MPS_CASES): the tilt is [tilts] if there is one, else the exact bisection of <O>_gs = 1/2
    (the stored k = 6 runs used it), then `run_mps_check` with the stored T, chi_max, eps and the TDVP step 0.5."""
    from ljtunnel.resonance import exact_tilt
    case = dict(MPS_CASES[name])
    N, k = case['N'], case['k']
    c = model().tilt(N, k)
    if c is None:
        c0 = model().tilt(N, CFG['basis']['k'])
        c = exact_tilt(model(), N, k, c0 - 0.2, c0 + 0.9)
        print(f"{name}: tilt {c} (exact bisection of <O>_gs = 1/2, theta_barrier = {theta_barrier():g})")
    return run_mps_check(N=N, k=k, tilt=float(c), T=case['T'], chi_max=case['chi_max'], eps=case['eps'], pairs=case['pairs'], dt=MPS_DT)


def run_tier1_parameter_map(workers=2):
    """Appendix A (the one-input-at-a-time parameter study): rerun ljtunnel/parameter_study.py with the weight of
    [model] theta_barrier; it writes data/studies/parameter_map/tier1_parameter_map.txt, which `run_parameter_choice` parses.
    19 independent points of N = 2 exact diagonalization on a 28x18 grid (~1.2 h on 2 workers, one BLAS thread each)."""
    import subprocess
    out = PARAMETER_MAP
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + '.new')
    one = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    subprocess.run([sys.executable, '-m', 'ljtunnel.parameter_study', '--workers', str(workers),
                    '--theta-barrier', f'{theta_barrier():g}', '--out', str(tmp)], check=True, env=one, cwd=str(ROOT))
    os.replace(tmp, out)
    return out


def run_fullgrid_check():
    """The N = 2 resonance on the full grid under the current convention (~10 min), next to [quantum] t_star_0, the switching time
    the paper quotes (from the earlier full-grid study): data/derived/fullgrid_resonance.npz.  The splitting does not depend on Theta;
    only the tilt moves, so t0 should agree to well below the printed precision."""
    r = run_fullgrid_resonance()
    t0, ref = float(np.pi / r['dE']), float(CFG['quantum']['t_star_0'])
    out = dict(c=r['c'], mO=r['mO'], dE=r['dE'], t0=t0, t_star_0_config=ref, rel_diff=t0 / ref - 1)
    store.save_object(DERIVED / 'fullgrid_resonance.npz', out)
    print(f"full grid, N = 2: c* = {r['c']:.4f} (<O>_gs = {r['mO']:.4f})  dE = {r['dE']:.6f}  t0 = pi/dE = {t0:.4f}  "
          f"[quantum] t_star_0 = {ref}  ({100 * (t0 / ref - 1):+.3f} %)")
    return out


SUBSETS = {None: 'every job', 'medium': 'without the four long ones (N5 k16, N5 k20, N6 k12, MPS N4 k12): ~2 h on 4 cores',
           'smoke': 'N <= 3 and k <= 16, one MPS case: a few minutes'}
_LONG = {'N5_k20', 'N6_k12', 'N5_k16', 'mps_N4_k12'}


def _keep(name, subset):
    """Is job `name` part of the rehearsal subset?  (None: of the real rerun.)"""
    if subset is None:
        return True
    if subset == 'medium':
        return name not in _LONG
    small = {f"N{N}_k{k}" for N, k, *_ in RERUN_RUNS if N <= 3 and k <= 16}
    return name in small | {'mps_N3_k6', 'dt_scan_N3', 'parameter_scan_N3'}


def rerun_jobs(subset=None):
    """The simulations of the rerun as [dict(name, threads, hours, call)]: everything that writes data/ and is independent of the
    other jobs.  `call` is a python statement for `from run import *`.  subset: see SUBSETS (rehearsals)."""
    jobs = []
    for N, k, starts, threads, hours in RERUN_RUNS:
        jobs.append(dict(name=f"N{N}_k{k}", threads=threads, hours=hours, call=f"run_production({N}, {k}, starts={starts!r})"))
    for name, c in MPS_CASES.items():
        jobs.append(dict(name=f"mps_{name}", threads=c['threads'], hours=c['hours'], call=f"run_mps_case({name!r})"))
    for N, hours in ((3, 0.01), (4, 0.07)):
        jobs.append(dict(name=f"dt_scan_N{N}", threads=1, hours=hours, call=f"run_sampling_study(Ns=({N},), force=True)"))
    jobs += [dict(name='parameter_scan_N3', threads=1, hours=0.02, call="run_parameter_scan(N=3)"),
             dict(name='parameter_scan_N4', threads=1, hours=0.08, call="run_parameter_scan(N=4, As=(6.0, 8.0, 10.0, 12.0))"),
             dict(name='control_N5_offres', threads=1, hours=0.42, call="run_off_resonance(N=5, force=True)"),
             dict(name='tier1_parameter_map', threads=2, hours=1.2, call="run_tier1_parameter_map(workers=2)"),
             dict(name='fullgrid_N2', threads=1, hours=0.02, call="run_fullgrid_N2()"),
             dict(name='fullgrid_resonance', threads=1, hours=0.17, call="run_fullgrid_check()")]
    return sorted((j for j in jobs if _keep(j['name'], subset)), key=lambda j: -j['hours'])


def run_theta_effect(old_dir=None):
    """What the Theta(x = 0) convention changed: the runs of data/runs next to the file of the same name (same N, k, tilt, start) of
    the earlier version (default previous-data/runs, made with theta_barrier = 0) -> data/derived/theta_effect.csv.
    Invariants that must hold to rounding: the splitting dE (H has no Theta) and the entropies of every PRODUCT start (a product of
    valley orbitals; Theta enters only the observable <O>)."""
    old_dir = Path(old_dir) if old_dir else PREVIOUS / 'runs'
    rows = []
    for p in store.list_runs(str(RUNS)):
        q = old_dir / os.path.basename(p)
        if not q.exists():
            continue
        new, old = store.load_run(p), store.load_run(str(q))
        if new['meta']['protocol']['kind'] != 'uniform' or new['S'].shape != old['S'].shape:
            continue
        a, b = analysis.summarize(old), analysis.summarize(new)
        ga, gb = analysis.graph_record(old, None, exhaustive_max_N=0), analysis.graph_record(new, None, exhaustive_max_N=0)
        r = dict(N=b['N'], k=b['k'], start=new['meta']['start'], c=b['c'],
                 dE_diff=new['meta']['diag']['dE'] - old['meta']['diag']['dE'],
                 mO_old=old['meta']['diag']['mO'], mO_new=new['meta']['diag']['mO'],
                 S_cen_old=a['S_cen'], S_cen_new=b['S_cen'], dS_cen=b['S_cen'] - a['S_cen'],
                 tS_cen_old=a['tS_cen'], tS_cen_new=b['tS_cen'],
                 S_nat_old=ga['S_nat'], S_nat_new=gb['S_nat'], dS_nat=gb['S_nat'] - ga['S_nat'],
                 S_max_old=ga['S_max'], S_max_new=gb['S_max'], dS_max=gb['S_max'] - ga['S_max'],
                 chi3_old=a.get('chi3'), chi3_new=b.get('chi3'), S1_old=a['S1mean'], S1_new=b['S1mean'],
                 Omax_old=a['Omax'], Omax_new=b['Omax'], Omin_old=a['Omin'], Omin_new=b['Omin'],
                 max_dS_t=float(np.nanmax(np.abs(new['S'] - old['S']))), max_dO_t=float(np.abs(new['O'] - old['O']).max()))
        rows.append(r)
    for r in rows:
        print(f"N={r['N']} k={r['k']:<2} {r['start']:<11} c={r['c']:.3f}  <O>_gs {r['mO_old']:.4f} -> {r['mO_new']:.4f}   S_cen {r['S_cen_old']:.4f} -> "
              f"{r['S_cen_new']:.4f} ({r['dS_cen']:+.4f})  S_nat {r['dS_nat']:+.4f}  S_max {r['dS_max']:+.4f}  chi3 {r['chi3_old']} -> {r['chi3_new']}  "
              f"max|dS(t)| {r['max_dS_t']:.3f}  max|d<O>(t)| {r['max_dO_t']:.3f}")
    if not rows:
        print(f"no run of data/runs has a counterpart in {old_dir}")
        return rows
    tntables.write_csv(DERIVED / 'theta_effect.csv', rows)
    bad = [r for r in rows if abs(r['dE_diff']) > 1e-9 or (r['start'] == 'product' and r['max_dS_t'] > 1e-8)]
    cor = [r for r in rows if r['start'] == 'interacting']
    print(f"correlated start, {len(cor)} runs: max |dS_cen| {max(abs(r['dS_cen']) for r in cor):.4f}, max |dS_nat| "
          f"{max(abs(r['dS_nat']) for r in cor):.4f}, max |dS_max| {max(abs(r['dS_max']) for r in cor):.4f} bit; "
          f"<O>_gs shift {min(r['mO_new'] - r['mO_old'] for r in rows):+.4f} .. {max(r['mO_new'] - r['mO_old'] for r in rows):+.4f}")
    if bad:
        raise RuntimeError(f"INVARIANT BROKEN (dE or a product-start entropy changed with theta_barrier): {[(r['N'], r['k'], r['start']) for r in bad]}")
    return rows


def run_check_rerun(since=None, subset=None):
    """After a rerun: every expected file exists, was written after `since` (epoch seconds), carries the config's Theta convention
    and tilt, is not duplicated, and <O>_gs at the tilt agrees with the independent anchors of the new convention.
    subset: the rehearsal subset (SUBSETS) whose files are expected.  Returns the list of problems (empty: all good)."""
    problems = []
    smoke = subset == 'smoke'
    kept = lambda name: _keep(name, subset)
    theta, T, dt = theta_barrier(), CFG['protocol']['T'], CFG['protocol']['dt_snap']

    def need(path, what=None):
        if not Path(path).exists():
            problems.append(f"missing: {os.path.relpath(path, ROOT)}")
            return False
        if since is not None and Path(path).stat().st_mtime < since:
            problems.append(f"stale (older than the rerun start): {os.path.relpath(path, ROOT)}")
            return False
        return True

    def meta_of(path):
        return store.load_run(str(path))['meta']

    # data/runs
    want = set()
    for N, k, starts, _, _ in RERUN_RUNS:
        if not kept(f"N{N}_k{k}"):
            continue
        c = model().tilt(N, k)
        for st in starts:
            f = RUNS / (store.run_name(dict(N=N, k=k, start=st, c=c, protocol=dict(T=T, dt_snap=dt))) + '.npz')
            want.add(f.name)
            if not need(f):
                continue
            m = meta_of(f)
            if abs(records.theta_of(m) - theta) > 1e-12 or m['protocol']['kind'] != 'uniform' or m.get('how_tilt') != 'fixed':
                problems.append(f"{f.name}: theta_barrier {records.theta_of(m)}, protocol {m['protocol']['kind']}, tilt {m.get('how_tilt')}")
            if abs(m['info']['norm'] - 1) > 1e-5:
                problems.append(f"{f.name}: norm {m['info']['norm']} after the series")
            if (N, k) in EXPECT_MO and abs(m['diag']['mO'] - EXPECT_MO[(N, k)]) > 1.5e-3:
                problems.append(f"{f.name}: <O>_gs {m['diag']['mO']:.4f}, the weight-1/2 anchor is {EXPECT_MO[(N, k)]:.4f}")
    extra = sorted(set(p.name for p in RUNS.glob('*.npz')) - want)
    if extra:
        problems.append(f"unexpected files in data/runs: {', '.join(extra)}")
    print(f"data/runs: {len(want)} expected files, {len(extra)} unexpected")
    # studies, control, mps
    for N in (n for n in (3, 4) if kept(f"dt_scan_N{n}")):
        for d in CFG['studies']['dt_scan']:
            f = STUDIES / 'dt_scan' / (store.run_name(dict(N=N, k=12, start='interacting', c=model().tilt(N, 12), protocol=dict(T=T, dt_snap=float(d)))) + '.npz')
            if need(f) and abs(records.theta_of(meta_of(f)) - theta) > 1e-12:
                problems.append(f"{f.name}: wrong theta_barrier")
    if kept('control_N5_offres'):
        f = CONTROLS / (store.run_name(dict(N=5, k=12, start='interacting', c=CFG['control']['N5_off_c'], protocol=dict(T=T, dt_snap=dt))) + '.npz')
        if need(f) and abs(records.theta_of(meta_of(f)) - theta) > 1e-12:
            problems.append(f"{f.name}: wrong theta_barrier")
    n_mps = 0
    for name, case in MPS_CASES.items():
        if not kept(f"mps_{name}"):
            continue
        n_mps += 1
        found = sorted((DATA / 'mps').glob(f"N{case['N']}_k{case['k']}_mps_*.npz"))
        refs = sorted((DATA / 'mps' / 'ed_ref').glob(f"N{case['N']}_k{case['k']}_int_*.npz"))
        if len(found) != 1 or len(refs) != 1:
            problems.append(f"mps case {name}: {len(found)} MPS files, {len(refs)} exact references (expected 1 + 1)")
            continue
        if need(found[0]) and need(refs[0]):
            m = meta_of(found[0])
            if abs(records.theta_of(m) - theta) > 1e-12 or abs(m['mps']['dt'] - MPS_DT) > 1e-12 or abs(m['c'] - meta_of(refs[0])['c']) > 1e-9:
                problems.append(f"mps case {name}: theta_barrier {records.theta_of(m)}, dt {m['mps']['dt']}, tilt differs from the exact reference")
    if len(list((DATA / 'mps').glob('*.npz'))) != n_mps:
        problems.append(f"data/mps holds {len(list((DATA / 'mps').glob('*.npz')))} MPS files, expected {n_mps}")
    # derived
    names = ['ledger.npz', 'graph.npz', 'protocol_tables.npz', 'cosine_check.npz', 'quantum.npz', 'theta_effect.csv']
    if not smoke:
        names += ['larger_N.npz', 'parameter_choice.npz', 'parameter_scan_N3.npz', 'parameter_scan_N4.npz', 'fullgrid_N2.npz',
                  'fullgrid_resonance.npz', 'tn_tables.npz']
    for n in names:
        need(DERIVED / n)
    if not smoke:
        if need(PARAMETER_MAP) and f"--theta-barrier {theta:g}" not in PARAMETER_MAP.read_text()[:400]:
            problems.append('data/studies/parameter_map/tier1_parameter_map.txt was not made with the config theta_barrier')
        if (DERIVED / 'fullgrid_resonance.npz').exists():
            r = store.load_object(DERIVED / 'fullgrid_resonance.npz')
            if abs(r['rel_diff']) > 5e-3:       # not a failure: [quantum] t_star_0 is the earlier full-grid study's own number (see REVIEW.md)
                print(f"NOTE: full-grid t0 = {r['t0']:.4f} at the exact <O>_gs = 1/2 resonance c* = {r['c']:.4f}; [quantum] t_star_0 = "
                      f"{r['t_star_0_config']} ({100 * r['rel_diff']:+.2f} %).  dE depends steeply on c, so t0 depends on the exact c.")
    print(('PROBLEMS:\n  ' + '\n  '.join(problems)) if problems else 'rerun check: all expected files present, current, in the config convention')
    return problems


# %% 1b. follow-up (followup/TASK_followup_rival_and_mps.md): outputs go to data/followup/, criteria in data/followup/PREREG.md --------
FOLLOWUP = DATA / 'followup'
# Stage 1 case groups: (N, k, first guess of the tilt when it is not stored); the tilt is [tilts] if stored, else <O>_gs = 1/2 (ljtunnel.followup_spectra.tilt_half)
STAGE1_GROUPS = {
    'd6':  [(3, 6, 1.622), (4, 6, 2.456), (5, 6, 3.170)],                              # k = 6 (the stored exact_tilt values are the guesses and the check)
    'a':   [(N, k, None) for N in (2, 3, 4, 5) for k in (12, 16, 20)] + [(6, 12, None)],   # reproduction at the stored tilts
    'd8':  [(3, 8, 1.60), (4, 8, 2.40), (5, 8, 3.10), (6, 6, 3.85), (6, 8, 3.86), (7, 6, 4.02), (8, 6, 4.05)],         # (7, 8) and (8, 8): the tilt search alone is hours (13 min per ground state at N = 7, k = 8): group 'd78x'
    'b4':  [(4, 24, 1.92), (4, 28, 1.80), (4, 32, 1.68)],
    'b5':  [(5, 24, 2.52), (5, 28, 2.42), (5, 32, 2.32)],
    'c6a': [(6, 14, 3.56)],
    'c6b': [(6, 16, 3.46), (6, 20, 3.26)],           # projected above 4 h each: ask the user first (PREREG rule)
    'd88': [(8, 8, 4.05)],
    'd78x': [(7, 8, 4.60), (8, 8, 4.60)],             # not run in the first pass (amendment 4)
}
STAGE1_OPTS = {'b5': dict(krylov_only=True), 'c6a': dict(Lmax=80), 'c6b': dict(Lmax=80), 'd88': dict(krylov_only=True)}    # PREREG amendment 3
STAGE1_UNTILTED = [(N, 12, C0) for N in (3, 4, 5) for C0 in (None, 0.0)]       # C0 None = the configured model


def run_followup_stage1(groups=('d6', 'a'), untilted=False, force=False, Lmax=160, gb_cap=30.0):
    """Stage 1: the lowest L >= 40 levels of H, the weights of the correlated (and the product) start, B2, omega*, D, n95 (ljtunnel.followup_spectra).
    One case at a time; writes data/followup/stage1/N{N}_k{k}_c{c}.npz per case and tilts.json.  groups: keys of STAGE1_GROUPS ('d6', 'a', 'd8', 'b4', 'b5', 'c6a', 'c6b', 'd88');
    untilted=True also runs (e).  The table and the gate: run_followup_stage1_table()."""
    import json
    from ljtunnel import followup_spectra as fs
    out = FOLLOWUP / 'stage1'
    out.mkdir(parents=True, exist_ok=True)
    tj = out / 'tilts.json'
    tilts = json.loads(tj.read_text()) if tj.exists() else {}
    from ljtunnel.log import tee
    with tee(str(out / 'stage1.log')):
        for g in groups:
            for N, k, c0 in STAGE1_GROUPS[g]:
                key = f"N{N}_k{k}"
                c = model().tilt(N, k)
                if c is None:
                    if key not in tilts:
                        tilts[key] = fs.tilt_half(model(), N, k, c0)
                        tj.write_text(json.dumps(tilts, indent=1, sort_keys=True))
                    c = tilts[key]
                stored = None
                if g == 'a':
                    hits = sorted((RUNS).glob(f"N{N}_k{k}_int_c*_T200_dt1.npz"))
                    stored = str(hits[0]) if hits else None
                opts = STAGE1_OPTS.get(g, {})
                if opts.get('krylov_only'):                                          # PREREG amendment 3
                    fs.krylov_case(model(), N, k, float(c), W=CFG['protocol']['W_penalty'], force=force, product=False)
                    continue
                r = fs.analyse_case(model(), N, k, float(c), W=CFG['protocol']['W_penalty'], Lmax=opts.get('Lmax', Lmax), gb_cap=gb_cap, stored=stored, force=force)
                if r['meta']['summary']['interacting']['coverage'] < 0.99:          # PREREG amendment 2: add the spectral measure of the start
                    fs.krylov_case(model(), N, k, float(c), W=CFG['protocol']['W_penalty'], force=force, product=(N < 6))
        if untilted:
            for N, k, C0 in STAGE1_UNTILTED:
                m = model() if C0 is None else fs.model_with_C0(CFG, C0)
                fs.analyse_untilted(m, N, k, c=0.0, tag='' if C0 is None else f"_C0{C0:g}", force=force)


def run_followup_stage1_table():
    """The table of every saved Stage 1 case, the Gate 1 quantities (data/followup/stage1/stage1_table.csv, gate1.json) and the untilted band counts."""
    import csv
    import json
    from ljtunnel import followup_spectra as fs
    rows = fs.collect()
    out = FOLLOWUP / 'stage1'
    keys = ['N', 'k', 'c', 'tag', 'L', 'coverage', 'B2', 'pair', 'omega', 'omega_sqrtN', 'D', 'n_between', 'n95', 'verdict', 'B2_prod', 'cov_prod', 'repro', 'wall_s']
    with open(out / 'stage1_table.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in rows:
            w.writerow([r[x] if x != 'pair' else '-'.join(map(str, r[x])) for x in keys])
    gate = fs.gate1(rows)
    (out / 'gate1.json').write_text(json.dumps(gate, indent=1, default=float))
    for r in rows:
        print(f"N={r['N']} k={r['k']:2d} c={r['c']:.4f}{r['tag']:>8s} L={r['L']:3d} cov={r['coverage']:.3f} B2={r['B2']:.3f} pair={r['pair']} "
              f"omega*={r['omega']:.4f} om/sqrtN={r['omega_sqrtN']:.4f} D={r['D']:.3f} nbtw={r['n_between']} n95={r['n95']} {r['verdict']}")
    print(json.dumps(gate, indent=1, default=float))
    return rows, gate


def _stage1_tilt(N, k, c0):
    """The tilt of a Stage 1 / 5a case: [tilts] if stored, else the <O>_gs = 1/2 point saved in data/followup/stage1/tilts.json (computed and saved if missing)."""
    import json
    from ljtunnel import followup_spectra as fs
    c = model().tilt(N, k)
    if c is not None:
        return float(c)
    tj = FOLLOWUP / 'stage1' / 'tilts.json'
    tilts = json.loads(tj.read_text()) if tj.exists() else {}
    key = f"N{N}_k{k}"
    if key not in tilts:
        tilts[key] = fs.tilt_half(model(), N, k, c0)
        tj.parent.mkdir(parents=True, exist_ok=True)
        tj.write_text(json.dumps(tilts, indent=1, sort_keys=True))
    return float(tilts[key])


def run_followup_stage5a(steps=('exact', 'repro', 'mps'), Ns_mps=(6, 7, 8), max_hours=6.0, chi_default=120, force=False):
    """Stage 5a: the shape of S_cen(N) at k = 6.  steps: 'exact' (N = 2..5, bond-only exact runs, T = 200, written to data/followup/stage5a/exact_k6/),
    'repro' (the stored N3_k6 MPS case with the new entry point, compared with data/mps), 'mps' (N = 6, 7, 8, T = 60, TDVP step 0.5, eps 1e-8; one N at a time,
    each projected from the previous one, stopped if a projection exceeds max_hours; chi_max 120, raised to 160 / 200 when the previous N reached its cap).
    The table and the laws: run_followup_stage5a_table()."""
    import json
    from ljtunnel import mps
    out = FOLLOWUP / 'stage5a'
    (out / 'exact_k6').mkdir(parents=True, exist_ok=True)
    (out / 'mps').mkdir(parents=True, exist_ok=True)
    plan_file = out / 'mps_plan.json'
    plan = json.loads(plan_file.read_text()) if plan_file.exists() else {}
    guess = {2: 0.70, 3: 1.622, 4: 2.456, 5: 3.170, 6: 3.85, 7: 4.50, 8: 5.10}
    if 'exact' in steps:
        for N in (2, 3, 4, 5):
            if not force and list((out / 'exact_k6').glob(f"N{N}_k6_int_*.npz")):
                continue
            c = _stage1_tilt(N, 6, guess[N])
            experiments.dynamics_run(model(), N, 6, starts=['interacting'], tilt=c, cuts='bonds', outdir=str(out / 'exact_k6'), force=True)
    if 'repro' in steps:
        t0 = time.time()
        stored = store.load_run(str(DATA / 'mps' / 'N3_k6_mps_c1.622_T60_dt1_chi120_eps1e-08.npz'))
        new = run_mps(N=3, k=6, tilt=float(stored['meta']['c']), T=60.0, dt_snap=1.0, chi_max=120, eps=1e-8, pairs=False, dt=MPS_DT,
                      start_key=None, save=False)
        wall = time.time() - t0
        col = {int(m_): j for j, m_ in enumerate(stored['masks'])}
        dS = max(np.nanmax(np.abs(new['S'][:, j] - stored['S'][:, col[int(m_)]])) for j, m_ in enumerate(new['masks']))
        dO = float(np.abs(new['O'] - stored['O']).max())
        plan['repro'] = dict(max_dS_bit=float(dS), max_dO=dO, wall_s=wall, stored_wall_s=float(stored['meta']['wall_s']),
                             speed_factor=wall / float(stored['meta']['wall_s']), tolerance=1e-6, passes=bool(dS <= 1e-6 and dO <= 1e-6))
        plan_file.write_text(json.dumps(plan, indent=1))
        print('reproduction of N3_k6:', plan['repro'])
    if 'mps' in steps:
        stored_wall = {3: 17.0, 4: 95.0, 5: 476.0}                         # data/mps/README.md (container timings)
        factor = plan.get('repro', {}).get('speed_factor', 1.0)
        walls = {N: stored_wall[N] * factor for N in stored_wall}
        for r in (out / 'mps').glob('N*_k6_mps_*.npz'):
            m_ = store.load_run(str(r))['meta']
            walls[m_['N']] = m_['wall_s']
        chi_prev = None
        for N in Ns_mps:
            ratio = walls[N - 1] / walls[N - 2]
            proj_h = walls[N - 1] * ratio / 3600.0
            existing = list((out / 'mps').glob(f"N{N}_k6_mps_*.npz"))
            if existing and not force:
                r = store.load_run(str(existing[0]))
                chi_prev = r['meta']['mps']['chi_reached'] >= r['meta']['mps']['chi_max']
                continue
            plan[f"N{N}"] = dict(projected_hours=proj_h, ratio_used=ratio, basis=f"N{N - 1} wall {walls[N - 1]:.0f} s")
            plan_file.write_text(json.dumps(plan, indent=1))
            print(f"N={N}: projected {proj_h:.2f} h (previous wall {walls[N - 1]:.0f} s x ratio {ratio:.2f})", flush=True)
            if proj_h > max_hours:
                print(f"N={N}: projection above {max_hours} h: not started; ask the user", flush=True)
                plan[f"N{N}"]['status'] = 'not started: projected above the limit'
                plan_file.write_text(json.dumps(plan, indent=1))
                break
            chi = chi_default if not chi_prev else {6: 160, 7: 160, 8: 200}[N]
            c = _stage1_tilt(N, 6, guess[N])
            runN = run_mps(N=N, k=6, tilt=c, T=60.0, dt_snap=1.0, chi_max=chi, eps=1e-8, pairs=False, dt=MPS_DT, save=False)
            print('wrote', os.path.relpath(mps.save(runN, outdir=str(out / 'mps')), ROOT))
            walls[N] = runN['meta']['wall_s']
            chi_prev = runN['meta']['mps']['chi_reached'] >= chi
            plan[f"N{N}"].update(status='done', wall_s=walls[N], chi_max=chi, chi_reached=runN['meta']['mps']['chi_reached'],
                                 trunc_err=runN['meta']['mps']['trunc_err'])
            plan_file.write_text(json.dumps(plan, indent=1))


def run_followup_stage5a_table():
    """Rows N = 2..8 at the window T <= 60, the three laws fitted on N <= 5 and their predictions (data/followup/stage5a/growth_table.json)."""
    import json
    from ljtunnel import followup_growth as fg
    rows = fg.growth_rows()
    res = fg.laws(rows)
    (FOLLOWUP / 'stage5a' / 'growth_table.json').write_text(json.dumps(dict(rows=rows, laws=res), indent=1, default=float))
    for r, q in zip(rows, res['rows']):
        print(f"N={r['N']} {r['kind']:5s} S_cen(T<=60)={r['S_cen']:.4f} S1mean={r['S1mean']:.4f}"
              + (f" S_cen(T<=200)={r['full']['S_cen']:.4f}" if 'full' in r else f" chi_reached={r['chi_reached']}/{r['chi_max']} trunc={r['trunc_err']:.1e}")
              + '  ' + ' '.join(f"{k_}:{v:+.1%}{'*' if q['consistent'][k_] else ''}" for k_, v in q['rel_err'].items()))
    return rows, res


def _cli(argv):
    """python run.py jobs [smoke|medium]          the simulation jobs of the rerun, one per line: name <TAB> threads <TAB> hours <TAB> call
       python run.py check [smoke|medium] [EPOCH]  the checks of `run_check_rerun` (exit status 1 if there is a problem)"""
    subset = next((a for a in argv[1:] if a in ('smoke', 'medium')), None)
    if argv[0] == 'jobs':
        for j in rerun_jobs(subset):
            print(f"{j['name']}\t{j['threads']}\t{j['hours']}\t{j['call']}")
        return 0
    if argv[0] == 'check':
        since = next((float(a) for a in argv[1:] if a not in ('smoke', 'medium')), None)
        return 1 if run_check_rerun(since=since, subset=subset) else 0
    print(_cli.__doc__)
    return 2


# %% main: comment lines in or out ------------------------------------------------------------------
def main():
    # --- before you start: what does it cost? -----------------------------------------------------
    run_cost_table()

    # --- simulations (each writes data/runs/*.npz) ------------------------------------------------
    # run_dynamics(N=2)                                 # correlated start, k = 12, uniform protocol
    # run_dynamics(N=3, starts=('product', 'interacting'))
    # run_dynamics_batch(Ns=(2, 3, 4), ks=(12, 16, 20))
    # run_dynamics(N=5, k=12)                           # ~ 20 min; N=5 at k=16, 20 are much longer
    # run_dynamics(N=6, k=12, c_guess=3.6, big_cut_stride=2)   # hours; read run_cost_table first
    # run_off_resonance(N=5)                            # control at the localized tilt, ~ 20 min
    # run_mps(N=5, k=6, T=60, chi_max=120)              # matrix-product-state series (TeNPy), no k^N state; see README
    # run_sampling_study()                              # snapshot-spacing study, N=3,4 (~ 10 min)
    # run_fullgrid_N2()                                 # N = 2 on the full grid, ~ 1 min

    # --- analysis of what is saved (no simulation) ------------------------------------------------
    # run_summary()                                     # all runs, one window for all N
    # run_summary(T_win=120)                            # the same series cut to T <= 120
    # run_window_scan(N=4, k=12)                        # window sensitivity from one series
    # run_cut_table(N=4, k=12)                          # every bipartition vs its graph cut
    # run_mps_summary()                                 # MPS cost (chi, S) of the saved MPS runs next to the exact ones
    # run_mps_check(N=4, k=6, tilt=2.455)               # exact run + MPS run of the same case, largest difference over every cut
    # run_mps_tables()                                  # tables for the tensor-network section -> data/derived/tn_tables/
    # run_graph_cuts(N_max=12)                          # enumeration cost of the graph cuts
    # run_records()                                     # ledger + graph tables from data/runs
    # run_protocol_tables()                             # window and spacing sensitivity
    # run_cosine_check()                                # how much of <O>(t) one splitting-cosine explains\n    # run_parameter_choice()                            # why A=8 etc.: the earlier one-input-at-a-time study
    # run_extrapolation()                               # sharing law at larger N and the walls
    # run_paper()                                       # all of the above + numbers.tex and the figures of the paper

    # --- counting models --------------------------------------------------------------------------
    # run_walls(); run_quantum(); run_montecarlo(); run_onebody_checks()

    # --- tests ------------------------------------------------------------------------------------
    # run_tests()                                       # exact machinery; python tests/test_mps.py for the MPS module


if __name__ == '__main__':
    if len(sys.argv) > 1:
        sys.exit(_cli(sys.argv[1:]))
    main()
