#!/usr/bin/env python3
"""make_numbers.py -- write numbers.tex and the generated tables from data/ and config.toml.

Every value that appears in the paper's text or tables and comes from a computation is a macro
\\num{key} defined here, from files in data/derived (built from data/runs by run.py) or
from the configuration file.  A value that needs a run which does not exist yet is written as
\\pending{...}.  numbers.tex lists, for each key, the file it was read from.

One measurement protocol applies to every N (window T, snapshot spacing: config.toml [protocol]).  A
point made with another window or spacing would be marked with a dagger in the tables and figures
(none is: every point is a uniform run of data/runs).
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')                       # paper/
REPO = os.path.join(ROOT, '..')
DER = os.path.join(REPO, 'data', 'derived')
RAW = os.path.join(REPO, 'data', 'studies', 'onebody_mc')      # mc_gamma_ledger.txt, tier1_checks.log (one-body and Monte Carlo checks)
sys.path.insert(0, REPO)
from ljtunnel import load_config, store, records  # noqa: E402

cfg = load_config(os.path.join(REPO, 'config.toml'))
led = store.load_object(os.path.join(DER, 'ledger.npz'))['rows']
gr = store.load_object(os.path.join(DER, 'graph.npz'))['rows']
L = store.load_object(os.path.join(DER, 'larger_N.npz'))
Q = store.load_object(os.path.join(DER, 'quantum.npz'))
PT = store.load_object(os.path.join(DER, 'protocol_tables.npz'))
PC = store.load_object(os.path.join(DER, 'parameter_choice.npz'))
CC = store.load_object(os.path.join(DER, 'cosine_check.npz'))['rows']
TN = store.load_object(os.path.join(DER, 'tn_tables.npz'))
qrow = {r['N']: r for r in Q['rows']}

NUM = {}
SRC = {}
sys.path.insert(0, HERE)
import followup_data as _fdt  # noqa: E402
_S1_tab = _fdt.stage1()


def put(key, val, src=''):
    NUM[key] = val
    SRC[key] = src


def f(x, n=2):
    return f"{x:.{n}f}"


def sci(x, n=1):
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    if round(m, n) >= 10:
        m, e = m / 10, e + 1
    return f"\\ensuremath{{{m:.{n}f}\\times10^{{{e}}}}}"


def pw(x):
    e = int(round(math.log10(x)))
    return f"\\ensuremath{{10^{{{e}}}}}"


PEND = lambda s: f"\\pending{{{s}}}"


def tilt(N):
    """Resonance tilt at k = 12 from config.toml [tilts] (N = 2: a single number); None if the file has none for this N."""
    t = cfg['tilts'].get(f'N{N}')
    return t.get('k12') if isinstance(t, dict) else t


def G(N, c=None, k=12):
    """Graph record of the correlated start at the resonance tilt (or at tilt c): uniform run if there is one."""
    c0 = tilt(N) if c is None else c
    if c0 is None:
        return None
    out = [r for r in gr if r['N'] == N and r['k'] == k and r['role'] == 'primary'
           and abs(r['c'] - c0) < 1e-9]
    out.sort(key=lambda r: r['protocol'] != 'uniform')
    return out[0] if out else None


def rec(N, k=12, start='correlated', role='primary'):
    return records.pick(led, N, k, start, role)


def dag(N, k=12, start='correlated'):
    """Dagger for a point that comes from the earlier protocol."""
    r = rec(N, k, start)
    return '$^\\dagger$' if r is not None and r['protocol'] != 'uniform' else ''


# --------------------------------------------------------------------------- configuration
mp, gp, bp = cfg['model'], cfg['grid'], cfg['budgets']
put('A', f(mp['A'], 0), 'config')
put('wy2', f(mp['omega_y'], 0), 'config')
put('eps_LJ', f(mp['epsilon'], 0), 'config')
put('sigma_LJ', f(mp['sigma'], 2), 'config')
put('Vcap', f(mp['V_cap'], 0), 'config')
put('Czero', f(mp['C0'], 2), 'config')
put('nx', str(gp['nx']), 'config')
put('ny', str(gp['ny']), 'config')
put('M_grid', str(gp['nx'] * gp['ny']), 'config')
put('Lx', f(gp['Lx'], 1), 'config')
put('Ly', f(gp['Ly'], 1), 'config')
put('dx', f(2 * gp['Lx'] / gp['nx'], 3), 'config')
put('dy', f(2 * gp['Ly'] / gp['ny'], 3), 'config')
put('smear', f(gp['sigma_smear'], 2), 'config')
put('Wpen', f(cfg['protocol']['W_penalty'], 0), 'config')
pp = cfg['protocol']
put('Tprot', f(pp['T'], 0), 'config [protocol]')
put('dtsnap', f(pp['dt_snap'], 1), 'config [protocol]')
put('nsnap', str(int(round(pp['T'] / pp['dt_snap'])) + 1), 'config [protocol]: T/dt + 1')
put('Bmem_txt', pw(bp['classical_mem']), 'config')
put('Bflop_txt', pw(bp['classical_flop']), 'config')
put('Qcap_txt', f"{bp['qubits']}", 'config')
put('Tcap_txt', pw(bp['t_gates']), 'config')
put('eps_prec', pw(bp['eps']), 'config')
put('sweeps', str(cfg['tn_wall']['sweeps']), 'config')
put('Asw', str(2 * cfg['tn_wall']['sweeps']), 'config')
put('vecs', str(cfg['ed_wall']['vectors']), 'config')
put('d_mode', str(cfg['tn_wall']['d_mode']), 'config')
put('d_grid', str(cfg['tn_wall']['d_grid']), 'config')

# --------------------------------------------------------------------------- quantum model
qm = Q['meta']
put('mut0', f(qm['mu_tilde_0'], 0), 'quantum.npz')
put('Szero', f(qm['S0'], 2), 'quantum.npz')
put('muzero', f(qm['mu_0'], 3), 'quantum.npz')
put('tstar0', f(qm['t_star_0'], 1), 'quantum.npz')
put('dE2grid', f(qm['dE2_grid'], 3), 'quantum.npz')
put('NQ', str(qm['N_Q']), 'quantum.npz')
put('n_p', str(qm['n_p']), 'quantum.npz')
put('b_aux', str(qm['b_aux']), 'quantum.npz')
put('C_sel', str(qm['C_sel']), 'quantum.npz')
put('lam_U', f(qm['lam_U'], 1), 'quantum.npz')
put('lam_W', f(qm['lam_W'], 0), 'quantum.npz')
put('lam_kin', f(qm['lam_kin'], 0), 'quantum.npz')
put('lam_two', sci(qm['lam2'], 1), 'quantum.npz')
put('mut_s26', f(qm['mu_tilde_s26'], 1), 'quantum.npz')
for N in (4, 8, 10, 16, 24, 40, 88):
    r = qrow[N]
    put(f'edge_N{N}', f(r['edge_qub'], 1), 'quantum.npz')
    put(f'edge_N{N}_full', f(r['edge_qub_complete'], 1), 'quantum.npz')
    put(f'Qb_N{N}', str(r['Q']), 'quantum.npz')
    put(f'Tqub_N{N}', sci(r['T_qub_mu0'], 1), 'quantum.npz')
    put(f'Ttrot_N{N}', sci(r['T_trot_mu0'], 1), 'quantum.npz')
    put(f'tstar_edge_N{N}', sci(r['tstar_at_edge'], 1), 'quantum.npz')
    put(f'edge_trot_N{N}', f(r['edge_trot'], 1) if r['edge_trot'] == r['edge_trot'] and r['edge_trot'] else 'none', 'quantum.npz')
put('edge_drop_8_88', f(qrow[8]['edge_qub'] - qrow[88]['edge_qub'], 1), 'quantum.npz')
put('edge_shift_8', f(qrow[8]['edge_qub'] - qrow[8]['edge_qub_complete'], 1), 'quantum.npz')
put('tstar_ratio_8', f"{qrow[8]['tstar_at_edge'] / qm['t_star_0']:.0f}", 'quantum.npz')
put('s_edge_8', f"{(qrow[8]['edge_qub'] / 3) ** 2:.0f}", 'quantum.npz')
put('s_edge_24', f"{(qrow[24]['edge_qub'] / 3) ** 2:.0f}", 'quantum.npz')
put('mu_budget_shift', f(1.5 * math.log(1e3), 1), 'formula (3/2) ln 10^3')
# T-gate ratio of the Trotter model to qubitization at N=4
put('trot_ratio_4', sci(qrow[4]['T_trot_mu0'] / qrow[4]['T_qub_mu0'], 1), 'quantum.npz')

# --------------------------------------------------------------------------- measured entropies
# The particle numbers of the per-N keys and rows: the established set N = 2..5 (a missing run is shown as pending) and every other
# N that has a correlated-start run at k = 12 and the tilt of config.toml [tilts] (N = 6: the run of the rerun).
N_BASE = (2, 3, 4, 5)
_untilted = sorted({r['N'] for r in gr if r['k'] == 12 and r['role'] == 'primary' and tilt(r['N']) is None})
if _untilted:
    raise KeyError(f"graph records exist for N={_untilted} at k=12 but config.toml has no [tilts.N<N>] k12 for them")
Ns = sorted(set(N_BASE) | {r['N'] for r in gr if r['k'] == 12 and r['role'] == 'primary' and abs(r['c'] - tilt(r['N'])) < 1e-9})
PAIRS = list(zip(Ns[:-1], Ns[1:]))                    # successive particle numbers (increments)
# The (N, k) cells of tab_k that the rerun is meant to fill: a missing one is shown as pending, the others as '--'.
PLANNED_K = {2: (12, 16, 20), 3: (12, 16, 20), 4: (12, 16, 20), 5: (12, 16, 20), 6: (12,)}
for N in Ns:
    g = G(N)
    r = rec(N)
    if g is None:
        for key in ('Snat', 'Sperp', 'Smax', 'S1', 'w', 'wmin', 'wmax', 'Wg', 'Wu', 'Omin', 'Omax'):
            put(f'{key}_N{N}', PEND(f'N={N} run'))
        continue
    src = f"{g['protocol']} run {g['file']}"
    put(f'Snat_N{N}', f(g['S_nat']), src)
    put(f'S{N}_cen', f(g['S_nat']), src)
    put(f'Sperp_N{N}', f(g['S_perp']), src)
    put(f'Smax_N{N}', f(g['S_max']), src)
    put(f'S1_N{N}', f(g['S1mean']), src)
    put(f'w_N{N}', f(g['w_mean']), src)
    put(f'wmin_N{N}', f(g['w_min']), src)
    put(f'wmax_N{N}', f(g['w_max']), src)
    put(f'Wg_N{N}', f(g['W_graph']), src)
    put(f'Wu_N{N}', f(g['W_unif']), src)
    put(f'Omin_N{N}', f(g['O_min']), src)
    put(f'Omax_N{N}', f(g['O_max']), src)
    # revival: the largest <O> after it has first fallen below one half (O_max above is the largest value in the
    # window, which is the initial value and says nothing about a return)
    _fn = os.path.join(REPO, 'data', 'runs', g['file'])
    if os.path.exists(_fn):
        _O = np.asarray(store.load_run(_fn)['O'])
        put(f'Orev_N{N}', f(_O[int(np.argmax(_O < 0.5)):].max()), src + ' (largest <O> after it first falls below 1/2)')
    else:
        put(f'Orev_N{N}', PEND(f'run {g["file"]}'))
    put(f'tcen_N{N}', f(g['t_cen'], 0) if g['t_cen'] is not None else '--', src)
    put(f'cstar_N{N}', f(g['c'], 2), src)
    put(f'monomin_N{N}', f(min(g['monogamy'])), src)
    put(f'monomax_N{N}', f(max(g['monogamy'])), src)
    d = [s_['exact'] - s_['graph'] for s_ in g['scatter']]
    put(f'devmin_N{N}', f(min(d)), src)
    put(f'devmax_N{N}', f(max(d)), src)
    lay = g['layout']
    put(f'layout_gain_N{N}', f(lay['worst_exact'] - lay['best_exact']) if 'worst_exact' in lay else '--', src)
    put(f'spearman_N{N}', f(lay['spearman']) if lay.get('spearman', float('nan')) == lay.get('spearman') else '--', src)
    put(f'norders_N{N}', str(lay.get('n_orders', '--')), src)
    put(f'natbest_N{N}', 'yes' if lay.get('natural_exact', 9) <= lay.get('best_exact', -1) + 1e-9 else 'no', src)
    put(f'bracket_N{N}', f(g['S_max'] - g['S_perp']), src)
    put(f'gbg_N{N}', f(lay['exact_cost_of_graph_best'] - lay['best_exact']) if 'best_exact' in lay else '--', src)
    put(f'dag_N{N}', dag(N), 'dagger if the point is from the earlier protocol')
    if r:
        put(f'chi3_N{N}', str(r.get('chi3', '--')), 'ledger')
        if 'chi3' in r:
            put(f'twoS_N{N}', f(2 ** g['S_nat'], 1), 'ledger')
            put(f'chiratio_N{N}', f(r['chi3'] / 2 ** g['S_nat'], 1), 'ledger')
            put(f'chi2_N{N}', str(r['chi2']), 'ledger')
            put(f'chi6_N{N}', str(r['chi6']), 'ledger')
            put(f'S2_N{N}', f(r['S2_cen']), 'ledger')
        put(f'T_N{N}', f(r['T'], 0), 'ledger')
        put(f'dE_N{N}', f(r['dE'], 3), 'ledger')
        put(f'ov_N{N}', f(r['ov'], 2), 'ledger')
        put(f'dbl_N{N}', 'yes' if r['doublet'] == 'yes' else 'no', 'ledger')
    put(f'nedge_N{N}', str(g['n_edge']), src + ': cuts whose maximum is at the last snapshot')
    put(f'ncuts_N{N}', str(g['n_cuts']), src + ': number of subsets with |A| <= N/2')
for N in Ns:
    for k in (12, 16, 20):
        r = rec(N, k)
        if r is None:
            continue
        put(f'S_N{N}_k{k}', f(r['S_cen']), 'ledger')
        if r.get('doublet') is None:
            continue
        put(f'dE_N{N}_k{k}', f(r['dE'], 3), 'ledger')
        put(f'ov_N{N}_k{k}', f(r['ov'], 2), 'ledger')
        put(f'dbl_N{N}_k{k}', r['doublet'], 'ledger')
        put(f'cstar_N{N}_k{k}', f(r['c'], 2), 'ledger')
        put(f'gN_N{N}_k{k}', f(float(qm['dE2_grid']) / r['dE'], 2), 'ledger and quantum.npz (dE2 grid / dE_N)')
_g = [float(NUM[k]) for k in NUM if k.startswith('gN_N') and NUM.get('dbl_' + k[3:]) == 'yes']
if _g:
    put('gN_lo', f(min(_g)), 'ledger (doublet rows)')
    put('gN_hi', f(max(_g)), 'ledger (doublet rows)')
for a_, b_ in PAIRS:
    if G(a_) and G(b_):
        put(f'dS1_{a_}{b_}', f(G(b_)['S1mean'] - G(a_)['S1mean']), 'graph records')
# Sharing-law test on the stored runs (no new simulation): measured central entropy against b(N-b)/(N-1) * S1mean, b = N//2, with the
# run's own S1mean.  Rows: uniform-protocol primary runs of the ledger at the resonant tilts, for each start.
def _share_ratio(r):
    b_ = r['N'] // 2
    return float(r['S_nat']) / (b_ * (r['N'] - b_) / (r['N'] - 1) * float(r['S1mean']))
_LR = {(r['N'], r['k'], r['start']): r for r in led if r['role'] == 'primary' and r['protocol'] == 'uniform'}
_rc = [_share_ratio(r) for (N_, k_, st_), r in _LR.items() if st_ == 'correlated']
put('shr_lo', f(min(_rc)), 'ledger: min over correlated primary uniform runs of S_nat / sharing law fed with the run\'s own S1mean')
put('shr_hi', f(max(_rc)), 'ledger: max over the same runs')
put('shr_n', str(len(_rc)), 'ledger: number of runs')
put('shr_Nmax', str(max(N_ for (N_, k_, st_) in _LR if st_ == 'correlated')), 'ledger')
for N_ in (3, 4):
    if (N_, 12, 'product') in _LR:
        put(f'shr_prod_N{N_}', f(_share_ratio(_LR[(N_, 12, 'product')])), 'ledger: product start, k=12')
if (4, 12, 'product') in _LR:
    put('Scen_prod_N4', f(float(_LR[(4, 12, 'product')]['S_nat'])), 'ledger: product start, k=12')
    put('Scen_corr_N4', f(float(_LR[(4, 12, 'correlated')]['S_nat'])), 'ledger: correlated start, k=12')
for k_ in (16, 20):
    _s1 = {N_: float(_LR[(N_, k_, 'correlated')]['S1mean']) for N_ in (2, 3, 4, 5) if (N_, k_, 'correlated') in _LR}
    if len(_s1) == 4:
        for a_, b_ in ((2, 3), (3, 4), (4, 5)):
            put(f'dS1k{k_}_{a_}{b_}', f(_s1[b_] - _s1[a_]), f'ledger: k={k_}')
for k_ in (12, 16, 20):
    if (5, k_, 'correlated') in _LR and (4, k_, 'correlated') in _LR:
        put(f'S1_N5_k{k_}', f(float(_LR[(5, k_, 'correlated')]['S1mean'])), f'ledger: k={k_}')
        put(f'dScen_45_k{k_}', f(float(_LR[(5, k_, 'correlated')]['S_nat']) - float(_LR[(4, k_, 'correlated')]['S_nat'])), f'ledger: k={k_}')
for N in Ns:
    if G(N) and N > 1:
        put(f'wshare_N{N}', f(2 * G(N)['S1mean'] / (N - 1)), 'graph records: 2 S1mean/(N-1)')
_mo = [float(NUM[f'monomin_N{N}']) for N in Ns if f'monomin_N{N}' in NUM and not NUM[f'monomin_N{N}'].startswith('\\pending')]
_mx = [float(NUM[f'monomax_N{N}']) for N in Ns if f'monomax_N{N}' in NUM and not NUM[f'monomax_N{N}'].startswith('\\pending')]
put('mono_lo', f(min(_mo)), 'graph records: min over N, i of sum_j I_ij / 2 S_i')
put('mono_hi', f(max(_mx)), 'graph records: max over N, i of sum_j I_ij / 2 S_i')
_m = [r['mO'] for r in led if r['role'] == 'primary' and r['start'] == 'correlated' and r.get('mO') is not None]
put('mO_lo', f(min(_m)), 'ledger: <O> of the ground state at the resonance tilt')
put('mO_hi', f(max(_m)), 'ledger')
put('b_fix', str(cfg['quantum']['b_fixed']), 'config')
put('k_gauss', str(cfg['quantum']['k_gauss']), 'config (Gaussians in the softened pair-potential fit)')
put('chi_ratio_N5', NUM.get('chiratio_N5', PEND('N=5 chi')))
put('Nmax_measured', str(max(L['Ns_measured'])), 'larger_N.npz')
mx = max(abs(x) for N in Ns if G(N) for x in [float(NUM[f'devmin_N{N}']), float(NUM[f'devmax_N{N}'])])
put('graph_maxdev', f(mx), 'graph records')
put('graph_maxunder', f(max(float(NUM[f'devmax_N{N}']) for N in Ns if G(N))), 'graph records: largest underestimate of an exact entropy by the graph')
for a_, b_ in PAIRS:
    if G(a_) and G(b_):
        put(f'd{a_}{b_}', f(G(b_)['S_nat'] - G(a_)['S_nat']), 'graph records')
if G(5) and G(2):
    put('savg', f((G(5)['S_nat'] - G(2)['S_nat']) / 3), 'graph records')
# product start against correlated start at k = 12 (both of the uniform protocol)
for N in (2, 3, 4, 5, 6):
    rp, rc = rec(N, 12, 'product'), rec(N, 12, 'correlated')
    if rp:
        put(f'prod_N{N}', f(rp['S_cen']), 'ledger')
    if rc:
        put(f'corr_N{N}', f(rc['S_cen']), 'ledger')
put('d45_note', ' (the last increment uses the $N=5$ point of the earlier protocol; its shorter window can only lower it)' if dag(5) else '',
    'empty: every N is a uniform run')
# the pair law and the linear law as predictions
put('pair_pred_N5', f(L['pair_pred']['5']))
put('pair_pred_N6', f(L['pair_pred']['6']))
put('pair_pred_N8', f(L['pair_pred']['8']))
for _n in (12, 16):
    put(f'pair_pred_N{_n}', f(L['pair_fit4']['S0'] + L['pair_fit4']['beta'] * ((_n // 2) * (_n - _n // 2))))
    put(f'lin_pred_N{_n}', f(L['lin_fit4']['S0'] + L['lin_fit4']['s'] * _n))
put('lin_slope', f(L['lin_fit4']['s']))
put('lin_pred_N6', f(L['lin_pred']['6']))
# The sharing-law value at N=6 is a PREDICTION only while N=6 is not a data point: once it is, the curves are fitted through it and
# coincide there.  run_extrapolation then also stores the curves fitted to the points below it (L['share_pred_prev']): use those.
_prev = L.get('share_pred_prev')


def share_pred(sc_, n):
    return _prev['values'][sc_] if _prev and n == _prev['N'] else L['share_pred'][sc_][str(n)]


put('share_lin_gap_N6', f(max(abs(share_pred(sc_, 6) - L['lin_pred']['6']) for sc_ in ('opt', 'sat', 'pess'))), 'larger_N.npz')
put('lin_pred_N8', f(L['lin_pred']['8']))
for n_ in (5, 6):                                      # the laws fitted to N <= 4 against the measured central bond
    if G(n_):
        put(f'pair_over_N{n_}', f(L['pair_pred'][str(n_)] - G(n_)['S_nat']))
        put(f'lin_over_N{n_}', f(L['lin_pred'][str(n_)] - G(n_)['S_nat']))
for sc_ in ('sat', 'opt', 'pess'):
    for n in (6, 8, 12, 16):
        put(f'share_{sc_}_N{n}', f(share_pred(sc_, n)))
for sc_ in ('sat', 'opt', 'pess'):                     # the N=6 prediction (made from N <= 5) against the measured central bond
    if G(6):
        put(f'share_over_{sc_}_N6', f(share_pred(sc_, 6) - G(6)['S_nat']), 'larger_N.npz')
for n_ in (5, 6):                                      # the sharing law fed with the measured one-particle entropy, above the natural-order cut
    if G(n_) and f'Wu_N{n_}' in NUM:
        put(f'Wu_over_N{n_}', f(float(NUM[f'Wu_N{n_}']) - G(n_)['S_nat']), 'graph records')
put('fit_npts', str(len(Ns) - 1), 'larger_N.npz: points N=3.. fitted (the curve is anchored at N=2)')
put('S1_last', f(L['fit']['S1_last']), 'larger_N.npz')
put('fit_Sinf', f(L['fit']['S_inf']))
put('fit_tau', f(L['fit']['tau'], 1))
put('slope_sat', f(L['slope']['sat']))
put('slope_opt', f(L['slope']['opt']))
put('inc_pess', f(L['fit']['inc']))
# N=6: pending until a run exists (with one, the per-N loop above has defined S6_cen)
if 'S6_cen' not in NUM:
    put('S6_cen', PEND('N=6 run'))
for k_ in (16, 20):
    r5k = rec(5, k_)
    if r5k:
        put(f'chi3_N5_k{k_}', str(r5k.get('chi3')), r5k['protocol'])

# --- the protocol and its sensitivity (tab_protocol): one stored series cut to several windows; spacing scan
put('Tw_lo', f(cfg['studies']['windows'][0], 0), 'config [studies]')
put('Tw_mid', f(cfg['studies']['windows'][1], 0), 'config [studies]')
put('Tw_hi', f(cfg['studies']['windows'][2], 0), 'config [studies]')
for N_, rows_ in PT['window'].items():
    for row in rows_:
        T_ = int(row['T'])
        put(f'win_cen_N{N_}_T{T_}', f(row['S_cen'], 3), 'protocol_tables.npz')
        put(f'win_max_N{N_}_T{T_}', f(row['S_max'], 3), 'protocol_tables.npz')
    put(f'tcen_win_N{N_}', f(rows_[-1]['tS_cen'], 0), 'protocol_tables.npz')
for row in PT['spacing']:
    dt_ = f"{row['dt']:g}".replace('.', 'p')
    put(f"sp_cen_N{row['N']}_dt{dt_}", f(row['S_cen'], 3), 'data/studies/dt_scan')
    put(f"sp_one_N{row['N']}_dt{dt_}", f(row['S_one'], 3), 'data/studies/dt_scan')
_sp = {(r['N'], r['dt']): r for r in PT['spacing']}
for N_ in (3, 4):
    if (N_, 0.5) in _sp and (N_, 1.0) in _sp:
        key_ = 'S_one' if N_ == 3 else 'S_cen'
        put(f'sp_gap_N{N_}', f(abs(_sp[(N_, 0.5)][key_] - _sp[(N_, 1.0)][key_]), 3), 'data/studies/dt_scan')
# --------------------------------------------------------------------------- walls
for sc_ in ('pess', 'sat', 'opt'):
    for dk, dn in ((f"d{L['budgets']['d_grid']}", 'grid'), (f"d{L['budgets']['d_mode']}", 'mode')):
        put(f'wall_{sc_}_{dn}', str(L['walls'][f'{sc_}_{dk}_both']), 'larger_N.npz')
        put(f'wall_{sc_}_{dn}_flop', str(L['walls'][f'{sc_}_{dk}_flop']), 'larger_N.npz')
        put(f'wall_{sc_}_{dn}_mem', str(L['walls'][f'{sc_}_{dk}_mem']), 'larger_N.npz')
        put(f'wallchi_{sc_}_{dn}', str(L['walls_chi'][f'{sc_}_{dk}_both']), 'larger_N.npz (chi shift)')
put('chi_shift', f(L['chi_shift'], 1), 'larger_N.npz: log2 of chi(1e-3)/2^S at the largest simulated N')
put('ED_grid', str(L['ed']['grid']), 'larger_N.npz')
put('ED_grid_last', str(L['ed']['grid'] - 1), 'larger_N.npz')
put('ED_mode20', str(L['ed']['mode']), 'larger_N.npz')
put('ED_grid_boson', str(L['ed']['grid_boson']), 'larger_N.npz')
put('ED_mode20_boson', str(L['ed']['mode_boson']), 'larger_N.npz')
for k_, v in L['s_crit'].items():
    dn = 'grid' if k_.startswith(f"d{L['budgets']['d_grid']}") else 'mode'
    put(f"scrit_{dn}_N{k_.split('N')[-1]}", f(v, 2), 'larger_N.npz')
mem4 = 16 * L['budgets']['v'] * float(gp['nx'] * gp['ny']) ** 4
put('ED_mem_N4', sci(mem4, 1))
put('ED_amp_N4', sci(float(gp['nx'] * gp['ny']) ** 4, 1))
put('ED_mem_N24', sci(16 * L['budgets']['v'] * float(gp['nx'] * gp['ny']) ** 24, 0))
put('ED_ratio_N24', sci(16 * L['budgets']['v'] * float(gp['nx'] * gp['ny']) ** 24 / L['budgets']['Bmem'], 0))
# entropy budget for the tensor-network cost model
import importlib
th = importlib.import_module('ljtunnel.theory')
for N in (8, 16, 24):
    put(f'Sbud_grid_N{N}', f(th.S_budget(N, L['budgets']['d_grid'], L['budgets']['Bmem'], L['budgets']['Bflop'],
                                          L['budgets']['sweeps']), 1))
    put(f'Sbud_mode_N{N}', f(th.S_budget(N, L['budgets']['d_mode'], L['budgets']['Bmem'], L['budgets']['Bflop'],
                                          L['budgets']['sweeps']), 1))
# the N = 24 instance: entropy of the central bond under each scenario and the cost against each budget
Bd = L['budgets']
for sc_ in ('opt', 'sat', 'pess'):
    S24 = L['curves'][sc_][24 - 2]
    put(f'S24_{sc_}', f(S24, 1), 'larger_N.npz')
    fl = 2.0 * Bd['sweeps'] * 24 * (Bd['d_grid'] * 2.0 ** S24) ** 3 / Bd['Bflop']
    me = 16.0 * 24 * Bd['d_grid'] * 4.0 ** S24 / Bd['Bmem']
    put(f'TNflop_N24_{sc_}', sci(fl, 1) if fl >= 1000 else (f(fl, 0) if fl >= 10 else f(fl, 1)), 'cost model / budget (grid)')
    put(f'TNmem_N24_{sc_}', sci(me, 1) if me >= 1000 else (f(me, 0) if me >= 10 else f(me, 1)), 'cost model / budget (grid)')
# largest N at which the qubitization edge still reaches a given mass (capped at N_Q)
for m_ in (12, 15, 18):
    for key, lab in (('edge_qub', ''), ('edge_qub_complete', '_full')):
        ok = [r['N'] for r in Q['rows'] if r[key] >= m_]
        put(f'Nhi_{m_}{lab}', str(min(qm['N_Q'], max(ok))) if ok else 'none', 'quantum.npz')
# Monte Carlo identity on the production grid
dx, dy = 2 * gp['Lx'] / gp['nx'], 2 * gp['Ly'] / gp['ny']
Gx = math.pi ** 2 / (3 * dx ** 2) * (1 - 1 / gp['nx'] ** 2)
Gy = math.pi ** 2 / (3 * dy ** 2) * (1 - 1 / gp['ny'] ** 2)
put('Ginf2D', f(Gx + Gy, 1), 'formula (Prop. 3) on the production grid')
tstar = qm['t_star_0']
put('MC_exp_N24', sci(2 * 24 * (Gx + Gy) * tstar - 24 * math.log(gp['nx'] * gp['ny']), 1))
# MC ledger (1D identity)
ml = [ln.split() for ln in open(os.path.join(RAW, 'mc_gamma_ledger.txt')) if ln.startswith(('ginf', 'p_UV'))]
put('p_UV', f(float([m for m in ml if m[0] == 'p_UV'][0][1]), 2), 'mc_gamma_ledger.txt')
put('Ginf128', f(float([m for m in ml if m[0] == 'ginf' and m[1] == '128'][0][3]), 1), 'mc_gamma_ledger.txt')

# --------------------------------------------------------------------------- off-resonance control
off = None
for r in gr:
    if r['N'] == 5 and r['k'] == 12 and r['role'] == 'primary' and abs(r['c'] - cfg['control']['N5_off_c']) < 1e-6:
        off = r
if off:
    put('off_c', f(off['c'], 2), 'cutdata')
    put('off_Snat', f(off['S_nat']), 'cutdata')
    put('dag_off', '' if off['protocol'] == 'uniform' else '$^\\dagger$', 'dagger if from the earlier protocol')
    put('off_Omin', f(off['O_min']), 'cutdata')
    put('off_Smax', f(off['S_max']), 'cutdata')
    put('off_S1', f(off['S1mean']), 'cutdata')
    put('off_w', f(off['w_mean']), 'cutdata')
    put('off_wmin', f(off['w_min']), 'cutdata')
    put('off_wmax', f(off['w_max']), 'cutdata')
    put('off_Wg', f(off['W_graph']), 'cutdata')
    put('off_Sperp', f(off['S_perp']), 'cutdata')
    put('off_tcen', f(off['t_cen'], 0) if off['t_cen'] is not None else '--', 'cutdata')
    put('off_T', f(off['T'], 0), 'cutdata')
    put('off_lb', ', so the off-resonant values are lower bounds' if off['t_cen'] is not None and off['t_cen'] >= 0.9 * off['T'] else '', 'cutdata: central maximum within 10% of the window end')
    put('off_note', ' This control is a result of the earlier protocol ($\\dagger$) and is replaced when it is rerun.' if off['protocol'] != 'uniform' else '', 'protocol of the off-resonance run')
    put('off_rho', f(off['layout']['spearman']), 'cutdata')
    put('off_diff', f(G(5)['S_nat'] - off['S_nat']), 'cutdata (on-resonance minus off-resonance)')
    put('off_dev', f(max(abs(x['exact'] - x['graph']) for x in off['scatter'])), 'cutdata')
else:
    put('off_lb', ''); put('off_note', '')
    for key in ('off_c', 'off_Snat', 'off_Omin', 'off_Smax', 'off_S1', 'off_w', 'off_wmin', 'off_wmax', 'off_Wg', 'off_Sperp', 'off_tcen', 'off_T', 'off_rho', 'off_diff', 'off_dev'):
        put(key, PEND('off-resonance N=5 log'))
put('on_Omin_N5', NUM.get('Omin_N5', PEND('N=5')))

# --------------------------------------------------------------------------- write numbers.tex

# ---- why these model parameters (the earlier one-input-at-a-time study; data/derived/parameter_choice.npz)
_PCsrc = 'parameter_choice.npz <- ' + PC['source']
_sens = {r['label']: r for r in PC['sensitivity']}
_one = {int(r['A']): r for r in PC['onebody']}
_ok = lambda r: r['pass_overlap'] and r['pass_cap']
put('ch_ov_min', f"{PC['limits']['overlap_min']:g}", _PCsrc)
put('ch_cap_max', f"{PC['limits']['cap_max']:g}", _PCsrc)
_Aok = [int(r['A']) for r in PC['sensitivity'] if r['label'].startswith('A=') and _ok(r)] + [int(_sens['production point']['A'])]
put('ch_A_deep', str(max(_Aok)), _PCsrc)
put('ch_A4_ov', f(_sens['A=4']['ov_sub'], 3), _PCsrc)
put('ch_A6_ov', f(_sens['A=6']['ov_sub'], 3), _PCsrc)
put('ch_A10_cap', f(_sens['A=10']['cap_shift'], 2), _PCsrc)
put('ch_A12_cap', f(_sens['A=12']['cap_shift'], 2), _PCsrc)
put('ch_A6_cap', f(_sens['A=6']['cap_shift'], 2), _PCsrc)
put('ch_prod_ov', f(_sens['production point']['ov_sub'], 3), _PCsrc)
put('ch_prod_dE', f(_sens['production point']['dE'], 4), _PCsrc)
put('ch_prod_cap', f(_sens['production point']['cap_shift'], 3), _PCsrc)
put('ch_sig05_ov', f(_sens['sigma=0.5']['ov_sub'], 2), _PCsrc)
put('ch_C0_0_ov', f(_sens['C0=0']['ov_sub'], 2), _PCsrc)
put('ch_C0_0_cap', f(_sens['C0=0']['cap_shift'], 2), _PCsrc)
put('ch_C0_05_ov', f(_sens['C0=0.05']['ov_sub'], 2), _PCsrc)
put('ch_C0_05_cap', f(_sens['C0=0.05']['cap_shift'], 2), _PCsrc)
put('ch_cap25_cap', f(_sens['cap=25']['cap_shift'], 3), _PCsrc)
_eps = [r for r in PC['sensitivity'] if r['label'].startswith('eps=')] + [_sens['production point']]
put('ch_eps_dE_lo', f(min(r['dE'] for r in _eps), 4), _PCsrc)
put('ch_eps_dE_hi', f(max(r['dE'] for r in _eps), 4), _PCsrc)
put('ch_eps_ov_lo', f(min(r['ov_sub'] for r in _eps), 3), _PCsrc)
put('ch_eps_ov_hi', f(max(r['ov_sub'] for r in _eps), 3), _PCsrc)
_eo_lo, _eo_hi = f(min(r['ov_sub'] for r in _eps), 3), f(max(r['ov_sub'] for r in _eps), 3)
put('ch_eps_ov_rng', _eo_lo if _eo_lo == _eo_hi else f'{_eo_lo}--{_eo_hi}', _PCsrc)
put('ch_eps_c_lo', f(min(r['c_star'] for r in _eps), 2), _PCsrc)
put('ch_eps_c_hi', f(max(r['c_star'] for r in _eps), 2), _PCsrc)
put('ch_eps_n', str(len(_eps)), _PCsrc)
_dEfine_gap = max(abs(r['dE_coarse'] / r['dE_fine'] - 1) for r in PC['onebody'])
put('ch_grid_gap', f(100 * _dEfine_gap, 1), _PCsrc)
put('ch_A8_dE1', f(_one[8]['dE_fine'], 4), _PCsrc)
for _A in (10, 12, 16):
    put(f'ch_t_A{_A}', f(math.pi / _one[_A]['dE_fine'], 0), _PCsrc)
put('ch_A8_t1', f(math.pi / _one[8]['dE_fine'], 1), _PCsrc)


# ---- the splitting-based rival: one cosine at the doublet splitting against the stored signal <O>(t)
_CCsrc = 'cosine_check.npz (residual of one cosine, as a fraction of the standard deviation of <O>(t))'
_cc = {(r['N'], r['k']): r for r in CC}
for _N in (2, 3):
    _v = [r['fixed'] for r in CC if r['N'] == _N]
    put(f'cr_N{_N}_lo', f(min(_v), 2), _CCsrc)
    put(f'cr_N{_N}_hi', f(max(_v), 2), _CCsrc)
    _w = [r['free'] for r in CC if r['N'] == _N]
    put(f'cr_N{_N}_free_lo', f(min(_w), 2), _CCsrc)
    put(f'cr_N{_N}_free_hi', f(max(_w), 2), _CCsrc)
put('cr_N3_nk', str(sum(1 for r in CC if r['N'] == 3)), _CCsrc)
for _k in (12, 16):
    for _s, _key in (('fix', 'fixed'), ('free', 'free')):
        put(f'cr_N5_k{_k}_{_s}', f(_cc[(5, _k)][_key], 2) if (5, _k) in _cc else PEND(f'N=5, k={_k} run'), _CCsrc)
for _N, _k in ((5, 20), (6, 12)):                      # runs of the rerun: defined once they exist
    if (_N, _k) in _cc:
        put(f'cr_N{_N}_k{_k}_fix', f(_cc[(_N, _k)]['fixed'], 2), _CCsrc)
        put(f'cr_N{_N}_k{_k}_free', f(_cc[(_N, _k)]['free'], 2), _CCsrc)
        put(f'cr_N{_N}_k{_k}_wratio', f(_cc[(_N, _k)]['w_free'] / _cc[(_N, _k)]['dE'], 1), _CCsrc)   # best frequency / doublet splitting
for _k in (12, 16, 20):
    if (4, _k) in _cc:
        put(f'cr_N4_k{_k}_fix', f(_cc[(4, _k)]['fixed'], 2), _CCsrc)
        put(f'cr_N4_k{_k}_free', f(_cc[(4, _k)]['free'], 2), _CCsrc)


# ---- the Theta(x = 0) convention: every run next to the same run of the earlier convention (data/derived/theta_effect.csv),
# and the first failing N recomputed from the earlier runs (previous-data/runs)
import csv                                              # noqa: E402
_TE = [{k: (float(v) if k not in ('start',) and v not in ('',) else v) for k, v in r.items()}
       for r in csv.DictReader(open(os.path.join(DER, 'theta_effect.csv')))]
_TEcor = [r for r in _TE if r['start'] == 'interacting']
_TEsrc = 'theta_effect.csv (run_theta_effect: data/runs against previous-data/runs)'
put('th_n', str(len(_TEcor)), _TEsrc)
put('th_dS_cen_max', f(max(abs(r['dS_cen']) for r in _TEcor), 3), _TEsrc)
put('th_dS_max_max', f(max(abs(r['dS_max']) for r in _TEcor), 3), _TEsrc)
put('th_mO_lo', f(min(r['mO_new'] - r['mO_old'] for r in _TE), 3), _TEsrc)
put('th_mO_hi', f(max(r['mO_new'] - r['mO_old'] for r in _TE), 3), _TEsrc)
put('th_chi_max', str(int(max(abs(r['chi3_new'] - r['chi3_old']) for r in _TEcor))), _TEsrc)
_te6 = [r for r in _TEcor if r['N'] == 6][0]
put('th_N6_chi_old', str(int(_te6['chi3_old'])), _TEsrc)
put('th_N6_chi_new', str(int(_te6['chi3_new'])), _TEsrc)
put('th_N6_S_old', f(_te6['S_cen_old'], 3), _TEsrc)
put('th_N6_S_new', f(_te6['S_cen_new'], 3), _TEsrc)
put('th_dE_max', sci(max(max(abs(r['dE_diff']) for r in _TE), 1e-14), 0), _TEsrc)
put('th_prod_max', sci(max(max(abs(r['dS_cen']) for r in _TE if r['start'] == 'product'), 1e-14), 0), _TEsrc)
_prev_runs = os.path.join(REPO, 'previous-data', 'runs')
_walls_same = None
if os.path.isdir(_prev_runs):
    from ljtunnel import extrapolation                  # noqa: E402

    def _walls(runs_dir, theta):
        led_, gr_ = records.collect(runs_dir, theta_barrier=theta, controls_dir=None)
        k_ = cfg['basis']['k']
        S1_ = {}
        for g in gr_:
            if g['role'] == 'primary' and g['k'] == k_:
                t_ = cfg.get('tilts', {}).get(f"N{g['N']}")
                t_ = t_.get(f'k{k_}') if isinstance(t_, dict) else t_
                if t_ is not None and abs(g['c'] - t_) < 1e-6:
                    S1_[g['N']] = g['S1mean']
        pts_ = {N: records.pick(led_, N) for N in sorted(S1_)}
        Ns_ = [N for N in sorted(S1_) if pts_[N]]
        o_ = extrapolation.scenarios(cfg, Ns_, [S1_[N] for N in Ns_], [pts_[N]['S_cen'] for N in Ns_],
                                     chi_last=(pts_[Ns_[-1]].get('chi3'), pts_[Ns_[-1]]['S_cen']))
        return Ns_, {k: v for k, v in o_['walls'].items() if k.endswith('_both')}
    _Nold, _wold = _walls(_prev_runs, 0.0)
    _walls_same = (_Nold == Ns and all(L['walls'][k] == v for k, v in _wold.items()))
put('th_walls_same', 'yes' if _walls_same else ('no' if _walls_same is not None else PEND('previous-data/runs')), 'recomputed from previous-data/runs')

# ---- the switching time t_0: the quoted value (cost model) against the full grid at its own tilt and at the exact resonance
_FGr = store.load_object(os.path.join(DER, 'fullgrid_resonance.npz'))
_FGp = store.load_object(os.path.join(DER, 'fullgrid_N2.npz'))
_FGsrc = 'fullgrid_resonance.npz / fullgrid_N2.npz'
put('tstar0_2', f(qm['t_star_0'], 2), 'quantum.npz')
put('c_res', f(_FGr['c'], 3), _FGsrc)
put('mO_res', f(_FGr['mO'], 3), _FGsrc)
put('dE_res', f(_FGr['dE'], 4), _FGsrc)
put('t0_res', f(_FGr['t0'], 2), _FGsrc)
put('c_prod', f(_FGp['c'], 2), _FGsrc)
put('mO_prod', f(_FGp['mO'], 3), _FGsrc)
put('dE_prod', f(_FGp['dE'], 4), _FGsrc)
put('t0_prod', f(_FGp['t0'], 2), _FGsrc)
put('t0_dpct', f(100 * abs(_FGr['t0'] / qm['t_star_0'] - 1), 1), _FGsrc)
put('t0_dpct_prod', f(100 * abs(_FGp['t0'] / qm['t_star_0'] - 1), 1), _FGsrc)
# the upper edge depends on t_0 through (3/2) ln t_0 (Eq. muqub; checked against quantum.edge_table with t_star_0 = 16.37 / 16.57: +0.014 / -0.005)
put('t0_edge_shift', f(1.5 * abs(math.log(_FGr['t0'] / qm['t_star_0'])), 2), 'Eq. (muqub), checked numerically')


# ---- does the barrier chosen at N=2 carry over?  static scans at N=3, 4 (k=12): resonance re-found for each A
PS = {}
for _N in (3, 4):
    _p = os.path.join(DER, f'parameter_scan_N{_N}.npz')
    if os.path.exists(_p):
        PS[_N] = {int(r['A']): r for r in store.load_object(_p)['rows'] if r['found']}
_PSsrc = 'parameter_scan_N*.npz (static scan: resonance found again for each A, k=12, correlated start)'
for _N, _d in PS.items():
    for _A, _r in _d.items():
        put(f'chs_N{_N}_A{_A}_ov', f(_r['ov_int'], 2), _PSsrc)
        put(f'chs_N{_N}_A{_A}_dE', f(_r['dE'], 3), _PSsrc)
        put(f'chs_N{_N}_A{_A}_c', f(_r['c'], 2), _PSsrc)
_pass3 = sorted(A for A, r in PS.get(3, {}).items() if r['doublet_ok'] and r['ov_int'] >= PC['limits']['overlap_min'])
put('chs_N3_pass_list', ', '.join(str(a) for a in _pass3), _PSsrc)
put('chs_N3_pass_lo', str(min(_pass3)) if _pass3 else PEND('N=3 scan'), _PSsrc)
if 4 in PS:
    _pass4 = sorted(A for A, r in PS[4].items() if r['doublet_ok'] and r['ov_int'] >= PC['limits']['overlap_min'])
    put('chs_N4_pass_list', ', '.join(str(a) for a in _pass4) if _pass4 else 'none', _PSsrc)


# ---- a matrix-product state, measured (data/mps, tables in data/derived/tn_tables.npz)
_TNsrc = 'tn_tables.npz (matrix-product-state runs against exact runs of the same case; data/mps)'
_val = {r['case']: r for r in TN['tables']['mps_validation']}
for _c, _r in _val.items():
    put(f'tn_{_c}_dS', sci(_r['max_dS'], 1) if _r['max_dS'] > 0 else '0', _TNsrc)
    put(f'tn_{_c}_slow', f(_r['wall_mps_s'] / _r['wall_exact_s'], 0), _TNsrc)
_k12 = _val['N4_k12']
put('tn_k12_Smps', f(_k12['S_cen_mps'], 3), _TNsrc)
put('tn_k12_Sex', f(_k12['S_cen_exact'], 3), _TNsrc)
put('tn_k12_dSc', f(_k12['S_cen_exact'] - _k12['S_cen_mps'], 3), _TNsrc)
put('tn_k12_chi3m', str(int(_k12['chi3_mps'])), _TNsrc)
put('tn_k12_chi3e', str(int(_k12['chi3_exact'])), _TNsrc)
put('tn_k12_chi6m', str(int(_k12['chi6_mps'])), _TNsrc)
put('tn_k12_chi6e', str(int(_k12['chi6_exact'])), _TNsrc)
put('tn_k12_chimax', str(int(_k12['chi_max'])), _TNsrc)
put('tn_k12_T', str(int(_k12['T'])), _TNsrc)
put('tn_k12_R', str(int(_k12['channel_R'])), _TNsrc)
put('tn_k12_D', str(int(_k12['channel_R']) + 2), _TNsrc)
_cr6 = [r for r in TN['tables']['channel_ranks'] if r['N'] == 6 and r['k'] == 12 and abs(r['tol'] - cfg['mps']['channel_tol']) < 1e-12]
put('tn_k12_D6', str(int(_cr6[0]['D_mpo'])), _TNsrc)
_slows = [_r['wall_mps_s'] / _r['wall_exact_s'] for _r in _val.values()]
put('tn_slow_lo', f(min(_slows), 0), _TNsrc)
put('tn_slow_hi', f(max(_slows), 0), _TNsrc)
_sc = {int(r['chi_cap']): r for r in TN['tables']['step_cost'] if r['k'] == 12}
put('tn_step_lo', f(_sc[16]['sec_per_step'], 1), _TNsrc)
put('tn_step_hi', f(_sc[64]['sec_per_step'], 1), _TNsrc)
put('tn_step_mv2', str(int(_sc[16]['matvec2_per_step'])), _TNsrc)
put('tn_step_mv1', str(int(_sc[16]['matvec1_per_step'])), _TNsrc)
put('tn_step_frac_lo', f(100 * min(r['frac_in_matvec'] for r in _sc.values()), 0), _TNsrc)
put('tn_step_frac_hi', f(100 * max(r['frac_in_matvec'] for r in _sc.values()), 0), _TNsrc)
put('tn_step_ratio16', sci(_sc[16]['ratio_measured_to_model'], 1), _TNsrc)
put('tn_step_ratio64', f(_sc[64]['ratio_measured_to_model'], 0), _TNsrc)
_rk = {(r['N'], r['k']): r for r in TN['tables']['rank_vs_entropy'] if r['start'] == 'interacting' and r['T'] == 200 and r['protocol'] == 'uniform'}
for (_N, _k), _r in _rk.items():
    if _k in (12, 16, 20):
        put(f'tn_rk_N{_N}_k{_k}', f(_r['ratio3'], 1), _TNsrc)


# ============================================================================ follow-up spectra (data/followup)
# Static spectra of the correlated start (Stage 1), the static checks and the multi-cosine analysis of the stored runs.
# All macros are prefixed fu_ ; the source column names the file.
sys.path.insert(0, HERE)
import followup_data as fd  # noqa: E402

_S1 = fd.stage1()
_S1src = 'data/followup/stage1/N*_k*_c*.npz'
for (_N, _k), _r in _S1.items():
    if _k < 12:
        continue
    put(f'fu_B2_N{_N}_k{_k}', f(_r['B2'], 3), _S1src)
    put(f'fu_dress_N{_N}_k{_k}', f(1 - _r['B2'], 3), _S1src)
    put(f'fu_J_N{_N}_k{_k}', f(_r['J'], 3), _S1src)
    put(f'fu_cs_N{_N}_k{_k}', f(_r['c'], 2), _S1src + ' (tilt of the file name)')
    put(f'fu_near_N{_N}_k{_k}', f(_r['near'], 3), _S1src)
    put(f'fu_mid_N{_N}_k{_k}', f(_r['mid'], 3), _S1src)
    put(f'fu_far_N{_N}_k{_k}', f(_r['far'], 3), _S1src)
    put(f'fu_cov_N{_N}_k{_k}', f(_r['cov'], 3), _S1src)
# k-convergence of the weight on the pair at N=4, and the same quantity at N=5 (k=24 from the Lanczos list only)
put('fu_B2_N4_kmax', NUM['fu_B2_N4_k32'], _S1src)
put('fu_dress_N4_kmax', NUM['fu_dress_N4_k32'], _S1src)
put('fu_N4_dB2_32_20', f(_S1[(4, 32)]['B2'] - _S1[(4, 20)]['B2'], 3), _S1src)
put('fu_N4_dB2_32_12', f(_S1[(4, 32)]['B2'] - _S1[(4, 12)]['B2'], 3), _S1src)
put('fu_N5_dB2_24_20', f(_S1[(5, 24)]['B2'] - _S1[(5, 20)]['B2'], 3), _S1src)
put('fu_N5_cov_k24', f(_S1[(5, 24)]['cov'], 2), _S1src)
put('fu_N4_Jlo', f(min(_S1[(4, k_)]['J'] for k_ in (20, 24, 28, 32)), 3), _S1src)
put('fu_N4_Jhi', f(max(_S1[(4, k_)]['J'] for k_ in (20, 24, 28, 32)), 3), _S1src)
put('fu_N5_dress_over_N4_k20', f((1 - _S1[(5, 20)]['B2']) / (1 - _S1[(4, 20)]['B2']), 1), _S1src)
# resonance tilt per particle
_cs = {k_: [(N_, _S1[(N_, k_)]['c']) for N_ in range(2, 7) if (N_, k_) in _S1] for k_ in (12, 16, 20)}
for k_ in (12, 20):
    _x, _y = zip(*_cs[k_])
    _sl, _ic = np.polyfit(_x, _y, 1)
    put(f'fu_cslope_k{k_}', f(_sl, 2), _S1src + ' (least-squares line through c*(N))')
    put(f'fu_cstep_lo_k{k_}', f(min(np.diff(_y)), 2), _S1src)
    put(f'fu_cstep_hi_k{k_}', f(max(np.diff(_y)), 2), _S1src)
_ctop = fd.c_top()
put('fu_ctop', f(_ctop, 1), 'followup_data.c_top (one-particle tilted well, finite differences)')
for k_ in (12, 20):
    _x, _y = zip(*_cs[k_])
    _sl, _ic = np.polyfit(_x, _y, 1)
    put(f'fu_Ntop_k{k_}', f((_ctop - _ic) / _sl, 0), 'line through c*(N) reaching fu_ctop')
put('fu_Ntop_lo', NUM['fu_Ntop_k12'] if float(NUM['fu_Ntop_k12']) <= float(NUM['fu_Ntop_k20']) else NUM['fu_Ntop_k20'], 'min over the two lines')
put('fu_Ntop_hi', NUM['fu_Ntop_k20'] if float(NUM['fu_Ntop_k12']) <= float(NUM['fu_Ntop_k20']) else NUM['fu_Ntop_k12'], 'max over the two lines')

# parity, sector exclusion and label fragmentation from the saved level lists
_LV = fd.levels()
_LVsrc = 'data/followup/static_checks/levels_N*_k*_c*.npz'
for (_N, _k), _d in _LV.items():
    _E, _w, _R, _n = _d['E'], _d['weight'], _d['R'], _d['n']
    _even = _R > 0
    put(f'fu_Rstart_N{_N}_k{_k}', f(float(_d['start_parity']), 4), _LVsrc)
    put(f'fu_oddmax_N{_N}_k{_k}', sci(float(_w[~_even].max()), 1), _LVsrc)
    _q = _n.sum(1)
    _q2 = [i for i in range(len(_E)) if _even[i] and _q[i] >= 1.5]
    if _q2:
        put(f'fu_q2_N{_N}_k{_k}', f(_E[_q2[0]], 2), _LVsrc)
    else:
        put(f'fu_q2_N{_N}_k{_k}', f'>{_E.max():.2f}', _LVsrc)
    for lab in range(1, _N):
        _c = [i for i in range(len(_E)) if _even[i] and _n[i, lab] > 0.5]
        if _c:
            put(f'fu_lab{lab}_N{_N}_k{_k}', f(_E[_c[0]], 2), _LVsrc)
    _hv = sorted(sorted(np.where(_even)[0], key=lambda i: -_w[i])[:2])
    put(f'fu_Jlv_N{_N}_k{_k}', f((_E[_hv[1]] - _E[_hv[0]]) / 2, 3), _LVsrc)
    put(f'fu_nlev_N{_N}_k{_k}', str(len(_E)), _LVsrc)
    put(f'fu_Emax_N{_N}_k{_k}', f(_E.max(), 2), _LVsrc)
put('fu_q2_ratio_N3_k12', f(float(NUM['fu_q2_N3_k12']) / (float(NUM['fu_Jlv_N3_k12']) * 3 ** 0.5), 0), _LVsrc + ' (gap / (J sqrt N))')
put('fu_q2_ratio_N3_k16', f(float(NUM['fu_q2_N3_k16']) / (float(NUM['fu_Jlv_N3_k16']) * 3 ** 0.5), 0), _LVsrc + ' (gap / (J sqrt N))')
put('fu_twoC0', f(2 * cfg['model']['C0'], 2), 'config.toml')
_mg = []
for (_N, _k), _d in _LV.items():
    _E, _R, _n, _w = _d['E'], _d['R'], _d['n'], _d['weight']
    _hv = sorted(sorted(np.where(_R > 0)[0], key=lambda i: -_w[i])[:2])
    _Jv = (_E[_hv[1]] - _E[_hv[0]]) / 2
    for lab in range(1, _N):
        _c = [i for i in range(len(_E)) if _R[i] > 0 and _n[i, lab] > 0.5]
        if _c:
            _mg.append((_E[_c[0]] - _E[_hv[1]]) / _Jv)
put('fu_marg_lo', f(min(_mg), 1), _LVsrc + ' (first level of a moved label, distance to the pair, in units of J)')
put('fu_marg_hi', f(max(_mg), 1), _LVsrc)

# multi-cosine analysis of the stored runs: the number of cosines needed, at the largest k of each N
_CO = fd.cosines()
_COsrc = 'data/followup/freq_signal_analysis.log'
for _N in range(2, 7):
    _rr = sorted([r for r in _CO if r['N'] == _N], key=lambda r: r['k'])
    _r = _rr[-1]
    put(f'fu_cos_k_N{_N}', str(_r['k']), _COsrc)
    put(f'fu_cos_M05_N{_N}', '\\ensuremath{>}8' if _r['M'][2] is None else str(_r['M'][2]), _COsrc)
    put(f'fu_cos_M10_N{_N}', '\\ensuremath{>}8' if _r['M'][1] is None else str(_r['M'][1]), _COsrc)
    put(f'fu_cos_r3_N{_N}', f(_r['resid'][2], 2), _COsrc)

# the table argument: |f'| <= N ||K_x|| on the grid, so f can be tabulated to the allowed error on this many points (per N)
_dx = float(NUM['dx'])
_Kx = math.pi ** 2 / (2 * _dx ** 2)
_eps = float(cfg['budgets']['eps'])
put('win_ratio', f(float(cfg['protocol']['T']) / float(NUM['tstar0']), 0), 'config.toml [protocol] T over t_0 (quantum.npz)')
put('fu_Kx', f(_Kx, 0), 'pi^2 / (2 dx^2), dx from config.toml')
put('fu_tab_points', sci(float(NUM['tstar0']) * _Kx / (2 * _eps), 0), 'T_obs=t_0, step 2 eps / ||K_x|| (Lipschitz bound)')

# untilted dense band
_UT = fd.untilted()
for (_N, _c0), _E in _UT.items():
    if _c0 and _N in (3, 4):
        _gaps = np.diff(_E)
        _big = int(np.argmax(_gaps[:60]))
        put(f'fu_band_n_N{_N}', str(_big + 1), 'data/followup/stage1/untilted_*_C00.npz (level before the largest gap in the lowest 60)')
        put(f'fu_band_w_N{_N}', f(_E[_big], 2), 'same: width of the band')
        put(f'fu_band_gap_N{_N}', f(_gaps[_big], 2), 'same: gap above the band')

with open(os.path.join(ROOT, 'numbers.tex'), 'w') as fo:
    fo.write('% GENERATED by scripts/make_numbers.py -- do not edit.\n')
    for k in sorted(NUM):
        fo.write(f"\\defnum{{{k}}}{{{NUM[k]}}}%  {SRC.get(k, '')}\n")
print(len(NUM), 'macros -> numbers.tex')

# ============================================================================ tables
def tab_params():
    rows = [
        ('Trap', 'barrier strength $A$, $U=A(x^2-1)^2+\\tfrac12\\omega_y^2y^2$', f"${NUM['A']}$"),
        ('', 'transverse $\\omega_y^2$', f"${NUM['wy2']}$"),
        ('Pair potential', 'Lennard-Jones $\\epsilon$, $\\sigma$, cap $V_{\\rm cap}$', f"${NUM['eps_LJ']}$, ${NUM['sigma_LJ']}$, ${NUM['Vcap']}$"),
        ('Particles', 'tilt $c+c_i$, offsets $c_i=(i-1)C_0$, $C_0$', f"${NUM['Czero']}$; $c$ at resonance"),
        ('', 'mass, $\\hbar$ (simulation units)', '$1$, $1$'),
        ('Grid (basis 1)', 'cells $n_x\\times n_y$, $M$; $\\Delta x$, $\\Delta y$', f"${NUM['nx']}\\times{NUM['ny']}$, ${NUM['M_grid']}$; ${NUM['dx']}$, ${NUM['dy']}$"),
        ('Modes (basis 2)', 'modes per particle $k$ (local dimension $d$)', '$12$, $16$, $20$'),
        ('Qubits (basis 3)', 'qubits per particle $n_p$; auxiliary', f"${NUM['n_p']}$; ${NUM['b_aux']}$"),
        ('Starts', 'correlated: penalty $W$ on the wrong side', f"${NUM['Wpen']}$"),
        ('Dynamics', 'Chebyshev propagation; window $T$; snapshot spacing $\\Delta t$ (one protocol for all $N$)', f"${NUM['Tprot']}$; ${NUM['dtsnap']}$ (${NUM['nsnap']}$ snapshots)"),
        ('Working point', 'trap-units mass $\\mut_0$; action $S_0$; switching time $t_0$', f"${NUM['mut0']}$; ${NUM['Szero']}$; ${NUM['tstar0']}$"),
        ('Budgets', 'memory $\\Bmem$; compute $\\Bflop$', f"${NUM['Bmem_txt']}$ B; ${NUM['Bflop_txt']}$ operations"),
        ('', 'logical qubits $\\BQ$; T-gates $\\BT$; precision $\\varepsilon$', f"${NUM['Qcap_txt']}$; ${NUM['Tcap_txt']}$; ${NUM['eps_prec']}$"),
        ('Cost models', 'Krylov vectors $v$; sweeps (compute prefactor $A_{\\rm sw}$)', f"${NUM['vecs']}$; ${NUM['sweeps']}$ (${NUM['Asw']}$)"),
    ]
    s = ["\\begin{table}[ht]", "\\centering", "\\small",
         "\\caption{Parameters of the model, of the numerical representations and of the budgets. All values are read from the single configuration file of the simulations. The trap, the interaction and the budgets are choices, not fits; Appendix~\\ref{app:choice} says why these values were chosen and Appendix~\\ref{app:numerics} what each numerical setting controls.}",
         "\\label{tab:params}", "\\begin{tabular}{L{2.4cm}L{7.4cm}L{4.4cm}}", "\\toprule",
         "Group & Quantity & Value \\\\", "\\midrule"]
    for a, b, c in rows:
        s.append(f"{a} & {b} & {c} \\\\")
    s += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(s)


def _row(label, key, Ns_=(2, 3, 4, 5, 6)):
    """One table row: label and NUM[key.format(N=N)] for each N ('--' where the macro does not exist)."""
    cells = [NUM.get(key.format(N=N), '--') or '--' for N in Ns_]
    return label + " & " + " & ".join(cells) + " \\\\"


def tab_mc():
    g = lambda k: NUM.get(k, '--') or '--'
    big = 99

    def cm(rs, i):
        v = [big if r['M'][i] is None else r['M'][i] for r in rs]
        t = lambda x: '\\ensuremath{>}8' if x == big else str(x)
        return t(min(v)) if min(v) == max(v) else t(min(v)) + '--' + t(max(v))
    rows = []
    for N, rs, lab in ((2, [r for r in _CO if r['N'] == 2], "$N=2$, all $k$"),
                       (3, [r for r in _CO if r['N'] == 3], "$N=3$, %s basis sizes" % g('cr_N3_nk'))):
        rows.append((lab, "%s--%s" % (g('cr_N%d_lo' % N), g('cr_N%d_hi' % N)), "%s--%s" % (g('cr_N%d_free_lo' % N), g('cr_N%d_free_hi' % N)),
                     cm(rs, 0), cm(rs, 1), cm(rs, 2)))
    for N, ks in ((4, (12, 16, 20)), (5, (12, 16, 20)), (6, (12,))):
        for k in ks:
            rs = [r for r in _CO if r['N'] == N and r['k'] == k]
            rows.append(("$N=%d$, $k=%d$" % (N, k), g('cr_N%d_k%d_fix' % (N, k)), g('cr_N%d_k%d_free' % (N, k)),
                         cm(rs, 0), cm(rs, 1), cm(rs, 2)))
    s = ["\\begin{table}[ht]", "\\centering", "\\small",
         "\\caption{How many frequencies describe the signal. Residual, in units of the standard deviation of $\\langle\\Op\\rangle(t)$, of the best single cosine at the splitting $\\Delta E$ of the gated pair (what imaginary-time Monte Carlo of the splitting would predict) and at a free frequency; and the number of cosines of a free fit (greedy pursuit with joint refit of the amplitudes) needed for a residual below $20\\%$, $10\\%$ and $5\\%$ ($>8$: not reached with up to eight cosines). "
         "Where the doublet test passes ($N=2,3$ and $N=4$, $k=20$) a splitting-based prediction is adequate; elsewhere the gated splitting explains none of the signal, and the number of cosines tends to grow with $N$ and to fall with $k$ (the basis is not converged at $N=5,6$). "
         "At $N=5$, $k=20$ the best free cosine sits at $\\num{cr_N5_k20_wratio}$ times $\\Delta E$.}",
         "\\label{tab:mc}", "\\begin{tabular}{lcccccc}", "\\toprule",
         " & \\multicolumn{2}{c}{residual of one cosine} & \\multicolumn{3}{c}{cosines for residual below} \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}",
         "case & at $\\Delta E$ & free & $20\\%$ & $10\\%$ & $5\\%$ \\\\", "\\midrule"]
    s += [" & ".join(r) + " \\\\" for r in rows]
    s += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(s)

def tab_scrit():
    g = lambda k: NUM.get(k, '--') or '--'
    s = ["\\begin{table}[ht]", "\\centering", "\\small",
         "\\caption{Does a window exist at all? The growth per added particle $s_{\\rm crit}$ (bits, Eq.~\\eqref{eq:scrit}) that the central cut must exceed beyond $N=\\num{Nmax_measured}$ for the window at the simulated mass to be non-empty with its upper end at $N_{\\rm hi}$, against what the data give: the asymptotic slopes of the sharing law (saturated and saturating $\\bar S_1$) and the last measured step.}",
         "\\label{tab:scrit}", "\\begin{tabular}{lccc}", "\\toprule",
         "& $N_{\\rm hi}=24$ & $N_{\\rm hi}=40$ & $N_{\\rm hi}=88$ \\\\", "\\midrule",
         "$s_{\\rm crit}$, grid & %s & %s & %s \\\\" % (g('scrit_grid_N24'), g('scrit_grid_N40'), g('scrit_grid_N88')),
         "$s_{\\rm crit}$, $20$-mode basis & %s & %s & %s \\\\" % (g('scrit_mode_N24'), g('scrit_mode_N40'), g('scrit_mode_N88')),
         "\\midrule",
         "sharing-law slope, $\\bar S_1$ saturated & \\multicolumn{3}{c}{%s} \\\\" % g('slope_opt'),
         "sharing-law slope, $\\bar S_1$ saturating (fit) & \\multicolumn{3}{c}{%s} \\\\" % g('slope_sat'),
         "last measured step & \\multicolumn{3}{c}{%s} \\\\" % g('d56'),
         "\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(s)


def tab_measured():
    """One table for everything measured on the simulated runs (k=12, correlated start), by N."""
    s = ["\\begin{table}[!tp]", "\\centering", "\\small",
         "\\caption{Everything measured on the simulated runs, by cluster size ($k=12$ modes per particle, correlated start at the resonant tilt, one protocol for every $N$: maximum over $[0,T]$ with $T=\\num{Tprot}$, snapshot spacing $\\num{dtsnap}$; entropies in bits). "
         "\\emph{Signal} (Fig.~\\ref{fig:signal}): lowest value of $\\langle\\Op\\rangle/\\langle\\Op\\rangle(0)$ in the window; largest value after it first falls below $\\tfrac12$ (a clean oscillation returns to about one); weight of the start on the gated pair of levels (Table~\\ref{tab:start}); time of the central-bond maximum. "
         "\\emph{Cut entropies} (Eq.~\\eqref{eq:threeS}): mean one-particle entropy $\\bar S_1$; best balanced, natural-order and worst balanced cut; $\\Snat$ is the central bond; numerical Schmidt rank $\\chi$ of the central bond at discarded weight $10^{-3}$. "
         "\\emph{Growth}: increments with each added particle. "
         "\\emph{Layout}: how much more the worst order pays than the best; how much more the order chosen on the graph pays than the exact optimum; rank correlation between graph and exact costs over all orders. "
         "\\emph{Graph} (Section~\\ref{sec:graph}): mean edge weight $w_{ij}$ against the sharing-law weight $2\\bar S_1/(N-1)$ of Eq.~\\eqref{eq:sharing}, and the sharing-law central cut against the measured one. "
         "\\emph{Laws fitted to $N\\le4$}: excess of the pair-counting and of the linear law over the measured central bond. "
         "\\emph{Start}: central bond for the product start (a quench; no run exists for $N\\ge5$), to be compared with $\\Snat$. "
         "A dash marks a quantity that does not exist for that $N$ (an increment, a rank correlation over one order, a fit at the fitted sizes) or a run that was not made.\\num{d45_note}}",
         "\\label{tab:measured}", "\\begin{tabular}{lrrrrr}", "\\toprule",
         "$N$ & 2 & 3 & 4 & 5 & 6 \\\\", "\\midrule",
         "\\multicolumn{6}{l}{\\emph{Signal}} \\\\",
         _row("lowest value of $\\langle\\Op\\rangle/\\langle\\Op\\rangle(0)$", "Omin_N{N}"),
         _row("largest value after the first fall below $\\tfrac12$", "Orev_N{N}"),
         _row("weight of the start on the gated pair", "ov_N{N}"),
         _row("time of the central-bond maximum", "tcen_N{N}"),
         "\\multicolumn{6}{l}{\\emph{Cut entropies}} \\\\",
         _row("$\\bar S_1$", "S1_N{N}"),
         _row("best balanced cut $\\Sperp$", "Sperp_N{N}"),
         "central bond $\\Snat$ & " + " & ".join(NUM.get(f'S{N}_cen', '--') for N in (2, 3, 4, 5, 6)) + " \\\\",
         _row("worst balanced cut $\\Smax$", "Smax_N{N}"),
         _row("Schmidt rank $\\chi(10^{-3})$", "chi3_N{N}"),
         _row("$\\chi(10^{-3})/2^{\\Snat}$", "chiratio_N{N}"),
         "\\multicolumn{6}{l}{\\emph{Growth}} \\\\",
         "increment of $\\Snat$ & -- & " + " & ".join(NUM[k_] for k_ in ('d23', 'd34', 'd45', 'd56')) + " \\\\",
         "increment of $\\bar S_1$ & -- & " + " & ".join(NUM[k_] for k_ in ('dS1_23', 'dS1_34', 'dS1_45', 'dS1_56')) + " \\\\",
         "\\multicolumn{6}{l}{\\emph{Layout}} \\\\",
         _row("worst order minus best", "layout_gain_N{N}"),
         _row("graph-chosen order minus optimum", "gbg_N{N}"),
         _row("rank correlation, graph against exact", "spearman_N{N}"),
         "\\multicolumn{6}{l}{\\emph{Graph}} \\\\",
         _row("mean edge weight, measured", "w_N{N}"),
         _row("sharing-law weight", "wshare_N{N}"),
         _row("sharing-law central cut", "Wu_N{N}"),
         "\\multicolumn{6}{l}{\\emph{Laws fitted to $N\\le4$: excess over the measured central bond}} \\\\",
         "pair-counting law & -- & -- & -- & " + " & ".join(NUM[k_] for k_ in ('pair_over_N5', 'pair_over_N6')) + " \\\\",
         "linear law & -- & -- & -- & " + " & ".join(NUM[k_] for k_ in ('lin_over_N5', 'lin_over_N6')) + " \\\\",
         "\\multicolumn{6}{l}{\\emph{Start}} \\\\",
         "central bond, product start (correlated start: $\\Snat$ above) & " + " & ".join(NUM.get(f'prod_N{N}', '--') for N in (2, 3, 4, 5, 6)) + " \\\\",
         "\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(s)


def tab_control():
    w5 = f"{NUM['wmin_N5']}--{NUM['wmax_N5']}"
    s = ["\\begin{table}[ht]", "\\centering", "\\small",
         "\\caption{The control without tunneling: $N=5$, $k=12$, correlated start, away from resonance ($c=\\num{off_c}$, the start stays in one isomer) against the resonant run (Fig.~\\ref{fig:control}). Same window $T=\\num{off_T}$\\num{off_lb}.\\num{off_note}}",
         "\\label{tab:control}", "\\begin{tabular}{lrr}", "\\toprule",
         " & off resonance & resonance \\\\", "\\midrule",
         f"lowest value of $\\langle\\Op\\rangle/\\langle\\Op\\rangle(0)$ & {NUM['off_Omin']} & {NUM['Omin_N5']} \\\\",
         f"central bond $\\Snat$ & {NUM['off_Snat']} & {NUM['Snat_N5']} \\\\",
         f"mean one-particle entropy $\\bar S_1$ & {NUM['off_S1']} & {NUM['S1_N5']} \\\\",
         f"edge weights $w_{{ij}}$ (range) & {NUM['off_wmin']}--{NUM['off_wmax']} & {w5} \\\\",
         f"largest graph excess over the exact entropy & {NUM['off_dev']} & {NUM['graph_maxdev']}$^\\dagger$ \\\\",
         f"rank correlation, graph against exact orders & {NUM['off_rho']} & {NUM['spearman_N5']} \\\\",
         f"time of the central-bond maximum & {NUM['off_tcen']} & {NUM['tcen_N5']} \\\\",
         "\\bottomrule", "\\end{tabular}",
         "\\par\\smallskip\\footnotesize $^\\dagger$ largest over all $N\\le6$ (Fig.~\\ref{fig:entropy}a), not for $N=5$ alone.",
         "\\end{table}"]
    return "\n".join(s)


def tab_scen():
    def r(label, par, slope, k6, over, k8, k12, k16):
        return f"{label} & {par} & {slope} & {NUM[k6]} & {NUM[over]} & {NUM[k8]} & {NUM[k12]} & {NUM[k16]} \\\\"
    s = ["\\begin{table}[ht]", "\\centering", "\\small",
         "\\caption{The three scenarios for the one-particle entropy $\\bar S_1(N)$ beyond the simulated sizes, the growth laws fitted to smaller $N$, and what they predicted for the central bond (bits). "
         "The sharing-law rows were fitted to $N\\le5$ and give the central cut they predicted at $N=6$ before that run, and its excess over the measured $S_6=\\num{S6_cen}$; the pair-counting and linear laws were fitted to $N\\le4$. "
         "\\emph{Slope}: asymptotic growth per added particle (the sharing law with saturating $\\bar S_1$ is linear in $N$; with growing $\\bar S_1$ it grows as $N^2$). "
         "The last three columns are the extrapolated central bond. None of these is a measurement beyond $N=\\num{Nmax_measured}$.}",
         "\\label{tab:scen}", "\\begin{tabular}{llcccccc}", "\\toprule",
         "scenario & parameters & slope & $N=6$ & excess & $N=8$ & $N=12$ & $N=16$ \\\\", "\\midrule",
         r("sharing, $\\bar S_1$ saturating (fit)", "$S_\\infty=\\num{fit_Sinf}$, $\\tau=\\num{fit_tau}$", f"{NUM['slope_sat']}", 'share_sat_N6', 'share_over_sat_N6', 'share_sat_N8', 'share_sat_N12', 'share_sat_N16'),
         r("sharing, $\\bar S_1$ saturated at $N=6$", "$\\bar S_1=\\num{S1_last}$", f"{NUM['slope_opt']}", 'share_opt_N6', 'share_over_opt_N6', 'share_opt_N8', 'share_opt_N12', 'share_opt_N16'),
         r("sharing, $\\bar S_1$ keeps growing", "$+\\num{inc_pess}$ per particle", "$\\propto N^2$", 'share_pess_N6', 'share_over_pess_N6', 'share_pess_N8', 'share_pess_N12', 'share_pess_N16'),
         f"pair-counting law & fit to $N\\le4$ & $\\propto N^2$ & {NUM['pair_pred_N6']} & {NUM['pair_over_N6']} & {NUM['pair_pred_N8']} & {NUM['pair_pred_N12']} & {NUM['pair_pred_N16']} \\\\",
         f"linear law & fit to $N\\le4$ & {NUM['lin_slope']} & {NUM['lin_pred_N6']} & {NUM['lin_over_N6']} & {NUM['lin_pred_N8']} & {NUM['lin_pred_N12']} & {NUM['lin_pred_N16']} \\\\",
         "\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(s)


def tab_protocol():
    """One protocol for every N, and how much the maxima depend on the window and on the snapshot spacing."""
    wins = [int(w) for w in cfg['studies']['windows']]
    dts = [f"{d:g}" for d in cfg['studies']['dt_scan']]
    s = ["\\begin{table}[ht]", "\\centering", "\\small",
         "\\caption{Sampling sensitivity of the entropy maxima (bits; correlated start, $k=12$). \\emph{Window} (upper block): one stored series of the protocol ($T=\\num{Tprot}$, spacing $\\num{dtsnap}$) cut to a shorter window afterwards: central bond $S_{\\rm cen}$ and worst balanced cut $\\Smax$ (equal at $N=2$, where there is one cut). \\emph{Spacing} (lower block): the same state and window with another snapshot spacing $\\Delta t$; the maximum is taken over the snapshots.}",
         "\\label{tab:protocol}", "\\begin{tabular}{lcccc}", "\\toprule",
         " & \\multicolumn{3}{c}{window: $S_{\\rm cen}$ / $\\Smax$} \\\\", "\\cmidrule(lr){2-4}",
         "$N$ & " + " & ".join(f"$T\\le{w}$" for w in wins) + " & \\\\", "\\midrule"]
    for N in (2, 3, 4):
        if str(N) not in PT['window']:
            continue
        cells = [f"{NUM[f'win_cen_N{N}_T{w}']} / {NUM[f'win_max_N{N}_T{w}']}" for w in wins]
        s.append(f"{N} & " + " & ".join(cells) + " & \\\\")
    s += ["\\midrule", "spacing at $T=\\num{Tprot}$ & " + " & ".join(f"$\\Delta t={d}$" for d in dts) + " \\\\", "\\midrule"]
    for N, key, lab in ((3, 'one', '$N=3$, max one-particle entropy'), (3, 'cen', '$N=3$, $S_{\\rm cen}$'),
                        (4, 'cen', '$N=4$, $S_{\\rm cen}$')):
        cells = [NUM.get(f"sp_{key}_N{N}_dt{d.replace('.', 'p')}", '--') for d in dts]
        s.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    s += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(s)


def tab_walls():
    Nl = max(L['Ns_measured'])                         # the largest simulated N: where the scenarios start and chi is measured
    sc = {'pess': f'entropy keeps growing at the $N={Nl - 2}\\to{Nl}$ rate', 'sat': 'one-particle entropy saturates (fit)',
          'opt': f'one-particle entropy saturated at $N={Nl}$'}
    s = ["\\begin{table}[ht]", "\\centering", "\\small",
         "\\caption{Where each classical route first fails (first failing $N$) for the budgets of Table~\\ref{tab:params}. Exact diagonalization is an exact count; the tensor-network rows evaluate the cost model of Prop.~\\ref{prop:tn} on the graph law of Section~\\ref{sec:larger} under three assumptions about the one-particle entropy beyond the largest simulated $N$. In parentheses: the same with the numerical Schmidt rank $\\chi(10^{-3})$ charged instead of $2^{S}$, at the ratio measured at $N=" + str(Nl) + "$ ($+\\num{chi_shift}$ bits).}",
         "\\label{tab:walls}", "\\begin{tabular}{L{4.0cm}L{5.3cm}cc}", "\\toprule",
         "Route & Assumption & grid, $d=\\num{d_grid}$ & modes, $d=\\num{d_mode}$ \\\\", "\\midrule",
         f"Exact diagonalization, distinguishable & exact count & {NUM['ED_grid']} & {NUM['ED_mode20']} \\\\",
         f"Exact diagonalization, bosonic & exact count & {NUM['ED_grid_boson']} & {NUM['ED_mode20_boson']} \\\\", "\\midrule"]
    for k_ in ('pess', 'sat', 'opt'):
        s.append(f"Tensor networks & {sc[k_]} & {NUM[f'wall_{k_}_grid']} ({NUM[f'wallchi_{k_}_grid']}) & {NUM[f'wall_{k_}_mode']} ({NUM[f'wallchi_{k_}_mode']}) \\\\")
    s += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(s)


def tab_quantum():
    s = ["\\begin{table}[ht]", "\\centering", "\\small",
         "\\caption{Quantum price as a function of cluster size (leading-order model, Appendix~\\ref{app:quantum}). T-gates at the simulated mass $\\mut=\\num{mut0}$; $\\mutq$: largest mass whose qubitization count fits the T-gate budget (leading order, and with the two omitted terms restored).}",
         "\\label{tab:quantum}", "\\begin{tabular}{rrrrrr}", "\\toprule",
         "$N$ & qubits & T-gates, qubitization & T-gates, Trotter & $\\mutq$ & $\\mutq$, complete \\\\", "\\midrule"]
    for N in (4, 8, 16, 24, 40, 88):
        s.append(f"{N} & {NUM[f'Qb_N{N}']} & {NUM[f'Tqub_N{N}']} & {NUM[f'Ttrot_N{N}']} & {NUM[f'edge_N{N}']} & {NUM[f'edge_N{N}_full']} \\\\")
    s += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(s)


def tab_k():
    s = ["\\begin{table}[ht]", "\\centering", "\\small",
         "\\caption{Basis dependence of the central-cut entropy (bits, correlated start) and of the numerical Schmidt rank $\\chi(10^{-3})$ of that bond. Resonance tilts differ with $k$. " + ("Cells marked $\\dagger$ come from the earlier code (another window and spacing)." if any(dag(N_) for N_ in Ns) else "") + "}",
         "\\label{tab:k}", "\\begin{tabular}{rcccccc}", "\\toprule",
         " & \\multicolumn{3}{c}{$S$ (bits)} & \\multicolumn{3}{c}{$\\chi(10^{-3})$} \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}", "$N$ & $k=12$ & $16$ & $20$ & $k=12$ & $16$ & $20$ \\\\", "\\midrule"]
    for N in Ns:
        cells, chis = [], []
        for k in (12, 16, 20):
            r = rec(N, k)
            d_ = dag(N, k)
            planned = k in PLANNED_K.get(N, (12,))
            cells.append(f(r['S_cen']) + d_ if r else (PEND(f'N={N} k={k}') if planned else '--'))
            chis.append(str(r['chi3']) + d_ if r and 'chi3' in r else ('--' if r or not planned else PEND('')))
        s.append(f"{N} & " + " & ".join(cells) + " & " + " & ".join(chis) + " \\\\")
    s += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(s)


def tab_checks():
    """Numerical checks of the machinery, read from the log of the one-body and Monte Carlo tests."""
    txt = open(os.path.join(RAW, 'tier1_checks.log')).read()
    import re
    def grab(pat, cast=float):
        m = re.search(pat, txt)
        return cast(m.group(1)) if m else None
    A8 = grab(r"A=8\.0: c=0 dE=([\d.]+)")
    A1 = grab(r"A=1\.0: c=0 dE=([\d.]+)")
    fft = grab(r"dE_fft=([\d.]+)")
    err = grab(r"rel err ([\d.e+-]+)")
    inst = grab(r"A=8\.0: c=0 dE=[\d.]+\s+\(instanton ([\d.]+)\)")
    s = ["\\begin{table}[ht]", "\\centering", "\\small",
         "\\caption{Checks of the numerical machinery on one-particle problems with known answers.}",
         "\\label{tab:checks}", "\\begin{tabular}{L{6.8cm}ccc}", "\\toprule",
         "Check & computed & reference & relative difference \\\\", "\\midrule"]
    if None not in (A1, fft, err):
        s.append(f"Real-time propagation, weak barrier ($A=1$): splitting & {f(fft, 6)} & {f(A1, 6)} & ${sci(err, 1)}$ \\\\")
    if None not in (A8, inst):
        s.append(f"Instanton formula against exact diagonalization, $A=8$ & {f(inst, 4)} & {f(A8, 4)} & {f(100 * (inst / A8 - 1), 1)}\\% \\\\")
    s += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(s)



def tab_Asweep():
    s = ["\\begin{table}[ht]", "\\centering", "\\small",
         "\\caption{Barrier strength: one particle in $U=A(x^2-1)^2$ ($m=\\hbar=1$). $S_0=\\frac43\\sqrt{2A}$ is the barrier action, $\\mut=\\sqrt{8A}$ the trap mass (the level spacing $\\hbar\\omega$ of one valley), "
         "$\\Delta E$ the splitting of the two lowest levels (finite differences, converged in the grid spacing; the value on the coarse grid $\\Delta x=0.125$ is up to $\\num{ch_grid_gap}\\%$ away), "
         "$t^*=\\pi/\\Delta E$ the time of one complete transfer. $A=\\num{A}$ (bold) is the value used in the paper; its $N=2$ value, with the other valley and the pair interaction, is $t_0=\\num{tstar0}$ (Table~\\ref{tab:params}).}",
         "\\label{tab:Asweep}", "\\begin{tabular}{rccccc}", "\\toprule",
         "$A$ & barrier action $S_0$ & trap mass $\\mut$ & splitting $\\Delta E$ & transfer time $t^*=\\pi/\\Delta E$ & $\\Delta E\\,/\\,\\hbar\\omega$ \\\\", "\\midrule"]
    for r in PC['onebody']:
        A = int(r['A'])
        dE = r['dE_fine']
        cells = [str(A), f(r['S0'], 2), f(r['mut'], 1), f(dE, 3 if dE >= 0.1 else 4), f(math.pi / dE, 1), f"{dE / r['omega']:.2g}"]
        if A == int(float(NUM['A'])):
            cells = [f"\\textbf{{{c}}}" for c in cells]
        s.append(" & ".join(cells) + " \\\\")
    s += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(s)


def tab_choice():
    S = PC['sensitivity']

    def verdict(r):
        a, b = not r['pass_overlap'], not r['pass_cap']
        return 'pass' if not (a or b) else '\\textbf{fail: ' + ('both' if a and b else 'overlap' if a else 'cap') + '}'

    def one(label, name):
        r = _sens[label]
        return f"{name} & {f(r['c_star'], 3)} & {f(r['dE'], 4)} & {f(r['ov_sub'], 3)} & {f(r['cap_shift'], 3)} & {verdict(r)} \\\\"

    def pair(l1, l2, name):
        a, b = _sens[l1], _sens[l2]
        both = lambda k, n: f"{f(a[k], n)} / {f(b[k], n)}"
        v = verdict(a) if verdict(a) == verdict(b) else verdict(a) + ' / ' + verdict(b)
        return f"{name} & {both('c_star', 3)} & {both('dE', 4)} & {both('ov_sub', 3)} & {both('cap_shift', 3)} & {v} \\\\"

    def rng(prefix, name):
        rs = [r for r in S if r['label'].startswith(prefix)] + [_sens['production point']]
        def lo_hi(k, n):
            a, b = f(min(r[k] for r in rs), n), f(max(r[k] for r in rs), n)
            return a if a == b else f"{a}--{b}"
        v = 'pass' if all(_ok(r) for r in rs) else 'mixed'
        return f"{name} & {lo_hi('c_star', 3)} & {lo_hi('dE', 4)} & {lo_hi('ov_sub', 3)} & {lo_hi('cap_shift', 3)} & {v} \\\\"

    rows = [one('production point', 'production point'),
            one('A=4', 'barrier $A=4$'), one('A=6', '$A=6$'), one('A=10', '$A=10$'), one('A=12', '$A=12$'),
            pair('sigma=0.25', 'sigma=0.35', 'range $\\sigma=0.25$ / $0.35$'), one('sigma=0.5', '$\\sigma=0.5$'),
            rng('eps=', 'depth $\\epsilon=0.5,1,2,4,8$'),
            pair('wy=0.5 (omega_y=0.71)', 'wy=8 (omega_y=2.83)', 'transverse $\\omega_y^2=0.5$ / $8$'),
            one('C0=0', 'offset $C_0=0$'), one('C0=0.05', '$C_0=0.05$'), one('C0=0.3', '$C_0=0.30$'),
            pair('cap=25', 'cap=400', 'cap $V_{\\rm cap}=25$ / $400$')]
    s = ["\\begin{table}[ht]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}",
         "\\caption{Why these inputs: $N=2$ by exact diagonalization on a $28\\times18$ grid, one input changed at a time from the production point "
         "($A=8$, $\\epsilon=2$, $\\sigma=0.18$, $\\omega_y^2=2$, $C_0=0.15$, $V_{\\rm cap}=100$). $c^*$: tilt at which the ground state has $\\langle\\Op\\rangle=\\frac12$; $\\Delta E_2$: splitting of the two lowest levels; "
         "overlap: weight of the start state in those two levels (need $\\ge\\num{ch_ov_min}$); cap: relative change of $\\Delta E_2$ when the core cap is doubled (need $<\\num{ch_cap_max}$).}",
         "\\label{tab:choice}", "\\begin{tabular}{L{3.9cm}ccccc}", "\\toprule",
         "changed input & tilt $c^*$ & $\\Delta E_2$ & overlap & cap & verdict \\\\", "\\midrule"] + rows
    s += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(s)


def tab_Ascan():
    def verdict(r):
        if not r['doublet_ok']:
            return '\\textbf{fail: no doublet}'
        return 'pass' if r['ov_int'] >= PC['limits']['overlap_min'] else '\\textbf{fail: overlap}'
    s = ["\\begin{table}[ht]", "\\centering", "\\small",
         "\\caption{Does the barrier chosen at $N=2$ carry over? Static scan at $N=3$ and $4$ in the $k=12$ mode basis (no propagation): for each $A$ the resonant tilt is found again, "
         "and the correlated start is prepared as in the main runs. $c^*$: tilt; $\\Delta E$: splitting of the doublet that the doublet test selects; overlap: weight of the start in that doublet (need $\\ge\\num{ch_ov_min}$). "
         "A row marked \\emph{no doublet} has no pair of levels that passes the doublet test at the resonance found. The cap test of Table~\\ref{tab:choice} was not repeated for $N>2$.}",
         "\\label{tab:Ascan}", "\\begin{tabular}{rrcccc}", "\\toprule", "$N$ & $A$ & $c^*$ & $\\Delta E$ & overlap & verdict \\\\", "\\midrule"]
    for N in sorted(PS):
        for A in sorted(PS[N]):
            r = PS[N][A]
            bold = (A == int(float(NUM['A'])))
            cells = [str(N), str(A), f(r['c'], 2), f(r['dE'], 3), f(r['ov_int'], 3), verdict(r)]
            s.append(" & ".join(f"\\textbf{{{c}}}" if bold and i < 5 else c for i, c in enumerate(cells)) + " \\\\")
        s.append("\\midrule" if N != max(PS) else "")
    s = [x for x in s if x != ""]
    s += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(s)


def tab_mps():
    s = ["\\begin{table}[ht]", "\\centering", "\\small",
         "\\caption{A matrix-product state (density-matrix-renormalization-group start, two-site time-dependent variational evolution) against the exact propagation of the same case, correlated start, resonant tilt, window $T$ (shorter than $T=\\num{Tprot}$ so that the exact run is cheap). "
         "$S_{\\rm cen}$: central-bond entropy (bits); $\\max\\Delta S$: largest difference over every cut both methods measured and every snapshot; $\\chi_{10^{-3}}$: numerical Schmidt rank of the central bond (at odd $N$, of the two central bonds the one with the larger entropy); $\\chi_{\\max}$: cap on the bond dimension "
         "(binding where $\\chi_{10^{-6}}$ of the MPS equals it). Last column: wall time of the MPS over that of the exact code (a shared four-core machine; an order of magnitude only). The $k=6$ rows validate the method in a saturated basis; the $k=12$ row is capped at $\\chi_{\\max}$ and its $S_{\\rm cen}$ is $\\num{tn_k12_dSc}$ bit below the exact value.}",
         "\\label{tab:mps}", "\\begin{tabular}{rrrrrrrrr}", "\\toprule",
         "$N$ & $k$ & $T$ & $\\chi_{\\max}$ & $S_{\\rm cen}$ MPS & $S_{\\rm cen}$ exact & $\\max\\Delta S$ & $\\chi_{10^{-3}}$ MPS / exact & slower by \\\\", "\\midrule"]
    for c in ('N3_k6', 'N4_k6', 'N5_k6', 'N4_k12'):
        r = _val[c]
        s.append(f"{r['N']} & {r['k']} & {int(r['T'])} & {int(r['chi_max'])} & {f(r['S_cen_mps'], 4)} & {f(r['S_cen_exact'], 4)} & {sci(r['max_dS'], 1)} & {int(r['chi3_mps'])} / {int(r['chi3_exact'])} & {r['wall_mps_s'] / r['wall_exact_s']:.0f}$\\times$ \\\\")
    s += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(s)


def tab_start():
    """One table for the start state in the eigenbasis of H: the two heaviest levels (the pair) with the dressing classes, and the gated pair
    of the doublet test, for every case of data/followup/stage1 (k >= 12)."""
    _nodbl = any(rec(5, k_) is None or rec(5, k_).get('doublet') is None for k_ in (16, 20))
    s = ["\\begin{table}[ht]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3.2pt}",
         "\\caption{The correlated start in the eigenbasis of $\\hat H$, every case computed (method in the text of this appendix). "
         "$c^*$: resonance tilt of the mode basis; coverage: weight of the start on the computed levels. "
         "\\emph{Pair}: the two levels of largest weight, their splitting $2J$ and total weight $B_2$. "
         "\\emph{Outside the pair}: the weight on the other levels by distance to the nearer level of the pair (within $J$, $1$--$3J$, beyond $3J$); the remainder $1-B_2-\\text{sum}$ is on levels not computed. "
         "\\emph{Gated pair}: the pair of low levels of smallest splitting $\\Delta E$ that passes the doublet gates, the weight of the start on it, and whether it is a clean doublet (with the ratio $g_N=\\Delta E_2/\\Delta E_N$ of Eq.~\\eqref{eq:Tobs}, $\\Delta E_2$ the grid value of Table~\\ref{tab:params}); "
         "it is selected by splitting, not by weight, so it can differ from the pair. Rows marked $^\\ast$ are Lanczos lists only; $k=6,8$ are omitted (artefacts of the small basis)." + (" $N=5$, $k=16,20$: no gated-pair analysis in the logs." if _nodbl else "") + "}",
         "\\label{tab:start}", "\\begin{tabular}{rrcccccccccc}", "\\toprule",
         " & & & & \\multicolumn{2}{c}{pair} & \\multicolumn{3}{c}{outside the pair} & \\multicolumn{3}{c}{gated pair} \\\\ \\cmidrule(lr){5-6}\\cmidrule(lr){7-9}\\cmidrule(lr){10-12}",
         "$N$ & $k$ & $c^*$ & coverage & $2J$ & $B_2$ & $<J$ & $1$--$3J$ & $\\ge3J$ & $\\Delta E$ & weight & clean / $g_N$ \\\\", "\\midrule"]
    last = None
    for (N, k), r in sorted(_S1_tab.items()):
        if k < 12:
            continue
        if last is not None and N != last:
            s.append("\\addlinespace[2pt]")
        last = N
        star = "$^\\ast$" if r['krylov'] else ""
        rr = rec(N, k) if k in (12, 16, 20) else None
        if rr is None or rr.get('doublet') is None:
            g = "-- & -- & --"
        else:
            tail = f"yes / {NUM[f'gN_N{N}_k{k}']}" if rr['doublet'] == 'yes' else 'no'
            g = f"{f(rr['dE'], 3)} & {f(rr['ov'])} & {tail}"
        s.append(f"{N}{dag(N, k)} & {k}{star} & {f(r['c'], 2)} & {f(r['cov'], 3)} & {f(2 * r['J'], 3)} & {f(r['B2'], 3)} & {f(r['near'], 3)} & {f(r['mid'], 3)} & {f(r['far'], 3)} & {g} \\\\")
    s += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(s)


def tab_static():
    """Section 5: the start in the eigenbasis at k=20 and at the largest k, N=2..5, with the cosine counts."""
    ks = {N_: max(k_ for (n_, k_) in _S1_tab if n_ == N_) for N_ in (2, 3, 4, 5)}
    s = ["\\begin{table}[ht]", "\\centering", "\\small",
         "\\caption{What the prepared state is made of (Section~\\ref{sec:static}; every case is in Table~\\ref{tab:start}). Correlated start at the resonance tilt $c^*$, $k=20$ modes per particle unless stated. "
         "\\emph{Pair}: splitting $2J$ and weight $B_2$ of the two levels of largest weight, and $B_2$ at the largest basis computed (with that $k$). "
         "\\emph{Outside the pair}: weight within $J$, at $1$--$3J$ and beyond $3J$ of the nearer level of the pair. "
         "\\emph{Signal}: number of cosines of a free fit that brings the residual to $10\\%$ and to $5\\%$ of the standard deviation of $\\langle\\Op\\rangle(t)$ (largest basis stored; the $N=6$ fit did not reach $5\\%$ with eight, $k=12$, not converged).}",
         "\\label{tab:static}", "\\begin{tabular}{rcccccccccc}", "\\toprule",
         " & & \\multicolumn{3}{c}{pair} & \\multicolumn{3}{c}{outside the pair} & \\multicolumn{2}{c}{cosines for} \\\\ \\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-10}",
         "$N$ & $c^*$ & $2J$ & $B_2$ & $B_2$ at largest $k$ & $<J$ & $1$--$3J$ & $\\ge3J$ & $10\\%$ & $5\\%$ \\\\", "\\midrule"]
    for N_ in (2, 3, 4, 5):
        r = _S1_tab[(N_, 20)]
        rm = _S1_tab[(N_, ks[N_])]
        s.append(f"{N_} & {f(r['c'], 2)} & {f(2 * r['J'], 3)} & {f(r['B2'], 3)} & {f(rm['B2'], 3)} ($k={ks[N_]}$) & {f(r['near'], 3)} & {f(r['mid'], 3)} & {f(r['far'], 3)} & {NUM[f'fu_cos_M10_N{N_}']} & {NUM[f'fu_cos_M05_N{N_}']} \\\\")
    s.append(f"6 ($k=12$) & {f(_S1_tab[(6, 12)]['c'], 2)} & {f(2 * _S1_tab[(6, 12)]['J'], 3)} & {f(_S1_tab[(6, 12)]['B2'], 3)} & -- & {f(_S1_tab[(6, 12)]['near'], 3)} & {f(_S1_tab[(6, 12)]['mid'], 3)} & {f(_S1_tab[(6, 12)]['far'], 3)} & {NUM['fu_cos_M10_N6']} & {NUM['fu_cos_M05_N6']} \\\\")
    s += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(s)


def tab_levels():
    s = ["\\begin{table}[ht]", "\\centering", "\\small",
         "\\caption{The lowest levels of $\\hat H$ that carry weight of the start ($N=3$, $k=16$, resonance tilt): energy above the ground state, weight of the start, reflection parity $\\langle R\\rangle$ and right-valley occupation $\\langle n_i\\rangle$ of each particle. The pair (levels 0 and 2) is carried mainly by particle $0$; level 1 is odd under the reflection and carries no weight; levels 5 and 9 are the first in which particle $1$, then particle $2$, has moved.}",
         "\\label{tab:levels}", "\\begin{tabular}{rccccccc}", "\\toprule",
         "level & $E-E_0$ & weight & $\\langle R\\rangle$ & $\\langle n_0\\rangle$ & $\\langle n_1\\rangle$ & $\\langle n_2\\rangle$ \\\\", "\\midrule"]
    d = _LV[(3, 16)]
    E, w, R, n = d['E'], d['weight'], d['R'], d['n']
    heavy = sorted(sorted(np.where(R > 0)[0], key=lambda i: -w[i])[:2])
    rows = list(heavy)
    for lab in (1, 2):
        c = [i for i in range(len(E)) if R[i] > 0 and n[i, lab] > 0.5]
        rows.append(c[0])
    odd = [i for i in range(len(E)) if R[i] < 0][:1]
    rows += odd
    for i in sorted(set(rows)):
        s.append(f"{i} & {f(E[i], 3)} & {w[i]:.4f} & {R[i]:+.3f} & {f(n[i, 0], 2)} & {f(n[i, 1], 2)} & {f(n[i, 2], 2)} \\\\")
    s += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(s)


for name, fn in (('tab_params', tab_params), ('tab_measured', tab_measured), ('tab_mc', tab_mc), ('tab_scrit', tab_scrit), ('tab_control', tab_control), ('tab_scen', tab_scen), ('tab_walls', tab_walls),
                 ('tab_quantum', tab_quantum), ('tab_k', tab_k), ('tab_protocol', tab_protocol),
                 ('tab_checks', tab_checks), ('tab_Asweep', tab_Asweep), ('tab_choice', tab_choice), ('tab_Ascan', tab_Ascan), ('tab_mps', tab_mps), ('tab_start', tab_start), ('tab_static', tab_static), ('tab_levels', tab_levels)):
    open(os.path.join(ROOT, name + '.tex'), 'w').write(fn() + '\n')
    print('wrote', name + '.tex')


# ============================================================================ hand-written claims
# The text states a few qualitative facts in words.  They are checked here against the data, so a rerun that
# changes one of them is reported instead of silently contradicting the text.
def _v(key):
    try:
        return float(NUM[key])
    except (KeyError, ValueError):
        return None


claims = [
    ('one splitting-cosine explains the signal where the doublet test passes (N=2,3, N=4 at k=20) and none of it at N=4, k=12, 16',
     max(r['fixed'] for r in CC if r['N'] in (2, 3) or (r['N'] == 4 and r['k'] == 20)) < 0.25 and all(r['fixed'] > 0.9 for r in CC if r['N'] == 4 and r['k'] in (12, 16))),
    (f"natural order is among the best of all orders at N={','.join(str(N) for N in Ns if N >= 3)}",
     all(NUM.get(f'natbest_N{N}') == 'yes' for N in Ns if N >= 3)),
    ('central-bond maximum lies inside the window (all N)', all(G(N) is None or G(N)['t_cen'] < G(N)['T'] - 1e-6 for N in Ns)),
    ('N=4 passes the doublet test only at k=20', NUM.get('dbl_N4_k12') == 'no' and NUM.get('dbl_N4_k16') == 'no' and NUM.get('dbl_N4_k20') == 'yes'),
    ('lowest value of the signal in the window is higher at N=4,5,6 than at N=2,3', None not in (_v('Omin_N3'), _v('Omin_N4'), _v('Omin_N5')) and min(_v('Omin_N4'), _v('Omin_N5'), _v('Omin_N6') or 1) > max(_v('Omin_N2'), _v('Omin_N3'))),
    ('central entropy falls with k at N=4 and the rank rises', None not in (_v('S_N4_k12'), _v('S_N4_k20')) and _v('S_N4_k12') > _v('S_N4_k20')
        and rec(4, 12) and rec(4, 20) and rec(4, 12)['chi3'] < rec(4, 20)['chi3']),
    ('product start has the larger central bond at N=3,4', all(_v(f'prod_N{N}') is not None and _v(f'prod_N{N}') > _v(f'corr_N{N}') for N in (3, 4))),
    (f'central bond grows with N ({Ns[0]}..{Ns[-1]})', all(G(a_) is None or G(b_) is None or G(b_)['S_nat'] > G(a_)['S_nat'] for a_, b_ in PAIRS)),
    ('increments are not constant (largest more than 1.5x smallest)', all(_v(f'd{a_}{b_}') is not None for a_, b_ in PAIRS)
        and max(_v(f'd{a_}{b_}') for a_, b_ in PAIRS) > 1.5 * min(_v(f'd{a_}{b_}') for a_, b_ in PAIRS)),
    ('mean edge weight falls with N', all(G(a_) is None or G(b_) is None or G(b_)['w_mean'] < G(a_)['w_mean'] for a_, b_ in PAIRS)),
    ('pair law fitted to N<=4 overshoots even the worst layout at N=5', G(5) is None or L['pair_pred']['5'] > G(5)['S_max']),
    ('linear law fitted to N<=4 lies at the level of the worst layout at N=5 (within 0.02 bit)', G(5) is None or abs(L['lin_pred']['5'] - G(5)['S_max']) < 0.02),
    ('graph overestimates (negative exact-minus-graph) at N=3', _v('devmin_N3') is not None and _v('devmin_N3') < 0),
    ('mode-basis s_crit at N_hi=24 lies above both sharing slopes and the last step (sec5 says the window is not guaranteed there)', float(NUM['scrit_mode_N24']) > max(float(NUM['slope_opt']), float(NUM['slope_sat']), float(NUM['d56']))),
    ('rank-charged grid wall is the same in all three scenarios (sec1 states one value)', len({NUM['wallchi_pess_grid'], NUM['wallchi_sat_grid'], NUM['wallchi_opt_grid']}) == 1),
    # the N=6 point against what the earlier points predicted (sec4 'The N=6 test', sec5, sec7)
    ('N=6: every sharing-law scenario predicted from N<=5 overshoots the measured central bond',
     G(6) is None or all(share_pred(sc_, 6) > G(6)['S_nat'] for sc_ in ('opt', 'sat', 'pess'))),
    ('N=6: pair law fitted to N<=4 overshoots even the worst layout', G(6) is None or L['pair_pred']['6'] > G(6)['S_max']),
    ('N=6: linear law fitted to N<=4 overshoots the measured bond and lies within 0.05 bit of the worst layout',
     G(6) is None or (L['lin_pred']['6'] > G(6)['S_nat'] and abs(L['lin_pred']['6'] - G(6)['S_max']) < 0.05)),
    ('N=5,6: the sharing law fed with the measured one-particle entropy lies inside the layout bracket',
     all(G(n_) is None or G(n_)['S_perp'] <= float(NUM[f'Wu_N{n_}']) <= G(n_)['S_max'] for n_ in (5, 6))),
    ('one-particle entropy increments shrink from N=3->4 on (dS1_34 > dS1_45 > dS1_56)',
     None not in (_v('dS1_34'), _v('dS1_45'), _v('dS1_56')) and _v('dS1_34') > _v('dS1_45') > _v('dS1_56')),
    ('the Theta(x=0) convention changes no first failing N (walls recomputed from previous-data/runs)', _walls_same is True),
    ('the Theta(x=0) convention moves the central bond by less than 0.02 bit and the product-start entropies not at all',
     max(abs(r['dS_cen']) for r in _TEcor) < 0.02 and max(abs(r['dS_cen']) for r in _TE if r['start'] == 'product') < 1e-8),
    ('t_0 in the cost model lies between the full-grid values at the exact resonance and at the grid tilt (within 2 per cent)',
     min(_FGr['t0'], _FGp['t0']) <= qm['t_star_0'] <= max(_FGr['t0'], _FGp['t0']) and abs(_FGr['t0'] / qm['t_star_0'] - 1) < 0.02),
    ('largest graph overcount is at the largest N', G(6) is None or abs(float(NUM['devmin_N6'])) >= max(abs(float(NUM[f'devmin_N{N}'])) for N in Ns if G(N)) - 1e-9),
    ('the measured N=6 step lies within 0.1 bit of the asymptotic sharing-law slope (saturating fit)', _v('d56') is None or _v('slope_sat') is None or abs(_v('d56') - _v('slope_sat')) < 0.1),
    ('N=5,6 central bonds are still rising at the end of the window (T<=120 value below T<=200 value)',
     all(_v(f'win_cen_N{N}_T120') is not None and _v(f'win_cen_N{N}_T200') > _v(f'win_cen_N{N}_T120') for N in (5, 6))),
    ('Schmidt rank over 2^S rises with N (k=12, N=2..6)',
     all(_v(f'chiratio_N{a_}') is not None and _v(f'chiratio_N{b_}') is not None and _v(f'chiratio_N{b_}') > _v(f'chiratio_N{a_}') for a_, b_ in PAIRS)),
    ('N=5 (k=20) and N=6 (k=12) fail the doublet test; the cosine at dE explains none of N=6 and the best single cosine of N=5, k=20 sits at a different frequency (>1.5 dE)',
     NUM.get('dbl_N5_k20') == 'no' and NUM.get('dbl_N6_k12') == 'no' and _v('cr_N6_k12_fix') is not None and _v('cr_N6_k12_fix') > 0.9
     and _v('cr_N5_k20_wratio') is not None and _v('cr_N5_k20_wratio') > 1.5),
]
claims += [
    ('follow-up: B2 at N=4 rises monotonically with k=12..32 and the last step (28 to 32) is below 0.01',
     all(_S1[(4, a_)]['B2'] < _S1[(4, b_)]['B2'] for a_, b_ in ((12, 16), (16, 20), (20, 24), (24, 28), (28, 32))) and _S1[(4, 32)]['B2'] - _S1[(4, 28)]['B2'] < 0.01),
    ('follow-up: B2 at k=20 falls with N (N=2..5)', all(_S1[(a_, 20)]['B2'] > _S1[(b_, 20)]['B2'] for a_, b_ in ((2, 3), (3, 4), (4, 5)))),
    ('follow-up: the weight outside the pair at k=20 sits mostly at 1-3 J for N=4,5 (mid larger than near+far)',
     all(_S1[(N_, 20)]['mid'] > _S1[(N_, 20)]['near'] + _S1[(N_, 20)]['far'] for N_ in (4, 5))),
    ('follow-up: c*(N) increases by 0.55-0.85 per particle at k=12 and k=20 (no bending through the measured N)',
     all(0.55 < d_ < 0.85 for k_ in (12, 20) for d_ in np.diff([c_ for _, c_ in _cs[k_]]))),
    ('follow-up: start parity > 0.999 and odd levels carry less than 1e-4 (N=3,4)',
     all(float(NUM[f'fu_Rstart_N{N_}_k{k_}']) > 0.999 and _LV[(N_, k_)]['weight'][_LV[(N_, k_)]['R'] < 0].max() < 1e-4 for (N_, k_) in _LV)),
    ('follow-up: the label states lie 2 to 9 J above the pair (first level of each moved label)', float(NUM['fu_marg_lo']) >= 2 and float(NUM['fu_marg_hi']) <= 9),
    ('follow-up: more cosines are needed at N=5 than at N=2 for a 5% residual (largest k)',
     (_CO and True) and (lambda m5, m2: (m5 is None) or (m5 > m2))([r for r in _CO if r['N'] == 5 and r['k'] == 20][0]['M'][2], [r for r in _CO if r['N'] == 2 and r['k'] == 20][0]['M'][2])),
]
def _label_weights():
    out = []
    for (N_, k_), d_ in _LV.items():
        for lab in range(1, N_):
            c_ = [i for i in range(len(d_['E'])) if d_['R'][i] > 0 and d_['n'][i, lab] > 0.5]
            if c_:
                out.append(float(d_['weight'][c_[0]]))
    return out


claims += [
    ('follow-up: the first level of each moved label carries at most 2 per cent of the start (N=3,4)', max(_label_weights()) <= 0.02),
    ('follow-up: the nearest moved-label level is 1.5-3 J from the pair', 1.5 <= float(NUM['fu_marg_lo']) <= 3.0),
    ('follow-up: the far part (>=3J) is at most 0.03 at k=20 for N=2..5 and within a factor 3 across N',
     max(_S1[(N_, 20)]['far'] for N_ in (2, 3, 4, 5)) <= 0.03 and max(_S1[(N_, 20)]['far'] for N_ in (2, 3, 4, 5)) <= 3 * min(_S1[(N_, 20)]['far'] for N_ in (2, 3, 4, 5))),
    ('follow-up: the 1-3J part at k=20 grows with N (N=2..5)', all(_S1[(a_, 20)]['mid'] < _S1[(b_, 20)]['mid'] for a_, b_ in ((2, 3), (3, 4), (4, 5)))),
    ('follow-up: the number of cosines for a 5% residual does not fall with N (N=2..5)',
     all(r_ is not None for r_ in [[r for r in _CO if r['N'] == N_ and r['k'] == max(x['k'] for x in _CO if x['N'] == N_)][0]['M'][2] for N_ in (2, 3, 4, 5)])
     and (lambda ms: all(a_ <= b_ for a_, b_ in zip(ms, ms[1:])))([[r for r in _CO if r['N'] == N_ and r['k'] == max(x['k'] for x in _CO if x['N'] == N_)][0]['M'][2] for N_ in (2, 3, 4, 5)])),
    ('follow-up: the pair of Table 1 columns agree with the gated pair where the doublet test passes (N=2,3)',
     all(abs(_S1[(N_, k_)]['B2'] - float(NUM[f'ov_N{N_}_k{k_}'])) < 0.006 for N_ in (2, 3) for k_ in (12, 16, 20))),
]
claims += [
    ('appendix: no cut of any run peaks at the last snapshot (N=2..6)', all(NUM[f'nedge_N{N_}'] == '0' for N_ in (2, 3, 4, 5, 6))),
    ('section 4: the pair-counting law fitted to N<=4 lies above the worst layout at N=5 and 6',
     all(float(NUM[f'pair_pred_N{N_}']) > float(NUM[f'Smax_N{N_}']) for N_ in (5, 6))),
]
bad = [c for c, ok in claims if not ok]
print(f"{len(claims) - len(bad)}/{len(claims)} hand-written claims confirmed by the data")
for c in bad:
    print('  CLAIM FAILS (edit the text):', c)
