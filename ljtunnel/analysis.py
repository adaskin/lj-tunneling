"""ljtunnel.analysis -- everything derived from the saved time series of a run, for ANY time window.

The simulation stores S_A(t) for the cuts it measured over the whole protocol window; here the
window is chosen afterwards (T_win), so all particle numbers are analysed with the same window
and the same snapshot spacing.  The window is [0, T_win]: the prepared start state is part of what a
representation has to hold, so t = 0 is included (include_start=False drops it).

Definitions (all entropies in bits, sigma(A) = max over the window of S_A(t)):
    S_cen      the central contiguous bond: the larger of the bonds floor(N/2), ceil(N/2)
    S_nat      max over the contiguous bonds (an MPS in the particle order 0..N-1)
    S_perp     min over the smaller-side sizes ceil(N/3) .. floor(N/2) of sigma(A): the best balanced cut
    S_max      max over the cuts with smaller side floor(N/2): the worst balanced cut
    w_ij       max over the window of I_ij(t) = S_i + S_j - S_ij       (edge weights)
    C(A)       1/2 sum_{i in A, j not in A} w_ij                        (graph cut)
"""
import itertools
import math

import numpy as np

from .entanglement import canon_mask, mask_of, members
from . import graph as G


# ------------------------------------------------------------------ access
def n_particles(run):
    return int(run['meta']['N'])


def column(run, A):
    """Index of the cut A (any subset) in run['S']."""
    N = n_particles(run)
    m = canon_mask(mask_of(A), N)
    idx = np.where(run['masks'] == m)[0]
    if len(idx) == 0:
        raise KeyError(f"cut {tuple(A)} not measured in {run.get('path')}")
    return int(idx[0])


def S_of(run, A):
    """S_A(t) for the cut A (any subset)."""
    return run['S'][:, column(run, A)]


def window(run, T_win=None, include_start=True):
    t = run['t']
    ok = (t >= 0) if include_start else (t > 0)
    if T_win is not None:
        ok &= t <= T_win + 1e-9
    return ok


def maxima(run, T_win=None, include_start=True):
    """(sigma, t_at): max over the window of every measured cut, and the first time it is attained."""
    ok = window(run, T_win, include_start)
    S = run['S'][ok]
    with np.errstate(all='ignore'):
        sig = np.nanmax(S, axis=0)
        arg = np.nanargmax(np.where(np.isnan(S), -np.inf, S), axis=0)
    return sig, run['t'][ok][arg]


def proper(run):
    """Columns of run['S'] that are proper bipartitions (older files of N = 2 also hold the whole system, S = 0)."""
    return run['masks'] != (1 << n_particles(run)) - 1


def sizes(run):
    N = n_particles(run)
    pc = np.array([bin(int(m)).count('1') for m in run['masks']])
    return np.minimum(pc, N - pc)


def one_particle_max(run, T_win=None, include_start=True):
    N = n_particles(run)
    ok = window(run, T_win, include_start)
    return np.array([np.nanmax(S_of(run, (i,))[ok]) for i in range(N)])


def mutual_information(run):
    """I_ij(t) = S_i + S_j - S_ij, shape (nt, N, N); NaN where the pair entropy was not measured."""
    N, nt = n_particles(run), len(run['t'])
    I = np.zeros((nt, N, N))
    for i, j in itertools.combinations(range(N), 2):
        if N == 2:                              # the pair is the whole (pure) state
            Sij = np.zeros(nt)
        else:
            try:
                Sij = S_of(run, (i, j))
            except KeyError:
                Sij = np.full(nt, np.nan)
        I[:, i, j] = I[:, j, i] = S_of(run, (i,)) + S_of(run, (j,)) - Sij
    return I


def edge_weights(run, T_win=None, include_start=True):
    ok = window(run, T_win, include_start)
    I = mutual_information(run)[ok]
    w = np.nanmax(I, axis=0) if I.size else np.zeros(I.shape[1:])
    np.fill_diagonal(w, 0.0)
    return w


# ------------------------------------------------------------------ the ledger record
def summarize(run, T_win=None, include_start=True):
    """One record per (run, window): the quantities of the entropy tables."""
    m = run['meta']
    N, k = int(m['N']), int(m['k'])
    d = m['diag']
    ok = window(run, T_win, include_start)
    T = float(run['t'][ok].max()) if ok.any() else 0.0
    sig, tat = maxima(run, T_win, include_start)
    nb = N - 1
    bonds = [float(sig[G_idx]) for G_idx in _bond_columns(run)]
    if not np.isfinite(bonds).all():
        raise ValueError(f"contiguous bond(s) {[b + 1 for b, v in enumerate(bonds) if not np.isfinite(v)]} were not measured "
                         f"in {run.get('path', 'this run')} (skipped by max_cut_dim?); S_cen and S_nat are undefined")
    jc = N // 2 - 1 if bonds[N // 2 - 1] >= bonds[(N + 1) // 2 - 1] else (N + 1) // 2 - 1
    bc = jc + 1
    cen = bonds[jc]
    col_c = _bond_columns(run)[jc]
    cap = min(bc, N - bc) * math.log2(k)
    tag = {'interacting': 'interacting', 'product': 'product'}[m['start']]
    sz = sizes(run)
    S1 = one_particle_max(run, T_win, include_start)
    w = edge_weights(run, T_win, include_start)
    iu = np.triu_indices(N, 1)
    rec = dict(
        source=run.get('path', ''), N=N, k=k, c=float(m['c']), T=T,
        start='correlated' if tag == 'interacting' else 'product',
        protocol=m['protocol']['kind'], dt=float(m['protocol']['dt_snap']),
        dE=float(d['dE']), mO=float(d['mO']), doublet='yes' if d['doublet'] else 'no',
        ov=float(d[f'ov_{tag}']), S_cen=cen, S_one=float(S1.max()), bonds=bonds,
        S_nat=float(max(bonds)),
        Omin=float(run['O'][ok].min()), Omax=float(run['O'][ok].max()),
        **_ranks(run, ok, jc),
        capr=float(cen / cap), tS_cen=float(tat[col_c]), S1mean=float(S1.mean()),
        Wmf=float(bc * (N - bc) / (N - 1) * S1.mean()),
        lowW_int=float(d[f'lowW_{tag}']), Eexc_int=float(d[f'Eexc_{tag}']),
        deepW_int=float(d[f'deepW_{tag}']), n_report=int(d['n_report']), role='primary')
    if np.isfinite(w[iu]).all():
        rec.update(Imean=float(w[iu].mean()), Imin=float(w[iu].min()), Imax=float(w[iu].max()),
                   Wgraph=float(G.cut_value(w, range(bc))))
    h = N // 2
    sel = sz == h
    fin = np.isfinite(sig) & proper(run)
    if (sel & fin).any():
        rec['S_max_layout'] = float(sig[sel & fin].max())
        rec['S_min_layout'] = float(sig[sel & fin].min())
    return rec


def bond_summary(run, T_win=None, include_start=True):
    """What every run (exact or matrix-product-state) has in common, for the window [0, T_win]: the maxima sigma of the
    contiguous bonds, the central bond (the larger of floor(N/2), ceil(N/2)) and its Schmidt ranks, the largest ranks over
    all bonds, the one-particle entropies.  Needs only the arrays of a run, so it works for any run layout of `store`."""
    N, k = n_particles(run), int(run['meta']['k'])
    ok = window(run, T_win, include_start)
    sig, tat = maxima(run, T_win, include_start)
    cols = _bond_columns(run)
    bonds = [float(sig[j]) for j in cols]
    if not np.isfinite(bonds).all():
        raise ValueError(f"contiguous bond(s) {[b + 1 for b, v in enumerate(bonds) if not np.isfinite(v)]} were not measured "
                         f"in {run.get('path', 'this run')}; S_cen and S_nat are undefined")
    jc = N // 2 - 1 if bonds[N // 2 - 1] >= bonds[(N + 1) // 2 - 1] else (N + 1) // 2 - 1
    out = dict(N=N, k=k, c=float(run['meta']['c']), T=float(run['t'][ok].max()), bonds=bonds, jc=jc, bond_central=jc + 1,
               S_nat=max(bonds), S_cen=bonds[jc], tS_cen=float(tat[cols[jc]]),
               Omin=float(run['O'][ok].min()), Omax=float(run['O'][ok].max()))
    for key, name in (('chi2', 'bond_chi2'), ('chi3', 'bond_chi3'), ('chi6', 'bond_chi6')):
        v = run[name][ok]
        out[key + '_nat'] = int(np.nanmax(v)) if np.isfinite(v).any() else None
        out[key + '_cen'] = int(np.nanmax(v[:, jc])) if np.isfinite(v[:, jc]).any() else None
    S1 = [float(np.nanmax(S_of(run, (i,))[ok])) for i in range(N)]
    out['S_one'], out['S1mean'] = max(S1), float(np.mean(S1))
    return out


def _ranks(run, ok, jc):
    """Second-order entropy and Schmidt ranks of the central bond."""
    out = {}
    for key, name, cast in (('S2_cen', 'bond_S2', float), ('chi2', 'bond_chi2', int),
                            ('chi3', 'bond_chi3', int), ('chi6', 'bond_chi6', int)):
        v = run[name][ok][:, jc]
        if np.isfinite(v).any():
            out[key] = cast(np.nanmax(v))
    return out


def _bond_columns(run):
    N = n_particles(run)
    return [column(run, tuple(range(b))) for b in range(1, N)]


# ------------------------------------------------------------------ graph record
def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else float('nan')


def graph_record(run, T_win=None, exhaustive_max_N=8, include_start=True):
    """Exact cut entropies against the graph estimate, for one run and window."""
    m = run['meta']
    N = n_particles(run)
    sig, tat = maxima(run, T_win, include_start)
    ok = window(run, T_win, include_start)
    w = edge_weights(run, T_win, include_start)
    S1max = one_particle_max(run, T_win, include_start)
    sz = sizes(run)
    sigma_of = {int(mk): float(s) for mk, s in zip(run['masks'], sig)}

    def sigma(A):
        return sigma_of[canon_mask(mask_of(A), N)]

    b0 = N // 2
    nat = [sigma(range(b)) for b in range(1, N)]
    perp_sizes = range(math.ceil(N / 3), N // 2 + 1)
    fin = np.isfinite(sig) & proper(run)
    perp = float(min(sig[(sz == a) & fin].min() for a in perp_sizes if ((sz == a) & fin).any()))
    smax = float(sig[(sz == b0) & fin].max())
    smin_cen = float(sig[(sz == b0) & fin].min())
    S1mean = float(S1max.mean())
    Tend = float(run['t'][ok].max())
    at_end = fin & (tat >= Tend - 1e-9)                    # cuts whose maximum is the last snapshot: lower bounds
    cen_bond = max(nat[b0 - 1], nat[(N + 1) // 2 - 1])
    Wunif = b0 * (N - b0) / (N - 1) * S1mean
    out = dict(N=N, k=int(m['k']), c=float(m['c']),
               T=float(run['t'][ok].max()), nsnap=int(ok.sum()), dt=float(m['protocol']['dt_snap']),
               protocol=m['protocol']['kind'], file=run.get('path', ''),
               O_min=float(run['O'][ok].min()), O_max=float(run['O'][ok].max()), O_start=float(run['O'][0]),
               S1max=S1max.tolist(), S1mean=S1mean,
               S_nat=float(max(nat)), S_nat_bonds=nat, S_cen_bond=float(cen_bond),
               S_perp=perp, S_max=smax, S_min_cen=smin_cen, W_unif=float(Wunif),
               n_cuts=int(fin.sum()), n_edge=int(at_end.sum()),
               edge_cuts=[list(members(int(mk), N)) for mk in run['masks'][at_end]])
    if not np.isfinite(w[np.triu_indices(N, 1)]).all():
        return out
    iu = np.triu_indices(N, 1)
    out.update(w=w.tolist(), w_mean=float(w[iu].mean()), w_min=float(w[iu].min()), w_max=float(w[iu].max()),
               W_graph=G.cut_value(w, range(b0)))
    cutcols = [j for j in range(len(sig)) if fin[j]]
    exact_all = [(members(int(run['masks'][j]), N), float(sig[j])) for j in cutcols]
    diffs = [s - G.cut_value(w, A) for A, s in exact_all]
    out['scatter'] = [dict(A=list(A), exact=s, graph=G.cut_value(w, A)) for A, s in exact_all]
    out['diff_min'], out['diff_max'] = float(min(diffs)), float(max(diffs))
    out['monogamy'] = [float(w[i].sum() / (2 * S1max[i])) for i in range(N)]
    tb = tat[run['masks'] == mask_of(range(b0))]
    out['t_cen'] = float(tb[0]) if len(tb) else None
    # layouts: exhaustive for small N, recursive bisection (graph) beyond
    natural_graph = float(max(G.prefix_cuts(w, list(range(N)))))
    lay = dict(natural_exact=float(max(nat)), natural_graph=natural_graph)
    if N <= exhaustive_max_N:
        orders = [p for p in itertools.permutations(range(N)) if p[0] < p[-1]]
        ce = np.array([max(sigma(p[:b]) for b in range(1, N)) for p in orders])
        cg = np.array([max(G.cut_value(w, p[:b]) for b in range(1, N)) for p in orders])
        ie, ig = int(np.argmin(ce)), int(np.argmin(cg))
        lay.update(n_orders=len(orders), best_exact=float(ce.min()), best_exact_order=list(orders[ie]),
                   best_graph_order=list(orders[ig]), exact_cost_of_graph_best=float(ce[ig]),
                   worst_exact=float(ce.max()), spearman=spearman(ce, cg),
                   n_ties_best=int(np.sum(ce <= ce.min() + 1e-9)))
    else:
        bt = G.bisection_tree(w)
        lay.update(bisection_order=bt['order'], bisection_tree_cost=bt['tree_cost'],
                   bisection_mps_cost=bt['mps_cost'])
    out['layout'] = lay
    return out


def cosine_residual(run, w_lo=0.02, w_hi=0.6, n_w=1500):
    """How well one cosine explains the signal <O>(t): the splitting-based prediction of the signal.

    O(t) ~ a + b cos(w t) + c sin(w t), amplitudes by linear least squares.  Returns the rms residual as a fraction of
    the standard deviation of <O>(t) (1 = nothing explained), for w = the measured doublet splitting dE ('fixed')
    and for the best w on a grid ('free', with that w)."""
    t = np.asarray(run['t'], float)
    O = np.asarray(run['O'], float)
    sd = float(np.std(O))

    def resid(w):
        X = np.c_[np.ones_like(t), np.cos(w * t), np.sin(w * t)]
        c, *_ = np.linalg.lstsq(X, O, rcond=None)
        return float(np.sqrt(np.mean((O - X @ c) ** 2)))

    dE = float(run['meta']['diag']['dE'])
    ws = np.linspace(w_lo, w_hi, n_w)
    rs = np.array([resid(w) for w in ws])
    j = int(np.argmin(rs))
    return dict(dE=dE, sd=sd, fixed=resid(dE) / sd, free=float(rs[j]) / sd, w_free=float(ws[j]))
