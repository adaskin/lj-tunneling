"""ljtunnel.followup_spectra -- Stage 1 of followup/TASK_followup_rival_and_mps.md: the spectral structure of the correlated start.

Static diagonalization only (no propagation): the lowest L levels of H on the k^N mode space, the weights of the start on them, the
two heaviest levels (B2, omega*), the levels strictly between them (D), and the star-hypothesis tests.  The definitions are in
data/followup/PREREG.md; every number written by this module comes from a saved file under data/followup/stage1/.
"""
import copy
import os
import time

import numpy as np
from scipy.optimize import brentq
from scipy.sparse.linalg import eigsh, ArpackNoConvergence, LinearOperator

from .config import ROOT
from .hamiltonian import ModeN, dense_H
from .log import log, tee
from .model import Model
from .starts import configs, product_state, interacting_state
from .store import save_run, load_run

MERGE_TOL = 1e-6          # levels closer than this (energy units) are one level with the summed weight
DENSE_MAX = 8000          # k^N up to this: dense eigh
OUTDIR = os.path.join(ROOT, 'data', 'followup', 'stage1')


def start_key(N):
    """The valley assignment of every stored run: particle 0 right, the others left."""
    return f"{N - 1}1_1"


def lanczos_gb(dim, L):
    """Memory of the ARPACK basis (ncv = 2L + 1 vectors of float64)."""
    return (2 * L + 1) * dim * 8 / 1e9


def levels(M, L, v0=None, tol=1e-9, dense_max=DENSE_MAX, seed=0, maxiter=2000):
    """The lowest L eigenpairs (ascending) of H on the k^N space, plus the number of H v products used (0 for dense)."""
    dim = M.dim
    if dim <= dense_max:
        w, v = np.linalg.eigh(dense_H(M))
        return w[:L], v[:, :L], 0
    L = min(L, dim - 2)
    ncv = min(max(2 * L + 1, 40), dim - 1)
    if v0 is None:
        v0 = np.random.default_rng(seed).standard_normal(dim)
    count = [0]

    def mv(x):
        count[0] += 1
        return M.matvec(x)
    op = LinearOperator((dim, dim), matvec=mv, dtype=float)
    try:
        ev, vc = eigsh(op, k=L, which='SA', tol=tol, v0=v0, ncv=ncv, maxiter=maxiter)
    except ArpackNoConvergence:
        log(f"      ARPACK did not converge at tol {tol:g} in {maxiter} restarts (L={L}): retrying at tol 1e-7")
        ev, vc = eigsh(op, k=L, which='SA', tol=1e-7, v0=v0, ncv=ncv, maxiter=maxiter)
    o = np.argsort(ev)
    return ev[o], vc[:, o], count[0]


def occupancy_gs(M, v0=None):
    """<O> of the exact ground state (and the vector, for warm starts)."""
    ev, vc, _ = levels(M, 2, v0=v0, tol=1e-8, dense_max=2000)
    return float(M.O_matrix(vc[:, :1])[0, 0].real), vc[:, 0]


def tilt_half(model, N, k, c0, step=0.1, xtol=1e-4, max_steps=40):
    """The tilt at which the exact ground state has <O> = 1/2 (the target of resonance.exact_tilt), by bracketing from c0 in steps of
    `step` (<O>_gs falls with c) and brentq; rounded to 4 decimals."""
    state = {'v': None, 'n': 0}
    cache = {}

    def f(c):
        if c in cache:                                       # brentq re-evaluates the ends of the bracket
            return cache[c]
        M = ModeN(model, c, k, N)
        o, state['v'] = occupancy_gs(M, state['v'])
        state['n'] += 1
        log(f"   [tilt N={N} k={k}] c={c:.5f} <O>_gs={o:.5f}")
        cache[c] = o - 0.5
        return o - 0.5
    a, fa = c0, f(c0)
    if fa > 0:
        b = a + step
        fb = f(b)
        for _ in range(max_steps):
            if fb <= 0:
                break
            a, fa, b = b, fb, b + step
            fb = f(b)
    else:
        b, fb = a, fa
        a = b - step
        fa = f(a)
        for _ in range(max_steps):
            if fa >= 0:
                break
            b, fb, a = a, fa, a - step
            fa = f(a)
    if not (fa > 0 >= fb):
        raise ValueError(f"<O>_gs = 1/2 not bracketed near c0={c0} at N={N}, k={k}")
    return round(brentq(f, a, b, xtol=xtol, rtol=1e-6), 4)


def merge_levels(ev, *weights, tol=MERGE_TOL):
    """Levels closer than `tol` become one (mean energy, summed weights, multiplicity)."""
    grp = np.concatenate([[0], np.cumsum(np.diff(ev) >= tol)])
    n = grp[-1] + 1
    mult = np.bincount(grp, minlength=n)
    E = np.bincount(grp, weights=ev, minlength=n) / mult
    return (E, mult) + tuple(np.bincount(grp, weights=w, minlength=n) for w in weights)


def pair_summary(E, w, N, coverage):
    """B2, omega*, D, n95 ... of one start, from merged levels E (ascending) with weights w; `coverage` = total weight found."""
    order = np.argsort(w)[::-1]
    a, b = sorted(order[:2].tolist())
    B2 = float(w[a] + w[b])
    omega = float(E[b] - E[a])
    mid = 0.5 * (E[a] + E[b])
    btw = list(range(a + 1, b))
    D = float(max(w[btw])) if btw else 0.0
    cum = np.cumsum(w[order])
    n95 = int(np.argmax(cum >= 0.95)) + 1 if cum[-1] >= 0.95 else None
    miss = max(1.0 - coverage, 0.0)
    pair_certain = bool(miss < float(np.sort(w)[-2]))          # an unfound level cannot outweigh the second heaviest found
    top6 = float(cum[min(5, len(cum) - 1)])
    holds = bool(B2 >= 0.80 and D <= 0.02 and pair_certain)
    tail = bool(top6 + miss < 0.95)                           # n95 > 6 even if every unfound level joined the top six
    breaks = bool((B2 + miss < 0.60) or tail)
    # the two pre-registered conditions can both be met (two levels hold B2 >= 0.8, a long thin tail makes n95 > 6): 'holds+tail' (PREREG amendment 1)
    verdict = ('holds+tail' if tail else 'holds') if holds else 'breaks' if breaks else ('intermediate' if coverage >= 0.99 else 'unresolved')
    return dict(B2=B2, a=int(a), b=int(b), w_a=float(w[a]), w_b=float(w[b]), omega=omega, omega_over_sqrtN=omega / np.sqrt(N),
                D=D, n_between=len(btw), between=[float(E[j] - mid) for j in btw], between_w=[float(w[j]) for j in btw],
                n95=n95, n95_gt6=tail, top6=top6, coverage=float(coverage), pair_certain=pair_certain, verdict=verdict)


def analyse_case(model, N, k, c, W=30.0, L0=40, Lmax=320, gb_cap=30.0, tag='', outdir=OUTDIR, stored=None, force=False):
    """One (N, k, c): levels, weights of the correlated and the product start, summary, saved to outdir/N{N}_k{k}_c{c}{tag}.npz.

    L doubles from L0 while the coverage (weight of the correlated start on the levels found) is below 0.99 and the ARPACK basis fits in gb_cap.
    stored  path of a stored run (data/runs/*.npz): its weights_interacting are compared with the new ones (reproduction)."""
    path = os.path.join(outdir, f"N{N}_k{k}_c{c:.4f}{tag}.npz")
    if os.path.exists(path) and not force:
        log(f"[N={N} k={k}{tag}] {os.path.relpath(path, ROOT)} exists: skipped")
        return load_run(path)
    t0 = time.time()
    M = ModeN(model, c, k, N)
    sides = configs(N)[start_key(N)]
    prod = product_state(model, c, M, sides)
    inter = interacting_state(M, prod, sides, W)
    L, nmv = (M.dim if M.dim <= DENSE_MAX else L0), 0           # dense: all levels come with the diagonalization
    while True:
        t1 = time.time()
        ev, vc, n1 = levels(M, L)
        nmv += n1
        wI = (vc.T @ inter.reshape(-1)) ** 2
        wP = (vc.T @ prod.reshape(-1)) ** 2
        cov = float(wI.sum())
        log(f"[N={N} k={k}{tag}] c={c:.4f}: L={len(ev)} levels in {time.time() - t1:.0f} s ({n1} H v), coverage correlated {cov:.4f}, product {wP.sum():.4f}")
        if cov >= 0.99 or M.dim <= DENSE_MAX or 2 * L > Lmax or lanczos_gb(M.dim, 2 * L) > gb_cap:
            break
        L *= 2
    E, mult, mwI, mwP = merge_levels(ev, wI, wP)
    summ = {'interacting': pair_summary(E, mwI, N, cov), 'product': pair_summary(E, mwP, N, float(wP.sum()))}
    repro = None
    if stored:
        st = load_run(stored)['meta']['diag']
        sw = np.array(st['weights_interacting'])
        n = min(len(sw), len(wI))
        repro = dict(stored=os.path.relpath(stored, ROOT), n=int(n), max_abs_diff=float(np.abs(sw[:n] - wI[:n]).max()),
                     stored_dE=float(st['dE']), stored_doublet_levels=st['doublet_levels'], stored_sum=float(sw.sum()))
        log(f"      reproduction against {repro['stored']}: max |dw| over {n} levels = {repro['max_abs_diff']:.2e}")
    s = summ['interacting']
    log(f"      correlated: B2={s['B2']:.3f} pair=({s['a']},{s['b']}) omega*={s['omega']:.4f} omega*/sqrtN={s['omega_over_sqrtN']:.4f} "
        f"D={s['D']:.3f} n_between={s['n_between']} n95={s['n95']} -> {s['verdict']}")
    meta = dict(N=N, k=k, c=c, W=W, tag=tag, start_key=start_key(N), L=int(len(ev)), n_h_v=nmv, merge_tol=MERGE_TOL, summary=summ,
                reproduction=repro, wall_s=time.time() - t0, dim=int(M.dim), lanczos_gb=lanczos_gb(M.dim, len(ev)),
                C0=float(model.mp['C0']), theta_barrier=float(model.mp.get('theta_barrier', 0.5)))
    save_run(path, dict(E=E, mult=mult, w_interacting=mwI, w_product=mwP, ev_raw=ev, w_interacting_raw=wI, w_product_raw=wP), meta)
    log(f"      wrote {os.path.relpath(path, ROOT)} ({time.time() - t0:.0f} s)")
    return load_run(path)


def lanczos_coeffs(M, psi, m):
    """m Lanczos steps from psi without reorthogonalization (three vectors): the diagonal and off-diagonal of the tridiagonal matrix."""
    v = psi.reshape(-1).astype(float)
    v = v / np.linalg.norm(v)
    vprev, beta, al, be = np.zeros_like(v), 0.0, [], []
    for _ in range(m):
        w = M.matvec(v)
        w -= beta * vprev
        a = float(np.dot(v, w))
        w -= a * v
        b = float(np.linalg.norm(w))
        al.append(a)
        be.append(b)
        if b < 1e-10:
            break
        vprev, v, beta = v, w / b, b
    return np.array(al), np.array(be)


def measure_from(al, be, m):
    """Nodes theta_j, weights |<e_1|s_j>|^2 and Ritz residuals |beta_m s_mj| of the leading m x m block (= an m-step Lanczos run).  Ghost copies of a
    converged level lie closer than MERGE_TOL and are merged with their weights summed (merge_levels); a node with a large residual belongs to the
    part of the measure that is not yet resolved (excited continuum)."""
    from scipy.linalg import eigh_tridiagonal
    m = min(m, len(al))
    th, S = eigh_tridiagonal(al[:m], be[:m - 1])
    return th, S[0] ** 2, be[m - 1] * np.abs(S[-1])


def krylov_case(model, N, k, c, W=30.0, ms=(500, 1000, 2000), res_tol=1e-6, tag='_krylov', outdir=OUTDIR, force=False, product=True):
    """Stage 1 summary of one (N, k, c) from the spectral measure of the start (for cases where the lowest-L-levels method cannot reach a coverage
    of 0.99).  One Lanczos run of max(ms) steps; the summary is taken at each m in ms (leading blocks) and the run is accepted at the first m at which
    B2 and omega* moved by < 1e-4 from the previous m (else at the last).  Coverage = weight on nodes with Ritz residual < res_tol (resolved levels).
    product=False skips the product start.  Saved as outdir/N{N}_k{k}_c{c}_krylov.npz (same layout as analyse_case)."""
    path = os.path.join(outdir, f"N{N}_k{k}_c{c:.4f}{tag}.npz")
    if os.path.exists(path) and not force:
        log(f"[N={N} k={k}{tag}] exists: skipped")
        return load_run(path)
    t0 = time.time()
    M = ModeN(model, c, k, N)
    sides = configs(N)[start_key(N)]
    prod = product_state(model, c, M, sides)
    inter = interacting_state(M, prod, sides, W)
    res = {}
    for name, psi in (('interacting', inter), ('product', prod)):
        if name == 'product' and not product:
            continue
        t1 = time.time()
        al, be = lanczos_coeffs(M, psi, max(ms))
        log(f"[N={N} k={k}{tag}] {name}: {len(al)} Lanczos steps in {time.time() - t1:.0f} s")
        prev = None
        for m in ms:
            th, w, rs = measure_from(al, be, m)
            good = rs < res_tol
            E, mult, wm = merge_levels(th[good], w[good])
            cov = float(w[good].sum())
            sm = pair_summary(E, wm, N, cov)
            log(f"      {name}: m={m}: resolved coverage {cov:.4f} in {len(E)} levels, B2={sm['B2']:.4f} pair=({sm['a']},{sm['b']}) "
                f"omega*={sm['omega']:.5f} D={sm['D']:.4f} n95={sm['n95']} -> {sm['verdict']}")
            if prev is not None and abs(sm['B2'] - prev['B2']) < 1e-4 and abs(sm['omega'] - prev['omega']) < 1e-4 and sm['a'] == prev['a'] and sm['b'] == prev['b']:
                break
            prev = sm
        res[name] = (E, mult, wm, sm, m, cov, th, w, rs)
    meta = dict(N=N, k=k, c=c, W=W, tag=tag, start_key=start_key(N), L=int(len(res['interacting'][0])), n_h_v=int(sum(r[4] for r in res.values())),
                merge_tol=MERGE_TOL, res_tol=res_tol, summary={n_: r[3] for n_, r in res.items()}, reproduction=None, wall_s=time.time() - t0, dim=int(M.dim),
                m_used={n_: r[4] for n_, r in res.items()}, method='krylov spectral measure', C0=float(model.mp['C0']),
                theta_barrier=float(model.mp.get('theta_barrier', 0.5)))
    if 'product' not in res:
        meta['summary']['product'] = dict(B2=float('nan'), omega=float('nan'), omega_over_sqrtN=float('nan'), D=float('nan'), n_between=0, n95=None,
                                          coverage=float('nan'), verdict='not computed')
    E, mult, wm = res['interacting'][:3]
    save_run(path, dict(E=E, mult=mult, w_interacting=wm, w_product=np.zeros_like(wm), theta=res['interacting'][6], w_theta=res['interacting'][7],
                        resid=res['interacting'][8]), meta)
    log(f"      wrote {os.path.relpath(path, ROOT)} ({time.time() - t0:.0f} s)")
    return load_run(path)


def first_large_gap(ev, ratio=3.0):
    """Index j+1 = size of the lowest band: the first j >= 1 with gap g_j > ratio * max(g_0 .. g_{j-1}); None if there is none."""
    g = np.diff(ev)
    for j in range(1, len(g)):
        if g[j] > ratio * g[:j].max():
            return j + 1
    return None


def analyse_untilted(model, N, k, c=0.0, L=64, tag='', outdir=OUTDIR, force=False):
    """(e): the lowest L levels at tilt c (default 0), the lowest band (first large gap), its size against N+1 and 2^N, and the weights of the
    all-left product state and of the N single-right product states on the band."""
    path = os.path.join(outdir, f"untilted_N{N}_k{k}_c{c:.4f}{tag}.npz")
    if os.path.exists(path) and not force:
        log(f"[untilted N={N} k={k}{tag}] exists: skipped")
        return load_run(path)
    t0 = time.time()
    M = ModeN(model, c, k, N)
    ev, vc, nmv = levels(M, L)
    nb = first_large_gap(ev)
    cf = configs(N)
    wband = {}
    for name, sides in cf.items():
        p = product_state(model, c, M, sides)
        w = (vc.T @ p.reshape(-1)) ** 2
        wband[name] = dict(band=float(w[:nb].sum()) if nb else None, total=float(w.sum()))
    g = np.diff(ev)
    meta = dict(N=N, k=k, c=c, tag=tag, L=int(len(ev)), band_size=nb, N_plus_1=N + 1, two_pow_N=2 ** N, C0=float(model.mp['C0']),
                gap_after_band=float(g[nb - 1]) if nb else None, max_gap_in_band=float(g[:nb - 1].max()) if nb and nb > 1 else None,
                band_width=float(ev[nb - 1] - ev[0]) if nb else None, weights_of_products=wband, n_h_v=nmv, wall_s=time.time() - t0)
    save_run(path, dict(ev=ev, E_minus_E0=ev - ev[0]), meta)
    log(f"[untilted N={N} k={k}{tag}] c={c}: C0={meta['C0']}: lowest band = {nb} levels (N+1 = {N + 1}, 2^N = {2 ** N}), width "
        f"{meta['band_width']}, gap after {meta['gap_after_band']}; all-left weight on the band {wband[f'{N}0']['band']} ({time.time() - t0:.0f} s)")
    return load_run(path)


def model_with_C0(cfg, C0):
    """A model with another offset C0 (an in-memory copy of the configuration; config.toml is not touched)."""
    cfg2 = copy.deepcopy(cfg)
    cfg2['model']['C0'] = float(C0)
    return Model(cfg2)


# -- the table and the gate (everything below reads saved files only) ------------------------------------------------------------
def collect(outdir=OUTDIR):
    """One row per saved case (tilted, correlated start) from outdir, sorted by (N, k, tag)."""
    rows = []
    for f in sorted(os.listdir(outdir)) if os.path.isdir(outdir) else []:
        if f.startswith('N') and f.endswith('.npz'):
            m = load_run(os.path.join(outdir, f))['meta']
            s, p = m['summary']['interacting'], m['summary']['product']
            rows.append(dict(N=m['N'], k=m['k'], c=m['c'], tag=m['tag'], L=m['L'], coverage=s['coverage'], B2=s['B2'], pair=(s['a'], s['b']),
                             omega=s['omega'], omega_sqrtN=s['omega_over_sqrtN'], D=s['D'], n_between=s['n_between'], n95=s['n95'],
                             verdict=s['verdict'], B2_prod=p['B2'], cov_prod=p['coverage'], wall_s=m['wall_s'], file=f,
                             repro=(m['reproduction'] or {}).get('max_abs_diff')))
    rows.sort(key=lambda r: (r['N'], r['k'], r['tag']))
    return rows


def best_rows(rows):
    """Main cases: per (N, k) the lowest-L-levels row, replaced by the spectral-measure row (tag '_krylov') when that resolves more of the start."""
    by = {}
    for r in rows:
        if r['tag'] in ('', '_krylov'):
            by.setdefault((r['N'], r['k']), {})[r['tag']] = r
    out = []
    for key in sorted(by):
        d = by[key]
        out.append(d['_krylov'] if '_krylov' in d and ('' not in d or d['_krylov']['coverage'] > d['']['coverage']) else d[''])
    return out


def gate1(rows):
    """The pre-registered Gate 1 quantities from the collected rows (main cases only: tag '')."""
    main = best_rows(rows)
    best = {}
    for r in main:
        best.setdefault(r['N'], []).append(r)
    for N in best:
        best[N].sort(key=lambda r: r['k'])
    out = {}
    # star-frequency test: omega*/sqrt(N) at the largest k of each N = 3..6
    vals = {N: best[N][-1]['omega_sqrtN'] for N in range(3, 7) if N in best}
    if len(vals) == 4:
        out['freq'] = dict(values=vals, k={N: best[N][-1]['k'] for N in vals}, spread=max(vals.values()) / min(vals.values()) - 1,
                           passes=bool(max(vals.values()) / min(vals.values()) - 1 <= 0.10))
    # k-convergence at N = 4, 5: the two largest k
    for N in (4, 5):
        rs = best.get(N, [])
        if len(rs) >= 2:
            r1, r2 = rs[-2], rs[-1]
            do, db = abs(r2['omega'] / r1['omega'] - 1), abs(r2['B2'] / r1['B2'] - 1)
            out[f'conv_N{N}'] = dict(k=(r1['k'], r2['k']), d_omega=do, d_B2=db, passes=bool(do <= 0.05 and db <= 0.05))
    # N = 6: the two largest k
    rs = best.get(6, [])
    if len(rs) >= 2:
        out['N6'] = dict(k=[r['k'] for r in rs[-2:]], verdict=[r['verdict'] for r in rs[-2:]], B2=[r['B2'] for r in rs[-2:]],
                         coverage=[r['coverage'] for r in rs[-2:]], holds=bool(all(r['verdict'] in ('holds', 'holds+tail') for r in rs[-2:])),
                         breaks=bool(all(r['verdict'] == 'breaks' for r in rs[-2:])))
    return out
