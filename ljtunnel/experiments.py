"""ljtunnel.experiments -- the simulations, one function each.  Every function returns what it
computed and writes an .npz (plus a .log) under data/; none of them reads a log file.

    dynamics_run   resonance + start states + real-time series of all cut entropies -> data/runs/
"""
import json
import os
import shutil
import time

import numpy as np

from . import __version__
from .config import protocol as make_protocol, ROOT
from .dynamics import spectral_bounds, snapshots, matvec_count
from .entanglement import CutPlan
from .hamiltonian import ModeN
from .log import log, tee
from .resonance import locate
from .starts import prepare
from .store import save_run, run_name


def state_gb(N, k, n_vectors=1):
    return 16.0 * k ** N * n_vectors / 1e9


def _save_atomic(path, **arrays):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path[:-4] + '.tmp.npz'
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def measure_series(M, psi0, prot, plan, label='', ckpt=None, ckpt_every=5):
    """Evolve psi0 over the protocol's snapshot grid and record, at every snapshot,
    <O>, the per-particle <Theta(x_i)> and the entropies of all cuts in `plan`.

    ckpt   path of a checkpoint file (.npz).  Every `ckpt_every` snapshots the state and everything measured so far are written
           there; a rerun with the same file continues from the last one (the propagation is stepwise from snapshot to snapshot, so the
           continuation is the same computation; the spectral bounds are reused).  A hour-long series then survives a restart.
    Returns (arrays, info).  arrays: t, O, Oi, S (nt, n_cuts), bond_S2/chi2/chi3/chi6 (nt, N-1)."""
    N = M.N
    ts = np.arange(prot['nsnap']) * prot['dt_snap']
    nt = len(ts)
    O = np.zeros(nt)
    Oi = np.zeros((nt, N))
    S = np.full((nt, plan.n_cuts), np.nan)
    bond = np.full((nt, N - 1, 4), np.nan)
    key = dict(N=N, k=plan.k, label=label, T=prot['T'], dt=prot['dt_snap'], ncuts=plan.n_cuts)
    sig0 = float(np.abs(psi0).sum())
    i0, psi_from, bounds, series_prev = 0, psi0, None, 0.0
    if ckpt and os.path.exists(ckpt):
        try:
            with np.load(ckpt, allow_pickle=False) as d:
                if json.loads(str(d['key'])) == key and abs(float(d['sig0']) - sig0) <= 1e-6 * sig0:
                    i0, psi_from, bounds, series_prev = int(d['i']), d['psi'], tuple(float(x) for x in d['bounds']), float(d['series_wall'])
                    O[:], Oi[:], S[:], bond[:] = d['O'], d['Oi'], d['S'], d['bond']
                else:
                    log(f"      [{label}] checkpoint {os.path.basename(ckpt)} belongs to another run: ignored")
        except Exception as e:                                   # noqa: BLE001  (a torn or foreign file: start from the beginning)
            log(f"      [{label}] checkpoint unreadable ({type(e).__name__}): starting from the beginning")
            i0, psi_from, bounds, series_prev = 0, psi0, None, 0.0
    if bounds is None:
        bounds = spectral_bounds(M)
    nmv = matvec_count(bounds, prot['T'], prot['dt_snap'])
    log(f"      [{label}] spectrum [{bounds[0]:.1f},{bounds[1]:.1f}], T={prot['T']:g}, dt={prot['dt_snap']:g}, "
        f"{prot['nsnap']} snapshots, ~{nmv} matvecs; {plan.n_cuts} cuts ({plan.n_skipped()} skipped: dimension)")
    if i0:
        log(f"      [{label}] RESUMED from the checkpoint at snapshot {i0} of {nt - 1} (t = {ts[i0]:g}); "
            f"{series_prev:.0f} s of an earlier process are kept")
    t0 = time.time()
    psi = psi_from
    for j, t, psi in snapshots(M, psi_from, ts[i0:], bounds):
        i = i0 + j
        if i0 and j == 0:
            continue                                             # measured before the restart
        Oi[i] = M.O_particle(psi)
        O[i] = Oi[i].sum()
        S[i], bond[i] = plan.measure(psi, i)
        if i and (i % 10 == 0 or i == nt - 1):
            el = time.time() - t0
            log(f"      [{label}] t={t:6.1f}  <O>={O[i]:.3f}  elapsed {el:6.0f}s  eta {el / max(i - i0, 1) * (nt - 1 - i):6.0f}s")
        if ckpt and i % ckpt_every == 0 and i < nt - 1:
            _save_atomic(ckpt, key=np.array(json.dumps(key)), sig0=sig0, i=i, psi=psi, bounds=np.array(bounds), O=O, Oi=Oi, S=S,
                         bond=bond, series_wall=series_prev + time.time() - t0)
    nrm = float(np.linalg.norm(psi))
    if abs(nrm - 1) > prot['norm_tol']:
        log(f"      WARNING: norm drift {nrm - 1:+.2e} -- Chebyshev expansion not converged")
    arrays = dict(t=ts, O=O, Oi=Oi, masks=plan.masks, S=S, bond_b=plan.bond_b,
                  bond_S2=bond[:, :, 0], bond_chi2=bond[:, :, 1], bond_chi3=bond[:, :, 2],
                  bond_chi6=bond[:, :, 3])
    return arrays, dict(norm=nrm, bounds=[float(bounds[0]), float(bounds[1])], matvecs=nmv, resumed_from=i0, series_wall_before=series_prev)


def dynamics_run(model, N, k=None, starts=None, tilt='auto', c_guess=None, T=None, dt_snap=None,
                 W=None, cuts=None, outdir=None, force=False, big_cut_stride=None, checkpoint=False):
    """One resonance + one (or two) start states + the full entropy time series.

    N, k         particle number and modes per particle (default: config [basis] k)
    starts       ['interacting'] (correlated start, default) and/or ['product']
    tilt         'auto' | 'scan' | float;  c_guess refines around a tilt instead of scanning
    T, dt_snap   override the protocol (the run is then stamped 'custom' and not mixed with
                 uniform runs); leave None for the uniform protocol of config.toml
    cuts         'all' | 'bonds+pairs' | 'bonds'  (default config)
    checkpoint   True (needs a float tilt): the prepared start states and, every few snapshots, the state and the entropies so far
                 are written to data/ckpt/; a rerun of the same call continues from them (a restart costs minutes, not the hours of
                 locate + prepare + series), and the directory is removed when the run is complete.
    Returns the list of written .npz paths."""
    cfg = model.cfg
    k = k or cfg['basis']['k']
    prot = make_protocol(cfg, T=T, dt_snap=dt_snap, cuts=cuts, W_penalty=W, starts=starts,
                         big_cut_stride=big_cut_stride)
    outdir = outdir or os.path.join(ROOT, 'data', 'runs')
    logdir = os.path.join(ROOT, 'data', 'logs')
    if state_gb(N, k) > prot['max_state_gb'] and not force:
        raise MemoryError(f"N={N}, k={k}: one state vector is {state_gb(N, k):.1f} GB "
                          f"(> max_state_gb = {prot['max_state_gb']}); a run needs ~5 vectors. "
                          f"Raise [protocol] max_state_gb or pass force=True.")
    t_all = time.time()
    if checkpoint and not isinstance(tilt, (int, float)):
        raise ValueError("checkpoint=True needs a fixed tilt (a float)")
    ckdir = os.path.join(ROOT, 'data', 'ckpt', f"N{N}_k{k}_c{float(tilt):.3f}_T{prot['T']:g}_dt{prot['dt_snap']:g}") if checkpoint else None
    tag = f"N{N}_k{k}"
    with tee(os.path.join(logdir, f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}_p{os.getpid()}.log")):    # pid: parallel jobs of one (N, k)
        log(f"[dynamics N={N} k={k}] protocol {prot['kind']}: T={prot['T']:g} dt={prot['dt_snap']:g} "
            f"starts={prot['starts']} cuts={prot['cuts']}")
        carry_start, prep = 0.0, None
        start_file = os.path.join(ckdir, 'start.npz') if ckdir else None
        if start_file and os.path.exists(start_file):
            try:
                with np.load(start_file, allow_pickle=False) as d:
                    if json.loads(str(d['key'])) == dict(N=N, k=k, c=float(tilt), W=float(prot['W_penalty'])):
                        prep = dict(diag=json.loads(str(d['diag'])), states={'product': d['product'], 'interacting': d['interacting']},
                                    key=json.loads(str(d['diag']))['start_key'])
                        carry_start = float(d['wall_start'])
                        loc = dict(M=ModeN(model, float(tilt), k, N), c=float(tilt), how='fixed')
                        log(f"[N={N} k={k}] RESUMED: resonance and start states loaded from {os.path.relpath(start_file, ROOT)} "
                            f"({carry_start:.0f} s of an earlier process are kept)")
            except Exception as e:                               # noqa: BLE001
                log(f"[N={N} k={k}] start checkpoint unreadable ({type(e).__name__}): recomputing")
                prep = None
        if prep is None:
            loc = locate(model, N, k, tilt=tilt, c_guess=c_guess)
            if loc is None:
                log(f"[N={N} k={k}] no resonance found")
                return []
            prep = prepare(model, loc, N, k, prot['W_penalty'])
            if start_file:
                _save_atomic(start_file, key=np.array(json.dumps(dict(N=N, k=k, c=float(tilt), W=float(prot['W_penalty'])))),
                             diag=np.array(json.dumps(prep['diag'], default=float)), product=prep['states']['product'],
                             interacting=prep['states']['interacting'], wall_start=time.time() - t_all)
        plan = CutPlan(N, k, prot['cuts'], prot['max_cut_dim'], prot['big_cut_stride'])
        paths = []
        for start in prot['starts']:
            ck = os.path.join(ckdir, f"series_{start}.npz") if ckdir else None
            arrays, info = measure_series(loc['M'], prep['states'][start], prot, plan, label=start, ckpt=ck)
            meta = dict(N=N, k=k, c=prep['diag']['c'], start=start, protocol=prot, cuts=prot['cuts'],
                        diag=prep['diag'], info=info, wall_s=time.time() - t_all + carry_start + info.get('series_wall_before', 0.0),
                        model=cfg['model'], grid=dict(cfg['grid'], sigma_smear=model.sigma_smear),
                        version=__version__, how_tilt=loc['how'])
            path = os.path.join(outdir, run_name(meta) + '.npz')
            if os.path.exists(path) and not force:
                path = path.replace('.npz', f"_{time.strftime('%H%M%S')}.npz")
            save_run(path, arrays, meta)
            if ck and os.path.exists(ck):
                os.remove(ck)
            log(f"[N={N} k={k}] {start}: wrote {os.path.relpath(path, ROOT)}  "
                f"<O> in [{arrays['O'].min():.3f}, {arrays['O'].max():.3f}]  ({time.time() - t_all:.0f}s)")
            paths.append(path)
        if ckdir and os.path.isdir(ckdir):
            shutil.rmtree(ckdir, ignore_errors=True)             # the run is complete: its checkpoints are no longer needed
    return paths
