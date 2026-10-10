"""Driver: ED reference + MPS run for a list of (N, k), same tilt, same start, same window; writes data/mps/*.npz and a json-lines summary.
usage: driver.py OUT.jsonl T chi_max eps dt  N:k[:pairs] ...
"""
import json, sys, time, os, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/lj-tunneling')
import ljtunnel as lj
from ljtunnel import mps, analysis, experiments, store
from ljtunnel.resonance import locate
from ljtunnel.starts import configs
from ljtunnel.log import log

out, T, chi_max, eps, dt = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
cfg = lj.load_config(); model = lj.Model(cfg)
ROOT = lj.ROOT
for spec in sys.argv[6:]:
    parts = spec.split(':'); N, k = int(parts[0]), int(parts[1]); pairs = 'pairs' in parts[2:]
    rec = dict(N=N, k=k, T=T, chi_max=chi_max, eps=eps, dt=dt)
    try:
        # tilt
        c = model.tilt(N, k)
        if c is None:
            from ljtunnel.resonance import doublet_at
            def occ(cc): return float(doublet_at(model, cc, k, N)[3][0, 0].real)
            t = time.time()
            c12 = cfg['tilts'].get(f'N{N}'); c12 = c12['k12'] if isinstance(c12, dict) else c12
            if c12 is None: c12 = cfg['tilts']['N5']['k12'] + 0.75 * (N - 5)
            lo, hi = c12 - 0.2, c12 + 0.9
            assert occ(lo) > 0.5 > occ(hi), (occ(lo), occ(hi))
            for _ in range(30):                               # bisection on <O>_gs = 1/2 (exact diagonalization)
                cm = 0.5 * (lo + hi); o = occ(cm)
                lo, hi = (cm, hi) if o > 0.5 else (lo, cm)
                if hi - lo < 1e-3: break
            c = round(0.5 * (lo + hi), 4); rec['t_locate'] = time.time() - t
            rec['tilt_source'] = 'ED bisection, <O>_gs=1/2'
        else:
            rec['tilt_source'] = 'config'
        rec['c'] = c
        # exact reference (same tilt, same window, all cuts)
        t = time.time()
        paths = experiments.dynamics_run(model, N, k, starts=['interacting'], tilt=c, T=T, dt_snap=1.0,
                                         cuts='bonds+pairs', outdir=os.path.join(ROOT, 'data', 'mps', 'ed_ref'), force=True)
        ed = store.load_run(paths[0]); rec['t_ed'] = time.time() - t
        key = ed['meta']['diag']['start_key']; rec['start_key'] = key
        # MPS from the same start
        prob = mps.MPSProblem(model, c, k, N)
        t = time.time()
        run = mps.run_series(prob, sides=configs(N)[key], T=T, dt_snap=1.0, dt=dt, chi_max=chi_max, eps=eps, pairs=pairs,
                             label=f'N{N}k{k}')
        rec['t_mps'] = time.time() - t
        mps.save(run)
        # comparison
        plan_ed = {int(m): j for j, m in enumerate(ed['masks'])}
        diffs = {}
        for j, m in enumerate(run['masks']):
            je = plan_ed[int(m)]
            d = np.nanmax(np.abs(run['S'][:, j] - ed['S'][:, je]))
            diffs[str(mps.CutPlan(N, k, 'bonds+pairs' if pairs else 'bonds').small[j])] = float(d)
        rec['max_dS_cuts'] = max(diffs.values()); rec['dS_by_cut'] = diffs
        rec['max_dO'] = float(np.abs(run['O'] - ed['O']).max())
        sm = mps.summarize(run); se = analysis.summarize(ed)
        ge = analysis.graph_record(ed)
        rec['mps'] = {x: sm[x] for x in ('S_nat', 'S_cen', 'tS_cen', 'chi2_cen', 'chi3_cen', 'chi6_cen', 'chi3_nat', 'chi6_nat',
                                         'chi_kept', 'chi_cen_kept', 'trunc_err', 'S1mean', 'Omin', 'Omax')}
        rec['ed'] = dict(S_nat=ge['S_nat'], S_cen=se['S_cen'], tS_cen=se['tS_cen'], chi2_cen=se.get('chi2'), chi3_cen=se.get('chi3'),
                         chi6_cen=se.get('chi6'), S1mean=se['S1mean'], Omin=se['Omin'], Omax=se['Omax'],
                         dE=ed['meta']['diag']['dE'], doublet=ed['meta']['diag']['doublet'])
        rec['start_variance'] = run['meta']['mps']['start_variance']; rec['drift_E'] = run['meta']['mps']['drift_E']
        rec['R'] = prob.R; rec['channel_err'] = prob.err
        rec['status'] = 'ok'
    except Exception as e:                                  # noqa: BLE001
        import traceback; traceback.print_exc()
        rec['status'] = f'FAILED {type(e).__name__}: {e}'
    with open(out, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    log('DONE', json.dumps({x: rec.get(x) for x in ('N', 'k', 'status', 'c', 't_ed', 't_mps', 'max_dS_cuts', 'max_dO')}))
