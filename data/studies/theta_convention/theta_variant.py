"""E3 test: Theta(x=0) assigned to the LEFT (repo default) vs weight 1/2 on the barrier column, exact code, k = 12, N = 2..4.
usage: theta_variant.py OUTJSON   (runs ~15 min on an idle 4-core machine)"""
import json, os, sys, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/lj-tunneling')
import ljtunnel as lj
from ljtunnel import analysis, experiments, store
from ljtunnel.resonance import doublet_at

SP = '/tmp/claude-0/-home-user-lj-tunneling/2726bd60-70ad-5a1e-a087-829d28a60558/scratchpad'
cfg = lj.load_config()
out_json = sys.argv[1]
K = 12


def make(variant):
    m = lj.Model(cfg)
    g = m.g
    if variant == 'half':
        w = np.where(np.abs(g['x']) < 1e-9, 0.5, (g['x'] > 0).astype(float))
        g['DR'] = np.repeat(w, g['ny'])
    return m


def occ(m, c, N):
    return float(doublet_at(m, c, K, N)[3][0, 0].real)


def bisect(m, N, lo, hi, tol=2e-4):
    assert occ(m, lo, N) > 0.5 > occ(m, hi, N)
    while hi - lo > tol:
        cm = 0.5 * (lo + hi)
        lo, hi = (cm, hi) if occ(m, cm, N) > 0.5 else (lo, cm)
    return 0.5 * (lo + hi)


def summarize(path):
    r = store.load_run(path)
    d = r['meta']['diag']
    out = dict(c=r['meta']['c'], dE=d['dE'], doublet=d['doublet'], mO=d['mO'], ov_int=d['ov_interacting'], Eexc_int=d['Eexc_interacting'],
               lowW_int=d['lowW_interacting'], start_key=d['start_key'], O0=float(r['O'][0]))
    for T in (60, 120, 200):
        s, g = analysis.bond_summary(r, T), analysis.graph_record(r, T, exhaustive_max_N=0)
        out[f'T{T}'] = dict(S_cen=s['S_cen'], tS=s['tS_cen'], S_nat=s['S_nat'], S1=g['S1mean'], chi3=s['chi3_cen'], chi6=s['chi6_cen'],
                            Omin=s['Omin'], Omax=s['Omax'], n_edge=g['n_edge'], Wgraph=g.get('W_graph'), w_mean=g.get('w_mean'))
    out['S_cen_t'] = analysis.S_of(r, tuple(range(analysis.bond_summary(r, 200)['bond_central']))).tolist()
    out['O_t'] = r['O'].tolist()
    return out


res = {}
for variant in ('left', 'half'):
    m = make(variant)
    res[variant] = {}
    for N, bracket in ((2, (0.2, 1.5)), (3, (1.1, 1.8)), (4, (1.9, 2.7))):
        t = time.time()
        c_cfg = m.tilt(N, K)
        c_star = bisect(m, N, *bracket)                  # exact <O>_gs = 1/2 under THIS convention
        rec = dict(c_cfg=c_cfg, c_star_exact=c_star, mO_at_cfg=occ(m, c_cfg, N))
        for label, c in (('cfgtilt', c_cfg),) + ((('relocated', round(c_star, 3)),) if variant == 'half' and N <= 4 else ()):
            paths = experiments.dynamics_run(m, N, K, starts=['interacting'], tilt=c, outdir=f'{SP}/theta_{variant}_{label}', force=True)
            rec[label] = summarize(paths[0])
        res[variant][N] = rec
        print(f'{variant} N={N}: done in {time.time()-t:.0f}s  c*={c_star:.4f} (cfg {c_cfg})', flush=True)
        json.dump(res, open(out_json, 'w'))
# 'left' at its own relocated tilt, for the like-for-like tilt comparison
m = make('left')
for N in (2, 3, 4):
    c = round(res['left'][N]['c_star_exact'], 3)
    paths = experiments.dynamics_run(m, N, K, starts=['interacting'], tilt=c, outdir=f'{SP}/theta_left_relocated', force=True)
    res['left'][N]['relocated'] = summarize(paths[0])
    json.dump(res, open(out_json, 'w'))
    print(f'left relocated N={N} done', flush=True)
print('ALL DONE')
