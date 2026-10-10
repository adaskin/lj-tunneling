"""Compare the uniform-protocol exact N=5, k=12 run (probe) with the legacy N=5 point used by the paper, and redo the TN walls."""
import glob, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/lj-tunneling')
import numpy as np
import ljtunnel as lj
from ljtunnel import analysis, extrapolation as ex, store
SP = '/tmp/claude-0/-home-user-lj-tunneling/2726bd60-70ad-5a1e-a087-829d28a60558/scratchpad'
cfg = lj.load_config()
p = sorted(glob.glob(SP + '/ed5_out/N5_k12_int*.npz'))[-1]
r = store.load_run(p)
print('probe run:', p.split('/')[-1], ' c =', r['meta']['c'], ' doublet start key', r['meta']['diag']['start_key'], ' dE', round(r['meta']['diag']['dE'], 5),
      ' t0 = pi/dE = %.1f' % (np.pi / r['meta']['diag']['dE']))
L = store.load_object('/home/user/lj-tunneling/data/derived/larger_N.npz')
print("legacy N=5 (T=120, 45 snapshots): S_cen %.4f  S1mean %.4f" % (L['S_cen_measured'][-1], L['S1_measured'][-1]))
rows = {}
for Tw in (30, 60, 120, 200):
    s = analysis.summarize(r, Tw); g = analysis.graph_record(r, Tw)
    rows[Tw] = (s, g)
    print(f"  uniform, window T<={Tw:3d}: S_cen {s['S_cen']:.4f} (t={s['tS_cen']:.0f})  S_nat {g['S_nat']:.4f}  S1mean {g['S1mean']:.4f}  chi3 {s.get('chi3')}  chi6 {s.get('chi6')}  n_edge {g['n_edge']}")
Ns = list(L['Ns_measured'])
for Tw in (120, 200):
    s, g = rows[Tw]
    S1 = list(L['S1_measured'][:-1]) + [g['S1mean']]
    Sc = list(L['S_cen_measured'][:-1]) + [s['S_cen']]
    o = ex.scenarios(cfg, Ns, S1, Sc, chi_last=(s.get('chi3'), s['S_cen']))
    w = o['walls']
    print(f"walls with the uniform N=5 point (window {Tw}): sat grid/mode {w['sat_d1200_both']}/{w['sat_d20_both']}  opt {w['opt_d1200_both']}/{w['opt_d20_both']}  "
          f"pess {w['pess_d1200_both']}/{w['pess_d20_both']};  fit S_inf={o['fit']['S_inf']:.2f} tau={o['fit']['tau']:.2f};  slopes sat {o['slope']['sat']:.2f} opt {o['slope']['opt']:.2f};  "
          f"chi walls (sat grid) {o['walls_chi'].get('sat_d1200_both')}")
w0 = L['walls']
print(f"stored (legacy N=5): sat grid/mode {w0['sat_d1200_both']}/{w0['sat_d20_both']}  opt {w0['opt_d1200_both']}/{w0['opt_d20_both']}  pess {w0['pess_d1200_both']}/{w0['pess_d20_both']};  S_inf={L['fit']['S_inf']:.2f} tau={L['fit']['tau']:.2f}; slopes sat {L['slope']['sat']:.2f} opt {L['slope']['opt']:.2f}")
