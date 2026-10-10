"""MPS pipeline vs ED, snapshot by snapshot.  usage: validate.py N k c T chi_max eps dt [pairs]"""
import sys, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/lj-tunneling')
import ljtunnel as lj
from ljtunnel import mps, analysis
from ljtunnel.hamiltonian import ModeN
from ljtunnel.starts import product_state, interacting_state
from ljtunnel.dynamics import snapshots, spectral_bounds
from ljtunnel.entanglement import CutPlan

N, k, c, T = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
chi_max, eps, dt = int(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7])
pairs = len(sys.argv) > 8 and sys.argv[8] == 'pairs'
cfg = lj.load_config(); model = lj.Model(cfg)
W = cfg['protocol']['W_penalty']
sides = mps.start_sides(N)

t0 = time.time()
prob = mps.MPSProblem(model, c, k, N)
print(f'MPSProblem: R={prob.R} channels, max pair error {prob.err:.1e}  ({time.time()-t0:.1f}s)')
t0 = time.time()
run = mps.run_series(prob, sides=sides, T=T, dt_snap=1.0, dt=dt, chi_max=chi_max, eps=eps, pairs=pairs, label='mps')
t_mps = time.time() - t0
print(f'MPS wall {t_mps:.0f}s; chi reached {run["meta"]["mps"]["chi_reached"]}, trunc err {run["meta"]["mps"]["trunc_err"]:.1e}, '
      f'energy drift {run["meta"]["mps"]["drift_E"]:.1e}')

# ED reference
t0 = time.time()
M = ModeN(model, c, k, N)
prod = product_state(model, c, M, sides)
inter = interacting_state(M, prod, sides, W)
plan = CutPlan(N, k, 'bonds+pairs' if pairs else 'bonds')
ts = np.arange(int(round(T)) + 1) * 1.0
S_ed = np.full((len(ts), plan.n_cuts), np.nan); O_ed = np.zeros(len(ts)); bond_ed = np.full((len(ts), N - 1, 4), np.nan)
for i, t, psi in snapshots(M, inter, ts, spectral_bounds(M)):
    S, bond = plan.measure(psi, i)
    S_ed[i] = S; bond_ed[i] = bond; O_ed[i] = sum(M.O_particle(psi))
print(f'ED wall {time.time()-t0:.0f}s')

# compare
Smps = run['S']
d = np.abs(Smps - S_ed)
print('max |S_mps - S_ed| over all measured cuts & snapshots: %.2e' % np.nanmax(d))
for j, X in enumerate(plan.small):
    print(f'   cut {str(X):10s} max diff {np.nanmax(d[:, j]):.2e}   (ED max S = {np.nanmax(S_ed[:, j]):.4f})')
print('max |O_mps - O_ed| = %.2e' % np.abs(run['O'] - O_ed).max())
for nm, key in (('S2', 0), ('chi(1e-2)', 1), ('chi(1e-3)', 2), ('chi(1e-6)', 3)):
    a = run['bond_' + ['S2', 'chi2', 'chi3', 'chi6'][key]]; b = bond_ed[:, :, key]
    print(f'   bond {nm:10s}: max |diff| = {np.nanmax(np.abs(a - b)):.3g}   (ED max {np.nanmax(b):.4g}, MPS max {np.nanmax(a):.4g})')
s_m = mps.summarize(run); print({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in s_m.items() if kk in ('S_nat', 'S_cen', 'chi3_cen', 'chi6_cen', 'chi_cen_kept', 'trunc_err')})
