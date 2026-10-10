"""Static checks behind data/followup/static_checks/README.md (no propagation; ground states and low levels only).

Run from the repository root:  python followup/static_checks_scaling_args.py [part ...]   parts: energy levels dressing untilted (default: all)

energy    pair-interaction energy of the all-left correlated state, per pair, pair concentration, marginal cost (k = 12, 16, 20; k = 6 is
          printed only to show that its energies are artefacts of the 6-mode basis).
levels    lowest levels of H at the Stage 1 tilt with: y-reflection parity <R>, weight of the correlated start, right-valley occupation
          <n_i> per particle; the bright pair, the first level of each labelled state, the first level with total occupation >= 1.5.
dressing  weight of the start outside the two heaviest levels, binned by distance to the nearest of the two in units of J = omega*/2
          (from data/followup/stage1/N*_k*_c*.npz).
untilted  the saved untilted level lists (data/followup/stage1/untilted_*.npz).

Definitions: R is the global reflection y_i -> -y_i, built in the mode basis as prod_i P^T R_grid P with R_grid: (ix, iy) -> (ix, (ny - iy) % ny);
the one-body parity closure of the truncated modes is about 0.05, so eigenstate parities are near, not exactly, +-1.
"""
import glob
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ljtunnel.config import load_config            # noqa: E402
from ljtunnel.hamiltonian import ModeN, dense_H    # noqa: E402
from ljtunnel.model import Model                   # noqa: E402
from ljtunnel.starts import configs, product_state, interacting_state   # noqa: E402

S1 = os.path.join(ROOT, 'data', 'followup', 'stage1')
model = Model(load_config())
g = model.g
NX, NY = g['nx'], g['ny']
PERM = np.array([ix * NY + ((NY - iy) % NY) for ix in range(NX) for iy in range(NY)])


def pair_energies(M, psi):
    k, N = M.k, M.N
    out = {}
    for (i, j), V in M.pairs.items():
        t = np.moveaxis(psi, (i, j), (0, 1)).reshape(k * k, -1)
        out[(i, j)] = float(np.sum(t * (V @ t)))
    return out


def part_energy():
    cases = [(2, 6, 0.70), (3, 6, 1.6219), (4, 6, 2.4562), (5, 6, 3.1696),
             (2, 12, 0.70), (3, 12, 1.46), (4, 12, 2.26), (5, 12, 2.92),
             (2, 16, 0.70), (3, 16, 1.36), (4, 16, 2.14), (2, 20, 0.70), (3, 20, 1.31)]
    print("# energy: E_int = <sum_{i<j} V_ij> in the correlated start of (N,0) (all left) and (N-1,1) with particle 0 right, at the Stage 1 tilt")
    print("N k c | E_int(N,0) | E_int(N-1,1_0) | E_int(N,0) per pair | marginal E_int(N,0)-E_int(N-1,0) | pairs for 90% / C(N,2) | sum_j <V_0j> in (N,0)")
    prev = {}
    for N, k, c in cases:
        t0 = time.time()
        M = ModeN(model, c, k, N)
        res = {}
        for key in (f"{N}0", f"{N - 1}1_1"):
            sides = configs(N)[key]
            prod = product_state(model, c, M, sides)
            psi = interacting_state(M, prod, sides, 30.0)
            pe = pair_energies(M, psi)
            res[key] = (sum(pe.values()), pe)
        Ein, Eout = res[f"{N}0"][0], res[f"{N - 1}1_1"][0]
        pe = np.array(sorted(res[f"{N}0"][1].values(), reverse=True))
        n90 = int(np.searchsorted(np.cumsum(pe) / pe.sum(), 0.9) + 1)
        s0 = sum(v for (i, j), v in res[f"{N}0"][1].items() if i == 0 or j == 0)
        mg = Ein - prev[k][1] if k in prev and prev[k][0] == N - 1 else float('nan')
        prev[k] = (N, Ein)
        print(f"{N} {k} {c} | {Ein:.4f} | {Eout:.4f} | {Ein / (N * (N - 1) / 2):.4f} | {mg:.4f} | {n90}/{N * (N - 1) // 2} | {s0:.4f}   ({time.time() - t0:.0f}s)", flush=True)


def part_levels():
    print("# levels: E-E0, weight of the correlated start (particle 0 right), parity <R>, right-valley occupations <n_i>")
    for N, k, c, nlev in [(3, 12, 1.46, 60), (3, 16, 1.36, 60), (4, 12, 2.26, 90)]:
        t0 = time.time()
        M = ModeN(model, c, k, N)
        R = [P.T @ P[PERM, :] for P in M.P]
        if M.dim <= 8000:
            w, vc = np.linalg.eigh(dense_H(M))
            w, vc = w[:nlev], vc[:, :nlev]
        else:
            w, vc = M.lowest(nlev)
        sides = configs(N)[f"{N - 1}1_1"]
        prod = product_state(model, c, M, sides)
        psi = interacting_state(M, prod, sides, 30.0).reshape(-1)
        ov = vc.T @ psi

        def Rt(v):
            x = v.reshape((k,) * N)
            for i, r in enumerate(R):
                x = np.moveaxis(np.tensordot(r, x, axes=(1, i)), 0, i)
            return x.reshape(-1)
        print(f"\nN={N} k={k} c={c}  ({nlev} levels, {time.time() - t0:.0f}s, coverage {np.sum(ov ** 2):.3f}, start parity <R> = {float(psi @ Rt(psi)):+.4f})")
        rows = []
        for j in range(len(w)):
            v = vc[:, j]
            rows.append((j, w[j] - w[0], ov[j] ** 2, float(v @ Rt(v)), np.array(M.O_particle(v.reshape((k,) * N)))))
        np.savez(os.path.join(ROOT, 'data', 'followup', 'static_checks', f'levels_N{N}_k{k}_c{c:.4f}.npz'),
                 E=np.array([r[1] for r in rows]), weight=np.array([r[2] for r in rows]), R=np.array([r[3] for r in rows]),
                 n=np.array([r[4] for r in rows]), start_parity=float(psi @ Rt(psi)), coverage=float(np.sum(ov ** 2)))
        print("lvl  E-E0    weight   <R>     <n_i>   (first 14)")
        for r in rows[:14]:
            print(f"{r[0]:3d} {r[1]:7.4f} {r[2]:8.4f} {r[3]:+7.3f}  {np.round(r[4], 2)}")
        odd = [r for r in rows if r[3] < 0]
        print(f"odd levels carrying weight > 1e-6: {sum(1 for r in odd if r[2] > 1e-6)} of {len(odd)}; even levels with weight > 1e-6: {sum(1 for r in rows if r[3] > 0 and r[2] > 1e-6)}; max weight on an odd level {max(r[2] for r in odd):.2e}")
        ev = [r for r in rows if r[3] > 0]
        b = sorted(sorted(ev, key=lambda r: -r[2])[:2], key=lambda r: r[1])
        print(f"bright pair (even, two heaviest): E = {b[0][1]:.3f}, {b[1][1]:.3f}; weights {b[0][2]:.3f}, {b[1][2]:.3f}; J = omega*/2 = {(b[1][1] - b[0][1]) / 2:.3f}")
        for lab in range(1, N):
            c_ = [r for r in ev if r[4][lab] > 0.5]
            print(f"first even level with <n_{lab}> > 0.5: " + (f"E-E0 = {c_[0][1]:.3f}  <n_{lab}> = {c_[0][4][lab]:.2f}  total {c_[0][4].sum():.2f}" if c_ else "none in list"))
        q2 = [r for r in ev if r[4].sum() >= 1.5]
        print("first even level with total right occupation >= 1.5: " + (f"E-E0 = {q2[0][1]:.3f}  total {q2[0][4].sum():.2f}" if q2 else f"none in {nlev} levels (highest E-E0 {max(r[1] for r in rows):.3f})"))


def part_dressing():
    print("# dressing: weight outside the two heaviest levels, by distance to the nearest of the two in units of J = omega*/2")
    print("case | J | pair | out | near(<J) | mid(1-3J) | far(>=3J) | coverage | 1-B2")
    for f in sorted(glob.glob(os.path.join(S1, 'N[2-5]_k*_c[0-9.]*.npz'))):
        if f.endswith('_krylov.npz'):
            continue
        d = np.load(f, allow_pickle=True)
        if 'E' not in d.files:
            continue
        E = d['E'] - d['E'][0]
        w = d['w_interacting']
        a, b = sorted(np.argsort(-w)[:2])
        J = (E[b] - E[a]) / 2
        m = np.ones(len(E), bool)
        m[[a, b]] = False
        dist = np.minimum(abs(E - E[a]), abs(E - E[b]))
        print(f"{os.path.basename(f)[:-4]:20s} {J:.3f} ({a},{b}) {w[m].sum():.3f} {w[m & (dist < J)].sum():.3f} {w[m & (dist >= J) & (dist < 3 * J)].sum():.3f} {w[m & (dist >= 3 * J)].sum():.3f} {w.sum():.3f} {1 - w[a] - w[b]:.3f}")


def part_untilted():
    print("# untilted level lists (E-E0), lowest 12 and gaps")
    for f in sorted(glob.glob(os.path.join(S1, 'untilted_*.npz'))):
        d = np.load(f, allow_pickle=True)
        e = d['E_minus_E0']
        print(os.path.basename(f)[:-4], np.round(e[:12], 3), "gaps", np.round(np.diff(e)[:11], 3))


if __name__ == '__main__':
    parts = sys.argv[1:] or ['energy', 'levels', 'dressing', 'untilted']
    for p in parts:
        print(f"\n===== {p} =====")
        {'energy': part_energy, 'levels': part_levels, 'dressing': part_dressing, 'untilted': part_untilted}[p]()
