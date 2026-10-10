"""ljtunnel.parameter_study -- Appendix A: how sensitive are the paper's answers to the model parameters?

Two cheap parts, no real-time propagation (minutes, not hours):

(a) one particle in U = A (x^2-1)^2: splitting dE, tunneling time t* = pi/dE, omega = sqrt(8A) and
    the barrier action S0 = (4/3) sqrt(2A) a, for A = 2..16, on the production x-grid (dx = 0.125) and a
    fine grid (Richardson-converged value);
(b) exact diagonalization of the two-particle model on the coarse 28x18 grid (the grid of experiment E2b),
    one parameter changed at a time from the production point (A, wy, eps, sigma, C0) = (8, 2, 2, 0.18, 0.15):
    resonance tilt c* (bisection of <O>_gs = 1/2), splitting dE, weight ov_sub of the valley-product start state
    in the two lowest levels, the gap of the third level above the second, and the relative change of dE when the
    core cap is doubled -- the static ingredients of gates H2a, H2e and cap of Table app:tab:gates.

Check: at the production point the script reproduces the grid28 ledger (dE = 0.18890, ov_sub = 0.9783).
Usage:  python -m ljtunnel.parameter_study [--part a|b|ab] [--workers 2] [--theta-barrier W] [--out PATH]   (from the repository root;
        `run_tier1_parameter_map()` in run.py does this with the weight of config.toml)
Writes tier1_parameter_map.txt (key-value ledger; default data/studies/parameter_map/), or to --out.
`parameter_map(path)` parses that ledger back (used by `run_parameter_choice()`).

--theta-barrier W  weight of the grid column x = 0 in Theta(x) (the right-valley indicator of <O>).  The default 0 is how the
                   ledger before the rerun was made (the column counted as left; the file is in git history); `rerun_all.sh` passes
                   the 1/2 of config.toml [model] theta_barrier.  Only c* (the bisection of <O>_gs = 1/2) and what is evaluated at it depend on it.
"""
import argparse, json, sys, time, os
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh_tridiagonal
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
BASE = dict(A=8.0, wy=2.0, eps=2.0, sig=0.18, C0=0.15, cap=100.0)     # wy = omega_y^2 (V_y = wy y^2/2)
THETA_BARRIER = 0.0                                                   # set by --theta-barrier (see the docstring)

# ------------------------------------------------------------------ part (a)
def split1d(A, dx, L=2.5):
    n = int(round(2*L/dx)); x = -L + dx*np.arange(n)
    d = 1/dx**2 + A*(x**2-1)**2; e = -np.full(n-1, 0.5/dx**2)
    w = eigh_tridiagonal(d, e, select='i', select_range=(0, 1))[0]
    return w[1]-w[0]

def part_a():
    rows = []
    for A in (2, 4, 6, 8, 10, 12, 16):
        S0 = 4/3*np.sqrt(2*A); d1 = split1d(A, 0.125); d2 = split1d(A, 0.125/8)
        rows.append(dict(A=A, S0=S0, mut=1.5*S0, omega=np.sqrt(8*A), dE_prod=d1, dE_fine=d2, tstar=np.pi/d1,
                         ratio=d1/np.sqrt(8*A)))
    return rows

# ------------------------------------------------------------------ part (b)
def make(nx=28, ny=18, Lx=2.5, Ly=3.0):
    x = np.linspace(-Lx, Lx-2*Lx/nx, nx); y = np.linspace(-Ly, Ly-2*Ly/ny, ny)
    dx, dy = 2*Lx/nx, 2*Ly/ny
    K = lambda n, h: sp.diags([-0.5/h**2*np.ones(n-1), 1/h**2*np.ones(n), -0.5/h**2*np.ones(n-1)], [-1, 0, 1], format="csr")
    kin = (sp.kron(sp.identity(nx), K(ny, dy)) + sp.kron(K(nx, dx), sp.identity(ny))).tocsr()
    X = np.repeat(x, ny); Y = np.tile(y, nx)
    R = np.sqrt((X[:, None]-X[None, :])**2 + (Y[:, None]-Y[None, :])**2)
    return dict(kin=kin, X=X, Y=Y, R=R, N=nx*ny, dx=dx, dy=dy)

def pair_pot(g, eps, sig, cap):
    r = np.maximum(g["R"], 1e-12); return np.minimum(4*eps*((sig/r)**12-(sig/r)**6), cap)

def H2(g, p, c, Vp):
    I = sp.identity(g["N"], format="csr")
    base = p["A"]*(g["X"]**2-1)**2 + 0.5*p["wy"]*g["Y"]**2
    Ha = g["kin"] + sp.diags(base + c*g["X"]); Hb = g["kin"] + sp.diags(base + (c+p["C0"])*g["X"])
    return (sp.kron(Ha, I) + sp.kron(I, Hb) + sp.diags(Vp.ravel())).tocsr()

def lowest(H, k):
    ev, vc = eigsh(H, k=k, which="SA", tol=1e-9, ncv=max(2*k+1, 24)); o = np.argsort(ev); return ev[o], vc[:, o]

def Odiag(g):
    D = np.where(np.abs(g["X"]) < 1e-9, THETA_BARRIER, (g["X"] > 0).astype(float))
    return np.kron(D, np.ones(g["N"])) + np.kron(np.ones(g["N"]), D)

def valley_orbital(g, p, tilt, side, W=1e3):
    pen = np.where((g["X"] < 0) if side == "R" else (g["X"] > 0), W, 0.0)
    Hs = g["kin"] + sp.diags(p["A"]*(g["X"]**2-1)**2 + tilt*g["X"] + 0.5*p["wy"]*g["Y"]**2 + pen)
    v = eigsh(Hs.tocsc(), k=1, which="SA")[1][:, 0]; return v/np.linalg.norm(v)

def point(args):
    label, over = args
    t0 = time.time(); p = {**BASE, **over}; g = make(); Od = Odiag(g)
    Vp = pair_pot(g, p["eps"], p["sig"], p["cap"])
    def f(c):
        ev, vc = lowest(H2(g, p, c, Vp), 2); return float(np.sum(Od*vc[:, 0]**2)) - 0.5
    c_hi = min(3.5, 0.5*p["A"]); lo, hi = 0.0, c_hi; flo, fhi = f(lo), f(hi)
    out = dict(label=label, **{k: p[k] for k in BASE})
    if flo*fhi > 0:
        out.update(c_star=None, note=f"no <O>=1/2 crossing in c in [0,{c_hi:g}] (<O>_gs = {flo+.5:.3f} .. {fhi+.5:.3f})", sec=time.time()-t0)
        return out
    while hi-lo > 2e-3:
        mid = 0.5*(lo+hi); fm = f(mid)
        if flo*fm <= 0: hi, fhi = mid, fm
        else: lo, flo = mid, fm
    c = 0.5*(lo+hi)
    ev, vc = lowest(H2(g, p, c, Vp), 6)
    best = 0.0
    for sa, sb in (("L", "R"), ("R", "L")):
        psi = np.kron(valley_orbital(g, p, c, sa), valley_orbital(g, p, c+p["C0"], sb)); psi /= np.linalg.norm(psi)
        best = max(best, float(np.sum((vc[:, :2].T @ psi)**2)))
    dE = ev[1]-ev[0]
    ev2, _ = lowest(H2(g, p, c, pair_pot(g, p["eps"], p["sig"], 2*p["cap"])), 2)
    out.update(c_star=c, two_c=2*c, dE=dE, ov_sub=best, leak=1-best, O0=float(np.sum(Od*vc[:, 0]**2)),
               gap2=ev[2]-ev[1], cap_shift=abs((ev2[1]-ev2[0])-dE)/dE, sec=time.time()-t0)
    out["gates"] = dict(H2a_static=bool(best >= 0.9 and 1-best <= 0.1), split_window=bool(5e-3 < dE < 0.5),
                        H2e=bool(2*c <= 4.0), cap=bool(out["cap_shift"] < 0.1))
    return out

def _init(theta):
    """Set the Theta(x = 0) weight (also the initializer of the pool workers, so that it holds under 'spawn' as well as 'fork')."""
    global THETA_BARRIER
    THETA_BARRIER = theta

def runs_b():
    r = [("production point", {})]
    r += [(f"A={A:g}", dict(A=A)) for A in (4.0, 6.0, 10.0, 12.0)]
    r += [(f"sigma={s:g}", dict(sig=s)) for s in (0.25, 0.35, 0.50)]
    r += [(f"eps={e:g}", dict(eps=e)) for e in (0.5, 1.0, 4.0, 8.0)]
    r += [(f"wy={w:g} (omega_y={np.sqrt(w):.2f})", dict(wy=w)) for w in (0.5, 8.0)]
    r += [(f"C0={c:g}", dict(C0=c)) for c in (0.0, 0.05, 0.30)]
    r += [(f"cap={c:g}", dict(cap=c)) for c in (25.0, 400.0)]
    return r

def parameter_map(path):
    """Parse the ledger written by main().

    part (a): one particle in U = A (x^2-1)^2, rows 'A S0 mut omega dE(dx=0.125) dE(fine) t* dE/omega';
    part (b): N = 2 exact diagonalization on a 28x18 grid, one input changed at a time (one JSON record per line).
    Returns {'onebody': [dict], 'sensitivity': [dict]}.
    """
    keys = ['A', 'S0', 'mut', 'omega', 'dE_coarse', 'dE_fine', 'tstar_coarse', 'dE_over_omega_coarse']
    one, sens = [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith('{'):
                sens.append(json.loads(line))
            elif line and not line.startswith('#') and not line.startswith('A '):
                vals = line.split()
                if len(vals) == len(keys):
                    one.append(dict(zip(keys, (float(v) for v in vals))))
    return dict(onebody=one, sensitivity=sens)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--part", default="ab"); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--theta-barrier", type=float, default=0.0); ap.add_argument("--out", default=os.path.join(REPO, "data", "studies", "parameter_map", "tier1_parameter_map.txt"))
    a = ap.parse_args(); lines = []
    _init(a.theta_barrier)
    if "a" in a.part:
        lines.append(f"# parameter-study ledger (ljtunnel.parameter_study --theta-barrier {a.theta_barrier:g}): Theta(x = 0) weight {a.theta_barrier:g}")
        lines.append("# part (a): one particle, U = A (x^2-1)^2, m = hbar = 1, a = 1")
        lines.append("A  S0  mut  omega  dE_dx0.125  dE_fine  tstar=pi/dE  dE/omega")
        for r in part_a():
            lines.append(f"{r['A']:g}  {r['S0']:.3f}  {r['mut']:.3f}  {r['omega']:.3f}  {r['dE_prod']:.5e}  {r['dE_fine']:.5e}  {r['tstar']:.1f}  {r['ratio']:.4f}")
        print("\n".join(lines), flush=True)
    if "b" in a.part:
        runs = runs_b(); lines.append("# part (b): N=2 exact diagonalization, 28x18 grid, one parameter changed at a time")
        t0 = time.time()
        with Pool(a.workers, initializer=_init, initargs=(a.theta_barrier,)) as P:
            res = []
            for r in P.imap(point, runs):
                res.append(r); print(json.dumps(r, default=float), flush=True)
        for r in res:
            lines.append(json.dumps(r, default=float))
        lines.append(f"# wall time {time.time()-t0:.0f} s")
    open(a.out, "w").write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
