"""ljtunnel.mps -- the same model as a matrix product state (TeNPy): DMRG for the correlated start, TDVP in real time.

The exact code (`hamiltonian`, `dynamics`) stores all k^N amplitudes and stops near N = 7.  A matrix product state in
the particle order 0..N-1 stores N tensors of size k chi^2; what it costs is set by the bond dimension chi that the
truncated dynamics needs, which is the quantity the counting models of `walls`/`theory` assume (chi = 2^S) and this
module measures.  Everything is the SAME Hamiltonian as `ModeN`: the same per-particle mode bases, tilts and pair
matrices; the tests compare the two on dense references.

Hamiltonian as an MPO.  In the mode basis  H = sum_i diag(E_i) + sum_{i<j} V_ij  with
V_ij = (P_i x P_j)^T VP (P_i x P_j).  Write VP = sum_s e_s phi_s phi_s^T on the span of all products of modes of all
particles (eigendecomposition; the span is what V_ij can see).  Then for EVERY pair, with the same channel index s,

        V_ij = sum_s e_s F_i^s F_j^s ,        F_i^s = P_i^T diag(phi_s) P_i  (k x k, symmetric),

so the all-to-all interaction is a finite-state machine with the states 'nothing yet', 'channel s open' (s = 1..R),
'done': an MPO of bond dimension R + 2 that does not grow with N.  The capped Lennard-Jones core is contact-like
(the e_s are nearly equal), so R is of the order of the size of the function space, k(k+1)/2 = 78 at k = 12 and 21 at k = 6
(the slightly different bases of the particles add to it; R = 68 and 35 at the default tolerance); channels beyond
`channel_tol` are dropped and the resulting error of every pair matrix is reported (~1e-5 energy units at the default).  R, not N, sets the price of one H_eff
application.  TeNPy applies it through the dense W, ~2 chi^2 k^3 (R+2)^2 per matvec; measured at N = 4, k = 12, chi = 64
(R = 68) the H_eff matvecs are 87% of a TDVP step (14.5 s; 10.5 s at chi = 16).  The sum over channels would cost ~3 R chi^3 k^2 instead; a straightforward
implementation of it was exact but only 1.4x faster at that size and slower at small chi, so it is not included (REVIEW.md).

What is measured.  Per snapshot: <O>, <Theta(x_i)>, the entropy of every contiguous bond (the cuts of this MPS), of
every single particle and optionally of every pair, in bits; for the bonds also S_2 and the numerical Schmidt rank
chi(eps) at the discarded weights of `entanglement.EPS_RANK`.  The truncation (`eps`, `chi_max`) is part of the
simulation, so chi(1e-6) cannot exceed what the run kept; compare two values of `eps` before reading a rank.
Runs are written to data/mps/ in the layout of the exact runs (`store`), with protocol kind 'mps'.

    prob = MPSProblem(model, c=2.26, k=12, N=4)
    run = run_series(prob, T=60, dt_snap=1.0)
    summarize(run)                      # S_nat, S_cen, chi(1e-3) of the central bond, truncation error
"""
import itertools
import os
import time

import numpy as np

try:
    import tenpy
    from tenpy.algorithms import dmrg, tdvp
    from tenpy.linalg import np_conserved as npc
    from tenpy.models.lattice import Lattice
    from tenpy.models.model import MPOModel
    from tenpy.networks.mpo import MPO
    from tenpy.networks.mps import MPS
    from tenpy.networks.site import Site
except ImportError as exc:                               # pragma: no cover
    raise ImportError("ljtunnel.mps needs TeNPy:  pip install physics-tenpy") from exc

from . import analysis, store
from .config import ROOT
from .entanglement import CutPlan, spectrum_stats
from .log import log
from .model import masked_orbital, modes, project_pair
from .starts import configs


# ------------------------------------------------------------------ the interaction as a sum of channels
def channels(VP, Ps, tol=1e-4):
    """VP ~ sum_s e_s phi_s phi_s^T on the span of {psi_p psi_q} of all the mode sets Ps.

    The span is cut where a singular value falls below tol times the largest.  Returns (e, phi) with phi of shape
    (sites, R); the dropped directions are the only approximation."""
    cols = [np.stack([P[:, p] * P[:, q] for p in range(P.shape[1]) for q in range(p, P.shape[1])], 1) for P in Ps]
    U, s, _ = np.linalg.svd(np.concatenate(cols, 1), full_matrices=False)
    G = U[:, :int((s > tol * s[0]).sum())]
    e, kap = np.linalg.eigh(G.T @ VP @ G)
    return e, G @ kap


def pair_error(VP, Ps, e, phi, F):
    """max over pairs of the operator-norm error of sum_s e_s F_i^s F_j^s against the exact projected pair matrix."""
    err = 0.0
    for i, j in itertools.combinations(range(len(Ps)), 2):
        k = Ps[i].shape[1]
        Vr = np.einsum('s,spP,sqQ->pqPQ', e, F[i], F[j], optimize=True).reshape(k * k, k * k)
        err = max(err, float(np.linalg.norm(project_pair(VP, Ps[i], Ps[j]) - Vr, 2)))
    return err


# ------------------------------------------------------------------ the problem: sites, operators, MPO
class MPSProblem:
    """H_N and O_N of `ModeN` as TeNPy objects, for particle number N, k modes and tilt c.

        sites     one TeNPy Site per particle (operators 'Theta' = Theta(x_i) in the mode basis)
        R, err    number of interaction channels and the worst pair-matrix error they cause
    """

    def __init__(self, model, c, k, N, channel_tol=1e-4, check=True):
        g, mp = model.g, model.mp
        self.model, self.c, self.k, self.N = model, c, k, N
        self.E, self.P = [], []
        for i in range(N):
            e_i, P_i = modes(g, mp, c, i * mp['C0'], k)
            self.E.append(e_i)
            self.P.append(P_i)
        self.e, self.phi = channels(g['VP'], self.P, channel_tol)
        self.R = len(self.e)
        self.F = [np.einsum('ap,as,aq->spq', P, self.phi, P, optimize=True) for P in self.P]
        self.O1 = [P.T @ (g['DR'][:, None] * P) for P in self.P]
        self.err = pair_error(g['VP'], self.P, self.e, self.phi, self.F) if check and N > 1 else 0.0
        leg = npc.LegCharge.from_trivial(k)
        self.sites = [Site(leg, [f'm{p}' for p in range(k)], Theta=self.O1[i]) for i in range(N)]
        self.lattice = Lattice([1], self.sites, bc='open', bc_MPS='finite')

    def mpo(self, extra=None):
        """H as an MPO of bond dimension R + 2; `extra` = optional list of k x k one-body operators added to each site."""
        k, N, R = self.k, self.N, self.R
        D = R + 2
        Ws = []
        for i in range(N):
            W = np.zeros((k, k, D, D))
            h = np.diag(self.E[i]) + (0.0 if extra is None else extra[i])
            W[:, :, 0, 0] = W[:, :, D - 1, D - 1] = np.eye(k)
            W[:, :, 0, D - 1] = h
            for s in range(R):
                W[:, :, 0, 1 + s] = self.e[s] * self.F[i][s]
                W[:, :, 1 + s, 1 + s] = np.eye(k)
                W[:, :, 1 + s, D - 1] = self.F[i][s]
            Ws.append(W)
        return MPO.from_Wflat(self.sites, Ws, bc='finite', IdL=[0] * (N + 1), IdR=[D - 1] * (N + 1))

    def hamiltonian(self, extra=None):
        return MPOModel(self.lattice, self.mpo(extra))

    def penalty(self, sides, W):
        """One-body operators confining particle i to the valley sides[i] (+1 right, -1 left); as `ModeN.penalty_ops`."""
        I = np.eye(self.k)
        return [W * (I - D) if s > 0 else W * D for D, s in zip(self.O1, sides)]

    def product_mps(self, sides):
        """Each particle in the lowest orbital of its own valley (the product start of `starts`), as an MPS."""
        g, mp, C0 = self.model.g, self.model.mp, self.model.mp['C0']
        vs = []
        for i, (P, s) in enumerate(zip(self.P, sides)):
            v = P.T @ masked_orbital(g, mp, s, self.c, i * C0)
            vs.append((v / np.linalg.norm(v)).astype(complex))
        return MPS.from_product_state(self.sites, vs, bc='finite', dtype=complex)

    def dense(self, psi):
        """The MPS as a (k,)*N array (small N only; reference for the tests)."""
        return psi.get_theta(0, self.N).to_ndarray().reshape((self.k,) * self.N)


def start_sides(N, key=None):
    """Valley assignment of the start: the (N-1, 1) configuration with particle 0 on the right unless `key` says otherwise
    (keys of `starts.configs`, e.g. '31_2'; the exact runs store theirs in meta['diag']['start_key'])."""
    return configs(N)[key or f'{N - 1}1_1']


# ------------------------------------------------------------------ start state: DMRG
def ground_state(prob, psi0, extra=None, chi_max=64, eps=1e-10, max_sweeps=40, E_tol=1e-10, var_tol=1e-7, max_rounds=8):
    """Two-site DMRG of H (+ extra one-body terms) from the MPS psi0, repeated until the energy variance <H^2> - <H>^2
    is below var_tol (the energy alone converges long before the state does when levels lie close: the state error is
    ~ variance / gap^2).  Returns (E, psi, info) with the variance reached and the number of sweeps."""
    model = prob.hamiltonian(extra)
    psi = psi0.copy()
    opts = dict(trunc_params=dict(chi_max=chi_max, trunc_cut=np.sqrt(eps), svd_min=1e-12),
                mixer=True, mixer_params=dict(amplitude=1e-4, decay=1.5, disable_after=8),
                max_sweeps=max_sweeps, min_sweeps=6, max_E_err=E_tol, max_S_err=1e-7)
    sweeps = 0
    for rounds in range(1, max_rounds + 1):
        eng = dmrg.TwoSiteDMRGEngine(psi, model, opts)
        E, psi = eng.run()
        sweeps += int(eng.sweeps)
        var = float(np.real(model.H_MPO.variance(psi)))
        if var <= var_tol:
            break
    return float(E), psi, dict(sweeps=sweeps, chi=int(max(psi.chi)), variance=var, rounds=rounds)


def correlated_start(prob, sides, W, **kw):
    """The correlated start of `starts.interacting_state` at MPS level: ground state of H + W * (valley penalty),
    from the product start.  Returns (psi, info) with the energy of H alone."""
    E, psi, info = ground_state(prob, prob.product_mps(sides), extra=prob.penalty(sides, W), **kw)
    info['E'] = float(energy(prob, psi))
    return psi, info


def energy(prob, psi, H=None):
    return float(np.real((H or prob.mpo()).expectation_value(psi)))


# ------------------------------------------------------------------ resonance tilt at any N
def gs_occupation(model, c, k, N, chi_max=48, eps=1e-8, channel_tol=1e-3, starts=None):
    """<O> of the DMRG ground state at tilt c, from two starts so that a localized isomer cannot hide the ground state:
    the (N-1, 1) assignment with particle 0 on the right (the lowest tilt offset keeps it there) and (N, 0); `starts`
    overrides.  Loose convergence (the tilt search needs <O> to ~1e-3).  Returns (<O>, E)."""
    prob = MPSProblem(model, c, k, N, channel_tol=channel_tol, check=False)
    best = None
    for sides in (starts or [start_sides(N), start_sides(N, f'{N}0')]):
        E, psi, _ = ground_state(prob, prob.product_mps(sides), chi_max=chi_max, eps=eps, max_sweeps=24, E_tol=1e-7)
        if best is None or E < best[1]:
            best = (float(np.sum(np.real(psi.expectation_value('Theta')))), E)
    return best


def locate_tilt(model, N, k, c_lo, c_hi, tol=2e-3, max_iter=14, **kw):
    """The tilt c in [c_lo, c_hi] at which the ground state has <O> = 1/2 (bisection; <O>_gs falls with c).
    The ED search of `resonance.locate` picks the closest of a grid of tilts to that point; this one refines it, and works
    for any N the MPS can reach."""
    o_lo, o_hi = gs_occupation(model, c_lo, k, N, **kw)[0], gs_occupation(model, c_hi, k, N, **kw)[0]
    if not (o_hi < 0.5 < o_lo):
        raise ValueError(f"<O>_gs = 1/2 not bracketed on [{c_lo}, {c_hi}]: {o_lo:.3f}, {o_hi:.3f}")
    for _ in range(max_iter):
        c = 0.5 * (c_lo + c_hi)
        o = gs_occupation(model, c, k, N, **kw)[0]
        log(f"   [mps tilt N={N} k={k}] c={c:.4f} <O>_gs={o:.4f}")
        c_lo, c_hi = (c, c_hi) if o > 0.5 else (c_lo, c)
        if c_hi - c_lo < tol:
            break
    return 0.5 * (c_lo + c_hi)


# ------------------------------------------------------------------ measurements
def pair_entropies(psi, k):
    """S_ij (bits) for every pair i < j of a finite MPS, as {(i, j): S}.  One sweep per i; the reduced density matrix
    of the two sites is built from the Schmidt values at bond i and the transfer through the sites between."""
    N = psi.L
    B = [psi.get_B(i, form='B').to_ndarray() for i in range(N)]            # (vL, p, vR), right-isometric
    sl = [np.abs(psi.get_SL(i)) ** 2 for i in range(N)]
    out = {}
    for i in range(N - 1):
        E = np.einsum('a,apb,aqc->bcpq', sl[i], B[i], B[i].conj(), optimize=True)
        for j in range(i + 1, N):
            rho = np.einsum('bcpq,brx,csx->prqs', E, B[j], B[j].conj(), optimize=True).reshape(k * k, k * k)
            out[(i, j)] = spectrum_stats(np.linalg.eigvalsh(rho / np.trace(rho).real))[0]
            if j < N - 1:
                E = np.einsum('bcpq,brx,crz->xzpq', E, B[j], B[j].conj(), optimize=True)
    return out


def measure(prob, psi, plan, pairs=False):
    """Everything recorded at one snapshot.  Returns (O_i, S[n_cuts], bond[N-1, 4], chi[N-1])."""
    N, k = prob.N, prob.k
    c = psi.copy()
    c.canonical_form()                                                    # exact Schmidt values of the stored state
    Oi = np.real(c.expectation_value('Theta'))
    S = np.full(plan.n_cuts, np.nan)
    bond = np.full((N - 1, 4), np.nan)
    chi = np.zeros(N - 1, dtype=int)
    pair_S = pair_entropies(c, k) if (pairs and N > 2) else {}
    for j, X in enumerate(plan.small):
        if len(X) == 1:
            rho = c.get_rho_segment([X[0]]).to_ndarray()
            S[j] = spectrum_stats(np.linalg.eigvalsh(rho / np.trace(rho).real))[0]
        elif len(X) == 2 and tuple(X) in pair_S:
            S[j] = pair_S[tuple(X)]
    for b in range(1, N):
        p = np.abs(c.get_SL(b)) ** 2
        s, s2, ranks = spectrum_stats(p / p.sum())
        S[plan.bond_idx[b - 1]] = s
        bond[b - 1] = [s2, *ranks]
        chi[b - 1] = len(p)
    return Oi, S, bond, chi


# ------------------------------------------------------------------ real-time series
def run_series(prob, sides=None, T=None, dt_snap=None, dt=None, chi_max=None, eps=None, W=None, pairs=False,
               psi0=None, label='', cfg=None):
    """Correlated start (DMRG) -> TDVP over [0, T] -> the series of every measured cut.

    Defaults come from config [mps] / [protocol]; T and dt_snap are recorded in meta['protocol'].  The protocol kind of every
    MPS run is 'mps', so that it is never mixed with the exact ('uniform' / 'custom') runs.  `trunc_err` in the result is the
    cumulative discarded weight of the TDVP truncations up to each snapshot.
    Returns the run dict (arrays + 'meta'), the same layout as `store.load_run`."""
    cfg = cfg or prob.model.cfg
    mc, pc = cfg['mps'], cfg['protocol']
    T = float(pc['T'] if T is None else T)
    dt_snap = float(pc['dt_snap'] if dt_snap is None else dt_snap)
    dt = float(mc['dt'] if dt is None else dt)
    chi_max = int(mc['chi_max'] if chi_max is None else chi_max)
    eps = float(mc['eps'] if eps is None else eps)
    W = float(pc['W_penalty'] if W is None else W)
    N, k = prob.N, prob.k
    n_sub = int(round(dt_snap / dt))
    if abs(n_sub * dt - dt_snap) > 1e-9:
        raise ValueError(f"dt_snap = {dt_snap} must be a multiple of the TDVP step dt = {dt}")
    sides = sides if sides is not None else start_sides(N)
    plan = CutPlan(N, k, 'bonds+pairs' if pairs else 'bonds')
    t_all = time.time()
    if psi0 is None:
        psi0, start = correlated_start(prob, sides, W, chi_max=min(int(mc['dmrg_chi']), chi_max), eps=float(mc['dmrg_eps']))
        log(f"[mps N={N} k={k}] start: DMRG {start['sweeps']} sweeps, chi {start['chi']}, <H> = {start['E']:.6f}, "
            f"variance {start['variance']:.1e}")
    else:
        start = dict(E=energy(prob, psi0))
    H = prob.hamiltonian()
    psi = psi0.copy()
    eng = tdvp.TwoSiteTDVPEngine(psi, H, dict(preserve_norm=True,
                                              trunc_params=dict(chi_max=chi_max, trunc_cut=np.sqrt(eps), svd_min=1e-12)))
    nt = int(round(T / dt_snap)) + 1
    ts = np.arange(nt) * dt_snap
    O, Oi = np.zeros(nt), np.zeros((nt, N))
    S = np.full((nt, plan.n_cuts), np.nan)
    bond, chi = np.full((nt, N - 1, 4), np.nan), np.zeros((nt, N - 1), dtype=int)
    E, terr = np.zeros(nt), np.zeros(nt)
    t0 = time.time()
    for i in range(nt):
        if i:
            eng.run_evolution(n_sub, dt)
        Oi[i], S[i], bond[i], chi[i] = measure(prob, eng.psi, plan, pairs)
        O[i] = Oi[i].sum()
        E[i] = energy(prob, eng.psi, eng.model.H_MPO)
        terr[i] = float(eng.trunc_err.eps)
        if i and (i % 10 == 0 or i == nt - 1):
            el = time.time() - t0
            log(f"      [{label}] t={ts[i]:6.1f}  <O>={O[i]:.3f}  chi_max={chi[i].max():4d}  trunc={terr[i]:.1e}  "
                f"elapsed {el:6.0f}s  eta {el / i * (nt - 1 - i):6.0f}s")
    prot = dict(kind='mps', T=T, dt_snap=dt_snap, nsnap=nt, W_penalty=W, cuts='bonds+pairs' if pairs else 'bonds')
    meta = dict(N=N, k=k, c=float(prob.c), start='interacting', protocol=prot,
                mps=dict(chi_max=chi_max, eps=eps, dt=dt, channel_R=prob.R, channel_err=prob.err, sides=list(sides),
                         E_start=start['E'], start_variance=start.get('variance'), drift_E=float(E.max() - E.min()), trunc_err=float(terr[-1]),
                         chi_reached=int(chi.max()), tenpy=tenpy.__version__),
                model=cfg['model'], grid=dict(cfg['grid'], sigma_smear=prob.model.sigma_smear), wall_s=time.time() - t_all)
    arrays = dict(t=ts, O=O, Oi=Oi, masks=plan.masks, S=S, bond_b=plan.bond_b, bond_S2=bond[:, :, 0],
                  bond_chi2=bond[:, :, 1], bond_chi3=bond[:, :, 2], bond_chi6=bond[:, :, 3],
                  chi=chi, energy=E, trunc_err=terr)
    return dict(arrays, meta=meta)


def save(run, outdir=None):
    """Write the run to data/mps/N{N}_k{k}_mps_c{c}_T{T}_dt{dt}_chi{chi_max}_eps{eps}.npz."""
    m = run['meta']
    outdir = outdir or os.path.join(ROOT, 'data', 'mps')
    name = (f"N{m['N']}_k{m['k']}_mps_c{m['c']:.3f}_T{m['protocol']['T']:g}_dt{m['protocol']['dt_snap']:g}"
            f"_chi{m['mps']['chi_max']}_eps{m['mps']['eps']:g}.npz")
    arrays = {key: v for key, v in run.items() if key not in ('meta', 'path')}
    return store.save_run(os.path.join(outdir, name), arrays, m)


# ------------------------------------------------------------------ what the series says
def summarize(run, T_win=None, include_start=True):
    """The MPS cost of a series over the window [0, T_win]: `analysis.bond_summary` (sigma(A) = max_t S_A of every contiguous
    bond, the central bond, the Schmidt ranks) plus what the truncated run kept: bond dimension and discarded weight."""
    out = analysis.bond_summary(run, T_win, include_start)
    ok = analysis.window(run, T_win, include_start)
    out.update(chi_kept=int(run['chi'][ok].max()), chi_cen_kept=int(run['chi'][ok][:, out['jc']].max()),
               trunc_err=float(run['trunc_err'][ok].max()))
    return out
