"""ljtunnel.starts -- the initial states.

product      the lowest orbital of each particle in its own valley (one valley assignment 'sides').
interacting  the correlated start: the ground state of H + W * (penalty that confines particle i to
             valley sides[i]) -- the interacting analogue of the product start, started from it.
The correlated start is the physically relevant one (it sits inside the low band of H); the product
start leaves the band as N grows (a quench) and is kept as a control.
"""
import numpy as np

from .log import log
from .model import masked_orbital
from .resonance import doublet_params


def configs(N):
    """Valley assignments: the (N-1, 1) states (particle i on the right) and the all-left (N, 0) state."""
    out = {f"{N - 1}1_{i + 1}": tuple(1 if j == i else -1 for j in range(N)) for i in range(N)}
    out[f"{N}0"] = (-1,) * N
    return out


def product_state(model, c, M, sides):
    g, mp, C0 = model.g, model.mp, model.mp['C0']
    vs = [P.T @ masked_orbital(g, mp, s, c, i * C0) for i, (P, s) in enumerate(zip(M.P, sides))]
    psi = vs[0]
    for v in vs[1:]:
        psi = np.multiply.outer(psi, v)
    return psi / np.linalg.norm(psi)


def interacting_state(M, prod, sides, W):
    e, v = M.lowest(1, v0=prod.reshape(-1), extra=M.penalty_ops(sides, W), tol=1e-8)
    psi = v[:, 0].reshape((M.k,) * M.N)
    return psi / np.linalg.norm(psi)


def prepare(model, loc, N, k, W):
    """Start states and diagnostics at the located resonance.

    Returns dict with the two start states, the doublet (dE, levels), the start key (the valley
    assignment with the largest weight on the doublet), overlaps and the low-band weights."""
    M, ev, vc, On, res, c = loc['M'], loc['ev'], loc['vc'], loc['On'], loc['res'], loc['c']
    nodoub = res is None
    if nodoub:
        log(f"[N={N} k={k}] c={c}: NO clean doublet among the lowest levels; using levels (0,1) for dE")
        dE, i0, i1 = float(ev[1] - ev[0]), 0, 1
    else:
        dE, i0, i1 = res[0], res[1], res[2]
    cf = configs(N)
    ov, prod = {}, {}
    for name, sides in cf.items():
        prod[name] = product_state(model, c, M, sides)
        w = vc.T @ prod[name].reshape(-1)
        ov[name] = float(abs(w[i0]) ** 2 + abs(w[i1]) ** 2)
    key = max(ov, key=ov.get)
    if ov[key] < model.cfg['doublet'].get('ov_floor', 0.05):
        # the gate-passing doublet is populated by no start: a spurious pair of higher levels.
        nl = vc.shape[1]
        wcfg = {n_: vc.T @ prod[n_].reshape(-1) for n_ in cf}
        i0, i1, _ = max(((i, j, n_) for i in range(nl) for j in range(i + 1, nl) for n_ in cf),
                        key=lambda t: abs(wcfg[t[2]][t[0]]) ** 2 + abs(wcfg[t[2]][t[1]]) ** 2)
        dE = float(ev[i1] - ev[i0])
        nodoub = True
        for n_ in cf:
            ov[n_] = float(abs(wcfg[n_][i0]) ** 2 + abs(wcfg[n_][i1]) ** 2)
        key = max(ov, key=ov.get)
        log(f"[N={N} k={k}] WARNING: the gate-passing doublet is empty for every start (spurious levels); "
            f"using the most populated pair ({i0},{i1}): dE={dE:.6f}, ov={ov[key]:.4f}.  Check <O>_gs ~ 0.5.")
    sides = cf[key]
    inter = interacting_state(M, prod[key], sides, W)
    wI = vc.T @ inter.reshape(-1)
    ovI = float(abs(wI[i0]) ** 2 + abs(wI[i1]) ** 2)
    wp = vc.T @ prod[key].reshape(-1)
    Eexc = {t_: float(np.real(np.vdot(p_, M.matvec(p_.reshape(-1)).reshape(p_.shape))) - ev[0])
            for t_, p_ in (('product', prod[key]), ('interacting', inter))}
    n_rep = doublet_params(model.cfg, N).get('n_report', vc.shape[1])
    if n_rep > vc.shape[1]:
        ev_r, vc_r = M.lowest(n_rep, v0=vc[:, 0])
    else:
        ev_r, vc_r = ev, vc
    deepW = {t_: float(np.sum(np.abs(vc_r.T @ p_.reshape(-1)) ** 2))
             for t_, p_ in (('product', prod[key]), ('interacting', inter))}
    wIr = np.abs(vc_r.T @ inter.reshape(-1)) ** 2
    top = np.argsort(wIr)[::-1][:4]
    log(f"[N={N} k={k}] c*={c:.3f} dE={dE:.6f} <O>_gs={loc['mO']:.3f} doublet=({i0},{i1}) start={key}  "
        f"overlap on doublet: product {ov[key]:.4f}, interacting {ovI:.4f}")
    log(f"      start energy above E0: product {Eexc['product']:.3f}, interacting {Eexc['interacting']:.3f} "
        f"(band of the lowest {len(ev_r)} levels: {ev_r[-1] - ev_r[0]:.3f}); weight on those levels: "
        f"product {deepW['product']:.3f}, interacting {deepW['interacting']:.3f}; main levels of the "
        f"interacting start: " + " ".join(f"#{j}(E-E0={ev_r[j] - ev_r[0]:.3f}, w={wIr[j]:.3f})" for j in sorted(top)))
    diag = dict(c=float(c), dE=float(dE), mO=float(loc['mO']), doublet=not nodoub,
                doublet_levels=[int(i0), int(i1)], start_key=key,
                ov_product=ov[key], ov_interacting=ovI,
                lowW_product=float(np.sum(np.abs(wp) ** 2)), lowW_interacting=float(np.sum(np.abs(wI) ** 2)),
                Eexc_product=Eexc['product'], Eexc_interacting=Eexc['interacting'],
                n_report=int(len(ev_r)), deepW_product=deepW['product'],
                deepW_interacting=deepW['interacting'], W=float(W),
                levels=[float(x) for x in ev], levels_report=[float(x) for x in ev_r],
                weights_interacting=[float(x) for x in wIr])
    return dict(diag=diag, states={'product': prod[key], 'interacting': inter}, sides=sides, key=key)
