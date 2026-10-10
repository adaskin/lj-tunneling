"""Unit tests of the numerical machinery: the Kronecker/BLAS pair projection, the matrix-free
Hamiltonian against its dense form, the eigensolver, the Chebyshev propagator, the entropy
bookkeeping, the cut enumeration and the graph cuts.  None of this is part of a measurement; it is
what you run after changing the code.

    python tests/test_numerics.py            # fast tests (~1 min)
    python tests/test_numerics.py --slow     # + the full-grid N=2 reference (~1 min more)
    pytest tests/                            # also works if pytest is installed
"""
import itertools
import math
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ljtunnel import load_config, Model, ModeN                       # noqa: E402
from ljtunnel import analysis, entanglement as ent, graph, store    # noqa: E402
import ljtunnel.entanglement as entanglement_module                 # noqa: E402
from ljtunnel.dynamics import spectral_bounds, cheb_step            # noqa: E402
from ljtunnel.hamiltonian import dense_H                            # noqa: E402
from ljtunnel.model import modes, project_pair, project_pair_ref    # noqa: E402
from ljtunnel.entanglement import CutPlan                           # noqa: E402

CFG = load_config()
MODEL = Model(CFG)
RNG = np.random.default_rng(3)


def test_project_pair():
    g, mp = MODEL.g, MODEL.mp
    _, Pa = modes(g, mp, 0.9, 0.0, 5)
    _, Pb = modes(g, mp, 0.9, mp['C0'], 5)
    d = np.max(np.abs(project_pair(g['VP'], Pa, Pb) - project_pair_ref(g['VP'], Pa, Pb)))
    assert d < 1e-8, d


def test_matvec_and_lowest_vs_dense():
    for N, k in ((2, 6), (3, 5), (4, 4)):
        M = ModeN(MODEL, 0.8, k, N)
        Hd = dense_H(M)
        x = RNG.standard_normal(k ** N)
        assert np.max(np.abs(M.matvec(x) - Hd @ x)) < 1e-8
        ev, _ = M.lowest(6)
        assert np.max(np.abs(ev - np.linalg.eigvalsh(Hd)[:6])) < 1e-6


def test_chebyshev_vs_eigenbasis():
    M = ModeN(MODEL, 0.8, 5, 3)
    w, U = np.linalg.eigh(dense_H(M))
    lo, hi = spectral_bounds(M)
    psi = RNG.standard_normal(M.dim)
    psi /= np.linalg.norm(psi)
    dt = 0.7
    exact = U @ (np.exp(-1j * w * dt) * (U.T @ psi))
    got = cheb_step(M, psi.reshape((5,) * 3).astype(complex), dt, lo, hi).reshape(-1)
    assert np.max(np.abs(exact - got)) < 1e-7


def test_chebyshev_at_production_step():
    """The dense test has z = b dt ~ 3; production runs have z ~ 20-100: unitarity and step halving."""
    dt = CFG['protocol']['dt_snap']
    for N, k in ((3, 12), (4, 8)):
        M = ModeN(MODEL, 1.4, k, N)
        lo, hi = spectral_bounds(M)
        x = RNG.standard_normal((k,) * N)
        x /= np.linalg.norm(x)
        one = cheb_step(M, x, dt, lo, hi)
        half = cheb_step(M, cheb_step(M, x, dt / 2, lo, hi), dt / 2, lo, hi)
        assert abs(np.linalg.norm(one) - 1) < 1e-10
        assert np.max(np.abs(one - half)) < 1e-9


def test_entropy_bookkeeping():
    k = 4
    bell = np.zeros((k,) * 4)
    for a in (0, 1):                               # (|00>+|11>)_{01} (x) (|00>+|11>)_{23}
        for b in (0, 1):
            bell[a, a, b, b] = 0.5
    S = [ent.entropy_bits(bell, tuple(range(b)), k) for b in (1, 2, 3)]
    assert np.allclose(S, [1, 0, 1])
    assert abs(ent.entropy_bits(bell, (0, 2), k) - 2.0) < 1e-12
    sp = [ent.spectrum_stats(ent.reduced_eigs(bell, (0,), k))]
    assert sp[0][2][0] == 2 and abs(sp[0][1] - 1) < 1e-12           # chi = 2, S_2 = 1
    prodv = np.zeros((k,) * 4)
    prodv[0, 1, 2, 3] = 1.0
    S0, S20, r0 = ent.spectrum_stats(ent.reduced_eigs(prodv, (0, 1), k))
    assert abs(S0) < 1e-12 and r0[0] == 1
    rs = RNG.standard_normal((k,) * 4) + 1j * RNG.standard_normal((k,) * 4)
    rs /= np.linalg.norm(rs)
    S, S2, r = ent.spectrum_stats(ent.reduced_eigs(rs, (0, 1), k))
    assert S >= S2 - 1e-12 and r[2] == k ** 2                        # S_vN >= S_2, full rank


def test_cut_plan_complete_and_symmetric():
    k = 3
    for N in (3, 4, 5, 6):
        plan = CutPlan(N, k, 'all')
        assert plan.n_cuts == 2 ** (N - 1) - 1
        assert all(int(m) & 1 for m in plan.masks) and len(set(plan.masks.tolist())) == plan.n_cuts
        psi = RNG.standard_normal((k,) * N) + 1j * RNG.standard_normal((k,) * N)
        psi /= np.linalg.norm(psi)
        S, bond = plan.measure(psi)
        for j, m in enumerate(plan.masks):
            A = ent.members(int(m), N)
            B = tuple(i for i in range(N) if i not in A)
            direct = ent.spectrum_stats(ent.reduced_eigs(psi, A, k))[0]          # larger side included
            other = ent.spectrum_stats(ent.reduced_eigs(psi, B, k))[0]
            assert abs(S[j] - direct) < 1e-9 and abs(direct - other) < 1e-9      # S(A) = S(A^c), pure state


def test_cut_plan_modes_and_limits():
    p = CutPlan(5, 12, 'bonds')
    assert {1, 2, 3, 4} <= {bin(int(m)).count('1') for m in p.masks}
    q = CutPlan(8, 12, 'all', max_cut_dim=3000)
    assert q.skip.sum() > 0 and not q.skip[[j for j, X in enumerate(q.small) if len(X) <= 2]].any()
    r = CutPlan(6, 12, 'all', big_cut_stride=3)
    assert r.big.sum() > 0 and not r.big[r.bond_idx].any()           # contiguous bonds always every snapshot


def test_graph_cuts():
    W = RNG.uniform(0, 1, (7, 7))
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0)
    masks, C = graph.all_cuts(W)
    assert len(masks) == 2 ** 6 - 1
    for j in RNG.choice(len(masks), 20, replace=False):
        A = ent.members(int(masks[j]), 7)
        assert abs(C[j] - graph.cut_value(W, A)) < 1e-12
    # uniform sharing weights: C(A) = a (N-a)/(N-1) S1
    N, S1 = 9, 1.7
    m2, C2 = graph.all_cuts(graph.sharing_weights(N, S1))
    a = np.array([bin(int(x)).count('1') for x in m2])
    assert np.allclose(C2, a * (N - a) / (N - 1) * S1)
    # two cliques joined by a weak edge: the recursive bisection finds them
    W3 = np.full((6, 6), 0.05)
    W3[:3, :3] = 1.0
    W3[3:, 3:] = 1.0
    np.fill_diagonal(W3, 0)
    bt = graph.bisection_tree(W3)
    assert set(bt['sets'][0]) in ({0, 1, 2}, {3, 4, 5})
    order, cost = graph.best_order(W3)
    assert cost <= max(graph.prefix_cuts(W3, [0, 3, 1, 4, 2, 5])) + 1e-12


def test_store_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        p = store.save_run(os.path.join(d, 'x.npz'), dict(t=np.arange(3.0), S=np.ones((3, 2))),
                           dict(N=2, k=3, a=dict(b=[1, 2])))
        r = store.load_run(p)
        assert r['meta']['a']['b'] == [1, 2] and r['S'].shape == (3, 2)


def test_start_overlaps_match_earlier_ledger():
    """Overlaps and splittings do not depend on the time sampling; they are the regression anchors (N=2,3,4 at k=12).
    The splitting and the product-start overlap do not depend on Theta(x = 0).  The correlated start does (through the penalty that
    defines it): theta_barrier = 0 reproduces the earlier ledger, the default 1/2 gives the numbers of the rerun."""
    import copy
    from ljtunnel.resonance import locate
    from ljtunnel.starts import prepare
    # theta_barrier: (dE, product overlap, correlated overlap, <O>_gs at the config tilt)
    want = {0.0: {2: (0.179637, 0.9772, 0.9759, 0.5639), 3: (0.188569, 0.6725, 0.9390, 0.5122), 4: (0.051736, 0.1542, 0.4631, 0.5384)},
            0.5: {2: (0.179637, 0.9772, 0.9761, 0.5699), 3: (0.188569, 0.6725, 0.9384, 0.5201), 4: (0.051736, 0.1542, 0.4642, 0.5484)}}
    for theta, rows in want.items():
        cfg = copy.deepcopy(CFG)
        cfg['model']['theta_barrier'] = theta
        model = MODEL if theta == CFG['model'].get('theta_barrier', 0.5) else Model(cfg)
        for N, (dE, ovp, ovi, mO) in rows.items():
            loc = locate(model, N, 12)
            d = prepare(model, loc, N, 12, CFG['protocol']['W_penalty'])['diag']
            assert abs(d['dE'] - dE) < 2e-6, (theta, N, d['dE'])
            assert abs(d['ov_product'] - ovp) < 2e-4, (theta, N, d['ov_product'])
            assert abs(d['ov_interacting'] - ovi) < 2e-4, (theta, N, d['ov_interacting'])
            assert abs(d['mO'] - mO) < 2e-4, (theta, N, d['mO'])


def test_theta_convention():
    """Theta(x) is 0 / 1 in the two valleys and [model] theta_barrier on the grid column at the barrier top (and only there)."""
    import copy
    g = MODEL.g
    on = np.abs(g['x']) < 1e-9
    assert on.sum() == 1
    w = g['DR'].reshape(g['nx'], g['ny'])
    assert np.all(w[g['x'] > 0] == 1.0) and np.all(w[g['x'] < 0] == 0.0)
    assert np.all(w[on] == CFG['model'].get('theta_barrier', 0.5))
    for theta in (0.0, 1.0):
        cfg = copy.deepcopy(CFG)
        cfg['model']['theta_barrier'] = theta
        assert np.all(Model(cfg).g['DR'].reshape(g['nx'], g['ny'])[on] == theta)


def test_records_convention_guard_and_control_directory():
    """A run made with another theta_barrier than the config is refused by records.collect (a rerun that leaves old files beside the
    new ones must not be mixed silently), and the off-resonance control (data/controls) enters the graph table only: pick() never
    returns it, and it is never a candidate for the paper's table."""
    import copy
    from ljtunnel import experiments, records
    N, k = 3, 4
    with tempfile.TemporaryDirectory() as d:
        saved_root = experiments.ROOT
        experiments.ROOT = d                                    # the run log goes to d/data/logs, not into the repository
        try:
            experiments.dynamics_run(MODEL, N, k, starts=['interacting'], tilt=1.40, outdir=os.path.join(d, 'runs'), force=True)
            experiments.dynamics_run(MODEL, N, k, starts=['interacting'], tilt=1.70, outdir=os.path.join(d, 'controls'), force=True)
        finally:
            experiments.ROOT = saved_root
        runs, ctl = os.path.join(d, 'runs'), os.path.join(d, 'controls')
        theta = float(CFG['model'].get('theta_barrier', 0.5))
        led, gr = records.collect(runs, theta_barrier=theta, controls_dir=ctl)
        assert [r['c'] for r in led if r['protocol'] == 'uniform'] == [1.40]               # the control is not in the ledger
        assert records.pick(led, N, k)['c'] == 1.40
        ctrl = [g for g in gr if g.get('control')]
        assert len(ctrl) == 1 and ctrl[0]['c'] == 1.70 and ctrl[0]['role'] == 'primary' and ctrl[0]['protocol'] == 'uniform'
        for other in (0.0, 1.0):                                                           # any other convention is refused
            if abs(other - theta) > 1e-12:
                try:
                    records.collect(runs, theta_barrier=other, controls_dir=ctl)
                except ValueError as e:
                    assert 'theta_barrier' in str(e)
                else:
                    raise AssertionError('a run of another Theta convention was accepted')


def test_checkpoint_resume_matches_uninterrupted_run():
    """A series that is killed in the middle and rerun with checkpoint=True continues from the last checkpoint and gives the same
    entropies and <O>(t) as the uninterrupted run; the checkpoints are removed at the end."""
    from ljtunnel import experiments
    N, k, c = 3, 4, 1.40
    orig = entanglement_module.CutPlan.measure
    state = dict(fail=True)

    def flaky(self, psi, snap_index=0):
        if state['fail'] and snap_index == 12:
            raise RuntimeError('simulated kill')
        return orig(self, psi, snap_index)

    with tempfile.TemporaryDirectory() as d:
        saved_root = experiments.ROOT
        experiments.ROOT = d
        try:
            ref = store.load_run(experiments.dynamics_run(MODEL, N, k, starts=['interacting'], tilt=c, outdir=os.path.join(d, 'ref'), force=True)[0])
            entanglement_module.CutPlan.measure = flaky
            try:
                experiments.dynamics_run(MODEL, N, k, starts=['interacting'], tilt=c, outdir=os.path.join(d, 'out'), force=True, checkpoint=True)
                raise AssertionError('the simulated kill did not happen')
            except RuntimeError:
                pass
            ckdir = [p for p in os.listdir(os.path.join(d, 'data', 'ckpt'))]
            assert len(ckdir) == 1 and set(os.listdir(os.path.join(d, 'data', 'ckpt', ckdir[0]))) == {'start.npz', 'series_interacting.npz'}
            state['fail'] = False
            got = store.load_run(experiments.dynamics_run(MODEL, N, k, starts=['interacting'], tilt=c, outdir=os.path.join(d, 'out'),
                                                          force=True, checkpoint=True)[0])
        finally:
            entanglement_module.CutPlan.measure = orig
            experiments.ROOT = saved_root
        assert got['meta']['info']['resumed_from'] == 10, got['meta']['info']
        assert np.allclose(got['O'], ref['O'], atol=1e-7) and np.allclose(got['S'], ref['S'], atol=1e-7, equal_nan=True)
        assert np.allclose(got['bond_chi3'], ref['bond_chi3'], equal_nan=True)
        assert got['meta']['diag']['dE'] == ref['meta']['diag']['dE'] or abs(got['meta']['diag']['dE'] - ref['meta']['diag']['dE']) < 1e-9
        assert not os.path.exists(os.path.join(d, 'data', 'ckpt', ckdir[0]))


def test_analysis_window_and_summary():
    """A synthetic five-snapshot run: the window is [0, T_win] and cuts at T_win."""
    N, k = 2, 3
    plan = CutPlan(N, k, 'all')
    t = np.arange(5.0)
    S = np.zeros((5, plan.n_cuts))
    S[:, :] = np.array([0, 0.2, 0.9, 0.5, 1.0])[:, None]
    run = dict(t=t, O=np.linspace(1, 0, 5), Oi=np.zeros((5, N)), masks=plan.masks, S=S,
               bond_b=plan.bond_b, bond_S2=np.zeros((5, 1)), bond_chi2=np.ones((5, 1)),
               bond_chi3=np.ones((5, 1)), bond_chi6=np.ones((5, 1)),
               meta=dict(N=N, k=k, c=0.7, start='interacting', protocol=dict(kind='uniform', dt_snap=1.0),
                         diag=dict(dE=0.1, mO=0.5, doublet=True, ov_interacting=0.9, lowW_interacting=0.9,
                                   Eexc_interacting=0.1, deepW_interacting=0.9, n_report=4)))
    assert abs(analysis.summarize(run, 2)['S_cen'] - 0.9) < 1e-12      # window t <= 2
    assert abs(analysis.summarize(run)['S_cen'] - 1.0) < 1e-12         # full window
    run['S'][0, :] = 5.0                                               # a prepared start that already holds 5 bits
    assert abs(analysis.summarize(run, 2)['S_cen'] - 5.0) < 1e-12      # t = 0 belongs to the window
    assert abs(analysis.summarize(run, 2, include_start=False)['S_cen'] - 0.9) < 1e-12


def test_n2_has_one_cut():
    """For N = 2 the only proper bipartition is one particle against the other (the pair is the pure state)."""
    assert CutPlan(2, 3, 'all').n_cuts == 1


def test_fullgrid_reference():
    from ljtunnel import fullgrid
    cfg = load_config()
    m = Model(cfg, sigma_smear=cfg['fullgrid']['sigma_smear'])
    r = fullgrid.reference(m, cfg['fullgrid']['c_ref'], cfg['fullgrid']['n_levels'])
    assert abs(r['dE'] - 0.189552) < 2e-6, r['dE']          # the earlier full-grid study: 0.18955


def test_skipped_contiguous_bond_is_reported_and_summarize_refuses():
    """The central bond of N = 8 at k = 8 has a smaller side of 8^4 > max_cut_dim: it must be flagged, and a run in which it
    is NaN must not yield an S_cen."""
    plan = CutPlan(8, 8, 'bonds', max_cut_dim=3000)
    assert plan.skipped_bonds == [4]
    assert CutPlan(8, 8, 'bonds', max_cut_dim=5000).skipped_bonds == []
    N, k = 4, 3
    p = CutPlan(N, k, 'bonds')
    t = np.arange(3.0)
    S = np.ones((3, p.n_cuts))
    S[:, p.bond_idx[1]] = np.nan                                       # the central bond was not measured
    run = dict(t=t, O=np.zeros(3), Oi=np.zeros((3, N)), masks=p.masks, S=S, bond_b=p.bond_b,
               bond_S2=np.zeros((3, N - 1)), bond_chi2=np.ones((3, N - 1)), bond_chi3=np.ones((3, N - 1)),
               bond_chi6=np.ones((3, N - 1)),
               meta=dict(N=N, k=k, c=1.0, start='interacting', protocol=dict(kind='uniform', dt_snap=1.0),
                         diag=dict(dE=0.1, mO=0.5, doublet=True, ov_interacting=0.9, lowW_interacting=0.9,
                                   Eexc_interacting=0.1, deepW_interacting=0.9, n_report=4)))
    try:
        analysis.summarize(run)
    except ValueError as e:
        assert 'not measured' in str(e)
    else:
        raise AssertionError('summarize accepted an unmeasured central bond')


def test_resonance_scan_finds_a_transition_that_jumps_over_the_grid():
    """N = 4, k = 6: <O>_gs falls 0.81 -> 0.15 between c = 2.4 and 2.5, so no scanned tilt lies in (0.2, 0.8); the scan used to
    return None.  It must now refine around the <O> = 1/2 crossing."""
    from ljtunnel.resonance import locate
    loc = locate(MODEL, 4, 6, tilt='scan')
    assert loc is not None and 2.4 < loc['c'] < 2.55 and 0.3 < loc['mO'] < 0.7, loc and (loc['c'], loc['mO'])


def test_tn_tables_reproduce_stored_numbers():
    """The tables for the tensor-network section read stored runs only: the zero-shift sensitivity row is the stored wall, the window
    table is `analysis.summarize`, and every stored MPS run stays within the documented distance of its exact reference."""
    from ljtunnel import tntables
    data = os.path.join(ROOT, 'data')
    L = store.load_object(os.path.join(data, 'derived', 'larger_N.npz'))
    rows = tntables.wall_sensitivity(CFG, L)
    z = [r for r in rows if abs(r['dS1']) < 1e-9][0]
    for name in ('sat', 'opt', 'pess'):
        assert z[f'wall_{name}_grid'] == L['walls'][f"{name}_d{CFG['tn_wall']['d_grid']}_both"]
        assert z[f'wall_{name}_mode'] == L['walls'][f"{name}_d{CFG['tn_wall']['d_mode']}_both"]
    path = os.path.join(data, 'runs', 'N4_k12_int_c2.260_T200_dt1.npz')
    for r in tntables.window_convergence([('N4', path)], windows=(60, 200)):
        assert abs(r['S_cen'] - analysis.summarize(store.load_run(path), r['T'])['S_cen']) < 1e-12
    rows, series = tntables.mps_validation(os.path.join(data, 'mps'), [os.path.join(data, 'mps', 'ed_ref'), os.path.join(data, 'runs')])
    for r in rows:
        capped = r['chi_kept'] >= r['chi_max']                 # the bond-dimension cap was binding (the k = 12 case): a truncation error
        dS, dO, dScen = r['max_dS'], r['max_dO'], abs(r['S_cen_mps'] - r['S_cen_exact'])
        if capped:                                             # chi = 64 against an exact rank 117: 0.02-0.03 bit is the measured size
            assert dS < 0.05 and dScen < 0.03 and dO < 0.01, (r['case'], dS, dScen, dO)
        else:                                                  # converged MPS: agreement to the DMRG / TDVP / MPO tolerances
            assert dS < 0.01 and dScen < 0.005 and dO < 0.001, (r['case'], dS, dScen, dO)
        assert sum(1 for x in series if x['case'] == r['case']) == int(r['T']) + 1


FAST = [test_project_pair, test_matvec_and_lowest_vs_dense, test_chebyshev_vs_eigenbasis,
        test_chebyshev_at_production_step, test_entropy_bookkeeping, test_cut_plan_complete_and_symmetric,
        test_cut_plan_modes_and_limits, test_graph_cuts, test_store_roundtrip,
        test_start_overlaps_match_earlier_ledger, test_theta_convention, test_records_convention_guard_and_control_directory, test_checkpoint_resume_matches_uninterrupted_run,
        test_analysis_window_and_summary,
        test_n2_has_one_cut,
        test_skipped_contiguous_bond_is_reported_and_summarize_refuses,
        test_resonance_scan_finds_a_transition_that_jumps_over_the_grid, test_tn_tables_reproduce_stored_numbers]
SLOW = [test_fullgrid_reference]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    tests = FAST + (SLOW if '--slow' in argv else [])
    bad = 0
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except Exception as e:                          # noqa: BLE001
            bad += 1
            print(f"[FAIL] {t.__name__}: {type(e).__name__}: {e}")
    print('TESTS ' + ('PASSED' if not bad else f'FAILED ({bad})'))
    return bad


if __name__ == '__main__':
    sys.exit(1 if main() else 0)
