"""Tests of ljtunnel.mps (TeNPy matrix-product states) against the exact machinery: the MPO against the dense Hamiltonian,
the DMRG start against the exact correlated start, TDVP against the Chebyshev propagator, the pair entropies against
dense reduced density matrices, truncation and the stored layout.  Needs TeNPy (pip install physics-tenpy); skipped otherwise.

    python tests/test_mps.py            # about 40 s
    pytest tests/test_mps.py
"""
import itertools
import os
import sys
import tempfile
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings('ignore', message='unit_cell_width')

try:
    import tenpy                                                                  # noqa: F401
    HAVE_TENPY = True
except ImportError:
    HAVE_TENPY = False

if HAVE_TENPY:
    from ljtunnel import load_config, Model, ModeN, analysis, entanglement as ent, mps, store
    from ljtunnel.dynamics import snapshots, spectral_bounds
    from ljtunnel.hamiltonian import dense_H
    from ljtunnel.starts import product_state, interacting_state

    CFG = load_config()
    MODEL = Model(CFG)
    RNG = np.random.default_rng(5)
    W_PEN = CFG['protocol']['W_penalty']


def dense_from_mpo(H, N, k):
    """The MPO contracted to a k^N x k^N matrix (site 0 slowest, as in ModeN)."""
    Ws = [H.get_W(i).to_ndarray() for i in range(N)]                              # p, p*, wL, wR
    full = Ws[0][:, :, H.IdL[0], :]
    for i in range(1, N):
        full = np.einsum('abx,cdxy->acbdy', full, Ws[i]).reshape(full.shape[0] * k, full.shape[1] * k, -1)
    return full[:, :, H.IdR[N]]


def test_mpo_equals_dense_hamiltonian():
    N, k, c = 3, 4, 1.2
    M = ModeN(MODEL, c, k, N)
    Hd = dense_H(M)
    exact = mps.MPSProblem(MODEL, c, k, N, channel_tol=1e-13)
    assert np.max(np.abs(dense_from_mpo(exact.mpo(), N, k) - Hd)) < 1e-9
    loose = mps.MPSProblem(MODEL, c, k, N, channel_tol=1e-3)
    d = np.max(np.abs(dense_from_mpo(loose.mpo(), N, k) - Hd))
    assert loose.R < exact.R and d < 3 * loose.err + 1e-9 and loose.err > 1e-8, (loose.R, exact.R, d, loose.err)


def test_channels_are_the_span_of_mode_products():
    """R cannot exceed the span of the products of modes, and the error falls as the tolerance does."""
    N, k = 4, 5
    errs, Rs = [], []
    for tol in (1e-2, 1e-4, 1e-8, 1e-12):
        p = mps.MPSProblem(MODEL, 1.0, k, N, channel_tol=tol)
        errs.append(p.err)
        Rs.append(p.R)
    assert Rs == sorted(Rs) and errs == sorted(errs, reverse=True), (Rs, errs)
    assert Rs[-1] <= N * k * (k + 1) // 2 and errs[-1] < 1e-9


def test_dmrg_start_matches_exact_correlated_start():
    N, k, c = 3, 5, 1.46
    prob = mps.MPSProblem(MODEL, c, k, N)
    sides = mps.start_sides(N)
    psi, info = mps.correlated_start(prob, sides, W_PEN, chi_max=40, eps=1e-12)
    M = ModeN(MODEL, c, k, N)
    ref = interacting_state(M, product_state(MODEL, c, M, sides), sides, W_PEN)
    assert abs(abs(np.vdot(ref, prob.dense(psi))) ** 2 - 1) < 1e-8
    E_ed = float(np.vdot(ref, M.matvec(ref.reshape(-1)).reshape(ref.shape)).real)
    assert abs(info['E'] - E_ed) < 5 * prob.err + 1e-9, (info['E'], E_ed, prob.err)       # the dropped channels


def test_tdvp_matches_chebyshev_series():
    """DMRG start -> TDVP at full rank against the Chebyshev series of the exact code: <O>, every bond, every single."""
    N, k, c, T = 3, 5, 1.46, 12
    prob = mps.MPSProblem(MODEL, c, k, N)
    sides = mps.start_sides(N)
    run = mps.run_series(prob, sides=sides, T=T, dt_snap=1.0, dt=0.5, chi_max=40, eps=1e-14, label='test')
    M = ModeN(MODEL, c, k, N)
    ref = interacting_state(M, product_state(MODEL, c, M, sides), sides, W_PEN)
    plan = ent.CutPlan(N, k, 'bonds')
    ts = np.arange(T + 1.0)
    for i, t, psi in snapshots(M, ref, ts, spectral_bounds(M)):
        S, bond = plan.measure(psi, i)
        assert np.max(np.abs(run['S'][i] - S)) < 2e-4, (t, np.abs(run['S'][i] - S).max())
        assert abs(run['O'][i] - sum(M.O_particle(psi))) < 1e-4
        assert np.max(np.abs(run['bond_chi3'][i] - bond[:, 2])) <= 1
    assert run['meta']['mps']['drift_E'] < 1e-8                       # full rank: H is conserved
    assert run['meta']['protocol']['kind'] == 'mps'


def test_pair_entropies_against_dense():
    from tenpy.linalg import np_conserved as npc
    from tenpy.networks.mps import MPS
    N, k = 5, 3
    prob = mps.MPSProblem(MODEL, 1.0, k, N)
    vec = RNG.standard_normal((k,) * N) + 1j * RNG.standard_normal((k,) * N)
    vec /= np.linalg.norm(vec)
    arr = npc.Array.from_ndarray(vec, [s.leg for s in prob.sites], labels=[f'p{i}' for i in range(N)])
    psi = MPS.from_full(prob.sites, arr, form='B', cutoff=0.0)
    psi.canonical_form()
    got = mps.pair_entropies(psi, k)
    assert len(got) == N * (N - 1) // 2
    for (i, j), s in got.items():
        assert abs(s - ent.entropy_bits(vec, (i, j), k)) < 1e-9, (i, j)


def test_truncation_is_reported_and_bounded():
    N, k, c, T, chi = 4, 5, 2.0, 8, 6
    prob = mps.MPSProblem(MODEL, c, k, N)
    sides = mps.start_sides(N)
    run = mps.run_series(prob, sides=sides, T=T, dt_snap=1.0, dt=0.5, chi_max=chi, eps=1e-6, label='test')
    assert run['chi'].max() <= chi and run['trunc_err'][-1] > 0 and np.all(np.diff(run['trunc_err']) >= -1e-15)
    full = mps.run_series(prob, sides=sides, T=T, dt_snap=1.0, dt=0.5, chi_max=40, eps=1e-14, label='test')
    assert full['chi'].max() > chi                                    # the cap was binding
    assert abs(mps.summarize(run)['S_cen'] - mps.summarize(full)['S_cen']) < 0.15


def test_saved_layout_works_with_analysis():
    N, k = 3, 4
    prob = mps.MPSProblem(MODEL, 1.4, k, N)
    run = mps.run_series(prob, T=4, dt_snap=1.0, dt=0.5, chi_max=16, eps=1e-12, label='test')
    with tempfile.TemporaryDirectory() as d:
        back = store.load_run(mps.save(run, d))
    for key in ('t', 'O', 'S', 'masks', 'chi', 'bond_chi3', 'energy', 'trunc_err'):
        assert np.allclose(back[key], run[key], equal_nan=True), key
    s = mps.summarize(back, T_win=2)
    assert s['T'] == 2.0 and s['N'] == N and len(s['bonds']) == N - 1
    sig, _ = analysis.maxima(back, 2)
    assert abs(s['S_nat'] - max(sig[analysis.column(back, tuple(range(b)))] for b in range(1, N))) < 1e-12
    assert analysis.column(back, (1,)) >= 0                           # singles are stored under their canonical mask


TESTS = [test_mpo_equals_dense_hamiltonian, test_channels_are_the_span_of_mode_products,
         test_dmrg_start_matches_exact_correlated_start, test_tdvp_matches_chebyshev_series,
         test_pair_entropies_against_dense, test_truncation_is_reported_and_bounded,
         test_saved_layout_works_with_analysis]

if not HAVE_TENPY:
    try:
        import pytest
        pytestmark = pytest.mark.skip(reason='TeNPy not installed (pip install physics-tenpy)')
    except ImportError:
        pass


def main(argv=None):
    if not HAVE_TENPY:
        print('SKIPPED: TeNPy not installed (pip install physics-tenpy)')
        return 0
    bad = 0
    for t in TESTS:
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
