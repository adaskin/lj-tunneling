"""followup_data.py -- readers for data/followup/ (the Stage 1 spectra and the static checks), shared by make_numbers.py and figs.py.

Nothing here is typed by hand: every quantity is computed from the saved npz files and logs.

    stage1()      {(N, k): record} from data/followup/stage1/N*_k*_c*.npz (dense-or-Lanczos level lists with the weight of the
                  correlated start on each level); a *_krylov.npz is used only for a case that has no other file.
    cosines()     rows parsed from data/followup/freq_signal_analysis.log (multi-cosine fit residuals of the stored runs)
    levels()      {(N, k): arrays} from data/followup/static_checks/levels_*.npz (parity and right-valley occupation per level)
    untilted()    {N: record} from data/followup/stage1/untilted_N*_k12_c0.0000*.npz
    c_top()       the tilt at which the upper-valley level of the one-particle tilted well reaches the barrier top
"""
import glob
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, '..', '..')
FU = os.path.join(REPO, 'data', 'followup')
S1 = os.path.join(FU, 'stage1')

_name = re.compile(r'N(\d+)_k(\d+)_c([0-9.]+?)(_krylov)?\.npz$')


def _record(d, N, k, c, krylov):
    E = np.asarray(d['E'], float)
    E = E - E[0]
    w = np.asarray(d['w_interacting'], float)
    a, b = sorted(int(i) for i in np.argsort(-w)[:2])
    J = (E[b] - E[a]) / 2
    keep = np.ones(len(E), bool)
    keep[[a, b]] = False
    dist = np.minimum(abs(E - E[a]), abs(E - E[b]))
    return dict(N=N, k=k, c=c, krylov=krylov, E=E, w=w, a=a, b=b, J=J, B2=float(w[a] + w[b]), cov=float(w.sum()),
                near=float(w[keep & (dist < J)].sum()), mid=float(w[keep & (dist >= J) & (dist < 3 * J)].sum()),
                far=float(w[keep & (dist >= 3 * J)].sum()))


def stage1():
    out = {}
    for f in sorted(glob.glob(os.path.join(S1, 'N*_k*_c*.npz'))):
        m = _name.search(f)
        if not m:
            continue
        N, k, c, kr = int(m.group(1)), int(m.group(2)), float(m.group(3)), bool(m.group(4))
        d = np.load(f, allow_pickle=True)
        if 'E' not in d.files:
            continue
        if (N, k) in out and kr:
            continue
        out[(N, k)] = _record(d, N, k, c, kr)
    return out


_cos = re.compile(r'N(\d+)_k(\d+)_int_c([0-9.]+)_T200_dt1\.npz sd=([0-9.]+) hankel rank\(([^)]*)\)= \[([^\]]*)\] '
                  r'M for resid<0\.2/0\.1/0\.05: \[([^\]]*)\] resid\(M=1,2,3,4\)= \[([^\]]*)\]')


def cosines():
    rows = []
    for line in open(os.path.join(FU, 'freq_signal_analysis.log')):
        m = _cos.match(line)
        if m:
            Ms = [None if t.strip() == 'None' else int(t) for t in m.group(7).split(',')]
            rows.append(dict(N=int(m.group(1)), k=int(m.group(2)), c=float(m.group(3)), sd=float(m.group(4)),
                             rank=[int(t) for t in m.group(6).split(',')], M=Ms, resid=[float(t) for t in m.group(8).split(',')]))
    return rows


def levels():
    out = {}
    for f in sorted(glob.glob(os.path.join(FU, 'static_checks', 'levels_N*_k*_c*.npz'))):
        m = re.search(r'levels_N(\d+)_k(\d+)_c([0-9.]+)\.npz', f)
        d = np.load(f)
        out[(int(m.group(1)), int(m.group(2)))] = {k: d[k] for k in d.files}
    return out


def untilted():
    out = {}
    for f in sorted(glob.glob(os.path.join(S1, 'untilted_N*_k12_c0.0000*.npz'))):
        m = re.search(r'untilted_N(\d+)_k12_c0.0000(_C00)?\.npz', f)
        d = np.load(f, allow_pickle=True)
        out[(int(m.group(1)), bool(m.group(2)))] = np.asarray(d['E_minus_E0'], float)
    return out


def c_top(A=8.0, n=500, L=2.5):
    """Tilt c at which the lowest level localized in the raised valley of V(x) = A(x^2-1)^2 + c x reaches the barrier top
    (finite differences, n points on [-L, L], hard walls: the rough single-particle picture of the static-checks README)."""
    x = np.linspace(-L, L, n)
    h = x[1] - x[0]
    T = (np.diag(np.full(n, 2.0)) - np.diag(np.ones(n - 1), 1) - np.diag(np.ones(n - 1), -1)) / (2 * h * h)

    def gap(c):
        V = A * (x ** 2 - 1) ** 2 + c * x
        w, v = np.linalg.eigh(T + np.diag(V))
        # barrier top: the maximum of V between its two minima
        i0, i1 = np.argmin(V[: n // 2]), n // 2 + np.argmin(V[n // 2:])
        itop = i0 + int(np.argmax(V[i0:i1 + 1]))
        right = (v ** 2)[itop:].sum(0)
        j = next(i for i in range(len(w)) if right[i] > 0.5)
        return w[j] - V[itop]
    lo, hi = 3.0, 8.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if gap(mid) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
