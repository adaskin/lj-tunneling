"""ljtunnel.onebody -- one-particle checks of the numerical machinery against problems with known
answers: the box lemma, exact diagonalisation against the instanton formula, the real-time
splitting readout, and the order of the finite-difference stencil."""
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from scipy.signal import find_peaks


def V_pot(x, c, A):
    return A * (x ** 2 - 1.0) ** 2 + c * x


def grid(M, L):
    x = np.linspace(-L, L - 2 * L / M, M)
    return x, x[1] - x[0]


def fd_H(x, dx, c, A):
    V = V_pot(x, c, A)
    return diags([-0.5 / dx ** 2 * np.ones(len(x) - 1), 1.0 / dx ** 2 + V,
                  -0.5 / dx ** 2 * np.ones(len(x) - 1)], [-1, 0, 1], format='csr')


def instanton_dE(A, hbar=1.0):
    """Full splitting E1 - E0 of the quartic double well: leading WKB prefactor (Garg) with the
    next-order factor 1 - 71/(72 S0) (Zinn-Justin & Jentschura) when S0 > 5."""
    S0 = np.sqrt(2 * A) * 4.0 / 3.0
    om = np.sqrt(8 * A)
    corr = (1.0 - 71.0 / (72.0 * S0)) if S0 > 5.0 else 1.0
    return 2.0 * om * np.sqrt(6 * S0 / np.pi) * np.exp(-S0 / hbar) * corr, S0, om


def exact_dE(A, M, L, c=0.0):
    ev = np.sort(eigsh(fd_H(*grid(M, L)[:1], grid(M, L)[1], c, A), k=4, which='SA')[0])
    return float(ev[1] - ev[0])


def sigmaH_analytic(A):
    """Energy width of the Gaussian (s = 1/4) start in the quartic well."""
    s = 0.25
    ET, ET2 = 0.5, 0.75
    EV = A * (4 * s + 3 * s ** 2)
    EV2 = A ** 2 * (48 * s ** 2 + 360 * s ** 3 + 105 * s ** 4)
    Vpp = A * (12 * (1 + s) - 4)
    Vp2 = A * 4 * (12 * s ** 2 + 15 * s ** 3)
    ETV = -0.25 * Vpp + 0.5 * Vp2
    return float(np.sqrt((ET2 - ET ** 2) + (EV2 - EV ** 2) + 2 * (ETV - ET * EV)))


def box_lemma(L, dx, A):
    """Measured energy width of the Gaussian start on a box of half-length L, and the model value."""
    M = int(round(2 * L / dx))
    x, d = grid(M, L)
    H = fd_H(x, d, 0.0, A)
    p = np.exp(-((x - 1.0) ** 2) / (4 * 0.5 ** 2))
    p /= np.sqrt(np.sum(p ** 2) * d)
    Hp = H.dot(p)
    E1, E2 = np.real(np.vdot(p, Hp)) * d, np.real(np.vdot(Hp, Hp)) * d
    sig = sigmaH_analytic(A)
    peak = (2.0 / np.pi) ** 0.25
    pe = peak * np.exp(-((x[-1] - 1.0) ** 2))
    return float(np.sqrt(max(E2 - E1 ** 2, 0))), float(np.sqrt(max(sig ** 2 + 0.25 * pe ** 2 / d ** 3, 0)))


def realtime_splitting(A, L, M, dt, T):
    """Splitting read from the spectrum of the autocorrelation of a real-time (Strang) evolution."""
    x, dx = grid(M, L)
    V = V_pot(x, 0.0, A)
    k = 2 * np.pi * np.fft.fftfreq(M, d=dx)
    p0 = np.exp(-((x - 1.0) ** 2) / (4 * 0.5 ** 2))
    p0 = p0 / np.sqrt(np.sum(p0 ** 2) * dx)
    psi = p0.astype(complex)
    n = int(T / dt)
    Ct = np.zeros(n + 1, complex)
    Ct[0] = np.vdot(p0, psi) * dx
    ph = np.exp(-0.5j * V * dt)
    kin = np.exp(-0.5j * k ** 2 * dt)
    for i in range(1, n + 1):
        psi = ph * np.fft.ifft(kin * np.fft.fft(ph * psi))
        Ct[i] = np.vdot(p0, psi) * dx
    t = np.arange(n + 1) * dt
    E = np.linspace(-0.5, 4.0, 1200)
    win = np.hanning(len(Ct))
    Cw, tw = Ct[::2] * win[::2], t[::2]
    spec = np.zeros(len(E))
    for b in range(0, len(E), 100):
        spec[b:b + 100] = np.abs(np.exp(1j * np.outer(E[b:b + 100], tw)) @ Cw * (tw[1] - tw[0])) ** 2
    pk, _ = find_peaks(spec, height=0.05 * spec.max(), distance=40)
    Ep = np.sort(E[pk[np.argsort(spec[pk])[-2:]]])
    return float(Ep[1] - Ep[0])


def grid_order(A, dxs, box=2.5):
    """Splitting of the one-particle trap against the grid spacing, and the Richardson reference."""
    from scipy.linalg import eigh_tridiagonal
    vals = []
    for dx in dxs:
        n = int(round(2 * box / dx))
        x = -box + dx * np.arange(n)
        d = 1 / dx ** 2 + A * (x ** 2 - 1) ** 2
        e = -np.full(n - 1, 0.5 / dx ** 2)
        w = eigh_tridiagonal(d, e, select='i', select_range=(0, 1))[0]
        vals.append(float(w[1] - w[0]))
    ref = vals[-1] + (vals[-1] - vals[-2]) / 3.0
    return np.array(vals), float(ref)


def run(cfg):
    ob = cfg['onebody']
    Aw, Ad = ob['A_weak'], ob['A_deep']
    out = {}
    out['dE_exact_weak'] = exact_dE(Aw, ob['rt_M'], ob['rt_L'])
    out['dE_exact_deep'] = exact_dE(Ad, ob['rt_M'], ob['rt_L'])
    out['dE_instanton_weak'] = instanton_dE(Aw)[0]
    out['dE_instanton_deep'] = instanton_dE(Ad)[0]
    out['dE_realtime_weak'] = realtime_splitting(Aw, ob['rt_L'], ob['rt_M'], ob['rt_dt'], ob['rt_T'])
    L_list = np.array(ob['box_L'])
    bx = np.array([box_lemma(L, ob['box_dx'], Aw) for L in L_list])
    out.update(box_L=L_list, box_measured=bx[:, 0], box_model=bx[:, 1], sigmaH=sigmaH_analytic(Aw))
    dxs = np.array(ob['grid_dx'])
    vals, ref = grid_order(Ad, dxs)
    out.update(grid_dx=dxs, grid_dE=vals, grid_dE_ref=ref)
    return out
