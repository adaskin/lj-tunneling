"""ljtunnel.model -- the physical model on the grid.

N distinguishable particles in the 2D double well  U = A (x^2-1)^2 + omega_y/2 y^2  with a
tilt c x (particle i: c + i*C0) and capped Lennard-Jones pair interaction.  This module
builds the one-particle problem (grid, Hamiltonian, mode basis, valley-masked orbitals) and
the pair-interaction matrix elements in the mode basis.  Nothing here evolves in time.

Convention: eigenvectors are Euclidean-normalised vectors on the nx*ny sites.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

_CACHE = {}


def make_grid(nx, ny, Lx, Ly, sigma_smear, mp):
    """Grid, kinetic operator and pair matrix.  mp = cfg['model']."""
    A, wy = mp['A'], mp['omega_y']
    EPS, SIG, V_CAP = mp['epsilon'], mp['sigma'], mp['V_cap']
    kin_pref = -0.5 * mp['hbar'] ** 2 / mp['m']
    x = np.linspace(-Lx, Lx - 2 * Lx / nx, nx)
    y = np.linspace(-Ly, Ly - 2 * Ly / ny, ny)
    dx, dy = 2 * Lx / nx, 2 * Ly / ny

    def lap(n, d):
        return sp.diags(
            [kin_pref / d ** 2 * np.ones(n - 1),
             -2 * kin_pref / d ** 2 * np.ones(n),
             kin_pref / d ** 2 * np.ones(n - 1)],
            [-1, 0, 1], format='csr')

    Kx, Ky = lap(nx, dx), lap(ny, dy)
    X2 = np.repeat(x[:, None], ny, 1)
    Y2 = np.repeat(y[None, :], nx, 0)
    g = dict(x=x, y=y, dx=dx, dy=dy, nx=nx, ny=ny, NB1=nx * ny, X2=X2, Y2=Y2,
             kin=(sp.kron(Kx, sp.identity(ny)) + sp.kron(sp.identity(nx), Ky)).tocsr())
    Xf, Yf = X2.ravel(), Y2.ravel()
    R = np.sqrt((Xf[:, None] - Xf[None, :]) ** 2 + (Yf[:, None] - Yf[None, :]) ** 2)
    if sigma_smear > 0:
        Rs = np.sqrt(R ** 2 + sigma_smear ** 2)
    else:
        Rs = np.maximum(R, 1e-12)
    g['VP'] = np.minimum(4 * EPS * ((SIG / Rs) ** 12 - (SIG / Rs) ** 6), V_CAP)
    g['SAME'] = (R < 1e-12).astype(float)
    # Theta(x) on the sites.  The grid x_j = -Lx + j dx has a column exactly at the barrier top x = 0; it carries the weight
    # [model] theta_barrier (1/2: symmetric; 0: counted as left, the convention of every run made before the rerun).
    on_barrier = np.abs(x) < 1e-9 * dx
    g['DR'] = np.repeat(np.where(on_barrier, float(mp.get('theta_barrier', 0.5)), (x > 0).astype(float)), ny)
    return g


def h1(g, mp, c, extra):
    """One-body Hamiltonian at tilt c (+ per-particle scalar offset `extra`)."""
    pot = (mp['A'] * (g['X2'] ** 2 - 1) ** 2 + 0.5 * mp['omega_y'] * g['Y2'] ** 2
           + (c + extra) * g['X2']).ravel()
    return (g['kin'] + sp.diags(pot)).tocsc()


def masked_orbital(g, mp, side, c, extra, W=1e3):
    """Lowest orbital with a wall on one side: side=+1 keeps x>0, side=-1 keeps x<0."""
    key = ('mask', id(g), side, round(c, 6), extra)
    if key not in _CACHE:
        pen = np.where((g['x'] < 0) if side > 0 else (g['x'] > 0), W, 0.0)
        H = h1(g, mp, c, extra) + sp.diags(np.repeat(pen[:, None], g['ny'], 1).ravel())
        v = eigsh(H, k=1, which='SA')[1][:, 0]
        _CACHE[key] = v / np.linalg.norm(v)
    return _CACHE[key]


def modes(g, mp, c, extra, k):
    """The k lowest one-body modes at tilt c + extra: (energies, (NB1, k) vectors)."""
    key = ('modes', id(g), round(c, 6), extra, k)
    if key not in _CACHE:
        H = h1(g, mp, c, extra)
        if k >= g['NB1']:
            ev, vc = np.linalg.eigh(H.toarray())
        else:
            ev, vc = eigsh(H, k=k, which='SA', tol=1e-10)
        o = np.argsort(ev)
        _CACHE[key] = (ev[o], vc[:, o])
    return _CACHE[key]


def project_pair(V, Pa, Pb):
    """Pair interaction in the mode basis, d[(p,q),(P,Q)] = sum_ij Pa[i,p] Pb[j,q] V[i,j] Pa[i,P] Pb[j,Q],
    as a (k^2, k^2) matrix, by two BLAS products."""
    k = Pa.shape[1]
    KA = (Pa[:, :, None] * Pa[:, None, :]).reshape(-1, k * k)
    KB = (Pb[:, :, None] * Pb[:, None, :]).reshape(-1, k * k)
    M1 = KA.T @ V
    Vp = (M1 @ KB).reshape(k, k, k, k)
    return np.ascontiguousarray(Vp.transpose(0, 2, 1, 3)).reshape(k * k, k * k)


def project_pair_ref(V, Pa, Pb):
    """Same by direct einsum (reference for the tests)."""
    k = Pa.shape[1]
    M1 = np.einsum('ip,ij,iP->pjP', Pa, V, Pa)
    Vp = np.einsum('pjP,jq,jQ->pqPQ', M1, Pb, Pb)
    return Vp.reshape(k * k, k * k)


class Model:
    """The configuration plus the grid: the one object every simulation starts from.

        from ljtunnel import load_config, Model
        model = Model(load_config())
    """

    def __init__(self, cfg, sigma_smear=None):
        self.cfg = cfg
        gp = cfg['grid']
        smear = gp['sigma_smear'] if sigma_smear is None else sigma_smear
        self.g = make_grid(gp['nx'], gp['ny'], gp['Lx'], gp['Ly'], smear, cfg['model'])
        self.sigma_smear = smear

    @property
    def mp(self):
        return self.cfg['model']

    def tilt(self, N, k):
        """Measured resonance tilt for (N, k) from [tilts], or None (-> scan)."""
        t = self.cfg.get('tilts', {}).get(f'N{N}')
        if isinstance(t, dict):
            return t.get(f'k{k}')
        return t
