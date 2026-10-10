"""ljtunnel.montecarlo -- sign-problem severity of absolute-value path sampling in position space.

For the Strang step U = D_V K D_V with unit-modulus diagonal phases D_V, |U_ij| = |K_ij|: the
column sum of absolute values does not depend on the potential, and
        lim_{dt->0} ln colZ(dt) / dt = Gamma_inf(M) = sum_{n != 0} |ifft(k^2/2)_n| .
Gamma_inf = pi^2 (M^2 - 1) / (3 L_tot^2) per direction (L_tot = grid length), i.e. it grows as M^2
(UV divergence) and is extensive in the number of particles.  A sample count of at least
(1/eps)^2 exp(2 N Gamma_inf t) M^(-N) follows for N particles up to time t.
"""
import numpy as np


def kinetic_col0(M, L):
    """Column 0 of the spectral kinetic matrix on a periodic grid of M points over [-L, L)."""
    dx = 2 * L / M
    k = 2 * np.pi * np.fft.fftfreq(M, d=dx)
    return np.fft.ifft(0.5 * k ** 2).real


def gamma_inf(M, L):
    """Plateau rate Gamma_inf(M) = sum_{n != 0} |col0_n|   (translation invariance; O(M log M))."""
    return float(np.abs(kinetic_col0(M, L)[1:]).sum())


def gamma_inf_exact(M, L_tot):
    """Closed form pi^2 (M^2 - 1) / (3 L_tot^2) = pi^2/(3 dx^2) (1 - M^-2)."""
    return float(np.pi ** 2 * (M ** 2 - 1) / (3 * L_tot ** 2))


def gamma_grid(nx, ny, Lx, Ly):
    """Gamma_inf of the production grid, summed over the two directions."""
    return gamma_inf_exact(nx, 2 * Lx) + gamma_inf_exact(ny, 2 * Ly)


def uv_exponent(Ms, G, Mmin=256):
    """Slope of ln Gamma_inf against ln M for M >= Mmin, and the prefactor."""
    Ms, G = np.asarray(Ms, float), np.asarray(G, float)
    m = Ms >= Mmin
    p, lnalpha = np.polyfit(np.log(Ms[m]), np.log(G[m]), 1)
    return float(p), float(np.exp(lnalpha))


def colZ_dense(M, L, dtau, V=None):
    """Mean column l1 norm of the dense Strang step (cross-check of the plateau identity)."""
    dx = 2 * L / M
    x = np.linspace(-L, L - dx, M)
    V = np.zeros(M) if V is None else V
    k = 2 * np.pi * np.fft.fftfreq(M, d=dx)
    U = np.empty((M, M), complex)
    for j in range(M):
        e = np.zeros(M)
        e[j] = 1.0
        p = e * np.exp(-0.5j * V * dtau)
        p = np.fft.ifft(np.exp(-0.5j * k ** 2 * dtau) * np.fft.fft(p))
        U[:, j] = p * np.exp(-0.5j * V * dtau)
    return float(np.abs(U).sum(axis=0).mean())


def ln_samples(N, t, Gam, M, eps):
    """Lower bound on ln(number of samples):  2 N Gamma t - N ln M + 2 ln(1/eps)."""
    return 2 * N * Gam * t - N * np.log(M) + 2 * np.log(1.0 / eps)


def run(cfg):
    """The Gamma_inf table, the UV exponent, the dense cross-checks and the production-grid value."""
    mc, g = cfg['montecarlo'], cfg['grid']
    L = mc['L']
    Ms = [2 ** p for p in range(mc['p_min'], mc['p_max'] + 1)]
    G = [gamma_inf(M, L) for M in Ms]
    p, alpha = uv_exponent(Ms, G, mc['fit_Mmin'])
    Md = mc['dense_M']
    V = (np.linspace(-L, L - 2 * L / Md, Md) ** 2 - 1.0) ** 2           # A = 1; V drops out of |U|
    cz_V, cz_0 = colZ_dense(Md, L, 0.01, V), colZ_dense(Md, L, 0.01)
    Gd = gamma_inf(Md, L)
    dtau = np.array(mc['dtau'])
    rate = np.array([np.log(colZ_dense(Md, L, d)) / d for d in dtau])
    return dict(M=np.array(Ms), Gamma=np.array(G), p_UV=p, alpha_UV=alpha,
                dense_M=Md, V_independence=abs(cz_V - cz_0), Gamma_dense_M=Gd,
                dtau=dtau, rate=rate, rate_over_Gamma=rate / Gd,
                Gamma_grid=gamma_grid(g['nx'], g['ny'], g['Lx'], g['Ly']),
                Gamma_1D_128=gamma_inf_exact(128, 2 * L))
