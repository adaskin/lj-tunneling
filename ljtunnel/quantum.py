"""ljtunnel.quantum -- leading-order quantum resource model.

Counts of logical qubits and T-gates for real-time simulation of the model on the position grid,
by qubitization (one-norm lambda = N lam_U + C(N,2) lam_W) and by a second-order product formula.
These are leading-order formulas with stated constants, not compiled circuits.  Every constant
comes from config.toml.

  Encoding   first quantization on the nx*ny grid, n_p = ceil(log2 M) qubits per particle,
             b-bit fixed point:  Q(N) = N n_p + b + 6
  Time       t*(mu) = pi / dE(mu) with dE from the WKB splitting, normalised to the measured
             t*_0 at the simulated mass
  Trotter-2  r = ceil(t* sqrt(Lambda/eps)) steps, Lambda(N) = N lam_1 + C(N,2) lam_2
  Qubit'n    q = ceil(lam t* + ln(1/eps)),  T_qub = q * C_sel

`CompleteModel` restores two terms the leading-order model omits: the trap frequency's m^(-1/2)
scaling in the WKB prefactor and the kinetic one-norm.
"""
import math

import numpy as np


class QuantumModel:
    def __init__(self, cfg):
        mp, qp, bp = cfg['model'], cfg['quantum'], cfg['budgets']
        self.A_TRAP, self.WY = mp['A'], mp['omega_y']
        self.EPS_LJ, self.SIG, self.V_CAP = mp['epsilon'], mp['sigma'], mp['V_cap']
        self.HBAR, self.M_SIM = mp['hbar'], mp['m']
        self.M_GRID = cfg['grid']['nx'] * cfg['grid']['ny']
        self.T_STAR_0 = qp['t_star_0']
        self.MU_0 = self.M_SIM * self.SIG ** 2 * self.EPS_LJ / self.HBAR ** 2
        self.B_FIXED = qp['b_fixed']
        self.N_ANC = self.B_FIXED + 6
        self.K_GAUSS, self.Q_PIECE, self.P_DEG = qp['k_gauss'], qp['q_piece'], qp['p_deg']
        self.EPS_T = bp['eps']
        self._Lx = cfg['grid']['Lx']
        self.Q_CAP, self.T_CAP = bp['qubits'], bp['t_gates']
        self.eg = qp['estimate_grids']
        # derived elementary costs
        self.N_P = int(np.ceil(np.log2(self.M_GRID)))
        self.C_QFT = self.N_P * int(np.ceil(np.log2(self.N_P)))
        self.C_KIN = 2 * self.C_QFT + self.N_P
        self.C_SUB, self.C_MUL, self.C_CMP = (2 * self.B_FIXED,
                                              4 * self.B_FIXED ** 2, self.N_P)
        self.C_W = (2 * self.C_SUB + 2 * self.C_MUL + self.C_CMP
                    + self.Q_PIECE * self.P_DEG * self.B_FIXED)
        self.C_SEL = 2 * self.C_QFT + self.C_W
        self.LAM1, self.LAM2 = self.commutator_norms()
        gp = cfg['grid']
        _xg = np.linspace(-gp['Lx'], gp['Lx'] - 2 * gp['Lx'] / gp['nx'], gp['nx'])
        _yg = np.linspace(-gp['Ly'], gp['Ly'] - 2 * gp['Ly'] / gp['ny'], gp['ny'])
        _XX, _YY = np.meshgrid(_xg, _yg, indexing='ij')
        self.LAM_U = float(np.max(self.A_TRAP * (_XX ** 2 - 1) ** 2
                                  + 0.5 * self.WY * _YY ** 2))
        self.LAM_W = self.V_CAP

    def S0_action(self, m=None, A=None, a=1.0):
        """WKB action of the quartic barrier, S0 = (4/3) sqrt(2 m A) a / hbar."""
        m = self.M_SIM if m is None else m
        A = self.A_TRAP if A is None else A
        return (4.0 / 3.0) * np.sqrt(2 * m * A) * a / self.HBAR

    def tunnel_time(self, mu):
        """t*(mu) = pi/dE via the WKB splitting, normalised to the measured
        t*_0 at mu_0 (prefactor kept, exponential exact in S0)."""
        S0_0 = self.S0_action()
        S0_m = S0_0 * np.sqrt(mu / self.MU_0)
        om = np.sqrt(8 * self.A_TRAP)
        dE = om * np.sqrt(6 * S0_m / np.pi) * np.exp(-S0_m)
        t_wkb = np.pi / dE
        t_wkb0 = np.pi / (om * np.sqrt(6 * S0_0 / np.pi) * np.exp(-S0_0))
        return self.T_STAR_0 * t_wkb / t_wkb0

    def commutator_norms(self):
        """lam_1 = max |grad U|^2 / m, lam_2 = max 2 |grad W|^2 / m (relative)."""
        cfg_lx = self._Lx
        x = np.linspace(-cfg_lx, cfg_lx, self.eg['trap_points'])
        U = self.A_TRAP * (x ** 2 - 1) ** 2
        lam1 = float(np.max(np.gradient(U, x) ** 2)) / self.M_SIM
        r = np.linspace(self.eg['pair_rmin'], self.eg['pair_rmax'],
                        self.eg['pair_points'])
        W = np.minimum(4 * self.EPS_LJ * ((self.SIG / r) ** 12
                                          - (self.SIG / r) ** 6), self.V_CAP)
        lam2 = float(2 * np.max(np.gradient(W, r) ** 2)) / self.M_SIM
        return lam1, lam2

    def q_log(self, N):
        return N * self.N_P + self.N_ANC

    def t_trot(self, N, mu):
        t = self.tunnel_time(mu)
        Lam = N * self.LAM1 + 0.5 * N * (N - 1) * self.LAM2
        r = int(np.ceil(t * np.sqrt(Lam / self.EPS_T)))
        return r * (N * (self.C_KIN + self.N_P)
                    + 0.5 * N * (N - 1) * self.K_GAUSS * self.C_W)

    def t_qub(self, N, mu):
        t = self.tunnel_time(mu)
        lam = N * self.LAM_U + 0.5 * N * (N - 1) * self.LAM_W
        q = int(np.ceil(lam * t + np.log(1.0 / self.EPS_T)))
        return q * self.C_SEL

    def mu_edge(self, N, which='qub', cap=None, lo=1e-3, hi=1e4):
        """Largest mu with T(N, mu) <= cap (bisection; T monotone in mu)."""
        cap = self.T_CAP if cap is None else cap
        f = self.t_qub if which == 'qub' else self.t_trot
        if f(N, lo) > cap:
            return float('nan')
        for _ in range(200):
            mid = np.sqrt(lo * hi)
            if f(N, mid) <= cap:
                lo = mid
            else:
                hi = mid
        return lo

    def mu_tilde_of_mu(self, mu):
        """trap-units mass mu_tilde = (3/2) S0 at fixed trap (= 8 sqrt(m))."""
        m = mu / self.MU_0
        return 8.0 * np.sqrt(m * self.A_TRAP / 8.0)

    @staticmethod
    def mu_tilde_recoil(s):
        """mu_tilde = (3/2) S0 of any species in a depth-s recoil lattice."""
        return 3.0 * np.sqrt(s)


class CompleteModel(QuantumModel):
    """Leading order plus the omega ~ m^(-1/2) prefactor and the kinetic one-norm in lambda."""

    def __init__(self, cfg):
        super().__init__(cfg)
        g = cfg['grid']
        dx, dy = 2 * g['Lx'] / g['nx'], 2 * g['Ly'] / g['ny']
        self.LAM_KIN = 0.5 * (math.pi / dx) ** 2 + 0.5 * (math.pi / dy) ** 2

    def t_qub(self, N, mu):
        m = mu / self.MU_0
        t = self.tunnel_time(mu) * math.sqrt(m)
        lam = N * (self.LAM_U + self.LAM_KIN / m) + 0.5 * N * (N - 1) * self.LAM_W
        q = int(np.ceil(lam * t + np.log(1.0 / self.EPS_T)))
        return q * self.C_SEL


def edge_table(cfg, N_max=None):
    """Per N: qubits, T-gates at the simulated mass, and the largest trap-units mass mu_tilde whose
    qubitization (leading and complete) / Trotter count fits the T-gate budget ('edge').

    Returns (meta, rows) with rows a list of dicts."""
    lead, comp = QuantumModel(cfg), CompleteModel(cfg)
    N_max = N_max or cfg['quantum']['N_max']
    rows = []
    for N in range(2, N_max + 1):
        et = lead.mu_edge(N, 'trot')
        rows.append(dict(
            N=N, Q=lead.q_log(N), T_qub_mu0=lead.t_qub(N, lead.MU_0), T_trot_mu0=lead.t_trot(N, lead.MU_0),
            edge_qub=lead.mu_tilde_of_mu(lead.mu_edge(N, 'qub')),
            edge_qub_complete=comp.mu_tilde_of_mu(comp.mu_edge(N, 'qub')),
            edge_trot=(lead.mu_tilde_of_mu(et) if et == et else float('nan')),
            tstar_at_edge=float(lead.tunnel_time(lead.mu_edge(N, 'qub')))))
    meta = dict(mu_tilde_0=float(lead.mu_tilde_of_mu(lead.MU_0)), mu_0=float(lead.MU_0),
                S0=float(lead.S0_action()), t_star_0=float(lead.T_STAR_0), n_p=int(lead.N_P),
                b_aux=int(lead.N_ANC), C_sel=int(lead.C_SEL), lam_U=float(lead.LAM_U),
                lam_W=float(lead.LAM_W), lam_kin=float(comp.LAM_KIN), lam2=float(lead.LAM2),
                Q_cap=float(lead.Q_CAP), T_cap=float(lead.T_CAP), eps=float(lead.EPS_T),
                N_Q=int((lead.Q_CAP - lead.N_ANC) // lead.N_P),
                mu_tilde_s26=float(lead.mu_tilde_recoil(26.0)),
                dE2_grid=float(math.pi / lead.T_STAR_0))
    return meta, rows
