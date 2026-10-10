"""ljtunnel.walls -- where the classical routes stop: exact diagonalisation (counting) and tensor
networks (cost as a function of the entropy law).  Pure arithmetic; no simulation.

ED memory  = 16 B * vectors * dim(N)        (Krylov workspace)
ED time    = iters * 2 * (5N+1) * dim(N)    (5-point stencil per particle)
TN memory  = 16 N d chi^2 ,  TN compute = sweeps * N * 2 (d chi)^3 ,  chi = 2^S  (S = central cut)
The TN wall is therefore a function of the entropy law S(N); see ljtunnel.theory and
ljtunnel.extrapolation.
"""
import math
from math import comb

from . import theory as th


def nnz_per_row(N):
    return 5 * N + 1


def ed_wall_dim(dim, B_mem, vectors):
    """First N whose `vectors`-vector workspace exceeds B_mem."""
    N = 2
    while 16.0 * vectors * dim(N) <= B_mem:
        N += 1
    return N


def ed_table(cfg, N_max=7):
    """Rows (N, dim, memory, flops, mem_ok, time_ok) on the position grid, distinguishable particles."""
    gp, ep, B = cfg['grid'], cfg['ed_wall'], cfg['budgets']
    M = gp['nx'] * gp['ny']
    rows = []
    for N in range(2, N_max + 1):
        dim = float(M) ** N
        mem = 16.0 * ep['vectors'] * dim
        fl = ep['iters'] * 2.0 * nnz_per_row(N) * dim
        rows.append((N, dim, mem, fl, mem <= B['classical_mem'], fl <= B['classical_flop']))
    return rows


def ed_walls(cfg):
    """First failing N: grid / mode basis, distinguishable / bosonic."""
    B_mem, v = cfg['budgets']['classical_mem'], cfg['ed_wall']['vectors']
    M = cfg['grid']['nx'] * cfg['grid']['ny']
    km = cfg['tn_wall']['d_mode']
    return dict(
        grid=ed_wall_dim(lambda N: float(M) ** N, B_mem, v),
        grid_boson=ed_wall_dim(lambda N: float(comb(M + N - 1, N)), B_mem, v),
        mode=ed_wall_dim(lambda N: float(km) ** N, B_mem, v),
        mode_boson=ed_wall_dim(lambda N: float(comb(km + N - 1, N)), B_mem, v))


def tn_budget(cfg, d, N):
    """Largest central entropy (bits) the memory and compute budgets allow at particle number N."""
    B, tp = cfg['budgets'], cfg['tn_wall']
    return th.S_budget(N, d, B['classical_mem'], B['classical_flop'], tp['sweeps'])


def tn_wall(cfg, law, d, N_from=2, which='both'):
    """First failing N of the entropy law `law` (callable N -> bits) for local dimension d."""
    B, tp = cfg['budgets'], cfg['tn_wall']
    return th.wall(law, d, B['classical_mem'], B['classical_flop'], tp['sweeps'],
                   N_from=N_from, nmax=tp['nmax'], which=which)


def run(cfg):
    rows = ed_table(cfg)
    return dict(ed_rows=rows, ed=ed_walls(cfg),
                tn_budget_grid=[tn_budget(cfg, cfg['tn_wall']['d_grid'], N) for N in range(2, 101)],
                tn_budget_mode=[tn_budget(cfg, cfg['tn_wall']['d_mode'], N) for N in range(2, 101)])
