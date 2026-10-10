#!/usr/bin/env python3
"""figs.py -- every figure of the paper, read from data/ only.

    python scripts/figs.py [name ...]      (default: all)

Inputs:  data/derived/{ledger,graph,larger_N,quantum}.npz, the saved runs (data/runs),
         data/studies/onebody_mc/mc_gamma_ledger.txt, config.toml.
A point from a run with another window or spacing than the protocol would be drawn open and dashed and labelled with a
dagger (none is: every point is a uniform run of data/runs).
A figure whose input is missing is replaced by a boxed placeholder that names the file it
needs, so a missing log is visible in the PDF and never filled by hand.
"""
import glob
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle, Polygon
from matplotlib.colors import to_rgb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')                       # paper/
REPO = os.path.join(ROOT, '..')
RAW = os.path.join(REPO, 'data', 'studies', 'onebody_mc')
DER = os.path.join(REPO, 'data', 'derived')
FIG = os.path.join(ROOT, 'figures')
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, REPO)
from ljtunnel import load_config, store, analysis, records  # noqa: E402

cfg = load_config(os.path.join(REPO, 'config.toml'))

# Okabe-Ito
BLUE, ORANGE, GREEN, RED, PURPLE, SKY, YEL, GREY = ('#0072B2', '#E69F00', '#009E73', '#D55E00',
                                                    '#CC79A7', '#56B4E9', '#F0E442', '#666666')
# One marker and one colour per particle number, the same in every figure (N >= 7 falls back to the last entry).
MK = {2: 'o', 3: 's', 4: '^', 5: 'D', 6: 'v'}
CL = {2: BLUE, 3: GREEN, 4: ORANGE, 5: RED, 6: PURPLE}


def mk_of(N):
    return MK.get(N, 'P')


def cl_of(N):
    return CL.get(N, SKY)


plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.labelsize': 9.5, 'legend.frameon': False, 'figure.dpi': 150,
                     'savefig.dpi': 300, 'pdf.fonttype': 42, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
                     'legend.fontsize': 8})


def jload(name, folder=DER):
    """A derived table (npz with a JSON record); the graph and ledger tables are lists of records."""
    p = os.path.join(folder, name.replace('.json', '.npz'))
    if not os.path.exists(p):
        return None
    o = store.load_object(p)
    return o['rows'] if set(o) >= {'rows'} and name in ('ledger.json', 'graph.json') else o


def placeholder(name, need, size=(6.4, 2.6)):
    fig, ax = plt.subplots(figsize=size)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linestyle('--')
        s.set_color(RED)
    ax.text(0.5, 0.5, f"PLACEHOLDER  {name}\nno log yet: {need}", ha='center', va='center',
            color=RED, fontsize=10, transform=ax.transAxes)
    save(fig, name)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + '.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG, name + '.png'), bbox_inches='tight', dpi=200)
    plt.close(fig)
    print('wrote figures/' + name)


def tilt12(N):
    """Resonance tilt at k = 12 from config.toml [tilts] (N = 2: a single number); None if the file has none for this N."""
    t = cfg['tilts'].get(f'N{N}')
    return t.get('k12') if isinstance(t, dict) else t


def series(N):
    """The saved correlated-start run of N particles at k=12 and the resonance tilt: a uniform run if there is one,
    else a run of another protocol.  Returns (run, is_uniform) or (None, None)."""
    found = []
    if tilt12(N) is None:
        return None, None
    for folder in (os.path.join(REPO, 'data', 'runs'),):
        for p in store.list_runs(folder, f'N{N}_k12_int_c{tilt12(N):.3f}_*.npz'):
            r = store.load_run(p)
            found.append((r['meta']['protocol']['kind'] != 'uniform', r))
    found.sort(key=lambda x: x[0])
    return (found[0][1], found[0][1]['meta']['protocol']['kind'] == 'uniform') if found else (None, None)


# ============================================================================ Fig. 1
def fig_regime():
    q, L = jload('quantum.json'), jload('larger_N.json')
    if q is None or L is None:
        return placeholder('fig_regime', 'data/derived/quantum.npz, larger_N.npz')
    rows = [r for r in q['rows'] if r['N'] >= 4]
    Nq = np.array([r['N'] for r in rows])
    e_lead = np.array([r['edge_qub'] for r in rows])
    e_full = np.array([r['edge_qub_complete'] for r in rows])
    w = L['walls']
    d = L['budgets']['d_grid']
    N_lo = min(L['walls_chi'][f'{k_}_d{d}_both'] for k_ in ('pess', 'sat', 'opt'))   # earliest wall of any scenario, charging chi(1e-3)
    N_est = w[f'sat_d{d}_both']
    N_far = max(w[f'opt_d{d}_both'], N_est)
    N_ED = L['ed']['grid']                    # first failing N on the grid
    N_ED_mode = L['ed']['mode']
    NQ = q['meta']['N_Q']

    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    xmin, xmax, ymax = 1.8, 100.0, 26.0
    ax.set_xscale('log')
    xs = np.logspace(math.log10(xmin), math.log10(xmax), 900)
    ys = np.linspace(0, ymax, 2)
    edge = np.interp(np.log(xs), np.log(np.r_[2, Nq]), np.r_[e_lead[0], e_lead])
    edge_lo = np.interp(np.log(xs), np.log(np.r_[2, Nq]), np.r_[e_full[0], e_full])
    edge[xs > Nq.max()] = np.nan
    edge_lo[xs > Nq.max()] = np.nan
    # qubit budget wall: beyond N_Q nothing fits
    ok = xs <= NQ + 1e-9

    # classical-reach weight g(N): 1 for exact diag, 0.5 for TN, fading to 0 across the band
    def g(N):
        if N < N_ED - 0.5:
            return 1.0
        if N < N_lo:
            return 0.45
        if N < N_far + 0.6 * (N_far - N_lo) + 3:
            x = (math.log(N) - math.log(N_lo)) / (math.log(N_far + 0.6 * (N_far - N_lo) + 3) - math.log(N_lo))
            return 0.45 * (1 - x) ** 1.2
        return 0.0

    gw = np.array([g(x) for x in xs])
    img = np.zeros((2, len(xs), 4))
    cg, cb = np.array(to_rgb('#9fd3b0')), np.array(to_rgb('#9dc3ec'))
    for i, (x, gg) in enumerate(zip(xs, gw)):
        base = cg if gg > 0 else cb
        col = cg * min(1, gg / 0.45) + cb * (1 - min(1, gg / 0.45))
        if x < N_ED - 0.5:
            col = np.array(to_rgb('#6fbf8b'))
        img[:, i, :3] = col
        img[:, i, 3] = 1.0
    # draw below the edge only
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path
    top = np.where(np.isnan(edge), 0, edge)
    verts = [(xs[0], 0)] + list(zip(xs, np.where(ok, top, 0))) + [(xs[-1], 0)]
    clip = Polygon(verts, closed=True, transform=ax.transData)
    im = ax.imshow(img, extent=[math.log10(xmin), math.log10(xmax), 0, ymax], aspect='auto',
                   origin='lower', zorder=1)
    # imshow with log axis: draw on a twin linear-log mapping instead
    im.remove()
    # draw as vertical strips
    for i in range(len(xs) - 1):
        col = img[0, i, :3]
        ax.fill_between([xs[i], xs[i + 1]], 0, np.where(ok[i], np.nan_to_num(edge[i], nan=0), 0),
                        color=col, lw=0, zorder=1)
    # hatch over the uncertain band
    xb = xs[(xs >= N_lo) & (xs <= N_far + 0.6 * (N_far - N_lo) + 3)]
    ax.fill_between(xb, 0, np.nan_to_num(np.interp(np.log(xb), np.log(xs), np.where(ok, edge, 0)), nan=0),
                    facecolor='none', edgecolor=BLUE, hatch='///', lw=0, alpha=0.35, zorder=2)
    # red above the edge
    ax.fill_between(xs, np.where(ok, edge, 0), ymax, color='#f3d2cf', lw=0, zorder=1)
    ax.fill_between(xs[~ok], 0, ymax, color='#f3d2cf', lw=0, zorder=1)
    # the quantum edge: line (firm) and the shift from the two omitted terms
    ax.plot(xs[ok], edge[ok], color='k', lw=2.0, zorder=5)
    ax.fill_between(xs[ok], edge_lo[ok], edge[ok], color='k', alpha=0.28, lw=0, zorder=4)
    ax.plot(xs[ok], edge_lo[ok], color='k', lw=0.8, ls=':', zorder=5)
    # measured points
    L_N = L['Ns_measured']
    ax.scatter(L_N, [q['meta']['mu_tilde_0']] * len(L_N), marker='s', s=34, color='k', zorder=6)
    ax.text(2.0, 7.2, 'simulated ($\\tilde\\mu=8$)', fontsize=7.5, ha='left', va='top')
    # labels
    ax.text(2.55, 3.2, 'exact\ndiag.', fontsize=7.5, color='#0f3d24', ha='center', va='center')
    ax.text(math.sqrt(N_ED * N_lo) * 0.95, 4.0, 'tensor networks\nreach here', fontsize=8.5, color='#1d5f3a',
            ha='center', va='center')
    ax.text(math.sqrt(N_lo * N_far) * 1.25, 11.0,
            'tensor-network wall:\nsomewhere in this band,\nmay lie further right', fontsize=8.5,
            color='#0b3d75', ha='center', va='center')
    ax.text(45, 6.0, 'quantum window as drawn\nby the bounds\n(not established, Sec. 5)', fontsize=9, color='#0b3d75',
            ha='center', va='center', fontweight='bold')
    ax.text(2.0, 24.0, 'too slow for any route:\nthe movie lasts $\\propto e^{2\\tilde\\mu/3}$',
            fontsize=8.5, color='#7a1f17', ha='left', va='center')
    ax.annotate('qubitization edge $\\tilde\\mu_{\\rm qub}(N)$\n(line: leading order; shaded:\nomitted terms restored)',
                xy=(24, np.interp(math.log(24), np.log(Nq), e_lead)), xytext=(25, 22.4), fontsize=7.5,
                arrowprops=dict(arrowstyle='-', lw=0.6), ha='left')
    ax.annotate('', xy=(N_far * 1.9, 14.5), xytext=(N_far * 1.15, 14.5),
                arrowprops=dict(arrowstyle='->', color='#0b3d75', lw=1.0))
    ax.axvline(NQ, color='k', lw=0.6, ls='--', zorder=3)
    ax.text(NQ * 1.04, 1.0, f'{NQ}\nqubit\nlimit', fontsize=7, va='bottom', ha='left')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, ymax)
    ax.set_xticks([2, 3, 4, 5, 8, 10, 16, 24, 40, 88])
    ax.set_xticklabels(['2', '3', '4', '5', '8', '10', '16', '24', '40', '88'])
    ax.minorticks_off()
    ax.set_xlabel('number of particles $N$')
    ax.set_ylabel('trap-units mass $\\tilde\\mu=3S_0/2$')
    save(fig, 'fig_regime')


# ================================================================ model schematic
def fig_model():
    from matplotlib.patches import Ellipse
    import matplotlib.pyplot as plt
    import numpy as np
    m = cfg['model']
    A, wy2, eps, sig, cap = m['A'], m['omega_y'], m['epsilon'], m['sigma'], m['V_cap']
    
    WB = dict(facecolor="white", alpha=0.9, edgecolor="none", pad=2)
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 6.2))
    x = np.linspace(-2.2, 2.2, 400)
    y = np.linspace(-1.6, 1.6, 300)
    X, Y = np.meshgrid(x, y)
    U = A * (X**2 - 1) ** 2 + 0.5 * wy2 * Y**2
    # valleys (U=0) dark, barrier (U=8) clipped bright: extend="max" paints it
    cf = ax[0].contourf(
        X, Y, U, levels=np.linspace(0, 3.0, 31), cmap="viridis", extend="max"
    )
    cb = fig.colorbar(cf, ax=ax[0], pad=0.02)
    cb.set_label("trap energy $U(x,y)$", fontsize=16)
    cb.ax.tick_params(labelsize=13)
    # valley / barrier labels
    ax[0].text(
        -1.0,
        1.32,
        "left valley",
        ha="center",
        fontsize=17,
        weight="bold",
        bbox=WB,
        zorder=7,
    )
    ax[0].text(
        1.0,
        1.32,
        "right valley",
        ha="center",
        fontsize=17,
        weight="bold",
        bbox=WB,
        zorder=7,
    )
    ax[0].text(
        1.02,
        -1.38,
        "barrier ($U=8$)",
        ha="center",
        fontsize=16,
        weight="bold",
        bbox=WB,
        zorder=7,
    )
    # isomer (1,1): one solid dot per valley
    ax[0].plot([-1.0], [0.35], "o", ms=26, color="#ff5555", mec="k", mew=1.8, zorder=6)
    ax[0].plot([1.0], [0.35], "o", ms=26, color="#5599ff", mec="k", mew=1.8, zorder=6)
    # isomer (2,0): bound pair (spaced by r_min ~ 0.2) ringed by a dashed ellipse
    ax[0].plot([1.0], [-0.55], "o", ms=26, color="#ff5555", mec="k", mew=1.8, zorder=6)
    ax[0].plot([1.0], [-0.82], "o", ms=26, color="#5599ff", mec="k", mew=1.8, zorder=6)
    ax[0].add_patch(
        Ellipse(
            (1.0, -0.685), 0.62, 0.85, fill=False, ls="--", lw=2.5, ec="k", zorder=6
        )
    )
    # tunneling: thick dashed double arrow through the barrier
    ax[0].annotate(
        "",
        xy=(-0.80, 0.35),
        xytext=(0.80, 0.35),
        arrowprops=dict(arrowstyle="<->", color="k", lw=4.5, ls=(0, (5, 3))),
        zorder=6,
    )
    ax[0].text(
        0.0,
        0.02,
        "tunneling",
        ha="center",
        fontsize=16,
        weight="bold",
        bbox=WB,
        zorder=7,
    )
    ax[0].annotate(
        "isomer $(1,1)$:\none particle per valley",
        xy=(-1.06, 0.48),
        xytext=(-2.14, 0.86),
        fontsize=15.5,
        weight="bold",
        bbox=WB,
        zorder=7,
        arrowprops=dict(arrowstyle="->", lw=2, color="k"),
    )
    ax[0].annotate(
        "isomer $(2,0)$: bound\npair in one valley",
        xy=(0.72, -0.72),
        xytext=(-2.16, -1.50),
        fontsize=15.5,
        weight="bold",
        bbox=WB,
        zorder=7,
        arrowprops=dict(arrowstyle="->", lw=2, color="k"),
    )
    ax[0].set_xlabel("$x$")
    ax[0].set_ylabel("$y$")
    ax[0].set_title("(a) the two-valley trap")
    r = np.linspace(0.135, 0.9, 800)
    V = 4 * eps * ((sig / r) ** 12 - (sig / r) ** 6)
    ax[1].plot(r, np.minimum(V, cap), color="k", lw=3.5)
    ax[1].axhline(0, color="gray", lw=1.2)
    rmin = sig * 2 ** (1 / 6)
    ax[1].plot([rmin], [-eps], "o", ms=15, color="#ff5555", mec="k", zorder=5)
    ax[1].annotate(
        "sticky minimum\n(weak attraction)", (rmin + 0.07, -1.9), fontsize=16
    )
    # hard-core label sits to the right of the repulsive wall; the arrow tip is
    # placed ON the curve (r where W(r) = y_tip), so it cannot miss the line.
    y_tip = 2.4
    r_tip = sig * (0.5 * (1 + np.sqrt(1 + (y_tip / eps)))) ** (
        -1.0 / 6
    )  # W=4e[(s/r)^12-(s/r)^6]=y
    ax[1].annotate(
        "hard core:\nstrong repulsion\nwhen too close",
        xy=(r_tip + 0.002, y_tip),
        xytext=(0.23, 1.15),
        fontsize=16,
        va="center",
        arrowprops=dict(arrowstyle="->", color="k", lw=2),
    )
    ax[1].set_xlabel("particle separation $r$")
    ax[1].set_ylabel("pair energy $W(r)$")
    ax[1].set_ylim(-2.7, 4.2)
    ax[1].set_title("(b) Lennard-Jones pair interaction")
    plt.tight_layout()
    save(fig, 'fig_model')
# ============================================================================ signal
def resonance_graphs(g):
    """One graph record per N: correlated start, k=12, resonance tilt; the uniform run if there is one."""
    out = {}
    for r in sorted(g, key=lambda r: r['protocol'] != 'uniform', reverse=True):
        if r['k'] == 12 and r['role'] == 'primary' and tilt12(r['N']) is None:
            raise KeyError(f"graph record for N={r['N']} at k=12 but config.toml has no [tilts.N{r['N']}] k12")
        if r['k'] == 12 and r['role'] == 'primary' and abs(r['c'] - tilt12(r['N'])) < 1e-9:
            out[r['N']] = r
    return out


def fig_signal():
    data = {N: series(N) for N in range(2, 13)}               # N = 2..5, and every larger N that has a k = 12 run at its tilt
    data = {N: v for N, v in data.items() if N <= 5 or v[0] is not None}
    if all(v[0] is None for v in data.values()):
        return placeholder('fig_signal', 'data/runs/N*_k12_int_*.npz')
    fig, (a, b) = plt.subplots(2, 1, figsize=(6.6, 4.6), sharex=True)
    for N, (run, uni) in data.items():
        if run is None:
            continue
        t, O = run['t'], run['O']
        ls = '-' if uni else '--'
        a.plot(t, O / O[0], color=cl_of(N), lw=1.5, ls=ls, label=f'$N={N}$' + ('' if uni else '$^\\dagger$'))
        b.plot(t, analysis.S_of(run, range(N // 2)), color=cl_of(N), lw=1.5, ls=ls)
    a.set_ylabel(r'$\langle\hat O\rangle(t)\,/\,\langle\hat O\rangle(0)$')
    a.legend(ncol=max(4, len(data)), loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize=8)
    a.set_ylim(0, 1.08)
    if any(not v[1] for v in data.values() if v[0] is not None):
        b.text(0.99, 0.04, 'dashed, $\\dagger$: earlier protocol (shorter window)', transform=b.transAxes, ha='right', fontsize=7, color=GREY)
    b.set_ylabel('central-cut entropy (bits)')
    b.set_xlabel('time $t$ (model units, $\\hbar=1$)')
    save(fig, 'fig_signal')


# ============================================================================ Venn / I-diagram
def ribbon(ax, p, q, width, color, alpha=0.9, z=3):
    p, q = np.array(p), np.array(q)
    d = q - p
    n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-12)
    h = width / 2
    poly = Polygon([p + n * h, q + n * h, q - n * h, p - n * h], closed=True, fc=color, ec='none',
                   alpha=alpha, zorder=z)
    ax.add_patch(poly)


RING = {2: 1.0, 3: 1.0, 4: 1.0, 5: 1.05, 6: 1.15, 8: 1.4}
DISK_K = 0.30          # disk radius = DISK_K * sqrt(shared entropy): the same area scale in every panel


def draw_sharing(ax, N, w, title, model=False, label=None):
    """Set diagram of the correlation graph. Particle i is a disk of area proportional to the
    entropy sum_j w_ij / 2 that it shares; the disk is cut into one petal per partner j, of angle
    proportional to w_ij, and the petals of i and j are joined by a ribbon of width ~ w_ij. A cut
    A|B (dashed line) crosses the ribbons drawn in orange (and colours the petals at their ends);
    its entropy is the summed area of those ribbons. The area scale is common to all panels."""
    rr = RING.get(N, 1.0)
    ang0 = np.pi / 2
    pos = [(rr * np.cos(ang0 + 2 * np.pi * i / N + np.pi / N * (N % 2 == 0)),
            rr * np.sin(ang0 + 2 * np.pi * i / N + np.pi / N * (N % 2 == 0))) for i in range(N)]
    b0 = N // 2
    A = set(range(b0))
    wmat = np.array(w)
    crossing = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            cross = (i in A) != (j in A)
            if cross:
                crossing += wmat[i, j] / 2
            ribbon(ax, pos[i], pos[j], width=0.10 * wmat[i, j],
                   color=ORANGE if cross else '#bbbbbb', alpha=0.95 if cross else 0.8, z=2)
    for i in range(N):
        Si = np.sum(wmat[i]) / 2
        R = DISK_K * math.sqrt(Si)
        partners = [j for j in range(N) if j != i]
        ang = np.degrees(np.arctan2([pos[j][1] - pos[i][1] for j in partners],
                                    [pos[j][0] - pos[i][0] for j in partners]))
        order = np.argsort(ang)
        wd = np.array([wmat[i, partners[o]] for o in order])
        wd = wd / wd.sum() * 360.0
        start = 0.0
        for o, dd in zip(order, wd):
            j = partners[o]
            cross = (i in A) != (j in A)
            ax.add_patch(Wedge(pos[i], R, start, start + dd, fc=ORANGE if cross else '#9db7d5',
                               ec='w', lw=0.8, alpha=0.95 if cross else 0.85, zorder=4))
            start += dd
        ax.add_patch(Circle(pos[i], R, fc='none', ec='k', lw=1.0, zorder=5, ls='--' if model else '-'))
        ax.text(pos[i][0], pos[i][1], str(i + 1), ha='center', va='center', fontsize=7, zorder=6,
                color='k', fontweight='bold')
    cA = np.mean([pos[i] for i in A], axis=0)
    cB = np.mean([pos[i] for i in range(N) if i not in A], axis=0)
    mid, nrm = (cA + cB) / 2, (cB - cA) / np.linalg.norm(cB - cA)
    tvec = np.array([-nrm[1], nrm[0]])
    ax.plot(*np.array([mid - tvec * 1.9, mid + tvec * 1.9]).T, color='k', lw=0.9, ls=(0, (4, 3)), zorder=7)
    ax.set_xlim(-1.95, 1.95)
    ax.set_ylim(-1.95, 2.0)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=9)
    ax.text(0, -1.95, label if label else f'cut $S_A\\approx{crossing:.2f}$ bits', ha='center', va='top', fontsize=8)
    return crossing


def fig_venn():
    g = jload('graph.json')
    L = jload('larger_N.json')
    if g is None or L is None:
        return placeholder('fig_venn', 'data/derived/graph.npz (from data/runs/*.npz)')
    Nv = sorted({2, 3, 4, 5} | set(resonance_graphs(g)))        # measured panels: N = 2..5 and every larger N with a run
    fig, axs = plt.subplots(1, len(Nv) + 1, figsize=(11.0 * (len(Nv) + 1) / 5, 2.9))
    shown = 0
    for N, ax in zip(Nv, axs[:-1]):
        r = resonance_graphs(g).get(N)
        if r is None:
            ax.axis('off')
            ax.text(0.5, 0.5, f'PLACEHOLDER\nN={N}: no run', ha='center', va='center', color=RED,
                    transform=ax.transAxes)
            continue
        draw_sharing(ax, N, r['w'], f'$N={N}$ (measured)' + ('' if r['protocol'] == 'uniform' else '$^\\dagger$'),
                     label=f"cut: ${r['W_graph']:.2f}$ (graph), ${r['S_nat']:.2f}$ (exact)")
    # N = 8 from the sharing law with the fitted S1
    N = 8
    Sinf, tau = L['fit']['S_inf'], L['fit']['tau']
    S1_8 = L['curves']['sat'][N - 2] / (N // 2 * (N - N // 2) / (N - 1))
    wmat = np.full((N, N), 2 * S1_8 / (N - 1))
    np.fill_diagonal(wmat, 0)
    draw_sharing(axs[-1], N, wmat, '$N=8$ (sharing law, no data)', model=True,
                 label=f'cut $\\approx{L["curves"]["sat"][N - 2]:.2f}$ bits (model)')
    save(fig, 'fig_venn')


# ============================================================================ entropy vs N and the graph
def fig_entropy():
    g, L, led = jload('graph.json'), jload('larger_N.json'), jload('ledger.json')
    if g is None or L is None or led is None:
        return placeholder('fig_entropy', 'data/derived/{graph,larger_N,ledger}.json')
    fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.2), gridspec_kw=dict(width_ratios=[1, 1, 1.25]),
                            constrained_layout=True)
    a, b, c = axs
    # (a) exact versus graph for every subset
    for N, r in sorted(resonance_graphs(g).items()):
        uni = r['protocol'] == 'uniform'
        a.scatter([s['graph'] for s in r['scatter']], [s['exact'] for s in r['scatter']], marker=mk_of(N),
                  s=22, facecolor=cl_of(N) if uni else 'none', edgecolor=cl_of(N), label=f'$N={N}$' + ('' if uni else '$^\\dagger$'),
                  zorder=3, alpha=0.9)
    top = max([3.2] + [x for r in resonance_graphs(g).values() for s in r['scatter'] for x in (s['graph'], s['exact'])])
    a.plot([0, top], [0, top], color=GREY, lw=0.8)
    a.set_xlabel('graph cut $C(A)$ (bits)')
    a.set_ylabel('exact $\\sigma(A)=\\max_t S_A$ (bits)')
    a.legend(fontsize=7.5, loc='upper left')
    a.set_title('(a) every subset $A$, $|A|\\leq N/2$', fontsize=9)
    # (b) edge weight vs N
    Ns, wm, wl, wh, s1 = [], [], [], [], []
    for N_, r in sorted(resonance_graphs(g).items()):
        Ns.append(r['N']); wm.append(r['w_mean']); wl.append(r['w_min']); wh.append(r['w_max']); s1.append(r['S1mean'])
    Ns = np.array(Ns)
    b.errorbar(Ns, wm, yerr=[np.array(wm) - np.array(wl), np.array(wh) - np.array(wm)], fmt='o', color='k',
               capsize=2.5, label='measured $w_{ij}$ (mean, range)', zorder=4)
    NN = np.arange(2, 13)
    S1fit = np.array([L['curves']['sat'][n - 2] / (n // 2 * (n - n // 2) / (n - 1)) for n in NN])
    b.plot(NN, 2 * S1fit / (NN - 1), color=BLUE, lw=1.6, label='$2\\bar S_1/(N-1)$ (sharing)')
    b.axhline(wm[0], color=RED, lw=1.2, ls='--', label='$N$-independent weight (pair law)')
    b.set_xlabel('$N$')
    b.set_ylabel('edge weight $w_{ij}$ (bits)')
    b.set_ylim(0, 2.9)
    b.legend(fontsize=7, loc='upper right')
    b.set_title('(b) each particle shares its entropy', fontsize=9)
    # (c) central cut vs N, with the model carried to larger N
    NNf = np.array(L['NN'])
    m = NNf <= 30
    c.plot(NNf[m], np.array(L['curves']['sat'])[m], color=BLUE, lw=1.8, label='sharing law, $\\bar S_1$ saturating')
    c.fill_between(NNf[m], np.array(L['curves']['opt'])[m], np.array(L['curves']['pess'])[m], color=BLUE, alpha=0.15,
                   lw=0, label='range of the two other cases')
    c.plot(NNf[m], np.array(L['pair_curve4'])[m], color=RED, lw=1.2, ls='--', label=f"pair law, fitted to $N\\leq{cfg['extrapolation']['pair_fit_max']}$")
    leg = ['none' if records.pick(led, n)['protocol'] != 'uniform' else 'k' for n in L['Ns_measured']]
    c.scatter(L['Ns_measured'], L['S_cen_measured'], facecolor=leg, edgecolor='k', s=26, zorder=5,
              label='measured (open: earlier protocol)' if 'none' in leg else 'measured')
    if 6 not in L['Ns_measured']:
        c.axvline(6, color=GREY, lw=0.8, ls=':')
        c.text(6.15, 0.25, '$N=6$:\nrun pending', fontsize=7, color=GREY, va='bottom')
    c.set_xlabel('$N$')
    c.set_ylabel('central-cut entropy $S_N$ (bits)')
    c.set_xlim(1.8, 30)
    c.set_ylim(0, 12)
    c.legend(fontsize=7, loc='lower right')
    c.set_title('(c) the law carried to larger $N$', fontsize=9)
    save(fig, 'fig_entropy')


# ============================================================================ walls
def fig_walls():
    L = jload('larger_N.json')
    if L is None:
        return placeholder('fig_walls', 'data/derived/larger_N.json')
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 3.1))
    NNf = np.array(L['NN'])
    m = NNf <= 40
    for nm, col, ls, lab in (('pess', RED, '-', 'sharing, $\\bar S_1$ keeps growing'),
                              ('sat', BLUE, '-', 'sharing, $\\bar S_1$ saturating (fit)'),
                              ('opt', GREEN, '-', f"sharing, $\\bar S_1$ saturated at $N={L['Ns_measured'][-1]}$")):
        a.plot(NNf[m], np.array(L['curves'][nm])[m], color=col, ls=ls, lw=1.6, label=lab)
    for dkey, ls, lab in (('d%d' % L['budgets']['d_grid'], '-', '$d=1200$ (grid)'),
                          ('d%d' % L['budgets']['d_mode'], ':', '$d=20$ (modes)')):
        a.plot(NNf[m], np.array(L['S_budget'][dkey])[m], color='k', ls=ls, lw=1.2, label='budget, ' + lab)
    a.plot(NNf[m], np.array(L['S_budget']['d%d' % L['budgets']['d_grid']])[m] - L['chi_shift'], color='k', ls='--', lw=0.9,
           label='budget, $d=1200$, $\\chi(10^{-3})$ charged')
    led_ = jload('ledger.json')
    leg = ['none' if led_ and records.pick(led_, n)['protocol'] != 'uniform' else 'k' for n in L['Ns_measured']]
    a.scatter(L['Ns_measured'], L['S_cen_measured'], facecolor=leg, edgecolor='k', s=24, zorder=5)
    a.set_xlabel('$N$')
    a.set_ylabel('bits')
    a.set_ylim(0, 22)
    a.legend(fontsize=6.8, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2)
    a.set_title('(a) entropy against the largest affordable entropy', fontsize=9)
    # (b) wall against asymptotic slope s for the linear law anchored at the last measured N
    import importlib
    th = importlib.import_module('ljtunnel.theory')
    Bd = L['budgets']
    Nref, Sref = L['Ns_measured'][-1], L['S_cen_measured'][-1]
    ss = np.linspace(0.15, 1.6, 120)
    for d, ls, lab in ((Bd['d_grid'], '-', 'grid, $d=1200$'), (Bd['d_mode'], '--', 'modes, $d=20$')):
        wl = [th.wall(th.law_linear(Sref, s_, Nref), d, Bd['Bmem'], Bd['Bflop'], Bd['sweeps'], N_from=Nref, nmax=400)
              for s_ in ss]
        b.plot(ss, wl, color=BLUE, ls=ls, lw=1.6, label=lab)
    sl = L['slope']['sat']
    b.axvspan(sl * 0.8, sl * 1.2, color=GREY, alpha=0.2, lw=0)
    b.text(sl * 1.25, 38, 'slope of the\nsharing law\n(saturating $\\bar S_1$)', fontsize=7.5, va='top')
    b.set_xlabel('entropy growth $s$ per added particle (bits)')
    b.set_ylabel('tensor-network wall (first failing $N$)')
    b.set_ylim(0, 45)
    b.legend(fontsize=7.5, loc='upper right')
    b.set_title('(b) the wall as a function of $s$', fontsize=9)
    save(fig, 'fig_walls')


# ============================================================================ appendix figures
def fig_mc():
    p = os.path.join(RAW, 'mc_gamma_ledger.txt')
    if not os.path.exists(p):
        return placeholder('fig_mc', 'data/studies/onebody_mc/mc_gamma_ledger.txt')
    M, G = [], []
    for ln in open(p):
        t = ln.split()
        if t and t[0] == 'ginf':
            M.append(int(t[1])); G.append(float(t[3]))
    M, G = np.array(M), np.array(G)
    fig, a = plt.subplots(figsize=(3.6, 2.7))
    a.loglog(M, G, 'o-', color=BLUE, ms=4)
    a.loglog(M, G[0] * (M / M[0]) ** 2, color=GREY, lw=0.8, ls='--', label='$M^2$')
    a.set_xlabel('grid points $M$ (one dimension)')
    a.set_ylabel('$\\Gamma_\\infty(M)$')
    a.legend(fontsize=8)
    save(fig, 'fig_mc')


def fig_kconv():
    led = jload('ledger.json')
    if led is None:
        return placeholder('fig_kconv', 'data/derived/ledger.npz')
    fig, a = plt.subplots(figsize=(4.4, 3.0))
    ymax, nser = 0.0, 0
    for N in range(2, 13):                                    # N = 2..5 and every larger N that has a record
        rs = [records.pick(led, N, k) for k in (12, 16, 20)]
        pts = [(k, r['S_cen'], r['protocol'] == 'uniform') for k, r in zip((12, 16, 20), rs) if r]
        if not pts:
            continue
        ymax, nser = max(ymax, max(p[1] for p in pts)), nser + 1
        a.plot([p[0] for p in pts], [p[1] for p in pts], color=cl_of(N), lw=1.2,
               ls='-' if all(p[2] for p in pts) else '--', label=f'$N={N}$' + ('' if all(p[2] for p in pts) else '$^\\dagger$'))
        for k_, v_, uni in pts:
            a.plot(k_, v_, marker='o', color=cl_of(N), mfc=cl_of(N) if uni else 'none', ls='none')
    a.set_xlabel('modes per particle $k$')
    a.set_ylabel('central-cut entropy (bits)')
    a.set_xticks([12, 16, 20])
    a.set_ylim(0.7, max(3.4, ymax + 0.5))
    a.legend(fontsize=7.5, ncol=4 if nser <= 4 else 3, loc='upper center')
    save(fig, 'fig_kconv')



# ============================================================================ the tensor-network measurement
def fig_tn():
    p = os.path.join(DER, 'tn_tables.npz')
    if not os.path.exists(p):
        return placeholder('fig_tn', 'data/derived/tn_tables.npz (run_mps_tables)')
    T = store.load_object(p)['tables']
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.7), constrained_layout=True)
    # (a) MPS against exact
    a = ax[0]
    for case, col, lab in (('N4_k12', BLUE, '$N=4$, $k=12$'), ('N5_k6', ORANGE, '$N=5$, $k=6$')):
        rows = [r for r in T['mps_series'] if r['case'] == case]
        t = [r['t'] for r in rows]
        a.plot(t, [r['S_cen_exact'] for r in rows], '-', color=col, lw=2.2, alpha=0.45, label=lab + ', exact')
        a.plot(t, [r['S_cen_mps'] for r in rows], '--', color=col, lw=1.3, label=lab + ', MPS')
    a.set_xlabel('time $t$')
    a.set_ylabel('central-bond entropy (bits)')
    a.set_title('(a) MPS against exact', fontsize=9)
    a.legend(loc='center right', fontsize=6.5)
    # (b) rank against 2^S
    b = ax[1]
    for k, col, mk in ((12, BLUE, 'o'), (16, ORANGE, 's'), (20, GREEN, '^')):
        rr = sorted([r for r in T['rank_vs_entropy'] if r['k'] == k and r['start'] == 'interacting' and r['T'] == 200 and r['protocol'] == 'uniform'], key=lambda r: r['N'])
        b.plot([r['N'] for r in rr], [r['ratio3'] for r in rr], mk + '-', color=col, ms=4.5, label=f'$k={k}$')
    b.axhline(1, color=GREY, lw=1, ls='--')
    b.text(2.05, 1.7, r'cost model: $\chi=2^{S}$', color=GREY, fontsize=7)
    b.set_xlabel('particles $N$')
    b.set_ylabel(r'$\chi(10^{-3})\,/\,2^{S_{\rm cen}}$')
    b.set_xticks(sorted({r['N'] for k in (12, 16, 20) for r in T['rank_vs_entropy'] if r['k'] == k and r['start'] == 'interacting' and r['T'] == 200 and r['protocol'] == 'uniform'}))
    b.set_ylim(0, None)
    b.set_title('(b) rank against the model', fontsize=9)
    b.legend(loc='upper left', fontsize=7)
    # (c) cost of a step
    c = ax[2]
    rr = sorted([r for r in T['step_cost'] if r['k'] == 12], key=lambda r: r['chi_cap'])
    c.plot([r['chi_cap'] for r in rr], [r['sec_per_step'] for r in rr], 'o-', color=BLUE, ms=4.5, label='measured, one TDVP step')
    c.plot([r['chi_cap'] for r in rr], [r['model_sec_at_zgemm_rate'] for r in rr], 's--', color=ORANGE, ms=4, label='counting model')
    c.set_yscale('log')
    c.set_xlabel(r'bond-dimension cap $\chi$')
    c.set_ylabel('seconds per step')
    c.set_title('(c) $N=4$, $k=12$', fontsize=9)
    c.legend(loc='center right', fontsize=6.5)
    save(fig, 'fig_tn')


# ============================================================================ the follow-up figures (data/followup)
sys.path.insert(0, HERE)
import followup_data as fd  # noqa: E402

NCOL = {2: BLUE, 3: GREEN, 4: ORANGE, 5: RED, 6: PURPLE}


def fig_spectrum():
    """Weight of the correlated start on each eigenlevel, k = 20, N = 2..5, against the distance from the lower level of the pair in units of J."""
    S = fd.stage1()
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 4.3), sharex=True, sharey=True, constrained_layout=True)
    for a, N in zip(ax.ravel(), (2, 3, 4, 5)):
        r = S[(N, 20)]
        x = (r['E'] - r['E'][r['a']]) / r['J']
        w = r['w']
        a.axvspan(-1, 3, color='#e6e6e6', lw=0, zorder=0)          # within J of either level of the pair (they are 2J apart)
        a.axvspan(-3, -1, color='#f6e3b4', lw=0, zorder=0)          # 1-3 J
        a.axvspan(3, 5, color='#f6e3b4', lw=0, zorder=0)
        pair = np.zeros(len(w), bool)
        pair[[r['a'], r['b']]] = True
        m = (~pair) & (w > 1e-5)
        a.vlines(x[m], 1e-5, w[m], color=GREY, lw=1.0)
        a.vlines(x[pair], 1e-5, w[pair], color=NCOL[N], lw=2.6)
        a.set_yscale('log')
        a.set_ylim(1e-4, 1.3)
        a.set_xlim(-3, 16)
        a.set_title(f"$N={N}$:  pair {r['B2']:.3f},  rest {1 - r['B2']:.3f}", fontsize=8.5, loc='left')
    for a in ax[1]:
        a.set_xlabel(r'level $-$ lower level of the pair, in units of $J$')
    for a in ax[:, 0]:
        a.set_ylabel('weight of the start')
    save(fig, 'fig_spectrum')


def fig_dressing():
    S = fd.stage1()
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True, gridspec_kw={'width_ratios': [1.15, 1]})
    a = ax[0]
    for N in (2, 3, 4, 5):
        ks = sorted(k for (n, k) in S if n == N and k >= 12)
        a.plot(ks, [S[(N, k)]['B2'] for k in ks], mk_of(N) + '-', color=NCOL[N], ms=4.5, lw=1.3, label=f'$N={N}$')
        if (N, 24) in S and S[(N, 24)]['krylov']:
            a.plot([24], [S[(N, 24)]['B2']], mk_of(N), mfc='white', color=NCOL[N], ms=5.5)
    a.set_xlabel('modes per particle $k$')
    a.set_ylabel('weight of the start on its two heaviest levels')
    a.set_title('(a) basis convergence', fontsize=9)
    a.set_xticks([12, 16, 20, 24, 28, 32])
    a.set_ylim(0.6, 1.0)
    a.legend(loc='lower right', ncol=2, fontsize=7.5)
    b = ax[1]
    cases = [(2, 20), (3, 20), (4, 20), (4, 32), (5, 20), (5, 24)]
    cols = (('near', '#999999', r'within $J$ of the pair'), ('mid', ORANGE, r'$1$--$3J$'), ('far', SKY, r'beyond $3J$'))
    for i, (N, k) in enumerate(cases):
        r = S[(N, k)]
        bot = 0.0
        for key, col, lab in cols:
            b.bar(i, r[key], bottom=bot, color=col, edgecolor='white', lw=0.6, label=lab if i == 0 else None)
            bot += r[key]
        b.bar(i, max(1 - r['B2'] - bot, 0), bottom=bot, color='white', edgecolor=GREY, hatch='////', lw=0.6, label='not resolved' if i == 0 else None)
    b.set_xticks(range(len(cases)))
    b.set_xticklabels([f'{N}\n{k}' for N, k in cases], fontsize=8)
    b.set_xlabel('$N$ (upper row), $k$ (lower row)')
    b.set_ylabel('weight outside the pair')
    b.set_title('(b) where the rest sits', fontsize=9)
    b.legend(loc='upper left', fontsize=7)
    save(fig, 'fig_dressing')


def fig_tilt():
    S = fd.stage1()
    ctop = fd.c_top()
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True, gridspec_kw={'width_ratios': [1.3, 1]})
    a = ax[0]
    Nx = np.arange(2, 13)
    for k, col in ((12, BLUE), (16, GREEN), (20, ORANGE)):
        pts = [(N, S[(N, k)]['c']) for N in range(2, 7) if (N, k) in S]
        x, y = zip(*pts)
        a.plot(x, y, 'o', color=col, ms=4.5, label=f'$k={k}$')
        if k in (12, 20):
            sl, ic = np.polyfit(x, y, 1)
            a.plot(Nx, sl * Nx + ic, '--', color=col, lw=1.0)
    a.axhline(ctop, color=RED, lw=1.2)
    a.text(12.2, ctop + 0.15, 'barrier top reached by the upper-valley level of one particle', color=RED, fontsize=6.8, ha='right')
    a.set_xlabel('particles $N$')
    a.set_ylabel(r'resonant tilt $c^\ast$')
    a.set_title('(a) the tilt grows linearly', fontsize=9)
    a.set_xlim(1.7, 12.3)
    a.set_ylim(0, ctop + 1.2)
    a.legend(loc='upper left', bbox_to_anchor=(0.0, 0.8), fontsize=7.5)
    b = ax[1]
    U = fd.untilted()
    for i, N in enumerate((3, 4)):
        E = U[(N, True)][:44]
        b.vlines(E, i + 0.1, i + 0.9, color=NCOL[N], lw=1.0)
    b.set_yticks([0.5, 1.5])
    b.set_yticklabels(['$N=3$', '$N=4$'])
    b.set_xlabel(r'level $-$ ground level ($c=0$, $C_0=0$)')
    b.set_title('(b) a dense band elsewhere', fontsize=9)
    b.set_xlim(-0.05, 2.5)
    b.spines['left'].set_visible(False)
    b.tick_params(axis='y', length=0)
    save(fig, 'fig_tilt')


def fig_parity():
    LV = fd.levels()
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.9), sharey=True, constrained_layout=True)
    for a, key, title in zip(ax, ((3, 12), (3, 16), (4, 12)), ('$N=3$, $k=12$', '$N=3$, $k=16$', '$N=4$, $k=12$')):
        d = LV[key]
        E, w, R, n = d['E'], d['weight'], d['R'], d['n']
        ev = R > 0
        heavy = sorted(np.argsort(-w)[:2])
        a.vlines(E[ev], 1e-5, np.maximum(w[ev], 1e-5), color=ORANGE, lw=1.4, label='even under $y\\to-y$')
        a.plot(E[~ev], np.full((~ev).sum(), 1.5e-5), '|', color=GREY, ms=7, label='odd (weight $<10^{-4}$)')
        a.vlines(E[heavy], 1e-5, w[heavy], color=BLUE, lw=3, label='the pair')
        q = n.sum(1)
        for lab in range(1, key[0]):
            c = [i for i in range(len(E)) if ev[i] and n[i, lab] > 0.5]
            if c:
                a.plot(E[c[0]], 0.3, 'v', color=GREEN, ms=5, label='first level of a moved label' if lab == 1 else None)
        q2 = [i for i in range(len(E)) if ev[i] and q[i] >= 1.5]
        if q2:
            a.axvline(E[q2[0]], color=RED, lw=1.0, ls='--', label='two particles right')
        a.set_yscale('log')
        a.set_ylim(1e-5, 1.2)
        a.set_xlabel('level $-$ ground level')
        a.set_title(title, fontsize=9)
        if key == (4, 12):
            a.text(E.max() * 0.98, 0.5, 'none up to\n%.2f' % E.max(), ha='right', va='top', color=RED, fontsize=7)
    ax[0].set_ylabel('weight of the start')
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=5, fontsize=7, bbox_to_anchor=(0.5, -0.1))
    save(fig, 'fig_parity')


def fig_cosines():
    rows = fd.cosines()
    top = {}
    for r in rows:
        if r['N'] not in top or r['k'] > top[r['N']]['k']:
            top[r['N']] = r
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True)
    a = ax[0]
    for N in sorted(top):
        r = top[N]
        a.plot([1, 2, 3, 4], r['resid'], mk_of(N) + '-', color=NCOL[N], ms=4.5, label=f"$N={N}$ ($k={r['k']}$)")
    a.axhline(0.05, color=GREY, lw=0.9, ls='--')
    a.set_yscale('log')
    a.set_xticks([1, 2, 3, 4])
    a.set_xlabel('number of cosines $M$')
    a.set_ylabel('residual / standard deviation of the signal')
    a.set_title('(a) residual of an $M$-cosine fit', fontsize=9)
    h_, l_ = a.get_legend_handles_labels()
    fig.legend(h_, l_, loc='lower center', ncol=5, fontsize=6.8, bbox_to_anchor=(0.5, -0.09))
    b = ax[1]
    Ns = sorted(top)
    M = [top[N]['M'][2] for N in Ns]
    for N, m in zip(Ns, M):
        if m is None:
            b.annotate('', xy=(N, 8.8), xytext=(N, 6.2), arrowprops=dict(arrowstyle='->', color=NCOL[N]))
            b.plot([N], [6.2], mk_of(N), mfc='white', color=NCOL[N], ms=5.5)
        else:
            b.plot([N], [m], mk_of(N), color=NCOL[N], ms=5.5)
    b.set_xlabel('particles $N$')
    b.set_ylabel('cosines for a 5% residual')
    b.set_xticks(Ns)
    b.set_ylim(0, 10)
    b.set_title('(b) number of cosines needed', fontsize=9)
    b.text(6, 5.3, 'beyond the\nfitted range', ha='center', va='top', fontsize=6.8, color=GREY)
    save(fig, 'fig_cosines')


def _numval(key):
    """A value written by make_numbers.py (numbers.tex): run make_numbers.py first."""
    import re
    for line in open(os.path.join(ROOT, 'numbers.tex')):
        m = re.match(r'\\defnum\{' + re.escape(key) + r'\}\{([^}]*)\}', line)
        if m:
            return float(m.group(1))
    raise KeyError(key)


def fig_control():
    """Central-bond entropy with and without the ingredients of the tunneling: how much of it is already there before anything moves."""
    fig, ax = plt.subplots(figsize=(4.6, 2.7), constrained_layout=True)
    groups = [('$N=3$', [('product start', _numval('prod_N3'), GREY), ('correlated start', _numval('corr_N3'), BLUE)]),
              ('$N=4$', [('product start', _numval('prod_N4'), GREY), ('correlated start', _numval('corr_N4'), BLUE)]),
              ('$N=5$', [('off resonance (no transfer)', _numval('off_Snat'), GREEN), ('on resonance', _numval('Snat_N5'), BLUE)])]
    x0 = 0
    seen = set()
    ticks = []
    for name, bars in groups:
        xs = []
        for j, (lab, val, col) in enumerate(bars):
            ax.bar(x0 + j * 0.85, val, width=0.78, color=col, label=lab if lab not in seen else None)
            seen.add(lab)
            ax.text(x0 + j * 0.85, val + 0.06, f'{val:.2f}', ha='center', fontsize=7)
            xs.append(x0 + j * 0.85)
        ticks.append((np.mean(xs), name))
        x0 += 2.2
    ax.set_xticks([t for t, _ in ticks])
    ax.set_xticklabels([n for _, n in ticks])
    ax.set_ylabel('central-bond entropy (bits)')
    ax.set_ylim(0, 6.6)
    ax.legend(loc='upper left', fontsize=7)
    save(fig, 'fig_control')


ALL = dict(regime=fig_regime, model=fig_model, signal=fig_signal, venn=fig_venn, entropy=fig_entropy,
           walls=fig_walls, mc=fig_mc, kconv=fig_kconv, tn=fig_tn,
           spectrum=fig_spectrum, dressing=fig_dressing, tilt=fig_tilt, parity=fig_parity, cosines=fig_cosines, control=fig_control)

if __name__ == '__main__':
    todo = sys.argv[1:] or list(ALL)
    for k in todo:
        ALL[k]()
