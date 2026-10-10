"""ljtunnel.records -- turn the saved runs into the record tables the paper reads.

    ledger   one record per (run, window): the quantities of the entropy tables (analysis.summarize)
    graph    one record per correlated-start run: exact cut entropies against the graph estimate
             (analysis.graph_record)

Every record says which protocol it comes from ('protocol'): 'uniform' is the protocol of config.toml (one T, one snapshot
spacing for every N), read from data/runs/.  The code of the earlier protocol (windows and spacings that depended on N) and its
converted results were removed from the repository once the uniform runs covered every point (git history, commit 497189c);
the field stays so that a run made with another protocol would still be told apart.

The off-resonance control (data/controls/) enters the graph table only (never the ledger), so that `pick` can never return
a run at the wrong tilt.
"""
import os

from . import analysis, store


def _start_label(meta):
    return 'correlated' if meta['start'] == 'interacting' else 'product'


def theta_of(meta):
    """Weight of the barrier column x = 0 in Theta(x) that a run was made with ([model] theta_barrier; runs made before the
    key existed counted the column as left: 0)."""
    return float(meta.get('model', {}).get('theta_barrier', 0.0))


def check_convention(paths, theta, what='runs'):
    """Raise if any file in `paths` was made with another Theta(x = 0) convention than `theta`: a rerun that leaves old files
    beside the new ones would otherwise be mixed silently."""
    bad = [os.path.basename(p) for p in paths if abs(theta_of(store.load_run(p)['meta']) - theta) > 1e-12]
    if bad:
        raise ValueError(f"{len(bad)} of the {what} were made with another [model] theta_barrier than the config ({theta:g}): "
                         f"{', '.join(bad[:6])}{' ...' if len(bad) > 6 else ''}.  Move them away (rerun_all.sh archives them) "
                         f"or set theta_barrier in config.toml to the value they were made with.")


def collect(runs_dir, T_win=None, exhaustive_max_N=8, theta_barrier=None, controls_dir=None):
    """(ledger, graph): lists of records.  T_win cuts every run to the same window (default: all of it).  theta_barrier (the config value) makes the call refuse
    runs of another Theta(x = 0) convention; controls_dir holds the off-resonance control (graph table only)."""
    ledger, graph = [], []
    paths = store.list_runs(runs_dir)
    if theta_barrier is not None:
        check_convention(paths, theta_barrier, 'runs in ' + os.path.basename(os.path.normpath(runs_dir)))
    for p in paths:
        run = store.load_run(p)
        m = run['meta']
        if m['protocol']['kind'] != 'uniform':
            continue
        s = analysis.summarize(run, T_win)
        s.update(role='primary', file=os.path.basename(p))
        ledger.append(s)
        if m['start'] == 'interacting':
            g = analysis.graph_record(run, T_win, exhaustive_max_N=exhaustive_max_N)
            g.update(role='primary', start='correlated', file=os.path.basename(p))
            graph.append(g)
    cpaths = store.list_runs(controls_dir) if controls_dir else []
    if theta_barrier is not None:
        check_convention(cpaths, theta_barrier, 'control runs')
    for p in cpaths:
        run = store.load_run(p)
        if run['meta']['protocol']['kind'] != 'uniform' or run['meta']['start'] != 'interacting':
            continue
        g = analysis.graph_record(run, T_win, exhaustive_max_N=exhaustive_max_N)
        g.update(role='primary', start='correlated', file=os.path.basename(p), control=True)
        graph.append(g)
    return ledger, graph


def pick(ledger, N, k=12, start='correlated', role='primary'):
    """The record the paper uses for (N, k, start): the uniform one if there is one."""
    out = [r for r in ledger if r['N'] == N and r['k'] == k and r['start'] == start and r['role'] == role]
    out.sort(key=lambda r: r['protocol'] != 'uniform')
    return out[0] if out else None


def window_scan(run, windows):
    """From ONE stored series: the central bond, the best/worst balanced cut and the time of the central maximum
    for each window (no re-simulation)."""
    rows = []
    for Tw in windows:
        s, g = analysis.summarize(run, Tw), analysis.graph_record(run, Tw, exhaustive_max_N=0)
        rows.append(dict(T=float(Tw), S_cen=s['S_cen'], S_nat=g['S_nat'], S_perp=g['S_perp'], S_max=g['S_max'],
                         S_one=s['S_one'], tS_cen=s['tS_cen']))
    return rows


def spacing_scan(paths):
    """The same quantities for runs that differ only in the snapshot spacing (data/studies/dt_scan)."""
    rows = []
    for p in sorted(paths):
        run = store.load_run(p)
        s = analysis.summarize(run)
        rows.append(dict(N=s['N'], k=s['k'], dt=float(run['meta']['protocol']['dt_snap']), T=s['T'],
                         S_cen=s['S_cen'], S_nat=s['S_nat'], S_one=s['S_one']))
    return sorted(rows, key=lambda r: (r['N'], r['dt']))
