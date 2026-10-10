"""ljtunnel.config -- load the single configuration file (stdlib tomllib)."""
import os
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path=None):
    """Read config.toml (default: the one at the repository root)."""
    path = path or os.path.join(ROOT, 'config.toml')
    with open(path, 'rb') as f:
        cfg = tomllib.load(f)
    cfg['_path'] = os.path.abspath(path)
    return cfg


def protocol(cfg, T=None, dt_snap=None, **over):
    """The measurement protocol, with optional overrides.

    The returned dict carries 'kind': 'uniform' when T and dt_snap equal the config values,
    'custom' otherwise.  The analysis only mixes runs of one kind.
    """
    p = dict(cfg['protocol'])
    p.update({k: v for k, v in over.items() if v is not None})
    base_T, base_dt = p['T'], p['dt_snap']
    if T is not None:
        p['T'] = float(T)
    if dt_snap is not None:
        p['dt_snap'] = float(dt_snap)
    p['kind'] = 'uniform' if (p['T'] == base_T and p['dt_snap'] == base_dt) else 'custom'
    p['nsnap'] = int(round(p['T'] / p['dt_snap'])) + 1
    return p
