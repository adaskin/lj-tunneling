"""ljtunnel.store -- every simulation result is one .npz file with a JSON metadata record.

Arrays are plain numpy arrays; metadata (parameters, protocol, resonance, start diagnostics,
provenance) is stored as a JSON string under the key 'meta_json'.  Nothing is pickled, so the
files load anywhere:

    run = store.load_run('data/runs/N4_k12_int_c2.260_T200_dt2.npz')
    run['t'], run['S'], run['meta']['c']
"""
import glob
import json
import os
import time

import numpy as np


def run_name(meta):
    tag = {'interacting': 'int', 'product': 'prod'}.get(meta['start'], meta['start'])
    return (f"N{meta['N']}_k{meta['k']}_{tag}_c{meta['c']:.3f}_"
            f"T{meta['protocol']['T']:g}_dt{meta['protocol']['dt_snap']:g}")


def save_run(path, arrays, meta):
    meta = dict(meta)
    meta.setdefault('saved', time.strftime('%Y-%m-%d %H:%M:%S'))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, meta_json=np.array(json.dumps(meta, default=_jsonable)), **arrays)
    return path


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def load_run(path):
    """Dict of arrays plus 'meta' (parsed) and 'path'."""
    with np.load(path, allow_pickle=False) as d:
        out = {k: d[k] for k in d.files if k != 'meta_json'}
        out['meta'] = json.loads(str(d['meta_json']))
    out['path'] = path
    return out


def list_runs(directory, pattern='*.npz'):
    return sorted(glob.glob(os.path.join(directory, pattern)))


def save_result(path, meta=None, **arrays):
    """Save a non-simulation result (walls, quantum costs, MC table, ...) the same way."""
    return save_run(path, {k: np.asarray(v) for k, v in arrays.items()}, meta or {})


def save_object(path, obj):
    """Save a nested dict (lists, numbers, strings) as an array-less .npz: the dict lives in 'meta_json'."""
    return save_run(path, {}, obj)


def load_object(path):
    """Inverse of save_object."""
    return load_run(path)['meta']
