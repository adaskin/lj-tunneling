# data/runs/ -- the exact runs (uniform protocol)

`N{N}_k{k}_{int|prod}_c{tilt}_T200_dt1.npz`: `int` = correlated start, `prod` = product start.  19 files, one per row:

| N | k | start | tilt c (from `[tilts]`) |
|---|---|---|---|
| 2 | 12 | int, prod | 0.70 |
| 2 | 16, 20 | int | 0.70 |
| 3 | 12 | int, prod | 1.46 |
| 3 | 16, 20, 24, 28, 32 | int | 1.36, 1.31, 1.32, 1.26, 1.21 |
| 4 | 12 | int, prod | 2.26 |
| 4 | 16, 20 | int | 2.14, 2.02 |
| 5 | 12, 16, 20 | int | 2.92, 2.72, 2.62 |
| 6 | 12 | int | 3.66 |

All with `[protocol]`: T = 200, snapshot spacing 1.0, `cuts = "all"` (every bipartition whose smaller side has dimension <= 3000; none was skipped),
`big_cut_stride = 1`, penalty strength W = 30, `[model] theta_barrier = 0.5`.  The tilts are grid picks near `<O>_gs = 1/2`, not the exact
crossing (the stored `<O>_gs` is `meta['diag']['mO']`, 0.42-0.59); `meta['how_tilt'] == 'fixed'`.

Written by `run_production(N, k, starts)` in `run.py`, one process per file, by `rerun_all.sh`; N = 5 at k = 16, 20 and N = 6 are the long runs
(hours; the last two with checkpoints, `meta['info']['resumed_from']` says whether a run continued from one).  `meta['wall_s']` is the wall time of the
call (for a two-start file the `int` file includes the product series).  Read a file with

```python
from ljtunnel import store, analysis
run = store.load_run('data/runs/N4_k12_int_c2.260_T200_dt1.npz')
run['t'], run['O'], run['S'], run['masks']    # time, <O>(t), S_A(t) for every cut A, bitmask of A (contains particle 0)
analysis.summarize(run, T_win=120)            # the ledger record for a shorter window
```

or `python -c "from run import *; run_summary()"` for one line per run.  The earlier-convention versions of the same files are in
`../../previous-data/runs/` (its N = 5 and N = 6 files come from the maintainer's own machine).
