# previous-data/ -- everything made before the Theta(x = 0) rerun

Moved here from `data/` (by the maintainer) so that the old results cannot be mixed up with the new ones.  All of it was made with
`[model] theta_barrier = 0` (the grid column x = 0 counted as left; REVIEW.md E3) and with the code of `main` before the rerun.  Nothing writes here;
treat it as the record of the earlier setting.  The layout is the one `data/` had.

| directory | what |
|---|---|
| `runs/` | the 19 exact runs of the earlier setting (same names as in `data/runs/`).  The N = 5 and N = 6 files were made on the maintainer's own machine (N = 5 at k = 20 and N = 6 at k = 12 finished there while the rerun was running); N = 2..4 in an earlier session of the project |
| `derived/` | the tables and records built from them (`ledger.npz`, `graph.npz`, `larger_N.npz`, ...; `tn_tables/`) |
| `mps/` | the four MPS validation cases and their exact references, the logs and the scripts that made them (`runlogs/`) |
| `studies/` | the snapshot-spacing study (`dt_scan/`) |
| `logs/` | the logs of the runs in `runs/` |

Who reads it: `run_theta_effect()` (`run.py`) and `paper/scripts/make_numbers.py` (the walls recomputed from the earlier runs), which compare `data/runs` with `previous-data/runs` run by run
and writes `data/derived/theta_effect.csv`.  The comparison in one line: the central entropy moves by at most 0.011 bit (N = 2..6), dE and the product-start
entropies do not move, `<O>_gs` at a fixed tilt moves by +0.006 .. +0.019.

A later full rerun (`bash rerun_all.sh`) archives the then-current `data/` into a new subdirectory `previous-data/<date>_before_rerun/` instead of overwriting anything.

**Removed 2026-10-09: `previous-data/legacy/`** (and the `data/legacy` link to it).  It held the results of the first code version (windows T = 70, 60, 60, 120 and spacings
that depended on N, converted to npz; the original logs, `.rows` and JSON files; the tier 1-3 ledgers and the tier 2 archives), a fallback that the pipeline used only for
points without a uniform run; since every point is now a uniform run, and the rerun reproduced the earlier numbers, nothing reads it.  The two files the paper still quotes
moved to `data/studies/onebody_mc/`.  Everything else is in git history (commit 497189c).  Mentions of `legacy` in the READMEs below this directory are historical.
