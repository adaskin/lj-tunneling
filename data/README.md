# data/ -- the results of the current setting

Everything here was produced by `bash rerun_all.sh` (2026-10-06/07; the record is `rerun/summary.txt` and `rerun/joblogs/`) with the code and the
`config.toml` of the same commit:

* `[model] theta_barrier = 0.5`: the grid column x = 0 (the barrier top) counts with weight 1/2 in the right-valley indicator Theta(x) of the
  observable `<O>` and of the penalty that defines the correlated start (REVIEW.md E3).  Every run file records it in `meta['model']`;
  `records.collect` refuses a run that was made with another value.
* the tilts are the frozen picks of `[tilts]` (never a scan), the protocol is the uniform one: T = 200, snapshot spacing 1.0 (201 snapshots),
  every bipartition of the state, correlated start (and the product start at k = 12 for N = 2, 3, 4).

The same simulations with the earlier convention (`theta_barrier = 0`) are in `../previous-data/`; the comparison, run by run, is
`derived/theta_effect.csv`.  Every log and data file is tracked in git (`.gitignore` ignores only `ckpt/`, the scratch checkpoints of a running job, and
`rerun/state/`, the scratch state of the scheduler).

| directory | what | written by |
|---|---|---|
| `runs/` | the exact runs of the paper: N = 2..6, k = 12..32, uniform protocol | `run_production` (`run_dynamics`) |
| `controls/` | the off-resonance control (N = 5, k = 12, c = 2.62) | `run_off_resonance` |
| `studies/` | snapshot-spacing study (`dt_scan/`), the parameter ledger of Appendix A (`parameter_map/`), the Theta-convention study made before the rerun, the one-body / Monte Carlo check logs the paper quotes (`onebody_mc/`) | `run_sampling_study`, `run_tier1_parameter_map` |
| `mps/` | matrix-product-state validation runs and their exact references (`ed_ref/`) | `run_mps_case` (`run_mps_check`) |
| `derived/` | the tables and records the paper reads, the tensor-network tables (`tn_tables/`), `theta_effect.csv` | `run_records`, `run_protocol_tables`, `run_extrapolation`, ... |
| `logs/` | the simulations' own logs (one per `dynamics_run` / `run_mps` call) | `ljtunnel.log.tee` |
| `rerun/` | the record of the rerun: one log per job and step, `summary.txt` | `rerun_all.sh` |
| `followup/` | the follow-up analysis of the start state (spectra, signal fits, static checks): the data behind Section 5 and Appendix D of the report (README there) | `run_followup_stage1`, `followup/*.py` |
| `ckpt/`, `rerun/state/` | scratch while jobs run (ignored; empty when a rerun is complete) | `dynamics_run(checkpoint=True)`, `rerun_all.sh` |

Which directory feeds what: `runs/`, `controls/`, `studies/`, `mps/` are simulation results; `derived/` is computed from them in seconds
(`run_records()` etc.); `followup/` is a separate analysis of the same model that the report reads directly.  Nothing in `data/` is written by hand.
The earlier code of the project (before the uniform protocol) and its converted results were removed on 2026-10-09, because every point of the paper is now a
uniform run (git history, commit 497189c, has them).

Format: every result is one `.npz` without pickles; the parameters, the resonance, the start-state diagnostics and the protocol are one JSON
string under `meta_json` (`ljtunnel.store.load_run(path)['meta']`).  See the README of the repository for the arrays.
