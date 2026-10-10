> Moved to `previous-data/mps/` (made with `theta_barrier = 0`, before the rerun).  Paths below are as they were in `data/mps/`; the current validation is `data/mps/README.md`.

# data/mps/ -- matrix-product-state runs, their exact references, logs

| path | what |
|---|---|
| `N{N}_k{k}_mps_c{c}_T{T}_dt1_chi{chi_max}_eps{eps}.npz` | an MPS run (TeNPy: DMRG start, two-site TDVP), same layout as `data/runs/` plus `chi` (bond dimensions), `energy`, `trunc_err`; `meta['protocol']['kind'] == 'mps'`; load with `ljtunnel.store.load_run` |
| `ed_ref/N{N}_k{k}_int_c{c}_T{T}_dt1.npz` | the exact run of the same case (every cut, protocol `custom` because T < 200), the reference for the MPS run of the same name |
| `runlogs/*.log.txt`, `runlogs/*.jsonl` | the stdout logs and the comparison summaries of the runs that produced the files above (`.txt` because `.gitignore` excludes `*.log` and `logs/`) |
| `runlogs/ed_logs/*.log.txt` | the logs the exact code writes itself (`data/logs/`, git-ignored there) |
| `runlogs/provenance/` | the scripts that produced those logs; superseded by `run_mps_check` / `run_mps` in `run.py`, kept because they are what was run (they contain this session's paths) |

Cases: k = 6: N = 3 (c = 1.6212), 4 (2.455), 5 (3.1639), tilts bisected to <O>_gs = 1/2; k = 12: N = 4 (c = 2.26, from `[tilts]`), T = 40, chi <= 64.

## Log files

| file | note |
|---|---|
| `validation_k6_N3_N4.log.txt` / `.jsonl` | N = 3, 4 at k = 6 (T = 60, chi <= 120).  Ends inside an N = 5 attempt (T = 60, at t = 20) that I stopped myself, because it was competing with a test run and I wanted to benchmark the engines; N = 5 was rerun with T = 40, see the next file. |
| `validation_k6_N5.log.txt` / `.jsonl` | N = 5 at k = 6 (T = 40, chi <= 100), complete.  The N = 6 entry fails: a bug in the driver script (no tilt table for N = 6), no data. |
| `validation_k12_N4.log.txt` / `.jsonl` | N = 4 at k = 12 (T = 40, chi <= 64).  Part of it ran while a test suite shared the CPU, hence the long wall time (2116 s). |
| `exact_N5_k12_uniform_reproduction.log.txt` | the uniform N = 5, k = 12 exact run done independently here; its arrays agree with `data/runs/N5_k12_int_c2.920_T200_dt1.npz` to 1.3e-10 (entropies) and exactly (ranks).  105 min on a shared 4-core container (21 min on yours). |
| `pilot_k12_N4_T12.log.txt` | an early k = 12 test, T = 12, from before the DMRG start was converged on the energy variance (its start was only converged in the energy); not used in any table. |
| `aborted_k6_N6.log.txt` | an N = 6, k = 6 run that was stopped after 2 h without producing data (it had been left running by mistake). |
| `ed_logs/` | the exact code's own logs: `N3_k6_*` (3 files) and `N5_k6_*` (3 files) are repeated starts of the same cases; `N4_k12_*` is the exact reference of the k = 12 MPS run; `N5_k12_*` is the reproduction run above; `N6_k6_*` belongs to the aborted run. |

Reproduce: `python -c "from run import *; run_mps_check(N=4, k=6, tilt=2.455, T=60, chi_max=120)"` (exact run + MPS run + the largest difference over every cut).
Tables for the figures: `data/derived/tn_tables/README.md`.
