# data/rerun/ -- the record of the rerun

`bash rerun_all.sh` writes here (the script and its options: `../../README.md`, "Rerunning everything"):

| file | what |
|---|---|
| `summary.txt` | when it started and ended, the git revision and the sha256 of `config.toml` it used, python / numpy / scipy / TeNPy versions, the CPU, one line per job (threads, seconds, exit status) and the result of the checks |
| `joblogs/rerun_all.log` | the master log: every job start and end, the heartbeat every 20 min, the derive and check steps |
| `joblogs/job_<name>.log` | stdout and stderr of one simulation job (`run.py jobs` lists them); `job_<name>.log.prev.<stamp>` is an earlier attempt of the same job that a container restart killed |
| `joblogs/post_<step>.log` | one derive step (`records`, `protocol_tables`, `extrapolation`, `quantum`, `cosine_check`, `parameter_choice`, `mps_tables` with the timing measurement, `theta_effect`) |
| `joblogs/check.log`, `tests_pre.log`, `tests_post.log`, `paper_*.log` | the checks of `run_check_rerun`, the two test suites (before, after) and the paper scripts |
| `paper_build/` | `numbers.tex` and the `tab_*.tex` that `paper/scripts/make_numbers.py` wrote from these data (a reference for the paper; the files in `paper/` belong to the paper branch) |
| `state/` | scratch state of the scheduler (done markers, pids, the config hash); ignored, and only needed to resume an interrupted rerun |

The wall times in `summary.txt` and in the logs were measured on a 4-core cloud container (the jobs ran two or four at a time, 1-2 BLAS threads each); the
long runs (N = 5 at k = 20, N = 6) took about 10.5-11 h each.
