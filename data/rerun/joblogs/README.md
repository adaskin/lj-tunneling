# data/rerun/joblogs/

`rerun_all.log` (the master log), `job_<name>.log` (one simulation job; `.prev.<stamp>` = an attempt that a container restart killed), `post_<step>.log` (derive steps; `post_mps_tables.log` holds the timing measurement),
`check.log`, `tests_pre.log`, `tests_post.log`, `paper_make_numbers.log`, `paper_figs.log`.  `tests_post.log` records the one test that failed at the end of the run: the bound for the capped MPS case
(`test_tn_tables_reproduce_stored_numbers`, N4_k12: 0.0209 against 0.02); the bound was then split into converged and capped cases and all tests pass.  Overview: `../README.md`, `../summary.txt`.
