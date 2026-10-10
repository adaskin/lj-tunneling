# data/logs/ -- the simulations' own logs

One file per call of the exact code or of the MPS code (written by `ljtunnel.log.tee`; times are the container's clock, UTC):

* `N{N}_k{k}_{YYYYmmdd}_{HHMMSS}_p{pid}.log`: one `dynamics_run` call (resonance, start state, the series with elapsed time and ETA every
  10 snapshots, the file written).  `_p{pid}` is there because the jobs of a rerun run in parallel.  The exact references of the MPS cases
  (`N{3,4,5}_k6`, and the `N4_k12` at T = 40) and the 4 + 4 files of the snapshot-spacing study (`dt_scan`, two extra logs per N = 3, 4 at k = 12) are in
  the same series.
* `N{N}_k{k}_mps_{stamp}.log`: one `run_mps` call (DMRG start, TDVP, bond dimension and truncation every 10 snapshots).
* Parameter scans, the full-grid runs and the Appendix-A ledger print only: their output is in `../rerun/joblogs/job_<name>.log`.

The container the rerun ran in was restarted twice (about 18:00 and 22:00 UTC on 2026-10-06), which killed the jobs that were running.  Their logs are
kept: `N5_k16_20261006_175029_p26951.log` (the first attempt, 8 min), `N6_k12_20261006_210719_p6279.log` and `N5_k20_20261006_211640_p6725.log`
(the first attempts of the two longest runs, about 50 min each, killed by the second restart; the code had no checkpoints yet).  The complete runs are the later files of the
same names (`N5_k16_..._185410_p520`, `N5_k20_..._221735_p514`, `N6_k12_..._221735_p511`).  The checkpoints (`../ckpt/`, removed when a run
completes) were added after that restart and are used by the second attempts of N = 5 at k = 20 and N = 6; `meta['info']['resumed_from']` in the run file says whether a run continued from one.

The logs of the earlier-convention runs are in `../../previous-data/logs/`.
