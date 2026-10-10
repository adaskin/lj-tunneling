> Moved to `previous-data/derived/tn_tables/` (made with `theta_barrier = 0`, before the rerun).  Paths below are as they were in `data/`; the current tables and their README are `data/derived/tn_tables/`.

# Tables for the tensor-network section (data/derived/tn_tables/)

One csv per table, all also in `data/derived/tn_tables.npz` (`ljtunnel.store.load_object(...)` -> `{'tables': {name: [row, ...]}, 'machine': {...}}`).
Rebuild: `python -c "from run import *; run_mps_tables(step_cost=False)"` (6 s; `channel_ranks` needs TeNPy; `step_cost=True` measures
for ~8 min and needs an idle machine).  The code is `ljtunnel/tntables.py`.  Entropies in bits, times in model units (hbar = m = 1) unless
the column says seconds.  Context and interpretation: `REVIEW.md` (finding ids T1-T8 below refer to it).

## What each table is

| file | rows | content | source data |
|---|---|---|---|
| `mps_validation.csv` | 4 | every matrix-product-state (MPS) run next to the exact run of the same (N, k, tilt, start) over the MPS run's window `T` | `data/mps/*.npz`, `data/mps/ed_ref/*.npz` |
| `mps_series.csv` | 204 | long format, per case and snapshot: central-bond entropy, <O>, central bond dimension, cumulative truncation, MPS and exact | same |
| `window_convergence.csv` | 92 | S_cen, S_nat, mean one-particle entropy, ranks, time of the maximum, number of cuts peaking at the window edge, against the window `T` (10, 20, ..., 200), k = 12, N = 2..5 (uniform runs; the legacy N = 5 for comparison) | `data/runs/N{2,3,4,5}_k12_int_*_T200_dt1.npz`, `data/legacy/runs/N5_k12_int_c2.920_T120_legacy.npz` |
| `rank_vs_entropy.csv` | 95 | central-bond Schmidt ranks chi(1e-2, 1e-3, 1e-6) against 2^S_cen and against the largest possible rank `cap` = k^min(b, N-b), for every stored exact run, windows 30, 40, 60, 120, 200 where the run is long enough | `data/runs/*.npz`, `data/mps/ed_ref/*.npz` |
| `wall_sensitivity.csv` | 15 | the first failing N of the three entropy scenarios (sat / opt / pess; grid d = 1200 / modes d = 20; both budgets) when S1(5) and S_cen(5) are shifted by `dS1` = -0.30 ... +0.30 (rows `shift ...`), and with the uniform N = 5 run at windows 120 and 200 (rows `uniform N=5 ...`, the only rows with `wallchi_*`) | `data/derived/larger_N.npz` (built from the legacy N = 5), `data/runs/N5_k12_...` |
| `channel_ranks.csv` | 40 | interaction channels R of the MPO (bond dimension `D_mpo` = R + 2) and the worst operator-norm error of any projected pair matrix, against the relative tolerance `tol`; (N, k, c) = (4,6,2.455), (4,12,2.26), (6,6,3.8), (6,12,3.6), (4,8,2.35); the tilts are representative only | computed (modes of `ljtunnel.model`) |
| `step_cost.csv` | 6 | measured seconds per two-site TDVP step (dt = 0.5) against the bond-dimension cap; the number and time of the H_eff matvecs; the counting model's operation count `2 N (d chi)^3` and its time at this machine's measured zgemm rate | measured, `machine` record in the npz |

## Columns that need a word

* `S_cen`, `S_nat`: maxima over the window [0, T] of the central bond / of the largest contiguous bond (paper definitions).  `tS_cen`: time of the first maximum.
* `chi2`, `chi3`, `chi6`: numerical Schmidt rank of the central bond at discarded weight 1e-2, 1e-3, 1e-6 (maximum over the window).  For an MPS run
  they can only be read below what the run kept: `chi_kept` (largest bond dimension) < `chi_max`, otherwise the cap was binding (`mps_validation`:
  the k = 12 case has `chi6_mps` = 64 = `chi_max` against `chi6_exact` = 115).
* `max_dS`: largest |S_mps - S_exact| over every cut measured by both (bonds, singles, pairs) and every snapshot; `max_dO` the same for <O>(t).
* `n_edge`: number of cuts whose maximum is the last snapshot of the window.  It is 0 for N = 5 at T = 200 although the maximum is at t = 197.
* `t_first_O_min`: first local minimum of <O>(t), the paper's fallback observation time.  `pi/dE` is NOT used: at N = 4 and 5 there is no clean doublet.
* `step_cost.ratio_measured_to_model`: measured step / (model operation count at the zgemm rate).  Nearly flat step time in chi (10.5 -> 14.5 s for
  chi = 16 -> 64) means the dense MPO contraction dominates at these sizes, so the ratio falls with chi and is not transferable to large chi (T4).
* `wall_sensitivity`: `S_cen_last` is shifted by the same `dS1` as `S1_last`; `wall_*` = first failing N (the paper's "wall"); 400 means "none below 400".

## Suggested figures (the numbers behind them are all here)

1. MPS against exact: `mps_series` (S_cen(t) and <O>(t), one panel per case) with the residuals; annotate `max_dS` from `mps_validation`.
2. Window convergence: `window_convergence` S_cen(T) and `tS_cen` for N = 2..5; the N = 5 legacy curve ends at T = 120.
3. Rank against 2^S: `rank_vs_entropy` ratio3 and ratio6 against N (k = 12: N = 5 has ratio3 10.9, ratio6 21.9; k = 16 at N = 5, also a uniform run: 17.3 and 41.6; chi6 is 99% and 98% of `cap`), with `cap`.
4. Wall against the last data point: `wall_sensitivity` walls against `dS1`, the uniform-N=5 rows marked.
5. Cost: `step_cost` seconds per step against chi next to the model; `channel_ranks` R against `tol` for the MPO.

## Caveats that belong in any caption

* The MPS runs are a validation, not an estimate of the k = 12 cost beyond N = 4: k = 6 is a saturated basis (S_cen(N=5) = 3.95 of at most 5.17 bits), and
  k = 12 with N >= 5 did not fit in this session.  The MPS was 30-100 times slower than the exact code at these sizes.
* The k = 12 MPS run is capped at chi = 64; its S_cen is 0.017 bit below the exact value and its chi(1e-6) is not a measurement.
* Wall times are from a 4-core container whose CPU was shared with another job for part of the time (`wall_*_s`); `step_cost` was measured on an idle machine.
* `data/derived/larger_N.npz` still holds the legacy N = 5 point; the `uniform N=5` rows show what regenerating it (`run_records(); run_extrapolation()`) does.
