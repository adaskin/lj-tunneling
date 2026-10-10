# Tables for the tensor-network section (data/derived/tn_tables/)

One csv per table, all also in `data/derived/tn_tables.npz` (`ljtunnel.store.load_object(...)` -> `{'tables': {name: [row, ...]}, 'machine': {...}}`).
Rebuild: `python -c "from run import *; run_mps_tables(step_cost=False)"` (seconds; `channel_ranks` needs TeNPy; it keeps the stored `step_cost` table; `step_cost=True`
re-measures for ~8 min and needs an idle machine, which is what `rerun_all.sh` does).  The code is `ljtunnel/tntables.py`.  Entropies in bits, times in model
units (hbar = m = 1) unless the column says seconds.  Context and interpretation: `REVIEW.md` (finding ids T1-T8 refer to it).  Made with `theta_barrier = 0.5`;
the earlier-convention tables are in `previous-data/derived/tn_tables/`.

## What each table is

| file | rows | content | source data |
|---|---|---|---|
| `mps_validation.csv` | 4 | every matrix-product-state (MPS) run next to the exact run of the same (N, k, tilt, start) over the MPS run's window `T` | `data/mps/*.npz`, `data/mps/ed_ref/*.npz` |
| `mps_series.csv` | 204 | long format, per case and snapshot: central-bond entropy, <O>, central bond dimension, cumulative truncation, MPS and exact | same |
| `window_convergence.csv` | 80 | S_cen, S_nat, mean one-particle entropy, ranks, time of the maximum, number of cuts peaking at the window edge, against the window `T` (10, 20, ..., 200), k = 12, N = 2..5 | `data/runs/N{2,3,4,5}_k12_int_*_T200_dt1.npz` |
| `rank_vs_entropy.csv` | 105 | central-bond Schmidt ranks chi(1e-2, 1e-3, 1e-6) against 2^S_cen and against the largest possible rank `cap` = k^min(b, N-b), for every exact run (N = 2..6, k = 12..32 and the MPS references), windows 30, 40, 60, 120, 200 where the run is long enough | `data/runs/*.npz`, `data/mps/ed_ref/*.npz` |
| `wall_sensitivity.csv` | 15 | the first failing N of the three entropy scenarios (sat / opt / pess; grid d = 1200 / modes d = 20; both budgets) when the LAST measured point (now N = 6) is shifted by `dS1` = -0.30 ... +0.30 in S1 and S_cen (rows `shift ...`), and with the data stopped at N = 5 and its values at windows 120 and 200 (rows `uniform N=5 ...`, the only rows with `wallchi_*`) | `data/derived/larger_N.npz`, `data/runs/N5_k12_...` |
| `channel_ranks.csv` | 40 | interaction channels R of the MPO (bond dimension `D_mpo` = R + 2) and the worst operator-norm error of any projected pair matrix, against the relative tolerance `tol`; (N, k, c) = (4,6,2.455), (4,12,2.26), (6,6,3.8), (6,12,3.6), (4,8,2.35); the tilts are representative only | computed (modes of `ljtunnel.model`) |
| `step_cost.csv` | 6 | measured seconds per two-site TDVP step (dt = 0.5) against the bond-dimension cap; the number and time of the H_eff matvecs; the counting model's operation count `2 N (d chi)^3` and its time at this machine's measured zgemm rate | measured on the 4-core container that ran the rerun, alone (`data/rerun/joblogs/post_mps_tables.log`); `machine` record in the npz |

## Numbers worth knowing (k = 12 unless said, window T = 200)

* Schmidt rank against 2^S_cen (`rank_vs_entropy`, ratio3 = chi(1e-3)/2^S, ratio6 = chi(1e-6)/2^S): N = 4: 7.5 and 23; N = 5: 11.3 and 21.8 (chi6 = 143 of cap 144);
  N = 6: 34.0 and 127.8 (chi3 = 325, chi6 = 1222 of cap 1728).  At k = 16, N = 5: 17.4 and 41.7 (chi6 = 251 of 256); at k = 20, N = 5: 19.3 and 61.8.
* TDVP step, N = 4, k = 12 (`step_cost`): 10.3, 11.4, 13.7, 15.0 s for chi = 16, 32, 48, 64: nearly flat, the dense MPO contraction dominates at these sizes.
* Walls with the data to N = 6 (`larger_N.npz`; first failing N on the grid d = 1200 / modes d = 20): saturating 15 / 25, optimistic 16 / 27, pessimistic 13 / 18; with the data
  stopped at N = 5 (rows `uniform N=5`): saturating 13 / 22, optimistic 17, pessimistic 10.

## Columns that need a word

* `S_cen`, `S_nat`: maxima over the window [0, T] of the central bond / of the largest contiguous bond (paper definitions).  `tS_cen`: time of the first maximum.
* `chi2`, `chi3`, `chi6`: numerical Schmidt rank of the central bond at discarded weight 1e-2, 1e-3, 1e-6 (maximum over the window).  For an MPS run
  they can only be read below what the run kept: `chi_kept` (largest bond dimension) < `chi_max`, otherwise the cap was binding (`mps_validation`:
  the k = 12 case has `chi6_mps` = 64 = `chi_max` against `chi6_exact` = 117).
* `max_dS`: largest |S_mps - S_exact| over every cut measured by both (bonds, singles, pairs) and every snapshot; `max_dO` the same for <O>(t).
* `n_edge`: number of cuts whose maximum is the last snapshot of the window.
* `t_first_O_min`: first local minimum of <O>(t), the paper's fallback observation time.  `pi/dE` is NOT used: at N = 4 and 5 there is no clean doublet.
* `step_cost.ratio_measured_to_model`: measured step / (model operation count at the zgemm rate).  Nearly flat step time in chi means the dense MPO contraction
  dominates, so the ratio falls with chi and is not transferable to large chi (T4).
* `wall_sensitivity`: `S_cen_last` is shifted by the same `dS1` as `S1_last`; `wall_*` = first failing N (the paper's "wall"); 400 means "none below 400".

## Caveats that belong in any caption

* The MPS runs are a validation, not an estimate of the k = 12 cost beyond N = 4: k = 6 is a saturated basis (S_cen(N=5) = 3.96 of at most 5.17 bits), and
  k = 12 with N >= 5 was not run as an MPS.  The MPS was 40-150 times slower than the exact code at these sizes.
* The k = 12 MPS run is capped at chi = 64; its S_cen is 0.021 bit below the exact value and its chi(1e-6) is not a measurement.
* Wall times in `mps_validation` (`wall_*_s`) are from a 4-core cloud container with other jobs running at the same time (two to four at once); `step_cost` was measured alone.
