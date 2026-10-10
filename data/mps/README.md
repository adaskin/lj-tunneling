# data/mps/ -- matrix-product-state runs against the exact code

Four validation cases of `ljtunnel.mps` (TeNPy: DMRG start, two-site TDVP): the same model, tilt, correlated start and window as an exact run, so that every
cut measured by both can be compared snapshot by snapshot.  Made with `[model] theta_barrier = 0.5` by `run_mps_case(name)` (`run.py`; `run_mps_check` underneath),
TDVP step 0.5, snapshot spacing 1.0.

| case | tilt c | T | chi_max | eps | pairs in the MPS | MPS file | exact reference (`ed_ref/`, bonds + pairs) |
|---|---|---|---|---|---|---|---|
| `N3_k6` | 1.6219 | 60 | 120 | 1e-8 | yes | `N3_k6_mps_c1.622_T60_dt1_chi120_eps1e-08.npz` | `N3_k6_int_c1.622_T60_dt1.npz` |
| `N4_k6` | 2.4562 | 60 | 120 | 1e-8 | yes | `N4_k6_mps_c2.456_T60_dt1_chi120_eps1e-08.npz` | `N4_k6_int_c2.456_T60_dt1.npz` |
| `N5_k6` | 3.1695 | 40 | 100 | 1e-8 | yes | `N5_k6_mps_c3.170_T40_dt1_chi100_eps1e-08.npz` | `N5_k6_int_c3.170_T40_dt1.npz` |
| `N4_k12` | 2.26 | 40 | 64 | 1e-6 | no (bonds) | `N4_k12_mps_c2.260_T40_dt1_chi64_eps1e-06.npz` | `N4_k12_int_c2.260_T40_dt1.npz` |

The k = 6 tilts are the exact `<O>_gs = 1/2` crossing of the new convention (bisection of the exact ground state to 2e-4: 1.6219, 2.4562, 3.1695; the
earlier-convention ones were 1.6212, 2.455, 3.1639), the k = 12 case uses the tilt of `[tilts]`.  The exact reference has protocol `custom` (T < 200).

Result (`../derived/tn_tables/mps_validation.csv`; largest difference over every cut measured in both and over all snapshots, `<O>(t)`, central bond):

| case | max dS (bit) | max d<O> | S_cen MPS - exact | chi(1e-3) MPS / exact | chi(1e-6) MPS / exact | kept chi | discarded weight | wall MPS / exact (s) |
|---|---|---|---|---|---|---|---|---|
| `N3_k6` | 1.8e-5 | 5e-6 | -1.5e-6 | 5 / 5 | 6 / 6 | 6 | 0 | 17 / 0.4 |
| `N4_k6` | 5.9e-4 | 1e-4 | -4.7e-4 | 18 / 18 | 31 / 31 | 34 | 1e-6 | 95 / 1.2 |
| `N5_k6` | 1.8e-3 | 1e-4 | +1.9e-4 | 32 / 32 | 36 / 36 | 36 | 0 | 476 / 6.4 |
| `N4_k12` | 2.8e-2 | 2.8e-3 | -2.1e-2 | 32 / 35 | **64** / 117 | **64 = chi_max** | 6.5e-3 | 2157 / 14.8 |

The three k = 6 cases are converged (agreement at the tolerances of the MPO, DMRG and TDVP).  The k = 12 case is limited by the bond-dimension cap: its
entropy is 0.021 bit below the exact one (0.017 with the earlier convention) and its chi(1e-6) is not a measurement.  The MPS is 40-150 times slower than the
exact code at these sizes (wall times of a shared cloud container, jobs running two to four at a time: only ratios and orders of magnitude mean anything).

Reproduce one case: `python -c "from run import *; run_mps_case('N4_k6')"` (runs the exact reference, then the MPS, and prints the differences).  Logs:
`../logs/N{N}_k{k}_mps_*.log` (MPS) and the `../logs/N{N}_k{k}_*_p*.log` of the exact references; per-job output `../rerun/joblogs/job_mps_*.log`.
The earlier-convention cases (and their logs and provenance scripts) are in `../../previous-data/mps/`.
