# data/mps/ed_ref/ -- the exact references of the MPS cases

`N{N}_k{k}_int_c{tilt}_T{T}_dt1.npz`: the exact run (every bond, single and pair cut; protocol `custom` because T < 200) of the same (N, k, tilt, start, window) as the MPS run of the same case in `..`; the
MPS file name is `N{N}_k{k}_mps_c{tilt}_T{T}_dt1_chi{chi}_eps{eps}.npz`.  Written first by `run_mps_check` / `run_mps_case`.  Comparison: `../README.md`, `../../derived/tn_tables/mps_validation.csv`.
