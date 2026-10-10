# data/followup/static_checks/ -- static checks of the scaling-path / conditional-reduction arguments

Written 2026-10-09. Not part of the pre-registered Stage 1-5 plan (`../PREREG.md`): these are exploratory, made to check claims of an outside analysis against data. Nothing here propagates in time.

| file | what | made by |
|---|---|---|
| `static_checks.log` | the complete output of the four parts below | `python followup/static_checks_scaling_args.py > data/followup/static_checks/static_checks.log` (repository root; about 2 min on 2 cores) |

Parts: `energy` (pair-interaction energy of the all-left correlated state), `levels` (low levels with y-reflection parity, start weight, per-particle right-valley occupation), `dressing` (where the weight outside the two heaviest levels sits, from `../stage1/N*_k*_c*.npz`), `untilted` (saved level lists).
The k = 6 and k = 8 rows are printed for completeness; k = 6 energies are artefacts of a 6-mode basis (the all-left state is crowded into too few modes) and are not evidence for anything.

## Results (numbers are in `static_checks.log`)

* **c*(N) is linear through the measured N.** k = 12: 0.70, 1.46, 2.26, 2.92, 3.66 (N = 2..6), increments 0.76, 0.80, 0.66, 0.74; k = 20: 0.70, 1.31, 2.02, 2.62, increments 0.61, 0.71, 0.60. The k = 6 value at N = 7 (4.02) is not converged and says nothing about bending.
* **The cluster is compact and all-to-all through N = 5.** E_int(N, 0) at k = 12: 0.339, 1.201, 3.122, 5.762 (N = 2..5); per pair 0.339, 0.400, 0.520, 0.576 (rising); k = 16: 0.325, 0.370, 0.475 (N = 2..4); k = 20: 0.316, 0.348 (N = 2, 3). Pairs holding 90 % of E_int: 1/1, 3/3, 6/6, 9/10. A coordination-saturation regime would show as a falling energy per pair; not seen.
* **y-reflection parity.** The start has <R> = +0.9999 (N = 3) and +0.9997 (N = 4). Every level of non-negligible weight is even; odd levels carry at most 1e-4 (the truncated modes close under R only to 0.05, so this is leakage, not physics). The "dark" levels between the bright pair are y-odd.
* **Sector exclusion.** The lowest even level with total right occupation >= 1.5 lies at E - E0 = 2.99 (N = 3, k = 12), 2.73 (N = 3, k = 16); none in the lowest 90 levels at N = 4, k = 12 (up to 2.14). J = omega*/2 = 0.092-0.095.
* **Label fragmentation.** N = 3: pair at 0 and 0.19; the state with particle 1 right at 0.40-0.43, particle 2 at 0.65-0.66 (2 C0 = 0.30). N = 4, k = 12: particles 1 and 2 hybridize near 0.67-0.71, particle 3 at 0.98. Margin 2-6 times J, weights <= 1-2 %.
* **Where the dressing sits.** Weight outside the two heaviest levels, k = 16 / 20: far (>= 3 J) 0.012/0.013, 0.030/0.028, 0.026/0.024, 0.010/0.010 for N = 2..5 (about constant); mid (1-3 J) 0.011/0.010, 0.018/0.017, 0.114/0.056, 0.151/0.114 (this is the growing part); near (< J) about 0 at k >= 16 (0.085 at N = 4 and 0.268 at N = 5 for k = 12: basis artefacts). 1 - B2 at k = 20: 0.022, 0.045, 0.094, 0.188 (N = 2..5). The growing part sits 0.1-0.35 from the pair, the same scale as J.
* **Dressing vs barrier (paper tables, not recomputed).** `tab_Asweep`: one-particle splitting 0.191, 0.125, 0.083, 0.039 for A = 8, 10, 12, 16. `tab_Ascan` (k = 12): N = 3 overlap 0.935, 0.960, 0.956 at A = 8, 10, 12; N = 4 0.409, 0.813, 0.760. J falls with A; the dressing does not.
* **Untilted (c = 0).** C0 = 0: N = 3 lowest band 12 levels (width 0.41, gap after it 1.05), N = 4 24 levels (width 0.29, gap 1.04), with permutation multiplets; with C0 = 0.15 the multiplets split and no band is found.

## What this does not settle
Whether the dressing is Gaussian (harmonic Huang-Rhys prediction of B against the measured B at k >= 16 is the test, WP4 of `followup/TASK_followup_v2_observable_first.md`); behaviour beyond N = 5; whether the transfer continues as a many-body peeling instanton past the single-particle barrier-top crossing (a 1-D estimate puts that at c ~ 6.3, N ~ 9-10 if c* keeps rising 0.6-0.7 per particle). The 1-D estimate is a rough single-particle picture (finite differences, 500 points on [-2.5, 2.5], hard walls).
