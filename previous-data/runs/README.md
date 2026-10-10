# previous-data/runs/ -- the exact runs of the earlier setting (theta_barrier = 0)

The same 19 files as `data/runs/` (`N{N}_k{k}_{int|prod}_c{tilt}_T200_dt1.npz`: N = 2..6, k = 12..32, uniform protocol T = 200, spacing 1.0, every bipartition),
with the grid column x = 0 counted as left.  Files of N = 5 and N = 6 come from the maintainer's own machine (`N5_k20_...` and `N6_k12_...` exist only there; logs `../logs/N5_k20_20261006_145858.log` and
`../logs/N6_k12_20261006_112438.log`; that machine's clock is UTC+3); N = 2..4 are from an earlier session of the project (origin not documented beyond their logs).
Their `meta['model']` has no `theta_barrier` key (read as 0 by `records.theta_of`), which is why `records.collect` refuses to mix them with the new runs.
`data/derived/theta_effect.csv` lists the difference to the new run of the same name.
