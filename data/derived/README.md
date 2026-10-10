# data/derived/ -- the tables and records the paper reads

Built from `../runs`, `../controls`, `../studies` and `../mps`
by the analysis functions of `run.py`; `paper/scripts/make_numbers.py` and `figs.py` turn them into `numbers.tex`, the tables and the figures.  No simulation
happens here.  Rebuild in this order (seconds, except the timing measurement of `run_mps_tables(step_cost=True)`, ~8 min on an idle machine; `rerun_all.sh` does it after the simulations):

| file | written by | content |
|---|---|---|
| `ledger.npz`, `graph.npz` | `run_records()` | one record per (run, window) and per correlated-start run (cut entropies against the graph estimate); every record is from a uniform run; the off-resonance control is in `graph` only |
| `protocol_tables.npz` | `run_protocol_tables()` | the window scan of every k = 12 run and the snapshot-spacing scan of `../studies/dt_scan` |
| `larger_N.npz` | `run_extrapolation()` | the sharing law carried to larger N, the three scenarios, the first failing N of each classical route (`walls`), the exact-diagonalisation walls, `share_pred_prev` (the sharing-law value at the last measured N as predicted from the points below it) |
| `quantum.npz` | `run_quantum()` | qubit and T-gate counts (config only) |
| `cosine_check.npz` | `run_cosine_check()` | how much of `<O>(t)` one splitting-cosine explains, per run |
| `parameter_choice.npz` | `run_parameter_choice()` | the Appendix-A tables, parsed from `../studies/parameter_map/tier1_parameter_map.txt` |
| `parameter_scan_N3.npz`, `parameter_scan_N4.npz` | `run_parameter_scan` | is the barrier A = 8 still right at N = 3, 4 (static scan; N = 4 at A = 6, 8, 10, 12) |
| `fullgrid_N2.npz`, `fullgrid_resonance.npz` | `run_fullgrid_N2()`, `run_fullgrid_check()` | N = 2 on the full grid at the production tilt 0.45, and at its own exact `<O>_gs = 1/2` resonance (c* = 0.4633, dE = 0.19186, t0 = pi/dE = 16.37; `[quantum] t_star_0` = 16.5212 is the earlier study's own number and is kept) |
| `tn_tables.npz`, `tn_tables/*.csv` | `run_mps_tables(step_cost=True)` | the tables of the tensor-network section (`tn_tables/README.md`) |
| `theta_effect.csv` | `run_theta_effect()` | every run of `../runs` next to the same run of `../../previous-data/runs` (`theta_barrier` 0): <O>_gs, S_cen, S_nat, S_max, chi(1e-3), time of the maximum, the largest change of S(t) and <O>(t) |

`theta_effect.csv` in one line: over the 16 correlated-start runs (N = 2..6) the central entropy moves by at most 0.011 bit, the largest cut entropy by at most
0.013 bit, `<O>_gs` at the fixed tilt by +0.006 .. +0.019, chi(1e-3) of the central bond by at most 13 (N = 6: 312 -> 325); dE and every product-start entropy are
unchanged (they do not depend on Theta).  `run_theta_effect()` raises if dE or a product-start entropy changed.
