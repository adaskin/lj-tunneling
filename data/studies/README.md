# data/studies/ -- studies around the main runs

| directory | what |
|---|---|
| `dt_scan/` | the snapshot spacing at the production window: `N{3,4}_k12_int_c{1.460,2.260}_T200_dt{0.5,1,2,4}.npz` (8 files; same state, same T = 200, different `dt_snap`; protocol `custom` except dt = 1).  Written by `run_sampling_study()`, read by `run_protocol_tables()`. |
| `parameter_map/` | `tier1_parameter_map.txt`: the ledger of Appendix A (one particle: splitting against the barrier A; N = 2 exact diagonalisation on a 28x18 grid, one input changed at a time).  Written by `ljtunnel/parameter_study.py --theta-barrier 0.5` (`run_tier1_parameter_map()`), parsed by `run_parameter_choice()` into `derived/parameter_choice.npz`. |
| `onebody_mc/` | `mc_gamma_ledger.txt`, `tier1_checks.log`: the one-body and Monte Carlo check logs quoted in Appendix C (inputs from the first code version; README there). |
| `theta_convention/` | the measured effect of the Theta(x = 0) convention at k = 12, N = 2-4, made BEFORE the rerun (see its README); `derived/theta_effect.csv` repeats it for every run of the rerun. |
