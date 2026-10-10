# data/studies/parameter_map/ -- the Appendix-A ledger of the current convention

`tier1_parameter_map.txt`: part (a) one particle in U = A (x^2-1)^2 (barrier action, trap mass, splitting, switching time pi/dE for A = 2..16), part (b) N = 2 exact diagonalisation on the
28x18 grid with one input changed at a time (A, sigma, epsilon, omega_y, C0, cap): the resonance tilt c* (bisection of `<O>_gs = 1/2`, with `theta_barrier = 0.5`), the splitting, the weight of the
valley-product start on the two lowest levels, the third level's gap and the cap sensitivity; one JSON record per line.  Written by `python -m ljtunnel.parameter_study --theta-barrier 0.5`
(`run_tier1_parameter_map()`, 19 points, 33 min on 2 workers), parsed by `run_parameter_choice()` into `derived/parameter_choice.npz`.  The earlier-convention ledger (`--theta-barrier 0`) is in git history (commit 497189c, `previous-data/legacy/tier1/`).
