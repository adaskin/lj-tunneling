# ljtunnel/ -- the library

Real-time tunnelling of N distinguishable particles in a 2D double well with a capped Lennard-Jones interaction: model, exact
propagation, entropies of every cut, tensor-network (TeNPy) validation, growth laws and resource counts.  Nothing here has constants of
its own: every number comes from `../config.toml`; every simulation is started from `../run.py`.

| module | what it is |
|---|---|
| `config` | `load_config`, `protocol` (the uniform measurement protocol; a run with another T or spacing is `custom`) |
| `model` | grid, one-particle problem, mode basis, capped LJ pair potential, tilt table; `make_grid` builds `Theta(x)` with the weight `[model] theta_barrier` on the barrier column |
| `hamiltonian` | `ModeN`: matrix-free H on the k^N mode space, the observable `O = sum Theta(x_i)`, penalty operators |
| `dynamics` | Chebyshev propagation, spectral bounds, snapshot generator |
| `entanglement` | all bipartitions of a state: entropies, second-order entropy, Schmidt ranks; `CutPlan` |
| `resonance`, `starts` | the tilt where the ground state has `<O>` = 1/2 (`locate`, `exact_tilt`); product and correlated start states with diagnostics |
| `experiments` | `dynamics_run`: resonance -> start -> real-time series -> npz; `checkpoint=True` makes an hours-long run restartable |
| `analysis` | everything derived from a stored series, for any window: maxima, edge weights, ledger and graph records |
| `graph` | cut values of a weighted graph for all 2^(N-1)-1 subsets, layouts, sharing-law weights |
| `records` | the saved runs (+ the off-resonance control) -> the ledger and graph tables the paper reads; refuses runs of another `theta_barrier` |
| `store` | npz + JSON-record storage (`save_run`, `load_run`, `save_object`, `load_object`) |
| `theory`, `extrapolation`, `walls` | growth laws, the three larger-N scenarios, exact-diagonalization and tensor-network walls |
| `mps` | the same model as a matrix product state (TeNPy): MPO, DMRG start, TDVP, bond entropies and Schmidt ranks at any N (optional import) |
| `tntables` | the tables of the tensor-network section as csv + npz |
| `quantum` | qubit and T-gate counts, the largest affordable mass |
| `cost` | what a simulation costs before you start it |
| `fullgrid` | N = 2 on the full grid (no mode truncation) |
| `montecarlo`, `onebody` | sign-problem severity; one-particle checks with known answers |
| `parameter_study` | Appendix A: one-particle splitting against the barrier, N = 2 on the 28x18 grid with one input changed at a time (`python -m ljtunnel.parameter_study`; `parameter_map` parses its ledger) |
| `followup_spectra`, `followup_growth` | the follow-up analysis (Stage 1 spectra of the start; Stage 5a growth law): see `../followup/README.md` and `../data/followup/` |
| `log` | timestamped console log that also goes to a file (`../data/logs/`) |

Import with `import ljtunnel as lj` (from the repository root); `ModeN`, `Model`, `load_config` are re-exported.
Needs numpy and scipy (Python >= 3.11); `mps` and `tntables.channel_ranks/step_cost` also need `physics-tenpy`.
Tests: `../tests/`.
