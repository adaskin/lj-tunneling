# tests/ -- checks of the machinery against dense references

```bash
python tests/test_numerics.py          # about 35 s: exact machinery (no TeNPy needed)
python tests/test_numerics.py --slow   # + the full-grid N = 2 reference (about a minute)
python tests/test_mps.py               # about 20 s, needs TeNPy (skipped otherwise)
python tests/lemma_checks.py           # the lemmas of the paper that are numerical statements
```

`test_numerics.py`: pair projection against `einsum`, matvec and lowest levels against dense matrices, Chebyshev propagation against the
eigenbasis (unitarity, step halving at the production step), entropy bookkeeping, completeness and symmetry of the cut plan, graph cuts,
the npz round trip, the start-state overlaps and splittings for BOTH `theta_barrier = 0` (the earlier ledgers) and `0.5` (current), the
`Theta(x)` convention itself, the convention guard of `records.collect` and the off-resonance control directory, checkpoint/restart of a
series (a run killed at snapshot 12 and continued equals the uninterrupted one), the analysis window, and the tables of the
tensor-network section against the stored runs (this one reads `../data/` and needs a finished rerun).

`test_mps.py`: the MPO against the dense Hamiltonian, the DMRG start against the exact correlated start, TDVP against the Chebyshev
series, pair entropies against dense reduced density matrices, truncation and the stored layout.

`rerun_all.sh` runs both suites before it changes anything and `test_numerics.py` again at the end.
