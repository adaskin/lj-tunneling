# data/studies/onebody_mc/ -- logs of the one-body and Monte Carlo checks

Two text files that the paper scripts read (`RAW` in `paper/scripts/make_numbers.py` and `figs.py`).  They are inputs, not outputs of the current rerun:
they come from the first version of the code (the scripts `tier1.py`, `measure_gamma_inf.py`; removed from the repository on 2026-10-09, last present in commit 497189c),
and `rerun_all.sh` does not touch them (the Hamiltonian is not involved, so `theta_barrier` does not matter).

| file | what | read by |
|---|---|---|
| `mc_gamma_ledger.txt` | the sign-problem ledger of the absolute-value path estimator in one dimension: `ginf M dx Gamma_inf(M)` for M = 64..4096 and `p_UV`; Gamma_inf grows as M^2 (exponent 2.00) | macros `Ginf128`, `p_UV`; Fig. `fig_mc` (Appendix C) |
| `tier1_checks.log` | log of the one-particle and Monte Carlo checks: exact splitting against the instanton formula (A = 1, 8), the splitting from a 20000-step evolution against exact diagonalization (relative error 1.7e-4), the box lemma, the plateau identity and the M^2 growth of Gamma_inf | table `tab_checks` (Appendix C) |

The same quantities are computed by the current code (`ljtunnel.montecarlo.gamma_inf`, `ljtunnel.onebody`; `run_montecarlo()`, `run_onebody_checks()` in `run.py`, results in `../../derived/`); these files are
kept because the paper quotes their numbers.
