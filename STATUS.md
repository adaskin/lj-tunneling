# Status (what is done, what is running, what is open)

Last updated 2026-10-09.  Keep this file short: one line per item, change the status when something moves, and say who has it.
Nothing is running.  Orientation for a new reader: `README.md` (top), then `paper/main.pdf` (cover page).

## In one paragraph

The simulations of the uniform protocol (T = 200, spacing 1.0, every bipartition, theta_barrier = 0.5) are complete, and the follow-up analysis of the start state (Stage 1 spectra, signal fits,
static checks) is complete for N <= 5.  The paper is a research report (37 pages; restructured in October 2026: follow-up merged into Appendix D, numbers moved from text into tables and figures, imaginary-time Monte Carlo highlighted) whose conclusion is negative: the bounds are right, the measured entropies are right,
but the classical-quantum window they draw is **not established as a statement about the dynamics**, because for N <= 5 the start is two eigenstates plus a dressing and most of the entropy is
static.  What would settle the question is listed under "Open" below.  No legacy code or data remain in the repository (removed 2026-10-09; the history has them).

## Done

| item | where |
|---|---|
| exact runs: N = 2, 3, 4 at k = 12, 16, 20 (N = 3 up to k = 32, product + correlated at k = 12), N = 5 at k = 12, 16, 20, N = 6 at k = 12; off-resonance control N = 5 (c = 2.62); rerun with the weight-1/2 convention (2026-10-06/07) | `data/runs/` (19 files), `data/controls/`, record in `data/rerun/summary.txt` |
| MPS validation (4 cases, TDVP step 0.5), snapshot-spacing study, parameter scans, Appendix-A ledger, full-grid N = 2 | `data/mps/`, `data/studies/`, `data/derived/` |
| follow-up: levels and weights of the start for N = 2..5 (N = 4 to k = 32, N = 5 to k = 24 from a Lanczos list), reflection parity and sector checks at N = 3, 4, signal fits (cosine counts 3, 3, 5, 8 for N = 2..5), untilted band, control run | `data/followup/` (README there), `followup/` |
| report restructure (PR #16): follow-up merged into Appendix D with one combined table; number lists moved into tables; imaginary-time Monte Carlo box in Section 6.3 and Fig. 1 caption | `paper/`, PR `report-restructure` |
| number-list sweep: remaining non-concluding number lists removed from text (N = 16, 24, 40, 88 lists, per-N lists, window and sampling counts); Section 6.3 rewritten (real-time Monte Carlo in a few sentences, imaginary-time Monte Carlo as the focus); cover no longer states Monte Carlo failure | `paper/` |
| report rewrite: cover page, Section 5 (spectrum of the start), claims fixed, figures instead of number lists, max-flow/min-cut citations, AI-usage statement | `paper/`, PR #14 merged (commit 497189c) |
| repository organisation: README in every directory, legacy removed, STATUS/REVIEW updated, report consistency pass (Fig. 1 relabelled, caption gives the width of the tensor-network band) | PR from branch `repo-organize` |
| numbers: `make_numbers.py` writes `numbers.tex` and the tables and confirms 43 hand-written claims; both test suites pass (the TeNPy tests need `pip install physics-tenpy`) | `paper/scripts/`, `tests/` |

## Open (in the order that would settle the question)

| item | what it would show | cost |
|---|---|---|
| Gaussianity of the dressing: the harmonic (Huang-Rhys) prediction of B2 from the two constrained isomers against the measured B2 at k >= 16 | whether the dressing is a spin-boson problem (heuristically easy) or not | static, no propagation; not run |
| a converged spectrum at N = 6, 7 (and the basis convergence of B2 at N = 5: 0.812 at k = 20, 0.836 at k = 24) | whether M(N), the number of cosines, stays small | hours to days per case; not run |
| MPS beyond N = 5 at k = 12 (and the growth law of the entropy) | whether the entropy growth falls below the critical s | hours to days per N on a 4-core machine; not run |
| Is A = 8 the right barrier at larger N? (Appendix A: N = 4 at k = 12 has no clean doublet at A = 8, 0.81 overlap at A = 10) | would change the working point and every run | not scanned with propagation |
| cap, sigma, epsilon, omega_y, C0 at N > 2 | only tested at N = 2 | static scans possible |
| where the regime ends at fixed trap depth (one-body estimate N = 10-11) | whether the tilted transfer exists at the sizes of the old window | needs N >= 7 spectra |

## Decisions pending

| item | what it changes |
|---|---|
| `[quantum] t_star_0` = 16.5212 (the earlier full-grid study's own number) against t0 = 16.37 at the exact `<O>_gs = 1/2` resonance (`data/derived/fullgrid_resonance.npz`) | the quoted switching time (16.5 or 16.4); the report keeps 16.52 and says so (Appendix C); dE depends steeply on c |

## Notes

* Every log and data file is tracked; only `data/ckpt/` (checkpoints of a running job) and `data/rerun/state/` (scheduler scratch) are ignored.
* Layout: `README.md` (top) and the README of every subdirectory.  Other documents: `REVIEW.md` (code review, findings T1-T8, E1-E4, status of each), `paper/README.md`, `followup/README.md`.
* Conventions: PRs go to `main` and the maintainer merges; every paper number is a `make_numbers.py` macro; no slides in the repository; the AI-usage statement on the cover is kept up to date.
