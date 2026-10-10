# Follow-up brief v2: establish observable complexity before extrapolating resource limits

Written 2026-10-09 for the agent working on branch `followup-rival-tests`. It REVISES `TASK_followup_rival_and_mps.md` (the "v1 brief"); it does not replace it. Where the two disagree, this file wins.
Sources: the v1 brief, the partial Stage 1 results already on this branch (`data/followup/stage1/`), two reviews of the problem's difficulty, and a third review of the plan (observable-first). Repo rules are
unchanged: no edits under `paper/`, no slides, no change to `config.toml` tilts or to `data/runs`, `data/derived`; PRs to `main` only when the user asks; commits keep the repo's AI-usage convention.

## 0. What changed and why

The Stage 1 numbers so far, plus three static diagonalizations made while checking them (N = 3 at k = 12, 16; N = 4 at k = 12; seconds each), overturned the v1 "star" reading:

* The low band is occupied on particle 0 (the start's particle, lowest tilt offset) at <n_0> ~ 0.45-0.52. The right-valley states of particles 1, 2 sit 0.3-0.4 and ~0.65 higher, i.e. the labelled states are spread by about 2 C0 = 0.3 per label, NOT degenerate. The start overlaps only the pair (all-left) <-> (particle 0 right).
* The N-1 levels "between the pair" are not other labelled states. They carry the same <n_0> as the ground state. They are **odd under the global reflection y_i -> -y_i** (checked below), so the start, which is even, cannot populate them.
* omega*/sqrt(N) = 0.136, 0.111, 0.109, 0.104 (N = 2..5, best k) was a coincidence of three points; omega* itself is 0.19-0.23. At N = 6, k = 12 the quantity is 0.178 and the v1 frequency test fails (gate1.json).
* The weight outside the two heaviest levels grows with N (k = 16: 0.02, 0.05, 0.16, 0.22 for N = 2..5). Whether it matters for the occupation signal is untested. That is the open question.

A review of the plan added: B2 and n95 are state-weight diagnostics and do not decide the difficulty of the task. The task is to predict f(t) = <O>(t) on [0, T_obs] to additive error eps*||O|| (eps = 1e-3, `config.toml` `[budgets] eps`; ||O|| = N for O = sum of Theta). **The decisive output is an error-controlled prediction of that signal together with the measured cost of obtaining it.**

Do not conclude "the quantum window survives" and do not conclude "classical methods close the window". Neither is established. Keep the original fixed-A model as the primary benchmark; any A(N) family is separate (WP7).

## 1. What happens to the v1 stages

| v1 stage | status under v2 |
|---|---|
| Stage 1 data (levels, weights, B2, n95, omega*, D, tilts, untilted band, k = 6/8 calibration) | KEEP all code and outputs. They are valid diagnostics. Finish jobs already running; do not launch new expensive diagonalizations (N = 6 at k = 14/16/20, k = 24..32) until WP1-WP2 show they are needed. |
| Stage 1 criteria: "star holds / breaks", omega*/sqrt(N) test, GATE 1 | RETIRED as decision rules. Keep printing the columns, with the label "diagnostic". Gate 1 no longer routes the project (the rival track is not stopped or continued on it). Record in PREREG amendment 5 (WP-1 below). |
| Stage 1b (T = 400, N <= 4) | KEEP (cheap). Report in terms of the signal: does f(t) change on [0, T_obs]; S_cen after t = 200 is secondary. |
| Stage 2 (imaginary-time noise on G(tau)) | REPLACED by WP5 (target = error of the predicted f(t), not of omega*). The "rival easy/hard" thresholds are retired. |
| Stage 3 (instanton J_N vs exact) | DEMOTED. Do it only after WP4 says which static quantities the effective model needs; the target is the transfer amplitude of the (all-left, particle 0 right) pair at its own resonance, not J sqrt(N). |
| Stage 4 (real imaginary-time MC) | unchanged: ASK THE USER FIRST; also needs the WP5 information requirement. |
| Stage 5a (k = 6 MPS series, N = 6, 7, 8) | KEEP, and add the observable error (WP6). |
| Stage 5b (k = 12, N = 7) | unchanged: ASK FIRST. |
| v1 "Final deliverables" options A / B / C | REPLACED by section 4 below. |

## 2. Work packages (do in this order; stop and write the result into `data/followup/REPORT.md` after each)

Compute rule (unchanged): ask the user before any single job projected above 4 h wall time or above 35 GB on the current 16-core / 90 GB machine (the PREREG machine note). WP0-WP4 need no new large jobs.

### WP-1. Pre-registration (before any WP0-WP6 result is computed)
Add PREREG amendment 5: say that the v2 criteria below were written after the Stage 1 table was seen, list what was seen (N <= 5 at k = 12, 16, 20; N = 6 at k = 12; k = 6 series; untilted band counts), and copy the criteria of section 3. Commit before running WP0.

### WP0. Reconcile the old and new "best pair" weights (cheap, first)
The paper's Table 11 (`paper/tab_doublet.tex`) reports "best pair" weights 0.34 (N = 4, k = 16) and 0.39 (N = 5, k = 16, 20); Stage 1 reports B2 = 0.845 and 0.780 at the same (N, k). Pre-checked: they are different definitions, not different simulations.
* Paper: the pair of levels that passes the doublet search (`ljtunnel/resonance.py: doublet_params / doublet_at`: lowest gap among the `n_low` = 8 lowest levels, `[doublet]` gates in `config.toml`, `n_low = 4` at N = 2), and its start weight. E.g. N = 4, k = 16: splitting 0.104, weight 0.34.
* Stage 1: the two most populated eigenstates, whatever their gap. E.g. N = 4, k = 16: omega* = 0.225, weight 0.845.
Define separately `B2_pop` (sum of the two largest eigenstate populations) and `B2_dbl` (weight of the pair chosen by the isolated-doublet rule). For every (N, k) in the paper's Table 11 and in `stage1_table.csv` write one row to `data/followup/reconcile_B2.csv` with: N, k, c, grid/cap/smearing identifiers (from `config.toml` hash), start definition (W, sides) and preparation residual, number of computed eigenpairs and their residuals, indices and energies of both selected pairs, individual weights, splittings, the rule. Explain every difference in two lines. Do not overwrite or rename old results.

### WP1. Convergence, separated into two questions; the observable is the acceptance criterion
* A, fixed Hamiltonian: at ONE common tilt c (use the k = 16 resonance of each N) increase k = 12, 16, 20 (24 where cheap) and compare f(t), the relevant energies, B2_pop, retained weight, <n_i>, central-cut Schmidt spectra.
* B, resonance: retune c*(k) with one documented definition (the existing `<O>_gs = 1/2`) and report whether c*, the signal, the splitting and the spectral composition converge. Comparisons at independently retuned tilts are NOT fixed-Hamiltonian convergence.
* Acceptance: E_f = max over 0 <= t <= T_obs of |f_k'(t) - f_k(t)| (unnormalized), against the budget in section 3. Define T_obs before computing: pi/dE_N when a clean doublet exists, otherwise the first minimum of <O>(t) (paper Section 2, Definition 1); state it per case.
* Say which target is meant: the fixed-grid Hamiltonian (first converge modes toward the grid H) or continuum LJ (then grid spacing, core smoothing and cap also need checking). Do not mix the two claims.
Deliverable: a convergence-status table: which (N, k) are trustworthy at the stated tolerance, which are not.

### WP2. Does the signal need many states? (cheap; uses stored and Stage 1 levels)
f(t) = sum_n |c_n|^2 O_nn + 2 Re sum_{m<n} c_m^* c_n O_mn exp(i (E_m - E_n) t). The objects that matter are c_m^* c_n O_mn, not |c_n|^2.
For each converged case compare the first-transfer-interval error of: (i) the selected two-state prediction; (ii) predictions keeping the M most populated eigenstates, M = 2, 3, 4, ...; (iii) predictions keeping the M dominant observable-frequency terms. Report M (or number of frequency terms) needed for E_f <= the budget. Plots: E_f vs M. A general bound for omitted probability eta is |delta f| <= 2 N sqrt(eta); it is loose, so test it against the measured error.
Incomplete spectra: if the L computed levels hold only 0.94 of the start, the energy-ordered 95 % count exceeds L and the population-ranked n95 is undetermined. Print "undetermined (coverage x)", never a number, and never treat n95 or B2 as a converged complexity measure.
Selecting exact eigenstates shows spectral compressibility only; it does not show they can be found cheaply. Report the two separately.

### WP3. Symmetry and internal states (replaces "selection rule" by a tested statement)
Preliminary result (reproduce it, do not just cite it): in the mode basis the global y-reflection operator R = prod_i (P^T R_grid P) with R_grid: grid row (ix, iy) -> (ix, (ny - iy) mod ny) (y = -Ly + iy dy; y = 0 at iy = ny/2). The correlated start has <R> = +0.9999 / +0.9997. At N = 3 (k = 16, c = 1.36) levels 1, 3, 4, 8, 10, 11 (and every other level of weight exactly 0) have <R> = -1.000, and all weighted levels have +1.000. At N = 4 (k = 12, c = 2.26) levels 1, 2, 3, 8, 10, 11 are odd (<R> = -0.999), all weighted ones even. The one-body parity closure error of the truncated modes is 0.05 (|R^2 - 1|), so eigenstate parities are near, not exactly, +-1. Therefore the "dark" levels are dark by an exact symmetry of the continuum model, and weights ~1e-5..1e-9 are numerical leakage. **Inside the even sector the next levels are not far from the pair**: at N = 4, k = 12 the even levels are 0, 0.184, 0.203, 0.210, 0.246, 0.273 and the upper bright weight is split 0.20 / 0.008 / 0.068 over the three at 0.18-0.21. That fragmentation is the dressing.
To do:
* <R> for every low eigenstate of every Stage 1 case where eigenvectors can be kept; redo B2, n95, D and the gap structure WITHIN the even sector. Mark exactly forbidden matrix elements, approximately small ones, and ones that only look small in an insufficient basis.
* For the relevant eigenstates: <n_i> per particle; where feasible P(n_R = q) or at least <n_i n_j> and the first moments of sum Theta (the mode-basis Theta_i are not exact projectors: state the truncation error); overlaps with the constrained isomer states of WP4; the O_mn matrix elements; a short list of internal excitation diagnostics (relative-coordinate vs centre-of-mass content). The y-centre-of-mass oscillator (omega = sqrt(2)) is exact in the continuum but NOT assumed exact in the finite-grid truncated implementation.
Deliverable: a symmetry- and occupation-resolved level diagram per case showing which levels contribute to the start and to O.

### WP4. Static effective model (build from static data, do not fit the movie)
Begin with the identified channel: correlated all-left state F, correlated state with particle 0 right, then add internal even-sector states and further labelled states only as required. Obtain candidates by constrained static calculations (`starts.interacting_state` with `sides`, penalty W = 30; `starts.configs` gives the assignments), document nonorthogonality and the orthogonalization used, get H and O matrix elements, and predict f(t) with NO fit to the real-time reference. Increase the effective space until E_f meets the budget. Report the diabatic ladder E_i - E_F for the labelled states (is it ~ 2 C0 per label?).
Choosing P to span exact eigenstates makes QHP = 0 by construction; that cannot be used as evidence that a cheaply built isomer subspace is weakly coupled to its complement. Report the coupling of the constrained states to the discarded even-sector states directly.
Deliverable: accuracy vs effective-space size, with the cost of constructing the effective model (diagonalizations, constrained solves).

### WP5. Imaginary-time methods, only against a concrete information requirement
First list the static quantities WP4's successful model needs (constrained energies, splitting, a few internal gaps, O matrix elements, couplings). Then, for each, ask what imaginary-time sampling would give. Keep the v1 Stage 2 code as the noise-sensitivity tool but change the target: error of the predicted f(t) on [0, T_obs] vs noise on G(tau) and on the matrix elements, not the error of omega*. Do not equate sign-free with cheap: include estimator variance, autocorrelation, projection-time dependence and required absolute precision. Positivity holds for the continuum or finite-difference kinetic operator, not for the paper's spectral-grid matrix; changing the discretization needs a demonstrated equivalence at the target accuracy. A pilot recovers a known gap or effective parameter against the existing exact reference (N <= 4); no N = 88 here. Deliverable: a validated parameter-estimation route with measured cost, or a named unresolved obstacle.

### WP6. Tensor networks against the actual observable
For selected converged reference cases (k = 6 first: exact is feasible for N <= 5): vary chi_max and truncation eps, TDVP step and local solver tolerance, and compare the error of f(t) on [0, T_obs]; record bond dimension, discarded weight, norm and energy drift, runtime and peak memory. Same start, Hamiltonian, window and error target as the reference. Profile before reading scaling: MPO/environment construction, effective-H applications, local eigensolver/exponential iterations, SVDs, other; inspect MPO compression and contraction order. Small-N MPS overhead cannot establish a large-N limit.
**Window consistency:** the entropy maxima of the paper's protocol are taken over T = 200 while the quantum price is for T_obs (about 16.5 at N = 2). From the stored series report max over t <= T_obs of S_cen and S_nat next to the T = 200 values for every stored case (cheap), and run the Stage 5a MPS cases with the observable error on [0, T_obs]. Do not price the classical method on one window and the quantum method on another.
Deliverable: cost vs achieved observable accuracy, not cost vs entropy alone.

### WP7. Fixed A and scaled A(N) are separate studies
The single-particle estimate used in review (lowest eigenstate of the tilted 1D well A(x^2 - 1)^2 + c x with majority weight on the upper side of the barrier; finite differences, 500 points on [-2.5, 2.5], hard walls; reaches the barrier top at c ~ 6.3, i.e. N ~ 9-10 if c* keeps rising by 0.6-0.7 per particle) is a rough picture with its boundary condition stated. Write down precisely what "upper-valley level" means before using it: constrained-well eigenstate, localized combination of full-system eigenstates, metastable resonance, or other. Do not infer the large-N regime from extrapolating c*(N) from unconverged small-mode calculations. For the fixed-A model determine, as far as feasible: do the isomers stay meaningful; is transfer tunneling-dominated; how does the transfer time change; does the few-state description improve or deteriorate. A(N) scaling needs its rule written in advance in a separate file and a recomputation of both classical and quantum costs; do not run it without the user's approval, and do not "amend" the original window by declaring A scaled.

## 3. Pre-registered v2 criteria (copy into PREREG amendment 5)
* Tolerance: E_f <= eps*N with eps = 1e-3 in total. Proposed split (change only by dated amendment): basis truncation <= 0.4 eps N, propagation/solver <= 0.2 eps N, resonance/tilt uncertainty <= 0.2 eps N, effective-model or other approximation <= 0.2 eps N. Agreement between two consecutive k is evidence, not a certified bound: add a further k or an independent representation where feasible.
* WP2 verdict wording: report M_needed(N) at each converged (N, k); say "M_needed grows from a to b over N = 2..N_max". No verdict about polynomial vs exponential from N <= 6.
* WP4: "static effective model sufficient" at (N, k) if E_f <= 0.5 eps N with D_eff <= N + 1 states built from constrained calculations; otherwise state which extra even-sector states were needed and their number.
* WP6: report chi (or cost) needed for E_f <= 0.25 eps N vs N; no extrapolation beyond the largest N actually run.
* Everything not listed here is labelled exploratory in the report.

## 4. Final report form (replaces the v1 options A / B / C)
`data/followup/REPORT.md`: per work package the table, the verdict against section 3, and one paragraph on what it means for the claim. Closing sections: (1) what survives of the paper's task, model and budgets and what does not; (2) a statement in one of these forms, whichever the data support: "the occupation signal is reproduced to eps*N by an effective model of D states built from static data for N <= N1; the construction cost scales as ...", or "...needs M(N) states with M growing as ... over N = 2..N1", or "inconclusive; the single measurement that decides is ..."; (3) the list of what was NOT done and why. A single-evolution gate count is a resource estimate, not an end-to-end cost, until state preparation, observable estimation and block encoding are included with one error budget.

## 5. Paper-side items are NOT part of this task
Claims in the current paper that the reviews flagged ("the firm edge", "above the curve nothing reaches the dynamics", "a finite entropy forces 1/(N-1) pair weights", the stabilizer-state monogamy sentence, product-formula expressions) are handled separately by the user and the Claude project. Do not edit `paper/`.
