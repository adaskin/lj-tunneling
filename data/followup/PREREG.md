# Pre-registration: Stage 1 (spectral structure of the start) and Stage 5a (growth law at k = 6)

Written 2026-10-08, before any Stage 1 or Stage 5a result exists, on branch `followup-rival-tests`.  The criteria are those of
`followup/TASK_followup_rival_and_mps.md`; this file adds the implementation choices the brief leaves open.  Later changes are dated amendments at the
end of the file; nothing above them is edited.  The only things run before this commit were timings of one `H v` product (no spectra, no results).

Scope of this pass: Stage 1 and Stage 5a, one job at a time.  Stages 1b, 2, 3 are not run; Stages 4 and 5b are not run without the user's approval.

## Machine and limits

16 cores, 90 GB RAM (about 75 GB free), numpy 2.2.6, scipy 1.16.3, physics-tenpy 1.1.1.  The brief's limit "6 GB" was written for a 7 GB container; on this machine
it is read as: ask the user before a job whose projected peak memory is above 35 GB or whose projected wall time is above 4 h.

## Stage 1: definitions

* Model, `config.toml`, `theta_barrier = 0.5`, `W_penalty = 30`, grid and `sigma_smear` as stored; nothing in `config.toml` or in `data/runs`, `data/derived` is changed.
* Levels: the lowest L eigenpairs of `ModeN` (real arithmetic, matrix-free H), by `scipy.sparse.linalg.eigsh` (`which = 'SA'`, thick-restart ARPACK,
  `ncv = 2L + 1`, tol 1e-9, a fixed random start vector) or, when k^N <= 8000, by dense `eigh`.  L starts at 40.
* Coverage = sum of the weights of the correlated start on the L levels.  If coverage < 0.99, L is doubled (40, 80, 160) while `ncv * k^N * 8 B` <= 30 GB.
  If the cap is reached with coverage < 0.99 the case is reported as **unresolved** with its coverage; its B2 is then only bracketed:
  B2_found <= B2 <= B2_found + (1 - coverage).  A case counts as "star holds" only if the lower bound passes, as "breaks" only if the upper bound fails.
* Start state: the valley assignment `{N-1}1_1` (particle 0 in the right valley, the others left), the one every stored run uses.  Correlated start =
  `starts.interacting_state` with W = 30; product start = `starts.product_state` for the same assignment.
* Weights are the squared overlaps with the eigenvectors; levels closer than 1e-6 (energy units) are merged into one level with the summed weight (the weight of a
  degenerate multiplet is basis independent).  Levels are counted after merging; the multiplicity is stored.
* B2 = weight of the two heaviest levels; (a, b) = that pair with energy order a < b; omega* = E_b - E_a; omega*/sqrt(N); n95 = smallest number of levels (heaviest
  first) holding 95 % of the start; D = largest weight of any level with index strictly between a and b in energy order; "between" = those levels' energies minus the
  pair's midpoint, and their number (the star predicts N - 1 of them).
* Tilt: the stored `[tilts]` value for the reproduction cases (a); for every other case the tilt c at which the exact ground state has <O>_gs = 1/2, by bracketing
  (steps of 0.1 from the stored or extrapolated tilt) and `scipy.optimize.brentq` on <O>_gs(c) - 1/2 (xtol 1e-4, result rounded to 4 decimals; <O>_gs falls with c).
  This is the same target as `resonance.exact_tilt`; it is checked against the three stored k = 6 values (N = 3, 4, 5: 1.622, 2.456, 3.170) to 5e-4.
* Reproduction (rule 3): for every case of (a) the stored `meta['diag']['weights_interacting']` (n_report levels, unmerged energy order) must agree with the new
  weights on the same levels to 1e-3 (max abs difference); the stored dE of the doublet is reported next to omega*.

Cases (each at its own tilt; N = number of particles, k = modes per particle):

* (a) N = 2, 3, 4, 5 at k = 12, 16, 20 and N = 6 at k = 12, stored tilts (reproduction and extension of the weights to L >= 40).
* (b) N = 4 and 5 at k = 24, 28, 32 (the state vector is <= 0.6 GB; the cost is time; the order is N = 4 first, then N = 5 by increasing k, and a case whose
  projected time is above 4 h is skipped and reported to the user).
* (c) N = 6 at k = 14 and 16, and at 20 if the memory rule allows.
* (d) calibration: k = 6 and 8 at N = 3, 4, 5, then k = 6 and 8 at N = 6, 7, 8.
* (e) untilted: c = 0 at N = 3, 4, 5, k = 12, in two variants: the configured model (particle i still carries the offset i*C0 = 0.15 i, so the wells are not exactly
  symmetric) and C0 = 0 (exactly symmetric wells; an in-memory copy of the configuration, `config.toml` unchanged).  Reported: the levels E_j - E_0 for the lowest 64,
  the lowest band = the levels below the first large gap, where the first large gap is the first gap g_j with g_j > 3 max(g_0 .. g_{j-1}) (j >= 1), its
  size next to N + 1 and 2^N, and the weight of the all-left product state (N, 0) on the band.  The rule is exploratory; the whole level list is saved so the
  band can be re-read by eye (an amendment, not an edit, records any change of the rule).

## Stage 1: criteria (verbatim from the brief)

* "few-level / star holds at (N, k)" if B2 >= 0.80 and D <= 0.02; "breaks" if B2 < 0.60 or n95 > 6.  A case that is neither is "intermediate".
* Star-frequency test: omega*/sqrt(N) constant to within 10 % over N = 3..6 at the best-converged k (the largest k computed for that N; max/min - 1 <= 0.10).
* k-convergence at N = 4 and N = 5: omega* and B2 change by <= 5 % (relative) between the two largest k computed.
* GATE 1: if the star holds at N = 6 for the two best-converged k and the k-convergence holds at N = 4, 5: "single-crossing task is few-level through N = 6", continue to
  Stage 2 (not run in this pass).  If it breaks at N = 6 in converged k: report it and STOP the rival track (go to Stage 5).  The untilted result (e) is reported either way.

## Stage 5a: definitions

* MPS (`ljtunnel.mps`, `run_mps_case` style): k = 6, T = 60, TDVP step 0.5, snapshot spacing 1.0, chi_max 120-200, eps 1e-8, pairs off, the correlated start of
  the stored cases (`start_key` `{N-1}1_1`), tilt from <O>_gs = 1/2 at k = 6 (stored for N = 3, 4, 5; computed as above for N = 6, 7, 8).  Cases N = 6, 7, 8 are new;
  N = 3, 4, 5 are the stored ones in `data/mps` (chi_max 120, T = 60) and are not rerun except the timing check described next.
* Reproduction (rule 3): the stored N = 3 (17 s) case is rerun with the new entry point; S_cen(t) and <O>(t) must agree with the stored run to 1e-6 bit / 1e-6.
* Calibration of the short window: exact runs at k = 6 (uniform protocol except T) with T = 60 and T = 200 for N = 3, 4, 5 (cuts: bonds only, written to
  `data/followup/stage5a/exact_k6/`), reporting max_t S_cen over T <= 60 and over T <= 200 and their ratio.
* Cost rule: each N is projected from the previous N (ratio of the last two wall times); if the projection exceeds 6 h the run is not started and the user is asked.
  chi_max: 120 for N = 6, raised to 160 for N = 7 and 200 for N = 8 if the previous run reached its cap (the cap is reported either way).
* Growth-law test: S_cen(N) = max over the window of the central-bond entropy, N = 2..8 at k = 6 (N = 2 from an exact k = 6 run if there is none stored), compared with the
  three laws of `ljtunnel.extrapolation` (pair, linear, sharing with the measured S1mean) fitted on N <= 5 and tested at N = 6, 7, 8.  Criterion (not in the brief,
  stated here so it can be judged): a law is "consistent" at N if its prediction is within 15 % of the measured value or within the spread
  of the measurement (S_cen at chi_max versus 0.8 chi_max, if both exist), "inconsistent" otherwise.  The test is of the shape of the growth at k = 6, T = 60.

## Amendments

1. 2026-10-08, after the k = 6 calibration cases N = 3, 4, 5 of group (d) (the only results seen), before any other Stage 1 case: the brief's "holds" (B2 >= 0.80 and D <= 0.02)
   and "breaks" (B2 < 0.60 or n95 > 6) can both be met (N = 5, k = 6: B2 = 0.889, D = 0.005, but a long thin tail gives n95 = 9).  Rule: the verdict is `holds` if the
   "holds" condition is met and n95 <= 6, `holds+tail` if it is met and n95 > 6 is certain, `breaks` if only a "breaks" condition is met, else `intermediate` (or `unresolved`
   when coverage < 0.99).  Gate 1 counts `holds` and `holds+tail` as "the star holds" (the brief's first definition) and always reports the tail (n95, top-six weight)
   next to it.  Also: for k^N <= 8000 the dense diagonalization gives all levels, so L = k^N and the coverage is 1 there.

2. 2026-10-08, after the cases N = 2, 3, 4 (k = 12, 16, 20) of group (a) and the first two level-count steps of N = 5, k = 12, before any other N >= 5 result: (i) the
   lowest-L-levels method was run to L = 320 for N = 4 (more than the 160 written above; it only adds levels) and is capped at L = 160 for N >= 5, because the coverage
   creeps up slowly (0.95 -> 0.97 at N = 5, k = 12: the rest of the start sits in a thin excited continuum) while the cost per doubling is hours at N = 5, k = 20.
   (ii) For every case whose coverage is below 0.99 at the cap, the spectral measure of the start is computed as well (`followup_spectra.krylov_case`: Lanczos from the start
   state without reorthogonalization, m = 500, 1000, 2000 until B2 and omega* move by < 1e-4; ghost copies merged by the 1e-6 rule; a node counts as a resolved level if its
   Ritz residual is < 1e-6, and the coverage is the weight on resolved nodes).  The table and Gate 1 use, for each (N, k), the row with the larger resolved coverage; both rows
   are kept and printed.  The two methods are checked against each other at N = 4, k = 12, 16, 20 (omega* to 1e-4 relative, B2 to 1e-3) before the Krylov rows are used for N >= 5.

3. 2026-10-09, after groups (d6) and (a) finished (Gate 1 quantities not yet evaluated; N = 5 k = 20 and N = 6 k = 12 seen): (i) at N = 5 the level-count and the spectral-measure rows
   agree to 1e-11 in B2 and omega* (k = 16, 20), so for N = 5 at k = 24, 28, 32 and for N = 8 at k = 8 only the spectral measure is computed (one Lanczos run of 2000 steps, interacting
   start only); the level-count method would take hours to days there (N = 5 k = 20: 3.2 h).  At N = 6 the spectral measure alone is NOT enough (k = 12: resolved coverage 0.44 against
   0.94 from 160 levels), so N = 6 cases use the level count with L capped at 80 (memory/time) plus the spectral measure.  (ii) Group (c) is split: N = 6, k = 14 is run now (projected 2-3 h); N = 6 at
   k = 16 and k = 20 are projected above 4 h each and wait for the user's approval, as rule 4 of the brief requires.  (iii) The product start is not computed for N >= 6 or for spectral-
   measure-only cases (the criteria concern the correlated start).  (iv) The spectral measure now comes from one Lanczos run whose leading blocks give m = 500, 1000, 2000.

4. 2026-10-09, during group (d): the calibration cases N = 7 and N = 8 at k = 8 are not run in this pass.  The resonance tilt at N = 7, k = 8 lies far above the k = 6 value (4.0189;
   <O>_gs is still 0.93 at c = 4.03 and 0.91 at c = 4.13) and one ground-state evaluation takes 13 min, so the tilt search alone is several hours for a calibration case.  Stage 5a needs only k = 6
   (done: N = 3..7; N = 8 follows), and the calibration question (does k = 6 reproduce the star pattern of k = 16, 20?) is answered by N = 3, 4, 5 at k = 6 and 8 and N = 6 at k = 6, 8.
   The group is kept in `run.py` as 'd78x'.
