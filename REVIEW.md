# Code review of `ljtunnel` (accuracy, performance, the tensor-network part)

Scope (of the original review, 2026-10-04..06): every module in `ljtunnel/` except `onebody.py` and `legacy.py` (not read; `legacy.py` was removed on 2026-10-09), `run.py`, `config.toml`, the tests, the stored results under
`data/`, and the framework / walls / open-problems sections of the paper, read in full; the tests run; the numerical claims below re-derived
or measured here.  "Evidence" says how each finding was established.  Severity
is about the effect on the paper's numbers, not about code style.

## Status on 2026-10-09 (read this first; the sections below are the original review and are kept as written)

What happened to each finding since the review.  "Report" means `paper/main.pdf`.

| id | status now |
|---|---|
| T1 (saturating fit ill-conditioned) | mitigated: N = 6 is measured and enters the fit (first failing N 13 / 15 / 16 on the grid for pess / sat / opt); the report states the walls as a band, not the saturating value alone, and Section 5 says the walls price the static entanglement of the isomers |
| T2 (mixed protocols) | **fixed**: every point of every table is a uniform run (`data/runs/`); the legacy fallback and the legacy data were removed on 2026-10-09 and the numbers did not change |
| T3 (`chi = 2^S` against the Schmidt rank) | measured: chi(1e-3) / 2^S is 3.7, 4.1, 7.5, 11.3, 34.0 for N = 2..6 at k = 12; the report charges it as a second wall (N = 7 on the grid) |
| T4, T5 (step cost, dense MPO) | measured (Section 6.2, Table 4, Fig. 13); the counting model's prefactor is a lower bound |
| T6 (physical caps of the pessimistic scenario) | open (the walls lie before the violation) |
| T7 (k = 12 entropies charged at d = 20, 1200) | open, stated in the report (Limitations) |
| T8 (window not converged for N = 2, 5) | open, stated: maxima at t = 163, 89, 114, 197, 193 for N = 2..6; window 12 transfer times; the report calls every maximum a lower bound |
| E1, E2 | fixed (with tests) |
| E3 (`Theta(x = 0)`) | fixed: weight 1/2, everything rerun; effect <= 0.011 bit on the central entropy |
| E4 (`big_cut_stride`) | open; matters only for N >= 6 runs with a stride (none exist) |
| P1, P2 | open (recommendations) |

New since the review:

* **F1 (the main finding).**  The follow-up analysis (`data/followup/`, report Section 5) shows that for N <= 5 the correlated start lies on two eigenstates plus a dressing at 1-3 J, an exact y-reflection symmetry removes the other levels, and the
  entropy of the evolved state is mostly static (N = 5: 2.20 bit off resonance against 2.72 bit on resonance).  The tensor-network walls of T1-T3 are therefore walls for the static entanglement of the isomers, not for the dynamics, and the classical-quantum window is not established.
  This is a finding about what the review's counting model prices, not an arithmetic error in it.
* **F2.**  The "best pair" of the doublet test (smallest splitting that passes the gates, `resonance.pick_doublet`) is not the pair that carries the weight at N >= 4; the report quotes both (the combined Table `tab:start` in Appendix D) and uses the two heaviest levels in Section 5.
* **F3.**  Text claims that the data did not support were removed or reworded during the report rewrite (for example "the frequency found is 2J" at N = 5, k = 20: the fitted ratio is 1.81, not 2).  `make_numbers.py` now checks 43 hand-written claims against the data.
* **F4.**  Not verified: Gaussianity of the dressing; any spectrum at N >= 6; MPS beyond N = 5.  Bibliography entries still unchecked are listed in `STATUS.md`.
* **Recommended steps below:** 1 is done (derived tables regenerated from the uniform N = 5 and N = 6 runs); 2 is still open and is also the way to the open questions of the report; 3 and 4 are open; 5 is done.

## Verdict

1. **The exact side (ED + Chebyshev + entropy bookkeeping) is sound.**  An independent implementation (a matrix product state in
   TeNPy, new `ljtunnel/mps.py`) reproduces the exact series snapshot by snapshot: every bond, single and pair entropy to
   2e-5 (N=3, k=6), 7e-4 (N=4, k=6), 2e-3 (N=5, k=6) bits, with identical central-bond Schmidt ranks; and at the paper's k=12 (N=4,
   chi capped at 64) S_cen to 0.017 bit.  That checks `ModeN`, `project_pair`, the Chebyshev step,
   the start-state preparation and `CutPlan` against code that shares none of them.
2. **The tensor-network "wall" is not buggy arithmetic.**  I re-derived the first failing N (14 and 23 for the saturating
   scenario) from the cost formulas independently and got the same numbers.  What is fragile is what feeds it (findings T1-T3).
3. **There was no tensor-network simulation anywhere in the repository**, only a counting model fed with entropies of exact
   states.  That is the real gap; `ljtunnel/mps.py` closes it for the sizes this machine and TeNPy can reach, and measures the
   quantity the model only assumes (the bond dimension).
4. Where a tensor network was *measured* here, the counting model is optimistic about cost per step (T4, T5) and about the rank
   (T3; the paper's parenthesised walls already charge part of it): both push the wall to smaller N than the model says.
5. Four small defects in the exact code (E1-E4); E1 and E2 are fixed here, with tests; none changes a published number.
6. **What this did not do:** produce an MPS result beyond what exact diagonalization already reaches.  The module is validated (N <= 5 at
   k=6, N=4 at k=12), but the N=6-8 runs at k=12 that would settle the tensor-network wall cost hours to days per N on this 4-core container
   (T4), so they are left for a bigger machine; the numbers I could measure are in "What was measured" below.

## Findings

| id | what | severity | status |
|---|---|---|---|
| T1 | the saturating-entropy extrapolation is ill-conditioned: 2 parameters, 3 points | high (the band between opt and pess is robust) | open |
| T2 | the N=5 point comes from the earlier protocol (T=120, 45 snapshots) and is mixed with N=2-4 (T=200, spacing 1) | medium (measured: S1 +0.02 bit, saturating wall -1) | uniform run now in `data/runs/`; derived tables not yet regenerated |
| T3 | the cost charges `chi = 2^S`; measured Schmidt ranks are 4-11x larger and the ratio grows with N | medium | open; now measurable (`mps`) |
| T4 | the flop model counts one SVD-sized operation per bond update; a TDVP update is ~10^2 x that by operation count; measured ~10^3 x at N=4, k=12, dominated by the chi-independent MPO contraction | medium | open; measured |
| T5 | the Hamiltonian is all-to-all and contact-like: its MPO needs R+2 = 58-174 states at k=12 (N-independent), not counted | medium | measured |
| T6 | the pessimistic scenario ignores `S1 <= log2 d` and the cut's Hilbert-space cap | low | open |
| T7 | entropies measured at k=12 are charged at d=20 and d=1200 | low (stated in the paper) | open |
| T8 | the maxima over T=200 have not converged for N=2 (peak at t=163) and N=5 (t=197); `n_edge` does not flag this | medium | open |
| E1 | `resonance.locate` returns `None` when `<O>_gs` jumps over the 0.2-0.8 band between two scan points (N=4, k=6) | low | **fixed** + test |
| E2 | `CutPlan` skips a contiguous bond silently (N=8 at k=8 or 12), `summarize` then picks a wrong "central" bond | low-medium | **fixed** + test |
| E3 | `Theta(x=0)` is assigned to the left; the grid is not symmetric; `<Theta>` of a symmetric ground state is 0.496 | low | fixed: `[model] theta_barrier = 0.5`, everything rerun (`rerun_all.sh`) |
| E4 | `big_cut_stride > 1` makes the maxima of big cuts and of bonds use different snapshot sets | low | open (by reading) |
| P1 | `ModeN.apply_t` is 1.5x faster with real GEMMs on the real/imag parts | perf | open (recommendation) |
| P2 | `model._CACHE` is keyed by `id(g)` and not by the parameters | hygiene | open (could not trigger it) |

### Tensor-network part

**T1. The saturating extrapolation is ill-conditioned (high).**  `extrapolation.fit_sat` fits `S_inf` and `tau` through
S1(2) with the three points N=3,4,5.  The last increment is 0.11 bit against 0.41 and 0.43 before it, and everything rests on it.
Moving S1(5) alone (everything else fixed), the first failing N on the grid / in the 20-mode basis changes as

| dS1(5) | -0.15 | -0.05 | 0 | +0.05 | +0.15 | +0.30 |
|---|---|---|---|---|---|---|
| saturating, grid / modes | 16 / 28 | 15 / 25 | **14 / 23** | 12 / 21 | 11 / 16 | 9 / 12 |
| fitted `S_inf` (bits) | 1.90 | 2.14 | **2.33** | 2.59 | 3.66 | 20 (the bound) |

so a 0.05-bit change moves the wall by two particles (the actual N=5 correction, +0.02 bit, moved it by one: T2), and +0.3 bit sends the fit to
the upper bound of its parameter range.  The "opt" and "pess" walls move less (by 2 particles over +-0.15 bit, against 5 for "sat"),
so the statement that the soft edge lies between them is the safe one; the "sat" value is not a measurement.  It also feeds the
existence argument: the asymptotic slopes quoted there (0.58 and 0.46 bit per particle) are S_inf/4 and S1(5)/4, so S_inf between 1.9
and 3.7 spans slopes of 0.47-0.92, while the paper's "marginal at N_hi = 24 in the mode basis" rests on s_crit = 0.58 being level
with the fitted slope.  *Evidence:* `extrapolation.scenarios` re-run with S1(5) shifted.  *Fix:* measure N=6 and 7 (`run_mps`), or
report the saturating wall and slope as ranges.

**T2. Mixed protocols (medium, measured).**  N=2-4 are uniform runs (T=200, spacing 1.0, 201 snapshots); the N=5 point of the derived tables is
`N5_k12_int_c2.920_T120_legacy.npz` (T=120, 45 snapshots, spacing 2.7).  A uniform N=5 run has since been added to `data/runs/` (commit eb732d9,
21 min on your machine); `data/derived/` (ledger, graph, larger_N) has not been regenerated from it yet.  I had run the same case independently
on this container (105 min, CPU shared with another job): all 201 x 15 entropies agree with yours to 1.3e-10, <O> to 6e-11, and the Schmidt ranks
exactly, so the run is reproducible across machines (its log is in `data/mps/runlogs/`).  From `data/derived/tn_tables/window_convergence.csv`:

| window | S_cen | at t | S1mean | chi(1e-3) | chi(1e-6) |
|---|---|---|---|---|---|
| legacy (T=120, spacing 2.7) | 2.6898 | | 1.8574 | 69 | |
| uniform, T <= 60 | 2.6333 | 60 | 1.8700 | 68 | 143 |
| uniform, T <= 120 | 2.6854 | 63 | 1.8732 | 70 | 143 |
| uniform, T <= 200 | 2.7077 | 197 | 1.8790 | 71 | 143 |

At equal window (T <= 120) the legacy central bond is right (2.690 against 2.685); the coarse spacing costs 0.016 bit in the one-particle entropies
(1.857 against 1.873); the window to 200 adds 0.022 more in S_cen.  Feeding the uniform point to the extrapolation moves the saturating walls
14 -> 13 (grid) and 23 -> 22 (20 modes), the fitted `S_inf` 2.33 -> 2.43 and the sat slope 0.58 -> 0.61; the opt and pess walls (17/28, 10/13) and
the saturating Schmidt-rank wall (8) does not change.  So the legacy point was a small error, as T1's table predicts for +0.02 bit (13 / 22), and the band between
opt and pess stands.  chi(1e-6) = 143 of at most 144 = k^2: at k = 12 the central cut of N=5 is already full rank at 1e-6.
Running `run_records(); run_extrapolation()` carries the uniform point into the tables and changes them as above (`wall_sensitivity.csv`).

**T3. `chi = 2^S` against the numerical Schmidt rank (medium).**  chi(1e-3) / 2^S_cen from the stored runs:

| N | 2 | 3 | 4 | 5 (legacy) |
|---|---|---|---|---|
| chi(1e-3) | 7 | 9 | 37 | 69 (uniform run: 71) |
| 2^S_cen | 1.87 | 2.45 | 5.01 | 6.45 |
| ratio | 3.7 | 3.7 | 7.4 | 10.7 (uniform: 10.9) |

At k = 16 (the uniform N=5 run in `data/runs/`) chi(1e-3) = 104 is 17 times 2^S and chi(1e-6) = 250 of at most 256 = k^2 (`rank_vs_entropy.csv`):
the ranks rise with k, as the paper says, and the ratio with it.  The model applies one constant shift (+3.42 bits, from N=5, k=12) to every larger N,
but the ratio is rising.  An exact-representation
bound (`chi >= 2^S`) is a lower bound on cost, and the data say how far above it a truncated MPS sits; the right input is the
rank a truncated evolution needs, which `ljtunnel.mps` now measures directly (`summarize`: `chi3_cen`, `chi6_cen`, `chi_kept`).

**T4. Cost per step (medium).**  The model charges `sweeps * N * 2 (d chi)^3`, one SVD-sized operation per bond update.  A TDVP bond
update runs ~20 Lanczos matvecs of the effective Hamiltonian instead (counted: 100 two-site and 80 one-site matvecs per step at N=4), each
~2 chi^2 d^3 (R+2)^2 with the dense MPO (TeNPy) or ~3 R chi^3 d^2 in the channel form (T5): about 30 R/d ~ 10^2 times the model's count per
update for this model's own d and R, even with the best contraction.  That is the argument for large chi.  What I measured is the small-chi
regime (TeNPy 1.1.1, 4 idle cores, `data/derived/tn_tables/step_cost.csv`): at N=4, k=12 one TDVP step takes **10.5, 12.1, 13.6, 14.5 s at chi =
16, 32, 48, 64**, nearly flat in chi, because 85-87% of it is the H_eff matvecs, which are dominated by the dense MPO (D = R+2 = 70) and not by
chi (two-site matvec 81 -> 103 ms).  The model's count at this machine's measured zgemm rate (246 GFlop/s) is 0.2 ms at chi=16 and 15 ms at
chi=64, so the measured step is ~10^3 times it at chi=64 and ~5x10^4 at chi=16; at k=6 (R=32) the step takes 0.32 / 0.38 s at chi = 16 / 32.  So the
model's chi^3 is not what a TDVP step costs at these sizes, and the measured factor is not transferable to the large-chi regime, which I did not
reach.  For the operation-count factor `f` ~ 10^2 the wall moves by `log2(f) / (3 s)` particles (paper, Eq. Nflop): 6.6 bits, 2.2 / s, i.e. 4-5
particles at s ~ 0.5; the least certain number in this review.

**T5. The all-to-all, contact-like interaction (medium).**  `V_ij = (P_i x P_j)^T VP (P_i x P_j)` is a sum over channels
`e_s F_i^s F_j^s` with the *same* channel index for every pair, so the MPO has bond dimension R+2 independent of N.  But the capped
Lennard-Jones core makes the kernel nearly flat (eigenvalues 300 at s=0, 290 at s=20, 250 at s=80, 210 at s=120 of 180 channels),
so R stays large: the span of mode products is 78 at k=12 (21 at k=6) and the slightly different bases of the particles add to it.  Measured
R (relative cut `channel_tol` of the product span) and the worst pair-matrix error it causes, N=4:

| tol | 1e-10 | 1e-6 | 1e-4 | 1e-3 |
|---|---|---|---|---|
| k=12: R / error | 172 / 1e-12 | 100 / 1e-7 | 68 / 2e-5 | 56 / 2e-4 |
| k=6: R / error | 76 / 1e-12 | 49 / 7e-8 | 35 / 1e-5 | 28 / 1e-4 |

The default (1e-4) leaves errors far below any splitting in the paper (dE >= 0.05).  The model has no term for this factor.

**T6. Physical caps (low).**  `scenarios['pess']` lets `S1` grow past `log2 d` (from N=15 at d=20, N=36 at d=1200) and the central cut
past `min(a, N-a) log2 d` (N=29, 73), so the plotted curves and the stored slope (`larger_N['slope']['pess']` = 7.3 bit per
particle; not quoted in the text) are unphysical beyond those N.  The walls themselves (13, 10) lie before the violation, so they are unaffected.

**T7. k=12 entropies at d=20 and d=1200 (low, documented).**  The paper says entropies fall and ranks rise with k.  At N=3, S_cen
is 1.29 bits at k=12 and 1.73 at k=6 (same window, own resonance tilt): a 35% effect, so the mode-basis wall at `d_mode=20`
is not a statement about k=12 data.

**T8. The window (medium).**  A maximum over [0, T] is a lower bound, and at T=200 it has not converged for N=2 and N=5: S_cen peaks at
t=163 (N=2) and t=197 (N=5), 82% and 98.5% of the window; for N=5 it is 2.633, 2.685, 2.708 at T <= 60, 120, 200 (+0.075 over the last 140), the same
size as the 0.05-bit shifts of T1 that move the saturating wall by two particles.  (N=4: 2.257 at t=26, 2.324 at t=114, flat after.)  `n_edge`, the
number of cuts that peak at the last snapshot, is 0 for N=2-5 and does not flag a peak at t=197.  Relative to the first switching (first minimum of
`<O>(t)`: t = 19, 17, 12, 9 for N=2-5) the window is 10.5, 11.8, 16.7, 22.2 times as long, so it is not short at larger N; the slow creep is
multi-frequency dynamics.  I do not use `pi/dE` for this: at N=4 and 5 there is no clean doublet and its `dE` comes from fallback levels (60.7 and 20.7).
A window at which the maximum has stopped moving (or an extrapolation of it) would remove the 0.05-0.1 bit uncertainty.

### Exact code

**E1. `resonance.locate`, fixed.**  The scan accepts the first tilt with `0.2 < <O> < 0.8`.  At k=6, N=4 the ground-state occupation
falls 0.81 -> 0.15 between c=2.4 and 2.5, so no grid point qualifies and the function returned `None` ("no resonance") although a
resonance exists at c=2.45.  Now it falls back to the `<O> = 1/2` crossing between two scan points, only when the old rule found
nothing, so existing results are unchanged.  Test: `test_resonance_scan_finds_a_transition_that_jumps_over_the_grid`.

**E2. `CutPlan` skipping a contiguous bond, fixed.**  The config comment says singles and contiguous bonds "are always
measured", but a bond whose smaller side exceeds `max_cut_dim` (k^a > 3000) was skipped without a message.  At N=8 the central bond
(a=4: 8^4 = 4096 at k=8, 12^4 at k=12) is exactly that; `summarize` then compared NaN bonds and picked one.  In practice the
memory guard stopped N=8 at k=12 but not at k=8.  Now `CutPlan.skipped_bonds` + a logged warning, and `summarize` raises
`ValueError` if a contiguous bond is unmeasured.  Test: `test_skipped_contiguous_bond_is_reported_and_summarize_refuses`.

**E3. `Theta(x)` convention.**  `DR = (x > 0)` puts the grid column x=0 (the barrier top, where it is a site) on the left, and the
grid `[-Lx, Lx-dx]` is not symmetric.  For the symmetric ground state of the untilted trap `<Theta> = 0.49615` (0.77% of the weight sits
on that column), and the penalty of the correlated start treats the two valleys differently by the same amount.  The effect on
the resonance (`<O>` near 1/2) is ~0.004 per particle; on entropies it is not tested.  A weight 1/2 on the barrier column (or a
cell-centred grid) removes it, but it changes every stored number, so it is a decision for you, not a patch.

*Resolution (2026-10-07).*  Measured first (`data/studies/theta_convention/`, k = 12, N = 2-4): the exact resonance tilt moves by +0.0014 / +0.0022 / +0.0037, the splitting dE and
the product-start entropies do not change (H has no Theta), <O>_gs at a fixed tilt moves by +0.006..+0.013, the central-bond entropy of the correlated start by 0.002-0.011 bit
at the same tilt, the Schmidt rank chi(1e-3) by at most 1-2, and the times of the maxima not at all.  Since it touches every number the decision was to rerun everything:
`[model] theta_barrier` (weight of the x = 0 column; 0 reproduces the old runs bit for bit) is 0.5, the [tilts] are the old grid picks (frozen: no pick changes), and
`bash rerun_all.sh` regenerated every simulation and table (record: `data/rerun/summary.txt`; the old results: `previous-data/`).  The measured difference, run by run
(`data/derived/theta_effect.csv`, `run_theta_effect()`): over the 16 correlated-start runs of N = 2..6 (k = 12..32) the central entropy changes by at most 0.011 bit
(N = 6: 3.249 -> 3.258, N = 5: 2.708 -> 2.715), the largest cut entropy by at most 0.013 bit, <O>_gs at the fixed tilt by +0.006 .. +0.019, chi(1e-3) of the central
bond by at most 13 (N = 6: 312 -> 325); dE and every product-start entropy are identical (checked: `run_theta_effect` raises otherwise).  A convention, not physics:
with the same data (N <= 5) the first failing N of every scenario is identical in the two conventions (grid / modes: saturating 13 / 22, optimistic 17 / 28, pessimistic 10 / 13);
what moves the walls is the new N = 6 point (15 / 25, 16 / 27, 13 / 18).  Data left in the old convention on purpose: `previous-data/legacy` (the earlier code's results).
The full-grid `t0 = pi/dE` at the exact <O>_gs = 1/2 resonance (`data/derived/fullgrid_resonance.npz`, c* = 0.4633) is 16.37 against the 16.52 of [quantum] `t_star_0` (the earlier
study's own number, not the exact resonance: dE rises 0.002 per 0.013 in c); the paper quotes t_star_0 unchanged.

**E4. `big_cut_stride`.**  With a stride > 1 the cuts with >= 3 particles on a side are measured on every n-th snapshot while the
bonds are measured on every one, so `S_perp` (a minimum over cuts) and `S_max` (a maximum) use fewer snapshots than `S_nat`.  Found by
reading `CutPlan.measure`/`analysis.graph_record`; not measured.  It matters only for N >= 6 runs with a stride.

**P1. `apply_t`.**  Applying the real pair matrices to the real and imaginary parts as one real GEMM (view the complex state as a real
array with a trailing axis of 2) takes 0.246 -> 0.164 s at N=5, k=12 and agrees to 9e-14.  Not applied: it changes the summation
order of the reference data path and needs a branch for real input; recommended.

**P2. `model._CACHE`** keys on `id(g)`, so a garbage-collected model whose grid dict address is reused could serve another model's
modes.  I could not trigger it in 200 tries; keying on `(nx, ny, Lx, Ly, sigma_smear, A, omega_y, ...)` removes the question.

Checked and found correct (read, plus tests where they exist): `project_pair` (against einsum), `ModeN.apply_t` (against dense),
`cheb_step` (against the eigenbasis, unitarity, step halving; the Lanczos spectral bounds sit inside the spectrum but the
padding and the final norm check cover it), `spectrum_stats` (S, S_2 and chi(eps) definitions), `CutPlan` completeness and
symmetry, `analysis.summarize/maxima/graph_record`, `graph.*`, the wall formulas of `theory`, the first-failing-N convention (the
code and the figures use "first failing N", the proposition's `N_ED = floor(...)` is the last feasible one: consistent), the lemma
checks, `fullgrid.apply` and `quantum.*` (the leading-order model with `g_N = 1` is stated in the paper).

## What the new tensor-network module does (`ljtunnel/mps.py`, TeNPy)

The same Hamiltonian, tilts, per-particle mode bases and start state as `ModeN`, as a matrix product state in the particle
order `0..N-1` (the bonds are the prefix cuts of `analysis`):

* `MPSProblem` builds the MPO from the channel decomposition above (bond dimension R+2, N-independent) and reports the error of the
  dropped channels; `product_mps`, `penalty` as in `starts`.
* `correlated_start`: two-site DMRG of `H + W * penalty`, repeated until the **energy variance** is below `var_tol`
  (the energy alone converges long before the state: at c=2.4, N=4, k=6 the energy was converged to 1e-5 with an overlap deficit of
  8e-3; variance 4e-7 <-> deficit 9e-4, variance 1e-9 <-> 3e-8).
* `run_series`: two-site TDVP at step `dt`, snapshots every `dt_snap`; per snapshot `<O>`, `<Theta_i>`, the entropy of every bond,
  single and (optionally) pair, the Schmidt ranks chi(1e-2, 1e-3, 1e-6) of the bonds, bond dimensions, truncation error, energy.
  Written to `data/mps/` in the layout of the exact runs, protocol kind `mps`.
* `locate_tilt`: bisection for the tilt with `<O>_gs = 1/2` by DMRG, for N beyond what `resonance.locate` can do.  Its building block
  `gs_occupation` agrees with ED to ~1e-4 where converged (N=3, 4 at k=6 and 12); the bisection itself was not run end-to-end here (its cost at N >= 6 is hours).
* Tests: `python tests/test_mps.py` (17 s): MPO against the dense Hamiltonian, DMRG against the exact start, TDVP against Chebyshev,
  pair entropies against dense reduced density matrices, truncation and the stored layout.

Run it with `run_mps(N=5, k=6, T=40, chi_max=100)` and `run_mps_summary()` (see README).

## What was measured with it (this machine: 4 cores, TeNPy 1.1.1, stock engines)

**Against the exact code** (same tilt, same correlated start, same window; `data/mps/` and `data/mps/ed_ref/`; k = 6 because the exact
reference is cheap there; the tilt is the exact `<O>_gs = 1/2` point):

| N | c | T | S_cen MPS / exact | max dS over every bond, single, pair | chi(1e-3) | chi(1e-6) | chi kept | MPS wall | exact wall |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 1.621 | 60 | 1.7326 / 1.7326 | 1.8e-5 | 5 / 5 | 6 / 6 | 6 | 14 s | 0.4 s |
| 4 | 2.455 | 60 | 2.4366 / 2.4371 | 7.3e-4 | 18 / 18 | 31 / 31 | 34 | 88 s | 1.1 s |
| 5 | 3.164 | 40 | 3.9489 / 3.9486 | 1.7e-3 | 32 / 32 | 36 / 36 | 36 | 505 s | 5.4 s |

(<O>(t) agrees to 1e-4 or better.)  The residuals are consistent with the accuracy of the start state (DMRG variance 2e-7) and the dropped
channels rather than with the TDVP step: with a deliberately unconverged start (c = 2.4, N = 4, k = 6) the error was 0.095 bit and
changed by only 5e-3 between dt = 1, 0.5 and 0.25, and at c = 2.26 with a converged start (variance 1e-9) the same comparison gives 8e-5 at dt = 0.5.
The MPS is 30-100x *slower* than the exact code at these sizes: a Hilbert space of 7776 amplitudes does not need a tensor network.
It also shows why k=6 only validates the machinery: the central cut of N=5 holds 3.95 of at most 5.17 bits (S1 mean 2.30), so the basis
is saturated, and the entropies fall with k (N=3: 1.73 at k=6, 1.29 at k=12).

**At the paper's k = 12** (N=4, c=2.26, T <= 40, chi <= 64, eps = 1e-6, dt = 0.5, same start; the exact run of that window as reference):

| | S_cen | time of max | chi(1e-3) | chi(1e-6) | max dS, every bond / single / pair | <O> |
|---|---|---|---|---|---|---|
| exact | 2.2566 | 26 | 34 | 115 | | |
| MPS, chi <= 64 | 2.2396 | 26 | 31 | 64 (cap) | 0.022 | 2e-3 |

The entropy of the capped MPS is within 1% (0.017 bit), the central rank at 1e-3 is resolved to 3, and the rank at 1e-6 is not (the cap binds:
chi_kept = 64 < 115; cumulative discarded weight 5.4e-3, energy drift 0.023).  That is the behaviour the model should expect from a truncated
representation: S_cen converges at modest chi, the rank at a tight tolerance does not, so a rank is only a measurement where `chi_kept < chi_max`
(`run_mps_summary` prints it).  The start state (DMRG capped at 64, variance 4e-7) is part of the 0.02.  Cost: ~27-30 s per snapshot (two
TDVP steps; 13-15 s per step) and ~7 s per DMRG sweep (91 sweeps to variance 4e-7), against 30 s for the whole exact run: ~70x slower at N=4
(the wall time of this run, 35 min, includes periods when the CPU was shared).


## Recommended next steps, in order

1. Regenerate the derived tables from the uniform N=5 runs now in `data/runs/` (`run_records(); run_protocol_tables(); run_extrapolation()`): the legacy
   point was off by 0.02 bit and the saturating walls move 14 -> 13 and 23 -> 22 (T2).  Then run N=6 (README).
2. Use `run_mps` at k=12, N=5-8 on a machine with more cores/time than this one, and replace `chi = 2^S` by the measured rank (T3), the
   flop model by the measured step cost (T4), and the entropy law by the measured `S_nat(N)` (T1).  The cost is dominated by
   TeNPy's dense-W effective Hamiltonian; the profile (T4) says where an optimisation would pay (a chunked sum over channels:
   ~3 R chi^3 k^2 per matvec instead of ~2 chi^2 k^3 (R+2)^2; my straightforward version was exact but only 1.4x faster).
3. Check that the maximum has stopped moving within the window, or extrapolate it, and report the time of the maximum with every S_max (T8).
4. Cap the pessimistic scenario at the Hilbert-space bounds (T6) and report the saturating wall as a range (T1).
5. ~~Decide on E3~~ done: weight 1/2, rerun.
