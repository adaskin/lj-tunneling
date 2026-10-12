# ljtunnel

DASKIN, A. (2026). Quantum simulation window for real-time tunneling in light Lennard-Jones clusters: resource bounds, and why the classical–quantum window is not established. Zenodo. [https://doi.org/10.5281/zenodo.23281107](https://doi.org/10.5281/zenodo.23281107)  

Real-time tunneling of N distinguishable particles in a 2D double well with a capped Lennard-Jones interaction, and what it costs to simulate it classically
(exact diagonalization, tensor networks, Monte Carlo) or on a quantum computer.  The code, the data and the research report built from them live here.

**Status and open items:** [STATUS.md](STATUS.md) (one page).  **Code review and findings:** [REVIEW.md](REVIEW.md).  **The report:** [paper/main.pdf](paper/main.pdf) (~34 pages; the first page is a cover with the summary).

## What this repository shows, in five lines

1. Bounds first: exact diagonalization fails by a count at N = 4 on the position grid, the real-time absolute-value Monte Carlo estimator by an identity, a tensor network pays at least the cut entropy, and a quantum computer is priced by a leading-order gate count (`ljtunnel.theory`, `walls`, `quantum`).
2. Measured: the entropy of every bipartition of the evolved state for N = 2..6 (exact propagation, `data/runs/`), and a correlation graph that reproduces it.
3. Read at face value, the bounds open a classical-quantum window; **a follow-up analysis does not support that window as a statement about the dynamics**: for N <= 5 the start lies on two eigenstates plus a dressing, an exact symmetry removes the other levels, and most of the entropy is static (`data/followup/`, report Section 5 and Appendix D).
4. Open: whether the cost of the static dressed-transfer spectrum grows exponentially with N; no hardness proof exists or can exist for a fixed family.
5. Every number in the report is a macro computed from the stored data; the pipeline is reproducible end to end (`rerun_all.sh`).

## Where to start

| if you want to ... | read / run |
|---|---|
| understand the result | `paper/main.pdf`: cover page, then Sections 4 (measurements) and 5 (spectrum of the start) |
| know what is done and what is open | `STATUS.md`, then `REVIEW.md` (findings T1-T8, E1-E4) |
| find the data behind a number | `\num{key}` in the text -> `paper/numbers.tex` (the comment on each line names the source file) -> `data/` (README in every directory) |
| rerun one simulation | `run.py` (`from run import *`; `run_dynamics(N=4)` takes about half a minute, N = 6 hours); the table below |
| rerun everything | `bash rerun_all.sh --dry-run`, then `--yes` (about 45 core-hours) |
| rebuild the report from the stored data | `python -I paper/scripts/make_numbers.py && python -I paper/scripts/figs.py && (cd paper && latexmk -pdf main.tex)` |
| check the machinery | `python tests/test_numerics.py` (35 s), `python tests/test_mps.py` (needs TeNPy) |

## How the pieces fit

```
config.toml --> run.py (run_* functions) --> data/runs, controls, mps, studies  (npz, one per simulation)
                         |                          |
                         |                          +--> data/derived  (ledger, graph, larger_N, quantum, tn_tables, ...)
                         |                                      |
followup/*.py, run_followup_stage1 --> data/followup             |
                         |                                      v
                         +----------------------> paper/scripts/make_numbers.py, figs.py --> paper/numbers.tex, tab_*.tex, figures/ --> paper/main.pdf
```

## Layout

```
config.toml          every constant: model (incl. theta_barrier), grid, basis, the measurement protocol, tilts, budgets, studies
run.py               the run file: one function per simulation/analysis, a main() to pick what runs; `python run.py jobs` lists the simulations of a rerun
rerun_all.sh         one command that reruns every simulation and table in parallel (below)
ljtunnel/            the library, one module per concept (below; README there)
tests/               exact machinery and the MPS module against dense references (README there)
notebooks/           run.ipynb (generated from run.py by make_notebook.py), one cell per `# %%` block
data/                the results of the CURRENT setting (theta_barrier = 0.5), README in every subdirectory:
  runs/                the exact runs of the uniform protocol, one npz per (N, k, start)
  controls/            the off-resonance control
  mps/                 matrix-product-state (TeNPy) validation runs; ed_ref/ = their exact references
  studies/             snapshot-spacing study, the Appendix-A parameter ledger, the Theta-convention study, the one-body / Monte Carlo check logs
  derived/             tables built from the runs (ledger, graph, larger_N, protocol tables, tn_tables/, theta_effect.csv)
  followup/            the follow-up analysis of the start state: spectra, signal fits, static checks (Section 5 and Appendix D of the report)
  logs/                the simulations' own text logs
  rerun/               the record of the rerun: one log per job and step, summary.txt
followup/            the plans (briefs) and scripts of the follow-up analysis (README there)
previous-data/       everything made before the rerun (theta_barrier = 0): runs, derived, mps, studies, logs; read only by the Theta-convention comparison
paper/               the report: LaTeX source, and the scripts that turn data/ into numbers.tex, tables and figures (README there)
```


## Running

Three ways, same code.

* **Spyder / VS Code / PyCharm (interactive window):** open `run.py`; each `# %%` marker is a cell. Edit `main()`
  at the bottom (comment lines in or out) and run the file, or run a single cell, or call a function at the prompt.
* **Jupyter / IPython:** `%run run.py`, or `from run import *` and call a function, or open `notebooks/run.ipynb`
  (regenerate it with `python notebooks/make_notebook.py` after editing `run.py`).
* **Terminal:** `python run.py` runs `main()`.

```python
from run import *
run_cost_table()                       # what will it cost?  (seconds)
run_dynamics(N=4, starts=('product', 'interacting'))
run_summary()                          # one line per saved run, one window for every N
run_window_scan(N=4)                   # the window dependence, from the one stored series
run_parameter_choice()                 # why A=8 etc.: the appendix tables, from the earlier one-input-at-a-time study
```

Requirements: Python >= 3.11 (tomllib), numpy, scipy; matplotlib for the paper figures; LaTeX for the paper;
`physics-tenpy` (TeNPy >= 1.0, `pip install physics-tenpy`) only for `ljtunnel.mps`.

## Library (one module per concept)

| module | what it is |
|---|---|
| `config` | `load_config`, `protocol` |
| `model` | grid, one-particle problem, mode basis, capped LJ, tilt table |
| `hamiltonian` | `ModeN`: matrix-free H on the k^N mode space, the observable, penalty operators |
| `dynamics` | Chebyshev propagation, spectral bounds, snapshot generator |
| `entanglement` | all bipartitions of the state: entropies, second-order entropy, Schmidt ranks; `CutPlan` |
| `resonance`, `starts` | the tilt where the ground state has `<O>` = 1/2; product and correlated start states with diagnostics |
| `experiments` | `dynamics_run`: resonance -> start -> real-time series -> npz |
| `analysis` | everything derived from a stored series, for any window: maxima, edge weights, ledger and graph records |
| `graph` | cut values of a weighted graph for all 2^(N-1)-1 subsets, layouts, sharing-law weights |
| `records` | the saved runs (+ the off-resonance control) -> the tables the paper reads |
| `store` | npz + JSON-record storage (`save_run`, `load_run`, `save_object`, `load_object`) |
| `theory`, `extrapolation`, `walls` | growth laws, the three larger-N scenarios, exact-diagonalization and tensor-network walls |
| `mps` | the same model as a matrix product state (TeNPy): MPO, DMRG start, TDVP, bond entropies and Schmidt ranks at any N (optional) |
| `tntables` | the tables for the tensor-network section as csv + npz: MPS against exact, window convergence, ranks against 2^S, wall sensitivity, interaction channels, measured step cost |
| `quantum` | qubit and T-gate counts, the largest affordable mass |
| `cost` | what a simulation costs before you start it |
| `fullgrid` | N = 2 on the full grid (no mode truncation) |
| `montecarlo`, `onebody` | sign-problem severity; one-particle checks with known answers |
| `parameter_study` | Appendix A: the one-input-at-a-time parameter study (one particle; N = 2 on a 28x18 grid) |
| `followup_spectra`, `followup_growth` | the follow-up analysis of the start state (see `followup/README.md`) |

## The measurement protocol (why every N is treated alike)

Earlier, the window depended on N (T = 70, 60, 60, 120, and 200 in the runs made by hand) and 45 snapshots were
spread over it, so the snapshot spacing varied from 1.4 to 4.5. A maximum over snapshots depends on both, so the
points at different N were not comparable. Now one protocol applies to everything (`[protocol]` in `config.toml`):

* window T = 200, snapshot spacing 1.0 (201 snapshots), for every N and every k;
* the full time series of every cut is stored (`S` has shape `(n_snapshots, n_cuts)`), so a shorter window is cut
  afterwards in `analysis` (`summarize(run, T_win)`), never rerun;
* the window is [0, T] (the prepared start counts).

A run with another T or spacing is marked `custom` and is never mixed with the others.

How much it matters (from the stored series; `run_protocol_tables()`; correlated start, k = 12):

| N | central bond, T <= 60 | T <= 120 | T <= 200 |
|---|---|---|---|
| 2 | 0.869 | 0.879 | 0.905 |
| 3 | 1.297 | 1.298 | 1.298 |
| 4 | 2.264 | 2.335 | 2.335 |
| 5 | 2.642 | 2.692 | 2.715 |
| 6 | 3.093 | 3.253 | 3.258 |

Snapshot spacing at T = 200: the one-particle entropy at N = 3 is 1.382, 1.367, 1.367, 1.367 for spacing 0.5, 1, 2, 4;
the central bond at N = 4 is 2.3346, 2.3346, 2.3346, 2.3265. Spacing 1.0 is within 0.02 bit of the finest spacing tried.

## Data format

Every result is one `.npz` (no pickles). Arrays are plain numpy; the parameters, the resonance, the start-state
diagnostics and the protocol are one JSON string under `meta_json`:

```python
from ljtunnel import store, analysis
run = store.load_run('data/runs/N4_k12_int_c2.260_T200_dt1.npz')
run['t'], run['O'], run['S'], run['masks']       # time, <O>(t), S_A(t) for every cut A, bitmask of A (contains particle 0)
run['bond_S2'], run['bond_chi3']                 # second-order entropy and Schmidt rank (1e-3) of the contiguous bonds
run['meta']['protocol'], run['meta']['diag']     # {'kind': 'uniform', 'T': 200, 'dt_snap': 1.0, ...}, doublet data
analysis.summarize(run, T_win=120)               # the ledger record for a shorter window
```

File names: `N{N}_k{k}_{int|prod}_c{tilt}_T{T}_dt{dt}.npz` (`int` is the correlated start, `prod` the product start).

## Why graph cuts stop at N = 7 (and the cost of everything else)

Enumerating all 2^(N-1)-1 graph cuts is not the problem: it takes under a millisecond up to N = 13
(`run_graph_cuts(14)`: 14 ms at N = 14). The cost is the edge weights w_ij = max_t I_ij(t), which need the state:
k^N amplitudes, propagated over the window. `run_cost_table()` (k = 12, T = 200, spacing 1, 30 Gflop/s):

| N | amplitudes | GB per vector | matvecs | hours | largest reduced density matrix |
|---|---|---|---|---|---|
| 5 | 2.5e5 | 0.004 | 2.2e4 | 0.29 | 144^2 |
| 6 | 3.0e6 | 0.05 | 2.8e4 | 6.7 | 1728^2 (0.05 GB) |
| 7 | 3.6e7 | 0.57 | 3.5e4 | 141 | 1728^2 |
| 8 | 4.3e8 | 6.9 | 4.3e4 | 2.8e3 | 20736^2 (6.9 GB) |
| 9 | 5.2e9 | 83 | 5.2e4 | 5e4 | 20736^2 |

(the spectral half-width is measured up to N = 5 and extrapolated linearly in the number of pairs beyond). N = 6 is a
few hours, N = 7 about a week, N = 8 and beyond is out of reach for exact propagation at k = 12. Beyond N = 7 the
weights have to come from an approximate state (a matrix-product state with time evolution), or from a model for the
weights; neither is in this code. The guards in `config.toml` (`max_state_gb`, `max_cut_dim`, `big_cut_stride`) stop a run
that would not fit before it starts.

## Matrix-product-state runs (TeNPy): the cut an MPS pays, measured

Everything above holds the full k^N state.  `ljtunnel.mps` evolves the same Hamiltonian, tilt, mode bases and correlated start as a
matrix product state in the particle order 0..N-1 (the bonds are the prefix cuts of `analysis`), with TeNPy (DMRG for the start,
two-site TDVP in real time, bond dimension truncated at a discarded weight `eps`).  It is the way to the entropy and the rank
at N where the state does not fit, and it measures the bond dimension that the counting models of `walls`/`theory` assume
(`chi = 2^S`).  It is validated against the exact runs (`tests/test_mps.py`; REVIEW.md has the numbers) and is optional
(`pip install physics-tenpy`).

```python
from run import *
run_mps(N=5, k=6, T=40, chi_max=100)      # DMRG start + TDVP -> data/mps/N5_k6_mps_c..._T40_dt1_chi100_eps1e-08.npz
run_mps(N=6, k=12, c_guess=3.5, T=60)     # tilt not in [tilts]: located by DMRG (<O>_gs = 1/2) around c_guess (illustration; not run here, hours)
run_mps_summary(T_win=40)                 # S_cen, S_nat, chi(1e-3), chi(1e-6), bond dimension kept, truncation error; next to the exact run
run_mps_check(N=4, k=6, tilt=2.455)       # exact run + MPS run of one case, largest difference over every cut (how data/mps/ was made)
run_mps_tables()                          # data/derived/tn_tables/*.csv + tn_tables.npz for the paper's figures (README there)
```

What to read from a run (`mps.summarize`): `S_nat`/`S_cen` are the same maxima over the window as for the exact runs; `chi3_cen`, `chi6_cen`
are the Schmidt ranks at discarded weight 1e-3 and 1e-6 of the central bond; `chi_kept` and `trunc_err` say what the truncated run kept and
lost.  A rank can only be read where `chi_kept < chi_max` (otherwise the cap was binding: raise `chi_max` or compare two values of `eps`).
To replace the `chi = 2^S` charge of the wall by a measured rank, pass a law for log2 chi(N) (interpolating or extrapolating the measured ranks) as
`law` to `walls.tn_wall`.

Cost.  H is all-to-all with a contact-like core, so its MPO has bond dimension R + 2 with R = 56-68 (k = 12), 28-35 (k = 6), independent
of N, and TeNPy applies it densely: ~2 chi^2 k^3 (R+2)^2 per Lanczos matvec, 85-87% of a TDVP step.  Measured (4 cores, TeNPy 1.1):
N = 4, k = 12, chi = 64: 13-15 s per TDVP step of dt = 0.5 (10.5 s at chi = 16: the dense MPO, not chi, dominates) (a snapshot every 1.0 is two steps); the DMRG start needs ~90 sweeps of ~7 s to reach variance 4e-7.
The exact run does the same N = 4 series in half a minute (T = 40), so the MPS pays off only where k^N does not fit; at k = 12 that is a cluster-sized job, at k = 6 (cheap, but different entropies:
they fall with k) it is minutes to an hour per N.  `[mps]` in `config.toml` holds chi_max, eps, dt, the channel tolerance and the DMRG settings.

## The first code version (removed)

The project began with scripts per "tier" (`tier1`: one-particle and Monte Carlo checks; `tier2`: the N = 2 study; `tier3`: entropies, walls and quantum costs) whose
windows and snapshot spacings depended on N.  Their converted results were a fallback for the points that had no uniform run.  Since all points are uniform runs and
the rerun reproduced the earlier numbers, the scripts, notebooks and converted data were removed on 2026-10-09 (git history, commit 497189c).  What the paper still reads
from them is two log files, now `data/studies/onebody_mc/`; the Appendix-A parameter study became `ljtunnel/parameter_study.py`.

Old module -> new module: tier1 -> `onebody`, `montecarlo`, `parameter_study`; tier2 -> `model`, `hamiltonian`, `dynamics`, `fullgrid`; tier3 -> `entanglement`, `walls`,
`quantum`, `extrapolation`; theory checks -> `tests/`; the paper-side scripts -> `experiments`, `analysis`, `records`.

## Rerunning everything (`rerun_all.sh`)

`bash rerun_all.sh --yes` reruns every simulation and every derived table of the paper in parallel (the `run_*` functions of `run.py`,
one process per job, longest first, never more threads than `--cores`; default: the physical cores).  Why a full rerun: `[model] theta_barrier`
is the weight of the grid column x = 0 in Theta(x), the right-valley indicator of <O> and of the penalty that defines the correlated start
(REVIEW.md E3).  The runs before the rerun counted that column as left (`theta_barrier = 0`, still available and reproducing them exactly).
The effect is small (`data/studies/theta_convention/`, and `data/derived/theta_effect.csv` after the rerun) but it touches every number.

| step | what |
|---|---|
| 0 | preflight: python >= 3.11, numpy/scipy/TeNPy/matplotlib, disk, other simulations running in the repository, both test suites |
| 1 | the old results are MOVED to `previous-data/<date>_before_rerun/` (tracked).  Nothing is deleted. |
| 2 | the simulations (`python run.py jobs` lists them): 16 groups of `data/runs/` (N = 2..6, k = 12..32, tilts from `[tilts]`, never a scan), the snapshot-spacing study, the off-resonance control (`data/controls/`), the four MPS validation cases (`data/mps/`, exact reference in `ed_ref/`, TDVP step 0.5), the parameter scans, the Appendix-A ledger (`ljtunnel/parameter_study.py`, with the new weight), the full-grid N = 2 reference |
| 3 | `run_records`, `run_protocol_tables`, `run_extrapolation`, `run_quantum`, `run_cosine_check`, `run_parameter_choice`, `run_mps_tables(step_cost=True)` (a timing measurement, alone), `run_theta_effect` (old against new) |
| 4 | `run_check_rerun`: every expected file exists, is new, has theta_barrier = 0.5, the config tilt and <O>_gs at the independent anchors; no duplicates |
| 5 | `paper/scripts/make_numbers.py` (reports hand-written claims the new data contradict), `figs.py`, LaTeX if installed |

Checkpoints: the exact runs started by `run_production` (`dynamics_run(checkpoint=True)`) write the start states and, every 5 snapshots, the state and everything measured so far to `data/ckpt/`;
a rerun of the same call continues from there, so a container restart or a power cut costs minutes, not the hours of a long run.  `--dry-run` shows the plan and the estimated schedule, `--resume` continues an interrupted rerun (finished jobs are kept), `--help` lists the rest.
Cost: about 45 core-hours; the wall time is set by two jobs, N = 6 and N = 5 at k = 20 (about 12 h each on two BLAS threads), so it is ~9-12 h on
8 or more physical cores and ~15 h on 4.  Logs: `data/logs/` (the simulations' own) and `data/rerun/joblogs/` (one file per job and step, and `summary.txt`).
The old convention stays reproducible: `theta_barrier = 0` in `config.toml` and `previous-data/`.  Only `data/ckpt/` (checkpoints of a running job, removed when it completes) and
`data/rerun/state/` (scheduler scratch) are git-ignored: every log and data file is tracked.

Data that stays in the old convention on purpose: `previous-data/` (the comparison of the two conventions) and the logs in `data/studies/onebody_mc/` (no Theta involved).

## Tests

`python tests/test_numerics.py` (about 35 s) checks the machinery against dense references: pair projection against
`einsum`, matvec and lowest eigenstates against dense matrices, Chebyshev propagation against the eigenbasis (and
unitarity and step-halving at the production step), entropy bookkeeping (Bell pair, product state, random state),
completeness and symmetry of the cut plan, graph cuts, the npz round trip, the start-state overlaps and splittings
against the earlier ledgers (N = 2, 3, 4), and the analysis window. `--slow` adds the full-grid N = 2 reference
(0.18955 at c = 0.45, about a minute).  `python tests/test_mps.py` (about 20 s, needs TeNPy) checks the matrix-product-state module the same way:
the MPO against the dense Hamiltonian, the DMRG start against the exact correlated start, TDVP against the Chebyshev series (every bond and
single), the pair entropies against dense reduced density matrices, truncation and the stored layout.
