# data/followup/ -- results of the follow-up analyses

**What "follow-up" means.**  The first version of the report (bounds first) ended with a window between a tensor-network wall and the quantum edge.  The follow-up is the analysis, run afterwards, of what the *prepared state* is made of: the low eigenlevels of H at the resonance tilt and the weight of the correlated start on each (no propagation), the exact reflection symmetry, a multi-cosine fit of the stored signals, and a control run away from resonance.  It does not support the window as a statement about the dynamics.  In the paper it is one short Section 5 (the finding and figures) and one appendix, **Appendix D** (method, one combined table `tab_start.tex` of every case, structural checks); the Monte Carlo box of Section 6.3 uses the cosine fits.

Plans and scripts: `../../followup/` (see its README).  Pre-registered criteria and dated amendments: `PREREG.md`.  Everything the report says about the spectrum of the start
(Section 5, Appendix D, the `fu_*` macros) is computed from these files by `paper/scripts/followup_data.py`.

| path | what | written by |
|---|---|---|
| `stage1/N{N}_k{k}_c{c}.npz` | Stage 1: the lowest levels of H at the resonance tilt and the weight of the correlated start on each (dense diagonalization where the dimension allows); N = 2..5, k = 6..32 (not every pair), plus `*_krylov.npz` (a second, Lanczos-only list; used only where no other file exists, e.g. N = 5, k = 24) | `run_followup_stage1()` in `run.py` (`ljtunnel/followup_spectra.py`) |
| `stage1/untilted_N*_k12_c0.0000*.npz` | the untilted band (c = 0; `_C00`: also no offsets) at N = 3, 4, 5 | the same, `untilted=True` |
| `stage1/stage1_table.csv`, `gate1.json`, `tilts.json`, `stage1.log` | the Stage 1 table, the retired gate, the tilts used, the log | `run_followup_stage1_table()` |
| `freq_signal_analysis.log` | Hankel-rank and multi-cosine analysis of the stored runs in `../runs/` (one line per run) | `python followup/freq_signal_analysis.py` |
| `static_checks/` | energies, levels with reflection parity, dressing bins, untilted band: log, the level arrays `levels_N*_k*_c*.npz` behind Fig. 8 of the report, and the discussion (README there) | `python followup/static_checks_scaling_args.py` |
| `PREREG.md` | pre-registration of Stages 1 and 5a (written before any result) with dated amendments | by hand |

The Stage 5a growth-law outputs (k = 6 matrix-product-state series for N = 6..8) were not produced; `ljtunnel/followup_growth.py` is the code for them.
The optional pickle `freq_signal_analysis.pkl` is the one the script writes when given a path; the other `.npz` files carry no pickles.

## Which data feed which object of the report

| data | report object |
|---|---|
| `stage1/N*_k*_c*.npz` | `tab_start.tex` (Appendix D, one row per N and k), `fig_spectrum`, `fig_dressing`, the `fu_B2_*` macros |
| `stage1/untilted_*.npz` | the untilted-band remark in Appendix D |
| `static_checks/levels_*.npz` and log | `fig_parity` and `tab_levels.tex` (Appendix D), parity and sector-exclusion macros `fu_oddmax_*`, `fu_marg_*`, `fu_lab*` |
| `freq_signal_analysis.log` | `fig_cosines`, `tab_mc.tex` (Section 6.3 box), cosine counts `fu_cos_*` and residuals `cr_*` |
| `../controls/N5_k12_int_c2.620_T200_dt1.npz` (run away from resonance, not in this directory) | `fig_control`, `tab_control.tex` (Section 4.3) |
| `c_top()` in `followup_data.py` (one-body estimate) | `fig_tilt`, `fu_ctop`, `fu_Ntop_*` |
