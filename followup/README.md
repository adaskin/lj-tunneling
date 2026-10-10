# followup/ -- the follow-up analysis of the prepared state (October 2026)

After the bounds-first report was reviewed, we asked what the correlated start is made of in the eigenbasis of H and how many cosines the signal <O>(t) needs.
The answer changed the reading of the whole report (`paper/`, Section 5 and Appendix D).  This directory holds the plans and the scripts of that
analysis; its **data** are in `../data/followup/`, its library code in `../ljtunnel/followup_spectra.py` and `followup_growth.py`.

| file | what |
|---|---|
| `TASK_followup_rival_and_mps.md` | the original brief (v1): Stage 1 spectra of the start, the nearest classical rival, MPS growth law.  Superseded where the v2 brief disagrees |
| `TASK_followup_v2_observable_first.md` | the revised brief (v2): establish the complexity of the observable before extrapolating resource limits; says which v1 stages are kept, retired or demoted |
| `freq_signal_analysis.py` | Hankel singular values and greedy multi-cosine fits of the stored signals `../data/runs/N*_int_*.npz`; writes `../data/followup/freq_signal_analysis.log` (26 s) |
| `static_checks_scaling_args.py` | static checks (no propagation): pair-interaction energy, levels with reflection parity, dressing bins, untilted band; writes `../data/followup/static_checks/` (about 2 min) |
| `report_draft_reduction_section.tex` | the first draft of the conditional-reduction section, kept for the record; the text in the report is `../paper/sec4b_static.tex` |

Run from the repository root (`python followup/<script>.py`).  The Stage 1 spectra themselves are produced by `run_followup_stage1()` in `../run.py`
(`python -c "from run import *; run_followup_stage1()"`; hours for the largest cases) and the pre-registered criteria are in `../data/followup/PREREG.md`.
The report reads the results through `../paper/scripts/followup_data.py`; every `fu_*` number in the text is a macro computed from the saved files.

What is not here: a converged spectrum at N = 6, a test of the Gaussianity of the dressing (the Huang-Rhys prediction of B2), and the matrix-product-state
growth law beyond N = 5.  These are the open items of the report (Section 8) and of `../STATUS.md`.
