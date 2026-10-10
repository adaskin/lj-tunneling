# data/controls/ -- the off-resonance control

`N5_k12_int_c2.620_T200_dt1.npz`: the same run as `runs/N5_k12_int_c2.920_...` at a tilt (`[control] N5_off_c = 2.62`) where the ground state is
localised, i.e. the entanglement the correlated start has without tunnelling (paper, Sec. on the correlated start).

It is NOT in `runs/` on purpose: `records.pick` selects a run by (N, k) only, so a run at another tilt there would displace the resonant point.
`records.collect(controls_dir=...)` puts it into the graph table only (role `primary`, `control = True`), the ledger and the extrapolation never see it.  Written by `run_off_resonance(N=5)`.
