# Theta(x = 0) convention: measured effect (before the full rerun)

REVIEW.md E3: `DR = (x > 0)` puts the grid column x = 0 (the barrier top) on the left valley.  This study re-runs the exact code at k = 12,
N = 2, 3, 4 (correlated start, uniform protocol, T = 200) with weight 1/2 on that column ("half") against the repo default at the time ("left"),
by replacing `model.g['DR']` after construction; nothing else differs.  Script: `theta_variant.py` (~15 min on 4 idle cores); raw numbers:
`theta_variant.json` (incl. S_cen(t) and <O>(t)); log: `theta_variant.log.txt`; table: `summary.csv` (`*_left` / `*_half` column pairs).
The exact runs it made are not kept (they were written to a scratch directory); their tee logs are in `logs/` (twelve files, N2/N3/N4_k12_20261006_14*: the
four runs per N of the script, in the order of `theta_variant.log.txt`).

| quantity | effect of the change |
|---|---|
| exact resonance tilt (<O>_gs = 1/2) | +0.0014 (N=2), +0.0022 (N=3), +0.0037 (N=4); the scan grid step is 0.06 |
| <O>_gs at the stored tilt | +0.006, +0.008, +0.010 (0.5639 -> 0.5699, 0.5122 -> 0.5201, 0.5384 -> 0.5484) |
| splitting dE, doublet verdict | unchanged (dE to 5 digits; N=4 still has no clean doublet) |
| overlap of the start with the doublet | <= 0.001 (N=4: 0.4631 -> 0.4642) |
| S_cen (T <= 200) at the same tilt | +0.0016, +0.0061, +0.0108 bit |
| S_cen (T <= 200) at each convention's own tilt | -0.003 (N=2), +0.012 (N=3), +0.015 (N=4) bit |
| chi(1e-3), chi(1e-6) of the central bond | N=3: 9 -> 10; N=4: 37 -> 38 and 116 -> 118 |
| <O>(t) | mean +0.006 .. +0.010; range [0.091, 0.997] -> [0.098, 0.999] at N=2 |
| time of the maximum of S_cen | unchanged (163, 89, 114) |

Reading: the change is at the 0.01-bit level, below the window uncertainty already in the tables (0.02-0.07 bit, REVIEW.md T1, T8), and leaves
the splittings, the doublet verdicts and the times of the maxima untouched; ranks move by at most 1 and <O>(t) by about 1%.  It is a convention,
not physics.  k = 12 only; N = 5 was not run (the preparation alone takes 15 min).
