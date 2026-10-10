# data/rerun/paper_build/ -- what the paper scripts make of these data

`numbers.tex`, `tab_*.tex` and `figures/*.png` as `paper/scripts/make_numbers.py` and `figs.py` wrote them at the end of the rerun (log: `../joblogs/paper_make_numbers.log`: 13/13 hand-written claims confirmed,
every `\num` key defined, no `\pending` marker left).  A reference copy: `paper/` itself belongs to the paper branch and was not changed by the rerun.  To rebuild there: `cd paper && python -I scripts/make_numbers.py && python -I scripts/figs.py`.

Note (2026-10-09): these files are the record of the rerun of 2026-10-07 and are not rebuilt.  Their source comments mention `data/legacy` and the `leg_*` macros of the first code
version, which were removed afterwards (git history, commit 497189c); the current `paper/numbers.tex` has no such entries and every other number is identical.
