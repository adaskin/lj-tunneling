# notebooks/

`run.ipynb` is `../run.py` split at its `# %%` markers (one cell per marker), for Jupyter.  `run.py` is the source of truth:
regenerate the notebook after editing it with

```bash
python notebooks/make_notebook.py
```

`make_notebook.py` writes the notebook JSON by hand (no `nbformat` needed).  The notebooks of the first code version were removed on 2026-10-09 (git history, commit 497189c).
