"""make_notebook.py -- build notebooks/run.ipynb from run.py.

run.py is the source of truth: every `# %%` marker starts a cell.  Rerun this file after editing run.py:

    python notebooks/make_notebook.py

(The notebook JSON is written by hand: no nbformat needed.)
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE.parent / 'run.py').read_text()

parts = re.split(r'^# %%(.*)$', src, flags=re.M)      # [head, title1, body1, title2, body2, ...]
head, rest = parts[0], parts[1:]


def cell(kind, text):
    lines = text.strip('\n').splitlines(keepends=True)
    c = {'cell_type': kind, 'metadata': {}, 'source': lines}
    if kind == 'code':
        c.update(execution_count=None, outputs=[])
    return c


doc = head.strip().strip('"').strip()
cells = [cell('markdown', "# ljtunnel: the run file as a notebook\n\n"
                          "Generated from `run.py` (one cell per `# %%` block).  Edit `main()` in the last cell to choose what to run: "
                          "comment lines in or out.  All constants are in `config.toml`; results are written to `data/`.\n\n"
                          "```\n" + doc + "\n```")]
for title, body in zip(rest[::2], rest[1::2]):
    t = title.strip(' -')
    cells.append(cell('markdown', f"### {t}"))
    cells.append(cell('code', body))
nb = {'cells': cells,
      'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python'}},
      'nbformat': 4, 'nbformat_minor': 5}
out = HERE / 'run.ipynb'
out.write_text(json.dumps(nb, indent=1))
print('wrote', out, f'({len(cells)} cells)')
