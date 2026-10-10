"""Probe: the uniform-protocol exact N=5, k=12 run (README 'runs still to do'), stored OUTSIDE data/runs."""
import sys, os, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/lj-tunneling')
import ljtunnel as lj
from ljtunnel import experiments
cfg = lj.load_config(); model = lj.Model(cfg)
out = sys.argv[1]
t = time.time()
paths = experiments.dynamics_run(model, 5, 12, starts=['interacting'], tilt='auto', outdir=out, force=True)
print('DONE', paths, f'{time.time()-t:.0f}s', flush=True)
