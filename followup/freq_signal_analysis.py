"""Signal-frequency analysis of the stored runs (data/runs/N*_int_*.npz): Hankel singular values and greedy multi-cosine pursuit of <O>(t).

Run from the repository root (or set REPO): python followup/freq_signal_analysis.py [out.pkl].  Reads only; the figures quoted in
followup/TASK_followup_rival_and_mps.md come from it (dominant frequency and its variance share, number of cosines for a given residual).
"""
import sys, glob, os, numpy as np
WT=os.environ.get('REPO', '.')   # path of the lj-tunneling checkout
sys.path.insert(0, WT)
from ljtunnel import store

def hankel_sv(x, L):
    n = len(x); K = n - L + 1
    H = np.array([x[i:i+K] for i in range(L)])
    return np.linalg.svd(H, compute_uv=False)

def greedy(t, O, Mmax=8, wgrid=np.linspace(0.005, 3.1, 6000)):
    """greedy cosine pursuit with joint refit of amplitudes; frequencies refined one at a time"""
    sd = np.std(O); ws = []
    def design(ws):
        cols=[np.ones_like(t)]
        for w in ws: cols += [np.cos(w*t), np.sin(w*t)]
        return np.array(cols).T
    def res(ws):
        X=design(ws); c,*_=np.linalg.lstsq(X,O,rcond=None); return O-X@c, c
    out=[]
    r0 = O-O.mean()
    for m in range(Mmax):
        r,_ = res(ws)
        # scan new frequency on residual
        best=None
        for w in wgrid:
            X=np.c_[np.cos(w*t),np.sin(w*t)]
            c,*_=np.linalg.lstsq(X,r,rcond=None); e=np.sum((r-X@c)**2)
            if best is None or e<best[0]: best=(e,w)
        ws.append(best[1])
        # local refinement of all frequencies (coordinate descent)
        for _ in range(3):
            for j in range(len(ws)):
                cand=np.linspace(ws[j]-0.02, ws[j]+0.02, 81)
                es=[np.sum(res(ws[:j]+[w]+ws[j+1:])[0]**2) for w in cand]
                ws[j]=cand[int(np.argmin(es))]
        r,_=res(ws)
        out.append((m+1, float(np.sqrt(np.mean(r**2))/sd), sorted(ws)))
    return out

def analyse(path):
    run = store.load_run(path)
    t=np.asarray(run['t'],float); O=np.asarray(run['O'],float)
    sd=O.std(); x=O-O.mean()
    sv=hankel_sv(x, 100); sv=sv/sv[0]
    rank = {th:int(np.sum(sv>th)) for th in (1e-1,3e-2,1e-2,1e-3)}
    g=greedy(t,O)
    need = {}
    for thr in (0.2,0.1,0.05):
        need[thr]=next((m for m,r,_ in g if r<thr), None)
    return dict(name=os.path.basename(path), meta=run['meta'], sd=sd, rank=rank, g=g, need=need, sv=sv[:12], dE=run['meta']['diag']['dE'])

if __name__=='__main__':
    files=sorted(glob.glob(WT+'/data/runs/N*_int_*.npz'), key=lambda p:(int(os.path.basename(p)[1]),int(os.path.basename(p).split('_k')[1].split('_')[0])))
    import pickle
    res=[]
    for p in files:
        r=analyse(p); res.append(r)
        print(r['name'], 'sd=%.3f'%r['sd'], 'hankel rank(>1e-1,3e-2,1e-2,1e-3)=',list(r['rank'].values()),
              'M for resid<0.2/0.1/0.05:',list(r['need'].values()), 'resid(M=1,2,3,4)=',[round(g[1],3) for g in r['g'][:4]], flush=True)
    if len(sys.argv) > 1:
        pickle.dump(res, open(sys.argv[1], 'wb'))   # optional: python followup/freq_signal_analysis.py out.pkl
