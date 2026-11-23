import sys,copy,json,numpy as np,warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, __import__('os').path.dirname(__file__))
from run_macro import build
from multiprocessing import Pool
KW=json.loads(sys.argv[1]) if len(sys.argv)>1 else {}
H=24
def stats(m):
    L=np.bincount(m.job[m.job>=0], m.z[m.job>=0], m.mp.F)
    y=m.prod(L)
    uc=m.unit_cost(m.w, m.P_ref()*(m.r_lr_f+m.mp.delta))
    mc=m.mc_y(m.w, np.maximum(y,1e-9))
    return dict(p=np.mean(np.log(m.p)), w=np.mean(np.log(m.w)), uc=np.mean(np.log(uc)), mc=np.mean(np.log(mc)), lnA=np.mean(m.lnA), y=np.log(y.sum()), K=np.log(m.K.sum()), L=np.log(L.sum()), pie=float(np.mean(m.anchored(m.pi_e))), inv=np.log(m.inv.sum()))
def run(seed):
    m=build(seed=seed,**KW)
    for t in range(400): m.step()
    m._mrp=None; a,b=copy.deepcopy(m),copy.deepcopy(m); b.vshock=0.8
    out=[]
    for _ in range(H):
        ra=a.step(); rb=b.step(); sa,sb=stats(a),stats(b)
        d={k:100*(sb[k]-sa[k]) for k in sa}; d['S']=100*np.log(rb['GDP']/ra['GDP']); d['C']=100*np.log(rb['C']/ra['C']); d['I']=100*np.log(rb['I']/ra['I'])
        out.append(d)
    return out
with Pool(4) as p: R=p.map(run,[1,2,3,4])
keys=list(R[0][0].keys())
for k in keys:
    print('%-4s'%k,' '.join('%+6.2f'%np.mean([r[t][k] for r in R]) for t in range(0,H,2)))
