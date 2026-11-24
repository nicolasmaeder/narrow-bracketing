import sys, copy, numpy as np, warnings, json
sys.path.insert(0, __import__('os').path.dirname(__file__)); warnings.filterwarnings('ignore')
from multiprocessing import Pool
from run_macro import build
H=60
KW=json.loads(sys.argv[1]) if len(sys.argv)>1 else {}
SIZE=float(sys.argv[2]) if len(sys.argv)>2 else 0.3
SHOCK=__import__('os').environ.get('SHOCK','asset')
def run(seed):
    import os, pickle, hashlib
    key=hashlib.md5((json.dumps(KW,sort_keys=True)+hashlib.md5(open(os.path.join(os.path.dirname(__file__),'macro.py'),'rb').read()).hexdigest()).encode()).hexdigest()[:12]
    fn=os.path.join(os.path.dirname(__file__),'cache','burn_%s_%d.pkl'%(key,seed))
    if os.path.exists(fn):                          # burned-in economy for this config, code version and seed
        m=pickle.load(open(fn,'rb'))
    else:
        m=build(seed=seed, **KW)
        for t in range(100): m.step()   # warm start (warm.py): 100 quarters settle the run
        m._mrp=None; os.makedirs(os.path.dirname(fn),exist_ok=True); pickle.dump(m,open(fn,'wb'))
    m._mrp=None; a,b=copy.deepcopy(m),copy.deepcopy(m)
    if SHOCK=='rate': pass
    elif SHOCK=='credit': b.cred_target=SIZE         # banks cut credit lines by SIZE, phased in (decays at cred_shock_rho)
    elif SHOCK=='2008': b.vshock=1-SIZE; b.cred_target=0.5   # asset bust plus a phased-in halving of credit lines
    else: b.vshock=1-SIZE                           # asset values fall by SIZE
    A=[];B=[];tA=[];tB=[]
    for _ in range(H):
        k_=len(A)
        ish=(SIZE/400.0)*0.8**k_ if SHOCK=='rate' else 0.0   # policy-rate shock: SIZE points (annual), decaying 0.8 a quarter
        A.append(a.step()); B.append(b.step(i_shock=ish) if SHOCK=='rate' else b.step())
        tA.append(np.mean(np.log(a.AL))); tB.append(np.mean(np.log(b.AL)))
    g=lambda R,k: np.array([r[k] for r in R],float)
    return dict(u=100*(g(B,'u')-g(A,'u')), nocutB=g(B,'nocut_bind'), nocutA=g(A,'nocut_bind'),
                rw=100*(np.log(g(B,'W')/g(B,'P'))-np.log(g(A,'W')/g(A,'P'))), pi=400*(g(B,'pi')-g(A,'pi')), i=400*(g(B,'i')-g(A,'i')),
                C=100*np.log(g(B,'C')/g(A,'C')), I=100*np.log(g(B,'I')/g(A,'I')), Y=100*np.log(g(B,'Y')/g(A,'Y')), P=100*np.log(g(B,'P')/g(A,'P')), tech=100*(np.array(tB)-np.array(tA)), prodY=100*(np.log(g(B,'Y')/(1-g(B,'u')))-np.log(g(A,'Y')/(1-g(A,'u')))), laid=np.array([r.get('laid',0) for r in B],float))
SEEDS=list(range(1,int(__import__("os").environ.get("NSEEDS","4"))+1))
with Pool(2) as p: R=p.map(run,SEEDS)
print(KW)
for k in ['u','nocutB','rw','P','pi','i','C','I','Y','prodY','tech']:
    x=np.mean([r[k] for r in R],0); print('%-7s'%k,' '.join('%+.2f'%v for v in x[0:60:4]))
