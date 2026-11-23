import sys, copy, numpy as np, warnings, json
sys.path.insert(0, __import__('os').path.dirname(__file__)); warnings.filterwarnings('ignore')
from multiprocessing import Pool
from run_macro import build
H=40
KW=json.loads(sys.argv[1]) if len(sys.argv)>1 else {}
def run(seed):
    m=build(seed=seed, **KW)
    for t in range(400): m.step()
    m._mrp=None; a,b=copy.deepcopy(m),copy.deepcopy(m); b.vshock=0.7
    A=[a.step() for _ in range(H)]; B=[b.step() for _ in range(H)]
    g=lambda R,k: np.array([r[k] for r in R],float)
    return dict(u=100*(g(B,'u')-g(A,'u')), nocutB=g(B,'nocut_bind'), nocutA=g(A,'nocut_bind'),
                rw=100*(np.log(g(B,'W')/g(B,'P'))-np.log(g(A,'W')/g(A,'P'))), pi=400*(g(B,'pi')-g(A,'pi')), i=400*(g(B,'i')-g(A,'i')),
                C=100*np.log(g(B,'C')/g(A,'C')), I=100*np.log(g(B,'I')/g(A,'I')), Y=100*np.log(g(B,'Y')/g(A,'Y')), P=100*np.log(g(B,'P')/g(A,'P')), laid=np.array([r.get('laid',0) for r in B],float))
with Pool(2) as p: R=p.map(run,[1,2,3,4])
print(KW)
for k in ['u','nocutB','rw','P','pi','i','C','I','Y']:
    x=np.mean([r[k] for r in R],0); print('%-7s'%k,' '.join('%+.2f'%v for v in x[0:32:2]))
