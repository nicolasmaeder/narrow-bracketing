import sys, json, numpy as np, warnings
sys.path.insert(0, __import__('os').path.dirname(__file__)); warnings.filterwarnings('ignore')
from multiprocessing import Pool
from run_macro import build
def run(kw):
    m=build(**kw); G=[]; D=[]; pc=[]
    for t in range(350):
        r=m.step()
        if t>=200: G.append(m._gain_rel.copy()); D.append(m._dp_opt.copy()); pc.append(r['price_changes'])
    G=np.concatenate(G); D=np.abs(np.concatenate(D))
    return kw, 'freq %.2f | gain/rev pctl 25/50/75/90: %s | |optimal dp| pctl 25/50/75/90: %s' % (np.mean(pc), np.round(np.percentile(G,[25,50,75,90])*100,2), np.round(np.percentile(D,[25,50,75,90])*100,1))
with Pool(2) as p:
    for kw,o in p.map(run,[json.loads(a) for a in sys.argv[1:]]): print(kw,o)
