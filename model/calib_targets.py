import sys, json, numpy as np, warnings
sys.path.insert(0, __import__('os').path.dirname(__file__)); warnings.filterwarnings('ignore')
from multiprocessing import Pool
from run_macro import build
T=int(sys.argv[1]); B=int(sys.argv[2]); base=json.loads(sys.argv[3])
def run(kw):
    kw={**base,**kw}
    try:
        m=build(**kw); R=[]; MK=[]
        for t in range(T):
            r=m.step(); R.append(r)
            if r['u']>0.9 or not np.isfinite(r['u']): return kw,'BLEW UP %d'%t
            if t>=B and t%10==0:
                L=np.bincount(m.job[m.job>=0], m.z[m.job>=0], m.mp.F)
                MK.append(np.median(m.p/m.mc_y(m.w, m.prod(L))))
    except Exception as e: return kw,'FAILED '+repr(e)[:150]
    R=R[B:]; k=lambda n: np.array([r[n] for r in R],float); G=k('GDP')
    lY=np.log(k('Y')); t=np.arange(len(lY))
    return kw, 'util %.2f | u %.1f sd %.1f | pi %.1f | i %.1f | gY %.2f | p/mc %.2f | ws %.2f | prof %.2f | K/Y %.2f | I/Y %.2f | A/GDP %.1f | pchg %.2f | fill %.2f' % (
        k('util').mean(),100*k('u').mean(),100*k('u').std(),400*k('pi').mean(),400*k('i').mean(),400*np.polyfit(t,lY,1)[0],np.mean(MK),k('wage_share').mean(),k('profit_share').mean(),
        (k('K')/k('Y')).mean()/4,(k('I')/k('Y')).mean(),(k('A')/(4*G)).mean(),k('price_changes').mean(),k('fill').mean())
with Pool(2) as p:
    for kw,o in p.imap(run,[json.loads(a) for a in sys.argv[4:]]): print({k:v for k,v in kw.items() if base.get(k)!=v} or 'base',o,flush=True)
