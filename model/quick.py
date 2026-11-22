import sys, json, numpy as np
from multiprocessing import Pool
from run_macro import build
def run(kw):
    try:
        m=build(**kw); R=[m.step() for t in range(500)][150:]
    except Exception as e:
        return kw, "FAILED " + repr(e)[:80]
    k=lambda n: np.array([r[n] for r in R])
    return kw, 'u %.1f sd %.2f | pi %.1f sd %.1f | i %.1f | sd dlogy %.3f | fill %.3f | ws %.2f | wdisp %.3f | inv/s %.2f' % (100*k('u').mean(),100*k('u').std(),400*k('pi').mean(),400*k('pi').std(),400*k('i').mean(),np.std(np.diff(np.log(k('y')))), k('fill').mean(), k('wage_share').mean(), k('wage_disp').mean(), k('inv_sales').mean())
with Pool(4) as p:
    for kw,o in p.map(run,[json.loads(a) for a in sys.argv[1:]]): print(kw, o, flush=True)
