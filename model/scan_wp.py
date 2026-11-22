import numpy as np, sys, json
from multiprocessing import Pool
from run_macro import build
def run(kw):
    m=build(wage_posting=True, **kw); R=[m.step() for t in range(1000)][300:]
    k=lambda n: np.array([r[n] for r in R])
    return kw, "u %.1f sd %.2f | pi %.1f sd %.1f | i %.1f sd %.1f | ws %.2f mk %.2f I/Y %.3f jj %.3f" % (100*k('u').mean(),100*k('u').std(),400*k('pi').mean(),400*k('pi').std(),400*k('i').mean(),400*k('i').std(),k('wage_share').mean(),k('markup').mean(),(k('I')/k('Y')).mean(), k('wage_disp').mean())
cfgs=[json.loads(a) for a in sys.argv[1:]]
with Pool(4) as p:
    for kw,o in p.map(run,cfgs): print(kw, o, flush=True)
