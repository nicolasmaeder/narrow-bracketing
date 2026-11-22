import sys, json, numpy as np
from multiprocessing import Pool
from run_macro import build
def run(kw):
    m = build(**kw); Lf = m.z[m.age < m.hp.J_W].sum(); R = []
    try:
        for t in range(450): R.append(m.step())
    except Exception as e:
        return kw, "FAILED " + repr(e)[:60]
    R = R[150:]; k = lambda n: np.array([r[n] for r in R])
    emp = (1 - k('u')) * Lf
    return kw, 'shortfall %.3f | Ld/Lf %.3f | u %.1f sd %.2f | pi %.1f | i %.1f | ws %.2f | wdisp %.3f | fill %.3f' % (
        ((k('Ld_pre') - emp) / emp).mean(), (k('Ld_pre') / Lf).mean(), 100*k('u').mean(), 100*k('u').std(), 400*k('pi').mean(), 400*k('i').mean(), k('wage_share').mean(), k('wage_disp').mean(), k('fill').mean())
with Pool(4) as p:
    for kw, o in p.map(run, [json.loads(a) for a in sys.argv[1:]]): print(kw, o, flush=True)
