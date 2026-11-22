import sys, json, numpy as np
from multiprocessing import Pool
from run_macro import build
def run(kw):
    seed = kw.pop("seed", 1)
    m = build(seed=seed)
    for t in range(300): m.step(record=False)
    m.lnq[:] = np.log(0.8); m.q_bel = 0.8; m.Q_bel = 0.03; m.theta = 0.5
    for k, v in {**dict(vacancies=True, m_match=0.67, c_max=0.6, eta_L=3, kappa_h=6), **kw}.items(): setattr(m.mp, k, v)
    R = []
    try:
        for t in range(300): R.append(m.step())
    except Exception as e:
        return kw, "FAILED " + repr(e)[:60]
    R = R[60:]; k = lambda n: np.array([r[n] for r in R])
    W = k('W'); wg = 400 * np.diff(np.log(W))
    return kw, 'u %.1f sd %.1f [%.1f,%.1f] | pi %.1f sd %.1f | i %.1f | wage gr %.1f | th %.2f | fill %.2f' % (100*k('u').mean(), 100*k('u').std(), 100*k('u').min(), 100*k('u').max(), 400*k('pi').mean(), 400*k('pi').std(), 400*k('i').mean(), wg.mean(), np.median(k('theta')), k('fill').mean())
with Pool(4) as p:
    for kw, o in p.map(run, [json.loads(a) for a in sys.argv[1:]]): print(kw, o, flush=True)
