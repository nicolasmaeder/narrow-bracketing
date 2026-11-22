import sys, json, numpy as np
from multiprocessing import Pool
from run_macro import build
def run(kw):
    m = build()
    for t in range(300): m.step(record=False)
    for k, v in kw.items(): setattr(m.mp, k, v)
    R = [m.step() for t in range(300)][40:]
    u = np.array([r['u'] for r in R]); x = u - u.mean()
    f = np.abs(np.fft.rfft(x)) ** 2; fr = np.fft.rfftfreq(len(x)); k = f[1:].argmax() + 1
    pi = 400 * np.array([r['pi'] for r in R])
    return kw, 'u %.1f sd %.2f | pi %.1f sd %.1f | peak period %.0f q (power share %.2f)' % (100*u.mean(), 100*u.std(), pi.mean(), pi.std(), 1/fr[k], f[k]/f[1:].sum())
with Pool(4) as p:
    for kw, o in p.map(run, [json.loads(a) for a in sys.argv[1:]]): print(kw, o, flush=True)
