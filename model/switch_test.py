import sys, json, numpy as np
from multiprocessing import Pool
from run_macro import build
def run(kw):
    m = build(seed=kw.pop("seed", 1), i_star=kw.pop("i_star", 0.0075))
    for t in range(300): m.step(record=False)
    Lf = m.z[m.age < m.hp.J_W].sum()
    m.lnq[:] = np.log(0.8); m.q_bel = 0.8; m.Q_bel = 0.03; m.theta = 0.5
    for k, v in kw.items(): setattr(m.mp, k, v)
    out = []
    try:
        for t in range(200):
            r = m.step()
            if t % 10 == 0: out.append('%3d u %.3f pi %+.1f i %.1f th %.2f c %.2f vac %.3f fill %.2f W %.2f' % (t, r['u'], 400*r['pi'], 400*r['i'], r['theta'], r['contact'], r['vac']/Lf, r['fill'], r['W']))
    except Exception as e:
        out.append('ERR ' + repr(e)[:60])
    return kw, out
cfgs = [json.loads(a) for a in sys.argv[1:]]
with Pool(4) as p:
    for kw, o in p.map(run, cfgs):
        print(kw); [print('   ', x) for x in o]
