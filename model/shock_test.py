import copy, json, sys, numpy as np
from multiprocessing import Pool
from run_macro import build
KW = json.loads(sys.argv[1]); H = 16
def run(seed):
    m = build(seed=seed, **KW)
    for t in range(300): m.step(record=False)
    N = m.hp.N; chk = np.full(N, 0.04 * m.GDP_hist[-1] / N)
    out = {}
    for name, f in [("check", lambda k: dict(extra_y=chk) if k == 0 else {}), ("rate", lambda k: dict(i_shock=0.0025 * 0.8 ** k))]:
        a, b = copy.deepcopy(m), copy.deepcopy(m)
        ra = [a.step() for k in range(H)]; rb = [b.step(**f(k)) for k in range(H)]
        out[name] = dict(u=[100 * (y["u"] - x["u"]) for x, y in zip(ra, rb)], Y=[100 * (y["Y"] / x["Y"] - 1) for x, y in zip(ra, rb)],
                         pi=[400 * (y["pi"] - x["pi"]) for x, y in zip(ra, rb)])
    return out
with Pool(4) as p: res = p.map(run, [1, 2, 3, 4])
for sh in ("check", "rate"):
    for k in ("u", "Y", "pi"):
        print(sh, k, np.round(np.mean([r[sh][k] for r in res], 0), 2))
