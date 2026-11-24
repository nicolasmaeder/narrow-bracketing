"""Regression suite. Runs a fixed economy (seed 1) for 300 quarters, then three scenarios from that state (asset bust plus
credit cut, deficit surge, policy-rate shock), and records the key series every quarter.

  python3 regress.py save      store the current results as the baseline (regress_baseline.json)
  python3 regress.py           compare with the baseline

The model is deterministic given its state, so a pure refactor must reproduce the baseline exactly (max relative
difference about 1e-12). A model change shows which series moved and by how much."""
import sys, os, copy, json, warnings, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); warnings.filterwarnings('ignore')
from run_macro import build

KEYS = ['u', 'pi', 'i', 'Y', 'C', 'I', 'P', 'W', 'GDP', 'loans', 'debt', 'E_B', 'B', 'A', 'V', 'tau', 'wage_share',
        'markup', 'defaults', 'reserves']
EXTRA = lambda m: dict(TP=m.TP, yL=m.y_L, X_F=m.X_F, R=m.R_res, E_F=m.E_F, def_F=m.def_F, n_firms=float(len(m.p)))
T0, H = 300, 16
FN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'regress_baseline.json')

def rec(m, r):
    d = {k: float(r[k]) for k in KEYS}
    d.update({k: float(v) for k, v in EXTRA(m).items()})
    return d

def run():
    out = {}
    m = build(seed=1)
    out['base'] = [rec(m, m.step()) for _ in range(T0)]
    m._mrp = None
    a = copy.deepcopy(m); a.vshock = 0.7; a.cred_target = 0.5
    out['2008'] = [rec(a, a.step()) for _ in range(H)]
    b = copy.deepcopy(m); S = []
    for k in range(H):
        ex = np.full(b.hp.N, 0.15 * b.GDP_hist[-1] / b.hp.N) if k < 4 else None
        S.append(rec(b, b.step(extra_y=ex)))
    out['deficit'] = S
    c = copy.deepcopy(m)
    out['rate'] = [rec(c, c.step(i_shock=0.01 * 0.8 ** k)) for k in range(H)]
    return out

def compare(new, old):
    worst = 0.0
    for sc in old:
        rows = []
        for k in old[sc][0]:
            o = np.array([q[k] for q in old[sc]]); n = np.array([q[k] for q in new[sc]])
            d = np.max(np.abs(n - o) / (np.abs(o) + 1e-9 * (1 + np.abs(o).max())))
            worst = max(worst, d)
            if d > 1e-9:
                rows.append((d, k, o[-1], n[-1]))
        print(f'{sc:8s} ' + ('identical' if not rows else f'{len(rows)} series differ'))
        for d, k, o, n in sorted(rows, reverse=True)[:8]:
            print(f'   {k:11s} max rel diff {d:.2e}   last: {o:.4g} -> {n:.4g}')
    print('IDENTICAL' if worst < 1e-9 else f'CHANGED (max rel diff {worst:.2e})')

if __name__ == '__main__':
    res = run()
    if len(sys.argv) > 1 and sys.argv[1] == 'save':
        json.dump(res, open(FN, 'w')); print('baseline saved')
    else:
        compare(res, json.load(open(FN)))
