import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
import sys, copy, itertools, numpy as np
from multiprocessing import Pool
from household import Params, Economy
INF, RB, RA = 1.005, 0.0025, 0.0125
BASE = dict(version="sticky", psi_A=0.9, fixed_cost=0.15, entry_cost=0.05, G25=2.0)
def run(cfg):
    e = Economy(Params(**BASE, **cfg))
    for t in range(700):
        P = INF ** t; e.step(P, P, RB, RA)
    p = e.p; work = e.age < p.J_W
    draws = np.random.default_rng(7).random(p.N)
    g = 0.06 * e.yhat * work
    b0 = copy.deepcopy(e).step(INF ** e.t, INF ** e.t, RB, RA, shocks=draws, record=True)
    b1 = copy.deepcopy(e).step(INF ** e.t, INF ** e.t, RB, RA, shocks=draws, record=True, extra_y=g)
    d = (b1["spend"] - b0["spend"])[work]
    liq = (e.b / e.yhat)[work]
    bins = np.digitize(liq, np.quantile(liq, [0.2, 0.4, 0.6, 0.8]))
    q = [d[bins == k].sum() / g[work][bins == k].sum() for k in range(5)]
    yr = 25 + e.age / 4
    prof = [np.median(e.a[(yr >= A) & (yr < A + 1)] / (4 * e.yhat[(yr >= A) & (yr < A + 1)])) for A in [30, 40, 50, 60]]
    htm = np.mean(liq <= 1 / 6)
    return cfg, htm, np.median(liq), d.sum() / g[work].sum(), q, prof
cfgs = [dict(eps_B=e, psi_B=pb) for e, pb in itertools.product([0.0, 0.25, 0.5, 1.0], [0.9, 0.6])]
with Pool(2) as pool:
    for cfg, htm, med, mpc, q, prof in pool.imap(run, cfgs):
        print(cfg, f"HtM {htm:.2f} median buffer {med:.2f}q  MPC {mpc:.2f}  by liquidity {np.round(q,2)}  ret profile {np.round(prof,2)}", flush=True)
