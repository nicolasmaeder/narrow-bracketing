"""Which block makes the economy overshoot? Switch blocks off one at a time (switches apply from the shock on,
in both the baseline and the shocked run), and measure peak and reversal of unemployment after two shocks."""
import copy, json, numpy as np
from multiprocessing import Pool
from run_macro import build
H = 20
CFG = {"full model": {}, "policy rate frozen": dict(fix_policy=True), "wage path fixed": dict(fix_wage=True),
       "both frozen": dict(fix_policy=True, fix_wage=True), "profits to retirement accts": dict(div_to_fund=True),
       "slow demand learning (gain .03)": dict(g_A=0.03), "fast demand learning (gain .3)": dict(g_A=0.3)}

def run(args):
    seed, name = args
    m = build(seed=seed)
    for t in range(300): m.step(record=False)
    for k, v in CFG[name].items(): setattr(m.mp, k, v)
    N = m.hp.N; tot = 0.04 * m.GDP_hist[-1]; chk = np.full(N, tot / N)
    out = {}
    for shock, kw in [("check", lambda k: dict(extra_y=chk) if k == 0 else {}),
                      ("rate", lambda k: dict(i_shock=0.0025 * 0.8 ** k))]:
        a, b = copy.deepcopy(m), copy.deepcopy(m)
        out[shock] = [100 * (b.step(**kw(k))["u"] - a.step()["u"]) for k in range(H)]
    return name, seed, out

if __name__ == "__main__":
    jobs = [(s, n) for n in CFG for s in (1, 2)]
    with Pool(4) as p:
        res = p.map(run, jobs)
    agg = {}
    for name, seed, out in res:
        for sh, v in out.items():
            agg.setdefault(name, {}).setdefault(sh, []).append(v)
    summary = {}
    for name in CFG:
        summary[name] = {}
        for sh in ("check", "rate"):
            path = np.mean(agg[name][sh], 0)
            pk = np.argmax(np.abs(path))
            rev = -np.sign(path[pk]) * path[pk:].min() if path[pk] > 0 else path[pk:].max()
            summary[name][sh] = dict(path=path.tolist(), peak=float(path[pk]), peak_q=int(pk), reversal=float(rev))
        print(f"{name:34s} check: peak {summary[name]['check']['peak']:+.2f} (q{summary[name]['check']['peak_q']}), reversal {summary[name]['check']['reversal']:+.2f} | "
              f"rate: peak {summary[name]['rate']['peak']:+.2f} (q{summary[name]['rate']['peak_q']}), reversal {summary[name]['rate']['reversal']:+.2f}")
    json.dump(summary, open("results_diag.json", "w"))
