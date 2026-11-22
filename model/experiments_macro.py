"""Policy experiments in the period-by-period economy. Common random numbers: each scenario is a copy of the
same burned-in economy, so differences from the baseline are the policy's effect alone."""
import copy, json, numpy as np, sys
from multiprocessing import Pool
from run_macro import build

H = 16          # quarters shown
BURN = 400
KEYS = ["GDP", "C", "I", "Y", "u", "pi", "i", "P", "W", "tau", "debt", "B", "A", "spend_h"]


def path(m, n, **kw_by_q):
    out = []
    for k in range(n):
        kw = {key: f(k) for key, f in kw_by_q.items() if f(k) is not None}
        r = m.step(**kw)
        out.append({key: r[key] for key in KEYS})
    return out


def run_seed(seed):
    m0 = build(seed=seed)
    for t in range(BURN):
        m0.step(record=False)
    hp = m0.hp
    N = hp.N
    ra_ss = m0.rb_lr + m0.mp.erp        # fund's posted assumed return at the start; aligns the two planning rules
    gdp_q = m0.GDP_hist[-1]
    res = {}
    base = path(copy.deepcopy(m0), H)
    res["base"] = base
    # 1. stimulus check: equal amount to every household, total 1% of annual GDP, paid in quarter 0
    total = 0.01 * 4 * gdp_q
    check = np.full(N, total / N)
    res["check"] = path(copy.deepcopy(m0), H, extra_y=lambda k: check if k == 0 else None)
    # 2. withholding cut: same total, spread over 4 quarters as a lower payroll tax rate
    wb = m0.W * (m0.emp & (m0.age < hp.J_W)) @ m0.z
    cut = -(total / 4) / wb
    res["withhold"] = path(copy.deepcopy(m0), H, tax_shift=lambda k: cut if k < 4 else 0.0)
    # 3. policy-rate shock: +1 pp (annualized), decaying at 0.8 per quarter; under two planning rules
    shock = lambda k: 0.0025 * 0.8 ** k
    for name, rp in [("fixed", None), ("current", lambda ra: 1 + ra + (hp.R_plan - 1 - ra_ss))]:
        mb = copy.deepcopy(m0)
        res[f"base_{name}"] = path(copy.deepcopy(mb), H, R_plan=(lambda k: rp))
        res[f"rate_{name}"] = path(copy.deepcopy(mb), H, i_shock=shock, R_plan=(lambda k: rp))
    res["gdp_q"] = gdp_q
    res["P0"] = m0.P_prev
    return res


if __name__ == "__main__":
    seeds = [1, 2, 3, 4]
    with Pool(4) as p:
        out = p.map(run_seed, seeds)
    json.dump(out, open("results_macro.json", "w"))
    print("saved")
