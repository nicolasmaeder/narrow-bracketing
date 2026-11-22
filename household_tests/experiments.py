import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
"""Checks of the own-state household block against micro evidence. Both versions, same shocks."""
import copy, json, time, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from household import Params, Economy

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})
VERSIONS = {"fungible": dict(version="fungible", psi_A=1.05),
            "sticky": dict(version="sticky", psi_A=0.9, fixed_cost=0.15)}
COL = {"fungible": "#c0392b", "sticky": "#2471a3"}
LAB = {"fungible": "Re-optimizes contributions every quarter", "sticky": "Sticky contributions + drawdown rule"}
INF, RB, RA = 1.005, 0.0025, 0.0125
BURN = 700


def burned(kw):
    e = Economy(Params(**kw))
    for t in range(BURN):
        P = INF ** t
        e.step(P, P, RB, RA)
    return e


def run_forward(e, n, shock_draws, extra=None):
    """Advance n quarters with given employment draws; return list of records."""
    recs = []
    for k in range(n):
        P = INF ** e.t
        if extra is not None and k == 0:
            extra(e)
        recs.append(e.step(P, P, RB, RA, shocks=shock_draws[k], record=True))
    return recs


res, out = {}, {}
for name, kw in VERSIONS.items():
    e0 = burned(kw)
    p = e0.p
    N = p.N
    rng = np.random.default_rng(123)
    draws = rng.random((160, N))
    r = {}

    # ---- speed, scaling with population
    times = {}
    for npc in [20, 80, 320]:
        ee = Economy(Params(**kw, n_per_cohort=npc))
        tt = []
        for t in range(30):
            t0 = time.perf_counter(); ee.step(1.0, 1.0, RB, RA); tt.append(time.perf_counter() - t0)
        times[ee.p.N] = 1000 * np.mean(tt[5:])
    r["ms_per_quarter"] = times

    # ---- baseline panel (event study + profiles)
    base = run_forward(copy.deepcopy(e0), 160, draws)
    last = base[0]
    yr = 25 + last["age"] / 4
    ages = np.arange(25, 95)
    r["profile_A"] = [float(np.median(last["a"][(yr >= A) & (yr < A + 1)] / (4 * last["yhat"][(yr >= A) & (yr < A + 1)]))) for A in ages]
    r["profile_B"] = [float(np.median(last["b"][(yr >= A) & (yr < A + 1)] / last["yhat"][(yr >= A) & (yr < A + 1)])) for A in ages]
    r["profile_C"] = [float(np.median(last["spend"][(yr >= A) & (yr < A + 1)] / last["y"][(yr >= A) & (yr < A + 1)])) for A in ages]

    # job loss: employed for 4 quarters, working age with >= 5 years left, loses job at k
    C = np.array([b["spend"] / b["P"] for b in base]); E = np.array([b["emp"] for b in base])
    Bq = np.array([b["b"] for b in base]); AG = np.array([b["age"] for b in base]); Y = np.array([b["y"] for b in base])
    paths_c, paths_b, paths_y = [], [], []
    for k in range(4, 140):
        lose = E[k - 4:k].all(0) & ~E[k] & (AG[k] < p.J_W - 20) & (AG[k] > 4)
        idx = np.nonzero(lose)[0]
        if len(idx) == 0:
            continue
        pre_c = C[k - 4:k, idx].mean(0); pre_y = Y[k - 4:k, idx].mean(0)
        paths_c.append(C[k - 4:k + 13, idx] / pre_c)
        paths_y.append(Y[k - 4:k + 13, idx] / pre_y)
        paths_b.append(Bq[k - 4:k + 13, idx] / (pre_y))
    r["jobloss_c"] = np.concatenate(paths_c, 1).mean(1).tolist()
    r["jobloss_y"] = np.concatenate(paths_y, 1).mean(1).tolist()
    r["jobloss_b"] = np.concatenate(paths_b, 1).mean(1).tolist()

    # ---- MPC out of a transitory transfer (working age), same quarter, by liquid-wealth quintile
    work = e0.age < p.J_W
    gift = 0.06 * e0.yhat * work                # ~1.5% of annual income, like a tax rebate
    def give(e): e.b = e.b + gift
    shocked = run_forward(copy.deepcopy(e0), 4, draws, extra=give)
    d1 = shocked[0]["spend"] - base[0]["spend"]
    liq = (e0.b / e0.yhat)[work]
    q = np.quantile(liq, [0.2, 0.4, 0.6, 0.8])
    bins = np.digitize(liq, q)
    r["mpc_transfer_q"] = [float(d1[work][bins == k].sum() / gift[work][bins == k].sum()) for k in range(5)]
    r["mpc_transfer"] = float(d1[work].sum() / gift[work].sum())
    d4 = sum(s["spend"] - b_["spend"] for s, b_ in zip(shocked, base[:4]))
    r["mpc_transfer_annual"] = float(d4[work].sum() / gift[work].sum())

    # ---- MPC out of retirement wealth: +10% retirement balances, cumulative spending over 4 quarters
    bump = 0.10 * e0.a
    def raise_a(e): e.a = e.a + bump
    sh_a = run_forward(copy.deepcopy(e0), 4, draws, extra=raise_a)
    d4a = sum(s["spend"] - b_["spend"] for s, b_ in zip(sh_a, base[:4]))
    ret = ~work
    r["mpc_wealth_workers"] = float(d4a[work].sum() / bump[work].sum())
    r["mpc_wealth_retirees"] = float(d4a[ret].sum() / bump[ret].sum())
    r["mpc_wealth_all"] = float(d4a.sum() / bump.sum())

    # ---- price elasticity of real consumption (1% higher posted price, nominal state unchanged)
    ep = copy.deepcopy(e0); Pp = INF ** ep.t
    hi = ep.step(Pp * 1.01, Pp, RB, RA, shocks=draws[0], record=True)
    lo = copy.deepcopy(e0).step(Pp, Pp, RB, RA, shocks=draws[0], record=True)
    r["price_elasticity"] = float(np.log((hi["spend"] / hi["P"]).sum() / (lo["spend"] / lo["P"]).sum()) / np.log(1.01))

    # ---- nominal neutrality: double every nominal state variable, the goods price and the wage index
    en = copy.deepcopy(e0)
    en.b *= 2; en.a *= 2; en.yhat *= 2
    if en.W_prev:
        en.W_prev *= 2
    nn = en.step(2 * Pp, 2 * Pp, RB, RA, shocks=draws[0], record=True)
    r["neutrality_max_rel_diff"] = float(np.max(np.abs((nn["spend"] / nn["P"]) / (lo["spend"] / lo["P"]) - 1)))
    res[name] = r
    print(name, {k: (v if not isinstance(v, list) else None) for k, v in r.items()}, flush=True)

json.dump(res, open("results.json", "w"), indent=1)

# ================================================================ figures
ages = np.arange(25, 95)
fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
for n in res:
    ax[0].plot(ages, res[n]["profile_A"], color=COL[n], label=LAB[n])
    ax[1].plot(ages, res[n]["profile_B"], color=COL[n])
    ax[2].plot(ages, res[n]["profile_C"], color=COL[n])
ax[0].plot([30, 40, 50, 60, 67], [1, 3, 6, 8, 10], "ko", ms=5, label="Published benchmarks (Fidelity)")
ax[0].set_title("Retirement balance (× annual income), median", fontsize=10)
ax[1].axhline(1.5, color="grey", ls=":", lw=1); ax[1].text(26, 1.53, "target 1.5", fontsize=8, color="grey")
ax[1].set_title("Liquid buffer (quarters of income), median", fontsize=10)
ax[2].set_title("Spending / income, median", fontsize=10)
for a_ in ax:
    a_.set_xlabel("Age"); a_.axvline(65, color="grey", lw=0.6)
ax[0].legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig("fig_lifecycle.png", dpi=140)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
k = np.arange(-4, 13)
for n in res:
    ax[0].plot(k, 100 * (np.array(res[n]["jobloss_c"]) - 1), color=COL[n], marker="o", ms=3, label=LAB[n])
    ax[1].plot(k, res[n]["jobloss_b"], color=COL[n], marker="o", ms=3)
ax[0].plot(k, 100 * (np.array(res["sticky"]["jobloss_y"]) - 1), color="k", ls=":", label="Income")
ax[0].set_title("Consumption around job loss (% vs. prior year)", fontsize=10)
ax[1].set_title("Liquid buffer (quarters of prior income)", fontsize=10)
for a_ in ax:
    a_.axvline(0, color="grey", lw=0.6); a_.set_xlabel("Quarters since job loss")
ax[0].legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig("fig_jobloss.png", dpi=140)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
x = np.arange(5)
for i, n in enumerate(res):
    ax[0].bar(x + (i - 0.5) * 0.38, res[n]["mpc_transfer_q"], width=0.38, color=COL[n], label=LAB[n])
ax[0].axhspan(0.12, 0.30, color="grey", alpha=0.15, lw=0)
ax[0].text(4.5, 0.31, "Parker et al. (2013):\n12-30% in the quarter", ha="right", fontsize=8, color="grey")
ax[0].set_xticks(x, ["Lowest", "2", "3", "4", "Highest"])
ax[0].set_xlabel("Liquid-wealth quintile (working age)")
ax[0].set_title("Spending out of a one-time transfer, same quarter", fontsize=10)
ax[0].legend(fontsize=8, frameon=False, loc="upper center")
labels = ["Workers", "Retirees", "All"]
for i, n in enumerate(res):
    vals = [res[n]["mpc_wealth_workers"], res[n]["mpc_wealth_retirees"], res[n]["mpc_wealth_all"]]
    ax[1].bar(np.arange(3) + (i - 0.5) * 0.38, vals, width=0.38, color=COL[n])
ax[1].axhspan(0.02, 0.07, color="grey", alpha=0.15, lw=0)
ax[1].text(2.4, 0.075, "Evidence: ~3 cents (stock wealth),\n5-7 cents (housing wealth)", ha="right", fontsize=8, color="grey")
ax[1].set_xticks(np.arange(3), labels)
ax[1].set_title("Spending per dollar of retirement wealth, first year", fontsize=10)
fig.tight_layout(); fig.savefig("fig_mpc.png", dpi=140)
print("done")
