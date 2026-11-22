import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
"""Shefrin-Thaler tests of the own-state household (sticky version), without and with the entry cost."""
import copy, json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from household import Params, Economy

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})
INF, RB, RA = 1.005, 0.0025, 0.0125
CASES = {"one-quarter objective": dict(entry_cost=0.05, eps_B=0.5, G25=2.0),
         "calculator rule": dict(entry_cost=0.05, eps_B=0.5, G25=2.0, version="calculator", band=0.05, sigma_B=1.0, psi_B=1.0)}
COL = {"one-quarter objective": "#7f8c8d", "calculator rule": "#2471a3"}
BASE = dict(version="sticky", psi_A=0.9, fixed_cost=0.15)


def burned(kw):
    e = Economy(Params(**{**BASE, **kw}))
    for t in range(700):
        P = INF ** t
        e.step(P, P, RB, RA)
    return e


def forward(e, n, draws, extra_y=None, before=None):
    e = copy.deepcopy(e)
    if before:
        before(e)
    recs = []
    for k in range(n):
        P = INF ** e.t
        recs.append(e.step(P, P, RB, RA, shocks=draws[k], record=True, extra_y=extra_y if k == 0 else None))
    return recs


def cum_spend(recs, n):
    return sum(r["spend"] for r in recs[:n])


res = {}
for name, kw in CASES.items():
    e0 = burned(kw)
    p = e0.p
    work = e0.age < p.J_W
    draws = np.random.default_rng(7).random((24, p.N))
    base = forward(e0, 24, draws)
    out = {}

    # 1. Spending per dollar, by mental account: income vs buffer vs retirement balance
    g = 0.06 * e0.yhat * work                                   # ~1.5% of annual income
    inc = forward(e0, 4, draws, extra_y=g)
    def add_b(e): e.b = e.b + g
    buf = forward(e0, 4, draws, before=add_b)
    def add_a(e): e.a = e.a + g
    ret = forward(e0, 4, draws, before=add_a)
    for lab, recs in [("income", inc), ("buffer", buf), ("retirement", ret)]:
        out[f"mpc_{lab}_q"] = float((recs[0]["spend"] - base[0]["spend"])[work].sum() / g[work].sum())
        out[f"mpc_{lab}_y"] = float((cum_spend(recs, 4) - cum_spend(base, 4))[work].sum() / g[work].sum())

    # market-style gain: +10% on every retirement balance (workers and retirees), first-year spending per dollar
    bump = 0.10 * e0.a
    def add_pct(e): e.a = e.a + bump
    rp = forward(e0, 4, draws, before=add_pct)
    dd = cum_spend(rp, 4) - cum_spend(base, 4)
    out["mpc_market_gain_y"] = float(dd.sum() / bump.sum())
    # profiles and hand-to-mouth share
    yr = 25 + e0.age / 4
    out["ret_profile"] = [float(np.median(e0.a[(yr >= A) & (yr < A + 1)] / (4 * e0.yhat[(yr >= A) & (yr < A + 1)]))) for A in [30, 40, 50, 60]]
    out["htm_share"] = float(np.mean((e0.b / e0.yhat)[work] <= 1 / 6))
    out["median_buffer_q"] = float(np.median((e0.b / e0.yhat)[work]))

    # by liquid-wealth quintile (income transfer, same quarter)
    liq = (e0.b / e0.yhat)[work]
    bins = np.digitize(liq, np.quantile(liq, [0.2, 0.4, 0.6, 0.8]))
    d = (inc[0]["spend"] - base[0]["spend"])[work]
    out["mpc_income_by_liq"] = [float(d[bins == k].sum() / g[work][bins == k].sum()) for k in range(5)]

    # 2. Windfall size (paid as income), first-year spending per dollar
    sizes = [0.02, 0.07, 0.25, 0.66]                         # share of annual income
    out["windfall_sizes"] = sizes
    out["mpc_by_size"] = []
    for s in sizes:
        gs = s * 4 * e0.yhat * work
        rr = forward(e0, 4, draws, extra_y=gs)
        out["mpc_by_size"].append(float((cum_spend(rr, 4) - cum_spend(base, 4))[work].sum() / gs[work].sum()))

    # 3. Pension offset: mandatory contribution of 1% of wages
    def mandate(e): e.p.tau_m = 0.01
    pen = forward(e0, 24, draws, before=mandate)
    def sums(recs, n, sel):
        DS = sum((r["db"] - r["b"])[sel] for r in recs[:n])             # buffer (discretionary) saving
        PS = sum(r["da"][sel] for r in recs[:n])                        # retirement contributions
        MAND = sum((0.01 * r["wage"])[sel] for r in recs[:n])
        return DS, PS, MAND
    for n, lab in [(4, "1y"), (20, "5y")]:
        stay = work & (e0.age < p.J_W - n)                           # still working over the window
        DS1, PS1, M = sums(pen, n, stay); DS0, PS0, _ = sums(base, n, stay)
        dDS = DS1 - DS0; dPS = PS1 - PS0
        out[f"offset_DS_{lab}"] = float(dDS.sum() / dPS.sum())                      # change in buffer saving per $ of pension saving
        out[f"offset_total_{lab}"] = float((dDS.sum() + dPS.sum() - M.sum()) / M.sum())   # other saving per $ mandated
        z = e0.z[stay]
        terc = np.digitize(z, np.quantile(z, [1 / 3, 2 / 3]))
        out[f"offset_DS_by_income_{lab}"] = [float(dDS[terc == k].sum() / dPS[terc == k].sum()) for k in range(3)]
    res[name] = out
    print(name, json.dumps(out, indent=None))

json.dump(res, open("results_calc.json", "w"), indent=1)

# ------------------------------------------------------------ figure
fig, ax = plt.subplots(1, 4, figsize=(19, 4.4))
x = np.arange(3)
for i, n in enumerate(res):
    ax[0].bar(x + (i - 0.5) * 0.38, [res[n]["mpc_income_y"], res[n]["mpc_buffer_y"], res[n]["mpc_retirement_y"]], width=0.38, color=COL[n], label=n)
ax[0].set_xticks(x, ["Current income", "Rainy-day acct", "Retirement acct"])
ax[0].set_title("First-year spending per dollar, by account\n(Shefrin-Thaler: income > assets > future ≈ 0)", fontsize=10)
ax[0].axhline(0, color="k", lw=0.6)
ax[0].legend(fontsize=8, frameon=False)
x5 = np.arange(5)
for i, n in enumerate(res):
    ax[1].bar(x5 + (i - 0.5) * 0.38, res[n]["mpc_income_by_liq"], width=0.38, color=COL[n])
ax[1].set_xticks(x5, ["Lowest", "2", "3", "4", "Highest"])
ax[1].set_xlabel("Liquid-wealth quintile")
ax[1].set_title("Same-quarter spending of a transfer\n(evidence: highest for the cash-poor)", fontsize=10)
for n in res:
    ax[2].plot(np.arange(4), res[n]["mpc_by_size"], marker="o", color=COL[n])
ax[2].set_xticks(np.arange(4), ["2%", "7%", "25%", "66%"])
ax[2].set_xlabel("Windfall, % of annual income")
ax[2].set_ylim(-0.25, 0.8); ax[2].axhline(0, color="k", lw=0.6)
ax[2].set_title("First-year spending per dollar of windfall\n(Landsberger: falls steeply with size)", fontsize=10)
for i, n in enumerate(res):
    ax[3].bar(x + (i - 0.5) * 0.38, res[n]["offset_DS_by_income_5y"], width=0.38, color=COL[n])
ax[3].axhspan(-0.47, -0.10, color="grey", alpha=0.15, lw=0, label="evidence: -0.10 to -0.47")
ax[3].axhline(-1, color="k", ls=":", lw=1, label="life-cycle model: -1")
ax[3].set_ylim(-1.1, 0.25)
ax[3].legend(fontsize=8, frameon=False, loc="lower right")
ax[3].set_xticks(x, ["Low income", "Middle", "High income"])
ax[3].set_title("Change in rainy-day saving per $ of\nmandatory pension saving, 5 years\n(evidence: larger offset for the rich)", fontsize=10)
fig.tight_layout(); fig.savefig("fig_calculator.png", dpi=140)
print("done")
