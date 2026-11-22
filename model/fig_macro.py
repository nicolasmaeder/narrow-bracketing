import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
out = json.load(open("results_macro.json")); R = json.load(open("macro_run.json"))
d = lambda sc, b, key: np.mean([[s[sc][k][key] - s[b][k][key] for k in range(16)] for s in out], 0)
tot = np.mean([0.04 * s["gdp_q"] for s in out]); q = np.arange(16)
fig, ax = plt.subplots(2, 3, figsize=(16, 8))
# baseline
t = np.array([r["t"] for r in R]) / 4; k = lambda n: np.array([r[n] for r in R])
s = t > 75
ax[0, 0].plot(t[s] - 75, 100 * k("u")[s], color="#2471a3", label="Unemployment")
ax[0, 0].plot(t[s] - 75, 400 * k("pi")[s], color="#c0392b", lw=0.7, label="Inflation (annualized)")
ax[0, 0].plot(t[s] - 75, 400 * k("i")[s], color="k", label="Policy rate")
ax[0, 0].set_xlabel("Years"); ax[0, 0].set_ylabel("%"); ax[0, 0].legend(fontsize=8, frameon=False)
ax[0, 0].set_title("Baseline: 125 years, computed one quarter at a time", fontsize=10)
G = k("GDP")[s]
rat = {"Retirement fund (owns firms, bank)": k("A")[s] / (4 * G), "Bank loans to firms": k("loans")[s] / (4 * G),
       "Government debt": k("debt")[s] / (4 * G), "Liquid buffers": k("B")[s] / (4 * G)}
ax[0, 1].barh(list(rat), [v.mean() for v in rat.values()], color="#2471a3")
ax[0, 1].set_xlabel("× annual GDP")
ax[0, 1].set_title("Balance sheets (bank deposits = loans + gov. debt, exactly)\nReserves drift: max |Δ| = %.0e of deposits" %
                   (np.abs(k("reserves")).max() / (k("B") + k("debt")).mean()), fontsize=10)
inv = [(k("I") * k("P") / k("GDP"))[s].mean(), k("wage_share")[s].mean(), (k("K") / (4 * k("Y")))[s].mean() / 10]
ax[0, 2].axis("off")
rows = [["Unemployment", "%.1f%%" % (100 * k("u")[s].mean()), "target 5%"],
        ["Inflation", "%.1f%%" % (400 * k("pi")[s].mean()), "target 2%"],
        ["Policy rate", "%.1f%%" % (400 * k("i")[s].mean()), "neutral 3%"],
        ["Labor share of GDP", "%.2f" % k("wage_share")[s].mean(), "US ≈ 0.60"],
        ["Investment / GDP", "%.2f" % (k("I") * k("P") / k("GDP"))[s].mean(), "US ≈ 0.17–0.20"],
        ["Capital / annual output", "%.1f" % (k("K") / (4 * k("Y")))[s].mean(), "US ≈ 2.5–3"],
        ["Retirement wealth / GDP", "%.1f" % rat["Retirement fund (owns firms, bank)"].mean(), "US ≈ 4–5"],
        ["Payroll tax rate", "%.2f" % k("tau")[s].mean(), ""],
        ["Firm defaults / quarter", "%.2f" % k("defaults")[s].mean(), ""]]
tb = ax[0, 2].table(cellText=rows, colLabels=["", "Model", "Reference"], loc="center", cellLoc="left")
tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1.25, 1.5)
[c.set_width(w) for (r_, c_), c in tb.get_celld().items() for w in [[0.5, 0.2, 0.32][c_]]]
ax[0, 2].set_title("Long-run averages", fontsize=10)
# fiscal
for sc, col, lab in [("check", "#c0392b", "Stimulus check (all in quarter 0)"), ("withhold", "#2471a3", "Withholding cut (over 4 quarters)")]:
    ax[1, 0].plot(q, np.cumsum(d(sc, "base", "spend_h")) / tot, color=col, marker="o", ms=3, label=lab)
    ax[1, 1].plot(q, 100 * d(sc, "base", "u"), color=col, marker="o", ms=3)
ax[1, 0].set_title("Cumulative extra household spending\nper dollar of transfer (1% of annual GDP)", fontsize=10)
ax[1, 0].legend(fontsize=8, frameon=False); ax[1, 0].set_xlabel("Quarters")
ax[1, 1].set_title("Unemployment response (pp)", fontsize=10); ax[1, 1].set_xlabel("Quarters")
for pl, col, lab in [("fixed", "#2471a3", "Calculator at fixed planning return"), ("current", "#e67e22", "Calculator at current posted rate")]:
    ax[1, 2].plot(q, 100 * d(f"rate_{pl}", f"base_{pl}", "u"), color=col, marker="o", ms=3, label=lab)
ax[1, 2].set_title("Unemployment after a 1 pp policy-rate rise (pp)", fontsize=10)
ax[1, 2].legend(fontsize=8, frameon=False); ax[1, 2].set_xlabel("Quarters")
fig.tight_layout(); fig.savefig("fig_macro.png", dpi=140)
