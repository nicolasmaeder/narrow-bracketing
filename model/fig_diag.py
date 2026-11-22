import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
S = json.load(open("results_diag.json"))
show = [("full model", "Full model", "k", "-"), ("policy rate frozen", "Policy rate frozen", "#2471a3", "-"),
        ("wage path fixed", "Wage path fixed", "#c0392b", "--"), ("profits to retirement accts", "Profits credited to retirement accounts", "#27ae60", "-"),
        ("fast demand learning (gain .3)", "Firms learn demand faster", "#8e44ad", ":")]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
for j, (sh, title) in enumerate([("rate", "Unemployment after a 1 pp policy-rate rise (pp)"), ("check", "Unemployment after a stimulus check, 1% of GDP (pp)")]):
    for key, lab, col, ls in show:
        ax[j].plot(S[key][sh]["path"], color=col, ls=ls, lw=1.8 if key in ("full model", "profits to retirement accts") else 1.2, label=lab)
    ax[j].axhline(0, color="grey", lw=0.6); ax[j].set_xlabel("Quarters"); ax[j].set_title(title, fontsize=10)
ax[0].legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig("fig_diag.png", dpi=140)
