"""Responses (scenario minus its baseline) across seeds: mean and 90% band (5th-95th percentile of seeds)."""
import json, numpy as np, sys
R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results_many.json"))
def resp(scen, base, key, kind):
    out = []
    for r in R:
        s = np.array([q[key] for q in r[scen]]); b = np.array([q[key] for q in r[base]])
        if kind == "log": out.append(100 * np.log(s / b))
        elif kind == "pp": out.append(100 * (s - b))
        elif kind == "pp_ann": out.append(400 * (s - b))
        elif kind == "gdp": out.append(100 * (s - b) / r["gdp_q"])
    return np.array(out)
def show(name, scen, base, rows):
    print(f"\n{name}  (mean [5%, 95%] across {len(R)} seeds)")
    for key, kind, lab in rows:
        x = resp(scen, base, key, kind)
        q = [0, 1, 2, 3, 4, 7, 11, 15]
        cells = "  ".join(f"q{k}: {x[:, k].mean():+.2f} [{np.percentile(x[:, k], 5):+.2f},{np.percentile(x[:, k], 95):+.2f}]" for k in q)
        print(f"  {lab:24s} {cells}")
rows = [("Y", "log", "output (%)"), ("C", "log", "consumption (%)"), ("u", "pp", "unemployment (pp)"),
        ("pi", "pp_ann", "inflation (pp, ann.)"), ("i", "pp_ann", "policy rate (pp, ann.)")]
show("Stimulus check, 1% of annual GDP", "check", "base", rows)
show("Withholding cut, same total over 4 quarters", "withhold", "base", rows)
show("Policy-rate shock +1pp, fixed planning return", "rate_fixed", "base_fixed", rows)
show("Policy-rate shock +1pp, current planning return", "rate_current", "base_current", rows)
# cumulative output effects
for scen, base, lab in [("check", "base", "check"), ("withhold", "base", "withholding")]:
    x = resp(scen, base, "spend_h", "gdp")
    c = x[:, :8].sum(1)
    print(f"\ncumulative household spending, 8 quarters, {lab}: {c.mean():.2f}% of quarterly GDP [{np.percentile(c,5):.2f},{np.percentile(c,95):.2f}]  (transfer = 400%)")
