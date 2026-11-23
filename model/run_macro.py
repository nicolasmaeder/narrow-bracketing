import numpy as np, time, json, sys
from household import Params
from macro import Macro, MParams
HP = dict(version="calculator", psi_A=0.9, fixed_cost=0.15, entry_cost=0.05, eps_B=0.5, G25=2.0, band=0.05, sigma_B=1.0, psi_B=1.0, R_plan=1.01, need=0.33)
def build(need=None, hpk=None, **mk):
    hp = dict(HP) if need is None else {**HP, 'need': need}
    hp.update(hpk or {})
    return Macro(Params(**hp), MParams(**mk))
if __name__ == "__main__":
    m = build()
    m.Lf = m.z[m.age < m.hp.J_W].sum()
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    t0 = time.time(); R = []
    for t in range(T):
        r = m.step(); R.append(r)
        if t % 20 == 0 or t == T-1:
            print(f"{t:4d} i={r['i']:.4f} pi={r['pi']:+.4f} u={r['u']:.3f} Y={r['Y']:.0f} y={r['y']:.0f} Ld/Lf={r['Ld_pre']/m.Lf:.2f} C={r['C']:.0f} I={r['I']:.0f} fill={r['fill']:.3f} W={r['W']:.3f} P={r['P']:.3f} tau={r['tau']:.3f} debt/GDP={r['debt']/(4*r['GDP']):.2f} K/Y={r['K']/r['Y']:.1f} A/GDP={r['A']/(4*r['GDP']):.2f} dflt={r['defaults']} loss={r['loss_rate']:.4f} res={r['reserves']:.2e} ws={r['wage_share']:.2f} mk={r['markup']:.3f} inv/s={r['inv_sales']:.2f} sig={r['sig']:.3f}")
    print("sec", time.time()-t0)
    json.dump(R, open("macro_run.json","w"))
