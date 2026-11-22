import sys, json, numpy as np
from run_macro import build
T=int(sys.argv[1]); kw=json.loads(sys.argv[2]) if len(sys.argv)>2 else {}
m=build(wage_posting=True, **kw); Lf=m.z[m.age<m.hp.J_W].sum(); R=[]
for t in range(T):
    r=m.step(); R.append(r)
    if t%20==0 or t==T-1:
        print(f"{t:4d} u={r['u']:.3f} pi={400*r['pi']:+.1f} i={400*r['i']:.1f} W={r['W']:.3f} P={r['P']:.3f} Ld/Lf={r['Ld_pre']/Lf:.2f} ws={r['wage_share']:.2f} mk={r['markup']:.2f} wdisp={r['wage_disp']:.3f} I/Y={r['I']/r['Y']:.3f} A/GDP={r['A']/(4*r['GDP']):.2f} res={r['reserves']:.1e} th={r['theta']:.2f} vac={r['vac']/Lf:.3f} c={r['contact']:.2f}", flush=True)
json.dump(R, open("wp_run.json","w"))
