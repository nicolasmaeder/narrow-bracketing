import sys, json
from run_macro import build
kw = json.loads(sys.argv[1]); T = int(sys.argv[2])
m = build(**kw); m.trace_on = True; m.trace = []
for t in range(T): m.step()
print(' t   u    inv0   mu     S   yplan  Lflow Lkeep  Ltgt  wrkrs  V    Lprod   y   avail  dem  sold   p/w   om/w  qm   phi')
for r in m.trace:
    print('%3d %.3f %5.0f %5.0f %5.0f %5.0f %5.1f %5.1f %5.1f %5.1f %4.1f %5.1f %5.0f %5.0f %5.0f %5.0f %.3f %.3f %.2f %.2f'%(r['t'],r['u'],r['inv0'],r['mu'],r['S'],r['y_plan'],r['L_flow'],r['L_keep'],r['L_tgt'],r['workers'],r['V'],r['Lprod'],r['y'],r['avail'],r['dem'],r['sold'],r['p']/r['w'],r['omega']/r['w'],r['q_mkt'],r['phi']))
