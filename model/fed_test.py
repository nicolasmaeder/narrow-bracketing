"""Long bonds and the Fed's balance sheet. Burned-in economies (cached like bust_diag), then a shock, with the Fed's
term-premium rule on and with a passive Fed (holdings fixed, the market sets the premium).
SHOCK=deficit: transfers of SIZE x quarterly GDP for four quarters (2020: about 0.15 -> 15% of annual GDP).
SHOCK=2008: asset bust plus credit-line cut (as in bust_diag); the short-rate rule falls below zero -> QE.
Prints differences from the no-shock path (averages over seeds), every 2 quarters."""
import sys, os, copy, json, pickle, hashlib, numpy as np, warnings
sys.path.insert(0, os.path.dirname(__file__)); warnings.filterwarnings('ignore')
from multiprocessing import Pool
from run_macro import build
H = 24
KW = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
SIZE = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15
SHOCK = os.environ.get('SHOCK', 'deficit')
def burned(seed):
    key = hashlib.md5((json.dumps(KW, sort_keys=True) + hashlib.md5(open(os.path.join(os.path.dirname(__file__), 'macro.py'), 'rb').read()).hexdigest()).encode()).hexdigest()[:12]
    fn = os.path.join(os.path.dirname(__file__), 'cache', 'burn_%s_%d.pkl' % (key, seed))
    if os.path.exists(fn):
        return pickle.load(open(fn, 'rb'))
    m = build(seed=seed, **KW)
    for t in range(400): m.step()
    m._mrp = None; os.makedirs(os.path.dirname(fn), exist_ok=True); pickle.dump(m, open(fn, 'wb'))
    return m
def path(m, shock, rule):
    m = copy.deepcopy(m); m._mrp = None; m.mp.fed_tp_rule = rule
    if shock and SHOCK == '2008':
        m.vshock = 1 - 0.3; m.cred_target = 0.5
    out = []
    for k in range(H):
        G = sum(m.GDP_hist[-4:]); Gq = m.GDP_hist[-1]
        ex = np.full(m.hp.N, SIZE * Gq / m.hp.N) if (shock and SHOCK == 'deficit' and k < 4) else None
        r = m.step(extra_y=ex)
        G = sum(m.GDP_hist[-4:])
        out.append(dict(u=100 * r['u'], i=400 * r['i'], yL=400 * m.y_L, TP=400 * m.TP, debt=100 * r['debt'] / G,
                        FedL=100 * m.q_L * m.X_F / G, privL=100 * m.q_L * (m.X_L - m.X_F) / G,
                        res=100 * m.R_res / G, EF=100 * m.E_F / G, Y=100 * np.log(r['Y']), P=100 * np.log(r['P'])))
    return out
def run(seed):
    m = burned(seed)
    res = {}
    for rule in (True, False):
        A, B = path(m, False, rule), path(m, True, rule)
        res[rule] = {k: np.array([b[k] - a[k] for a, b in zip(A, B)]) for k in A[0]}
        res[rule]['TP_lvl'] = np.array([b['TP'] for b in B])
    return res
if __name__ == '__main__':
    SEEDS = list(range(1, int(os.environ.get('NSEEDS', '2')) + 1))
    with Pool(2) as p: R = p.map(run, SEEDS)
    print(KW, SHOCK, SIZE)
    for rule in (True, False):
        print('Fed term-premium rule' if rule else 'passive Fed')
        for k in ['u', 'i', 'yL', 'TP', 'TP_lvl', 'debt', 'FedL', 'privL', 'res', 'EF', 'Y', 'P']:
            x = np.mean([r[rule][k] for r in R], 0); print('  %-7s' % k, ' '.join('%+6.2f' % v for v in x[0:H:2]))
