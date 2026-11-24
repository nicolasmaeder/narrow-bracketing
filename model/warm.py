"""Warm start. The model's own initial state (identical firms at a stationary point of an old one-sector version,
households at the stationary distribution of the household block alone) is far from where the full economy settles:
the first quarters bring a price burst, a policy rate near 60% a year, and a slump that lasts about 60 years
(DECISIONS.md, start-up slump). A warm start begins instead from a stored state of a long run of the default model.

  python3 warm.py save [T]    run the default model (seed 1) for T quarters (default 500) and store its state
                              in warm_state.pkl.gz

build(..., warm=True) creates the model as usual (current code, parameters and options), then copies into it every
state variable of the stored run (if the model also has it, only with the same type and shape). State variables added since keep
their initial values; the random-number generator is re-seeded from the run's seed. Re-save after model changes
that move the stationary state a lot."""
import sys, os, gzip, pickle, copy, numpy as np
FN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'warm_state.pkl.gz')
SKIP = {'hp', 'mp', 'rng', '_mrp'}

def save(T=500):
    from run_macro import build
    m = build(seed=1, warm=False)
    for _ in range(T):
        m.step()
    state = {k: v for k, v in m.__dict__.items() if k not in SKIP}
    with gzip.open(FN, 'wb') as f:
        pickle.dump(state, f)
    print('saved state after', T, 'quarters:', len(state), 'variables')

# firm-level state that scales with the firm's size (split k ways when the stored run had k times fewer firms),
# and log beliefs about the firm's demand position (shifted by -log k); everything else per firm is a price, rate,
# ratio, belief about a rate, or productivity, and is copied to each of the k clones
EXT = {'K', 'inv', 'loan', 'Kb', 'm', 'Ehat', 'Eq', 's_lr', 'cust', 'share_prev', 'share_prev_used', 'hired_last',
       'L_plan_raw', '_cap', '_Spre', '_inv_lag', '_yprod', '_sold', '_avail', '_qd', '_mu_lag', '_adopt_L'}
LOGEXT = {'lnA', 'ln_a'}

def split_firms(state, F_old, k, rng):
    """Each stored firm becomes k identical firms of 1/k its size; workers are spread evenly over the clones."""
    out = {}
    for key, v in state.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == F_old and key not in ('job',):
            v = np.tile(v, (k,) + (1,) * (v.ndim - 1))
            if key in EXT:
                v = v / k
            elif key in LOGEXT:
                v = v - np.log(k)
        elif isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == F_old:
            v = np.tile(v, (1, k))
        out[key] = v
    if 'n_sec' in out:
        out['n_sec'] = state['n_sec'] * k
    job = state['job'].copy()
    has = job >= 0
    job[has] = job[has] + F_old * rng.integers(0, k, has.sum())
    out['job'] = job
    return out

def transplant(m):
    """Copy the stored state into a freshly built model m. Returns the number of variables copied (0: no file).
    If m has k times as many firms as the stored run, each stored firm is split into k clones first."""
    if not os.path.exists(FN):
        return 0
    with gzip.open(FN, 'rb') as f:
        state = pickle.load(f)
    F_old, F = len(state['p']), m.mp.F
    if F != F_old:
        if F % F_old:
            return 0
        state = split_firms(state, F_old, F // F_old, np.random.default_rng(m.mp.seed + 7))
    n = 0
    for k, v in state.items():
        if k not in m.__dict__:                  # created during the run (lazily initialized state)
            m.__dict__[k] = copy.deepcopy(v); n += 1
            continue
        cur = m.__dict__[k]
        if isinstance(cur, np.ndarray) != isinstance(v, np.ndarray):
            continue
        if isinstance(v, np.ndarray) and (cur.shape != v.shape or cur.dtype.kind != v.dtype.kind):
            continue
        m.__dict__[k] = copy.deepcopy(v)
        n += 1
    m.rng = np.random.default_rng(m.mp.seed)
    return n

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'save':
        save(int(sys.argv[2]) if len(sys.argv) > 2 else 500)
