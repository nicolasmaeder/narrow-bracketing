"""Shape of recessions caused by an aggregate shock: an asset-value bust (the fund's valuation of its firms falls 30%
and recovers at 5% a quarter), applied to a burned-in economy and compared with the same economy unshocked (common
random numbers). Reports the unemployment response: peak, quarters to peak, quarters from peak to half recovery."""
import sys, json, copy, numpy as np, warnings
from multiprocessing import Pool
warnings.filterwarnings("ignore")
from run_macro import build
BURN, H = 400, 60
def run(args):
    kw, seed, size = args
    m = build(seed=seed, **kw)
    for t in range(BURN): m.step()
    m._mrp = None
    a, b = copy.deepcopy(m), copy.deepcopy(m)
    b.vshock = 1.0 - size
    ua = np.array([a.step()["u"] for _ in range(H)]); ub = np.array([b.step()["u"] for _ in range(H)])
    return 100 * (ub - ua)
if __name__ == "__main__":
    kw = json.loads(sys.argv[1]); n = int(sys.argv[2]); size = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
    with Pool(2) as p:
        D = np.array(p.map(run, [(kw, s, size) for s in range(1, n + 1)]))
    d = D.mean(0); k = int(d.argmax()); pk = d[k]
    half = next((j - k for j in range(k, H) if d[j] <= pk / 2), None)
    print(kw, f"peak +{pk:.2f} pp at q{k}; half recovery {half} quarters after peak")
    print(" mean path:", " ".join(f"{x:+.2f}" for x in d[:40:2]))
