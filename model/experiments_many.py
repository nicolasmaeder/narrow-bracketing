"""Paper experiments on many seeds, with bands across seeds. Each seed: burn in, then every scenario is a copy of the
same economy (common random numbers), so a scenario minus its baseline is the policy's effect in that economy."""
import json, sys, numpy as np, warnings
from multiprocessing import Pool
warnings.filterwarnings("ignore")
import experiments_macro as E

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    with Pool(2) as p:
        out = []
        for r in p.imap(E.run_seed, range(1, n + 1)):
            out.append(r); print("seed done", len(out), flush=True)
    json.dump(out, open("results_many.json", "w"))
    print("saved")
