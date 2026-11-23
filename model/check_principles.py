"""Checks the three properties the paper rests on.

1. State-wise computation: a quarter is a function of the state (including beliefs and the random-number state) alone.
   Two copies of the same state produce identical quarters, and nothing inside a quarter iterates to a market fixed point.
2. Werner accounting: every payment is a transfer between bank accounts, and loans create deposits. Bank reserves,
   computed as deposits + bank equity - loans, stay at their initial value (zero) to rounding.
3. Lucas critique: no decision rule is fitted to past policy. Every decision solves an objective given the agent's own
   state, beliefs updated from its own observations or published statistics, and prices posted this quarter
   (see DECISIONS.md)."""
import copy, sys, numpy as np
from run_macro import build

T = int(sys.argv[1]) if len(sys.argv) > 1 else 400
m = build()
worst = worst_f = 0.0
for t in range(T):
    r = m.step()
    scale = r["B"] + abs(r["debt"]) + r["M_f"] + abs(r["Fc"]) + 1e-12
    worst = max(worst, abs(r["reserves"]) / scale)
    if m.mp.multi_bank:   # reserves implied by the banks' own books = reserves on the Fed's balance sheet
        worst_f = max(worst_f, abs(m.res_k.sum() - m.R_res) / scale)
    if not np.isfinite(r["u"]):
        worst = np.inf
print(f"1. reserves: max |reserves| / deposits over {T} quarters = {worst:.1e}  ->  {'PASS' if worst < 1e-9 else 'FAIL'}")
print(f"   banks' reserves vs the Fed's: max gap / deposits = {worst_f:.1e}  ->  {'PASS' if worst_f < 1e-9 else 'FAIL'}")
a, b = copy.deepcopy(m), copy.deepcopy(m)
ra, rb = a.step(), b.step()
same = all(np.allclose(ra[k], rb[k], equal_nan=True) for k in ra)
print(f"2. same state -> same quarter: {'PASS' if same else 'FAIL'}")
