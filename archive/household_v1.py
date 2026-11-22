"""Own-state household block, solved period by period.

Each quarter, every household solves a static problem using only
  its own state  (age j, liquid b, retirement a, income y, reference income yhat, contribution rate kappa)
  and posted prices  (goods price P, deposit rate rb, retirement-account rate ra).
No value functions, no policy functions, no beliefs about other agents.

Objective (working age):
  u(c) + psi_B * beta * v( (1+rb) d_b / (beta * yhat) ) + psi_A * G_j * v( (1+ra)(a + d_a) / (G_j * yhat) )
  s.t.  P c + d_b + d_a = b + y,   d_b >= 0,   d_a >= 0
  u, v CRRA(sigma).  beta: buffer target (quarters of income).  G_j: retirement target (quarters of income) by age.

Two versions:
  'fungible' : every quarter the household re-optimizes everything, including its retirement contribution d_a.
  'sticky'   : the retirement contribution is a rate kappa of wage income, re-optimized only when the gain
               exceeds a fixed cost (inattention; Chetty et al. 2014). Retirees draw down the retirement
               balance at rate 1/(remaining quarters), the solution of max sum ln c over remaining life with a
               perceived zero real return (the IRS minimum-distribution logic).
"""
from __future__ import annotations
import numpy as np


class Params:
    def __init__(self, **kw):
        self.J_W = 160          # working quarters (age 25-65)
        self.J_R = 120          # retired quarters (age 65-95)
        self.n_per_cohort = 20
        self.sigma = 2.0
        self.psi_B = 0.9        # weight on the liquid buffer
        self.psi_A = 1.05       # weight on the retirement balance
        self.beta = 1.5         # buffer target: 1.5 quarters of income (4.5 months)
        self.need = 0.33        # retirement target: share of reference income needed per remaining retired quarter
        self.sep, self.find = 0.03, 0.45          # job loss / job finding per quarter
        self.ui = 0.5           # unemployment benefit, share of own wage
        self.pension = 0.4      # pension, share of reference income at retirement
        self.g = 0.05           # reference-income gain
        self.kappa0 = 0.06      # default contribution rate at entry
        self.fixed_cost = 0.02  # re-optimization cost, share of quarterly consumption (sticky version)
        self.z_sd = 0.5         # permanent productivity dispersion (log)
        self.entry_cost = 0.0   # fixed cost of drawing the buffer below its starting balance (share of u'(c)c)
        self.tau_m = 0.0        # mandatory pension contribution rate (floor on contributions)
        self.version = "sticky"
        self.seed = 0
        self.__dict__.update(kw)
        self.J = self.J_W + self.J_R
        self.N = self.J * self.n_per_cohort


def retirement_target(p: Params) -> np.ndarray:
    """G_j in quarters of reference income. Working age: Fidelity-style benchmarks (annual multiples
    1x at 30, 3x at 40, 6x at 50, 8x at 60, 10x at 67). Retirement: need x remaining quarters."""
    age = 25 + np.arange(p.J) / 4
    G = np.interp(age, [25, 30, 40, 50, 60, 67], [0.25, 4, 12, 24, 32, 40])
    retired = np.arange(p.J) >= p.J_W
    remaining = p.J - np.arange(p.J)
    G[retired] = p.need * remaining[retired]
    return G


def crra(x, s):
    return np.log(x) if s == 1 else x ** (1 - s) / (1 - s)


def solve(m, a, yhat, G, P, rb, ra, p: Params, choose_a: bool, a_lb=None, b_lb=0.0, n_iter=70):
    """Static problem for a vector of households, by bisection on the multiplier mu (log).
    m: cash on hand net of any fixed retirement flow; a: retirement balance; choose_a: whether d_a is chosen;
    b_lb: lower bound on the buffer deposit. Returns c, d_b, d_a, utility."""
    s = p.sigma
    lo = np.full(m.shape, -40.0)
    hi = np.full(m.shape, 40.0)

    def spend(lmu):
        mu = np.exp(lmu)
        c = (mu * P) ** (-1 / s)
        xb = (p.psi_B * (1 + rb) / (mu * yhat)) ** (1 / s)
        db = np.maximum(xb * p.beta * yhat / (1 + rb), b_lb)
        if choose_a:
            xa = (p.psi_A * (1 + ra) / (mu * yhat)) ** (1 / s)
            da = np.maximum(xa * G * yhat / (1 + ra) - a, a_lb)
        else:
            da = np.zeros_like(m)
        return c, db, da, P * c + db + da

    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        tot = spend(mid)[3]
        over = tot > m          # spending too much -> raise mu
        lo = np.where(over, mid, lo)
        hi = np.where(over, hi, mid)
    c, db, da, _ = spend(0.5 * (lo + hi))
    xb = (1 + rb) * db / (p.beta * yhat)
    U = crra(c, s) + p.psi_B * p.beta * crra(xb, s)
    if choose_a:
        xa = (1 + ra) * (a + da) / (G * yhat)
        U = U + p.psi_A * G * crra(np.maximum(xa, 1e-12), s)
    return c, db, da, U


def solve_accounts(m, b, a, yhat, G, P, rb, ra, p: Params, choose_a: bool, a_lb=None, add_a_term=0.0):
    """Shefrin-Thaler mental accounts: spending out of this quarter's income is free, but drawing the
    buffer below its starting balance b ('invading the asset account') carries a fixed entry cost,
    p.entry_cost x u'(c) c. The household compares not invading (d_b >= b) with invading and paying."""
    if p.entry_cost <= 0:
        out = solve(m, a, yhat, G, P, rb, ra, p, choose_a, a_lb=a_lb)
        return out[0], out[1], out[2], out[3] + add_a_term, np.zeros(m.shape, bool)
    cN, bN, dN, UN = solve(m, a, yhat, G, P, rb, ra, p, choose_a, a_lb=a_lb, b_lb=b)
    cI, bI, dI, UI = solve(m, a, yhat, G, P, rb, ra, p, choose_a, a_lb=a_lb, b_lb=0.0)
    UI = UI - p.entry_cost * cI ** (1 - p.sigma)
    feasibleN = P * cN + bN + dN <= m * (1 + 1e-9)          # no-invasion needs income to cover spending
    UN = np.where(feasibleN, UN, -np.inf)
    inv = UI > UN
    pick = lambda x, y: np.where(inv, y, x)
    return pick(cN, cI), pick(bN, bI), pick(dN, dI), pick(UN, UI) + add_a_term, inv


class Economy:
    """Population of households; prices and rates are posted inputs each quarter."""

    def __init__(self, p: Params):
        self.p = p
        rng = np.random.default_rng(p.seed)
        self.rng = rng
        N, J = p.N, p.J
        self.age = np.repeat(np.arange(J), p.n_per_cohort)          # j = 0..J-1
        self.z = np.exp(p.z_sd * rng.standard_normal(N) - p.z_sd ** 2 / 2)
        self.emp = np.ones(N, bool)
        self.b = np.zeros(N)
        self.a = np.zeros(N)
        self.yhat = self.z.copy()
        self.kappa = np.full(N, p.kappa0)
        self.G = retirement_target(p)
        self.pens = np.zeros(N)
        self.W_prev = None
        self.t = 0

    def income(self, W):
        p = self.p
        work = self.age < p.J_W
        wage = np.where(work & self.emp, W * self.z, 0.0)
        benefit = np.where(work & ~self.emp, p.ui * W * self.z, 0.0)
        pension = np.where(~work, self.pens * W, 0.0)        # pensions indexed to the wage level
        return wage, benefit + pension + wage

    def step(self, P, W, rb, ra, shocks=None, record=False, extra_y=None):
        """One quarter. P, W: posted goods price and wage index; rb, ra: posted rates.
        extra_y: one-time income paid this quarter (coded as current income)."""
        p = self.p
        N = p.N
        work = self.age < p.J_W
        wage, y = self.income(W)
        if extra_y is not None:
            y = y + extra_y
        G = self.G[self.age]
        m = self.b + y
        c = np.empty(N); db = np.empty(N); da = np.empty(N)
        reopt = np.zeros(N, bool)
        invade = np.zeros(N, bool)

        if p.version == "fungible":
            lb = np.where(work, 0.0, -self.a)
            c, db, da, _ = solve(m, self.a, self.yhat, G, P, rb, ra, p, True, a_lb=lb)
        else:
            # working age: contribution fixed at (kappa + mandatory) * wage unless re-optimizing pays the fixed cost
            w = work
            mand = p.tau_m * wage[w]
            d_fix = (self.kappa[w] + p.tau_m) * wage[w]
            a_term = p.psi_A * G[w] * crra(np.maximum((1 + ra) * (self.a[w] + d_fix) / (G[w] * self.yhat[w]), 1e-12), p.sigma)
            c1, b1, _, U1, inv1 = solve_accounts(m[w] - d_fix, self.b[w], self.a[w] + d_fix, self.yhat[w], G[w], P, rb, ra,
                                                 p, False, add_a_term=a_term)
            c2, b2, d2, U2, inv2 = solve_accounts(m[w], self.b[w], self.a[w], self.yhat[w], G[w], P, rb, ra, p, True,
                                                  a_lb=mand)
            cost = p.fixed_cost * c1 ** (1 - p.sigma)        # = fixed_cost * u'(c) * c, consumption-equivalent
            go = (U2 - U1 > cost) & self.emp[w]
            c[w] = np.where(go, c2, c1); db[w] = np.where(go, b2, b1); da[w] = np.where(go, d2, d_fix)
            invade[w] = np.where(go, inv2, inv1)
            kap = self.kappa[w]
            kap = np.where(go, np.clip((d2 - mand) / np.maximum(wage[w], 1e-12), 0, 0.5), kap)
            self.kappa[w] = kap
            reopt[w] = go
            # retired: draw down 1/(remaining quarters), then the consumption/buffer problem
            r = ~work
            remaining = p.J - self.age[r]
            wd = self.a[r] / remaining
            cr, br, _, _, invr = solve_accounts(m[r] + wd, self.b[r], self.a[r], self.yhat[r], G[r], P, rb, ra, p, False)
            c[r], db[r], da[r] = cr, br, -wd
            invade[r] = invr

        out = None
        if record:
            out = dict(c=c.copy(), y=y.copy(), b=self.b.copy(), a=self.a.copy(), yhat=self.yhat.copy(),
                       age=self.age.copy(), emp=self.emp.copy(), reopt=reopt, invade=invade, spend=P * c, P=P,
                       db=db.copy(), da=da.copy(), wage=wage.copy())
        # end of quarter: returns, income reference, labor-market transitions, aging
        self.b = (1 + rb) * db
        self.a = (1 + ra) * (self.a + da)
        # reference income: own-income average while working; frozen in real (wage-indexed) terms in retirement
        self.yhat = np.where(work, (1 - p.g) * self.yhat + p.g * y, self.yhat * (W / self.W_prev if self.W_prev else 1.0))
        self.W_prev = W
        u = self.rng.random(N) if shocks is None else shocks
        self.emp = np.where(self.emp, u >= p.sep, u < p.find)
        retiring = self.age == p.J_W - 1
        self.pens = np.where(retiring, p.pension * self.yhat / W, self.pens)   # stored relative to the wage level
        self.age = self.age + 1
        dead = self.age >= p.J
        self.age[dead] = 0
        self.b[dead] = 0; self.a[dead] = 0; self.emp[dead] = True
        self.yhat[dead] = W * self.z[dead]
        self.kappa[dead] = p.kappa0
        self.pens[dead] = 0
        self.t += 1
        return out
