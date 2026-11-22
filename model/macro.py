"""One Period at a Time: a macroeconomy computed quarter by quarter.

Every agent decides from its own state and prices posted this quarter; every price is posted by
some agent; every payment has a payer and a payee. Each quarter is computed once, in a fixed order:

  1. Central bank posts the policy rate (rule on observed inflation and unemployment).
  2. Bank posts deposit, retirement-account and loan rates (margins over the policy rate, loan
     premium from its own loss history). Government posts the payroll tax (rule on last quarter's
     budget and its debt).
  3. Firms solve their quarter problem from own state and beliefs: price and stock maximize this
     quarter's expected profit plus the value of unsold stock (pricing newsvendor); next quarter's
     capital minimizes expected production cost plus user cost plus adjustment cost (Jorgenson).
     They borrow what their cash does not cover.
  4. Labor market: firms' labor demand is filled from the labor force; the rest are unemployed.
  5. Households decide spending, buffer deposits and retirement contributions (household.py).
  6. Goods market: buyers spend at posted prices, firms sell from stock; unmet demand stays in
     buyers' accounts.
  7. Settlement: interest, debt service, defaults, dividends, taxes, transfers, estates. The retirement
     fund owns the firms and the bank: it collects their dividends, posts their value (trailing earnings
     capitalized at its published assumed return), and invests or raises cash so that it holds only what
     it needs for withdrawals. Household retirement balances are claims on the fund.
  8. Learning: firms' demand beliefs (level and noise, discounted least squares on own orders),
     bank's loss rate, observed inflation; next quarter's wage.

Accounting check: all payments are transfers between bank accounts, and the bank's balance sheet
is loans (to firms, to the government) = deposits + equity. Bank reserves, computed as the
residual, must stay exactly at zero.
"""
from __future__ import annotations
import copy
import numpy as np
from scipy.special import ndtr
from household import Params as HParams, Economy, retirement_target, solve_accounts


class MParams:
    def __init__(self, **kw):
        # firms
        self.F = 20
        self.alpha = 0.3
        self.delta = 0.02           # depreciation per quarter
        self.Z = 1.0
        self.eps = 7.7              # buyers' elasticity of substitution across firms (logit choice)
        self.taste_sd = 0.10        # buyers' idiosyncratic taste shocks across firms (sd of log weight)
        self.inv_decay = 0.03       # inventory carrying loss per quarter (spoilage, obsolescence)
        self.phi_K = 2.0            # convex capital adjustment cost (x P K x (dK/K)^2 / 2)
        self.g_A = 0.10             # firms' forgetting rate on own-demand evidence (level)
        self.g_sig = 0.05           # forgetting rate on own forecast errors (noise)
        # bank
        self.m_b = 0.005            # deposit rate = policy - m_b (floor 0)
        self.m_a = 0.0              # retirement-account rate = policy + m_a
        self.m_L = 0.005            # loan rate margin
        self.g_loss = 0.05          # loss-rate gain
        self.cap_B = 0.08           # bank capital target, share of loans
        # government
        self.d_target = 1.0         # government debt target, share of annual GDP
        self.phi_G = 0.02           # tax response to debt gap
        self.g_tax = 0.05           # gain on the observed budget share used to set the tax rate
        # central bank
        self.i_star = 0.0075        # neutral policy rate per quarter (3% a year)
        self.pi_star = 0.005        # inflation target per quarter (2% a year)
        self.u_star = 0.05
        self.phi_pi, self.phi_u = 1.5, 0.1
        self.rho_i = 0.95           # interest-rate smoothing
        # wages (placeholder rule): indexation to observed inflation + response to unemployment, downward rigid
        self.phi_W = 0.05
        self.w_index = 0.5          # indexation to observed inflation; the rest anchored at the target
        self.lam_W = 0.995
        self.g_pi = 0.25            # gain on observed inflation
        self.fix_policy = False     # diagnostics: policy rate frozen (shocks still pass through)
        self.fix_wage = False       # diagnostics: wage grows at the inflation target
        self.div_to_fund = False    # diagnostics: firm and bank profits credited to retirement balances
        # retirement fund (owns firms and bank)
        self.fund = True
        self.erp = 0.0075           # fund's published equity premium per quarter (3% a year)
        self.g_lr = 0.02            # fund's forgetting rate on observed deposit rates (long-run rate belief)
        self.g_earn = 0.25          # forgetting rate on firms' earnings (trailing year)
        self.fund_cash_q = 0.5      # cash kept for withdrawals, in quarters of withdrawals
        # wage posting (replaces the wage rule)
        self.wage_posting = True    # False = placeholder wage rule (v3 results)
        self.eta_L = 6.0            # workers' elasticity of job choice to the relative wage (logit)
        self.ojs = 0.10             # share of employed workers searching on the job each quarter
        self.g_pub = 0.25           # forgetting rate on published labor statistics
        self.public_labor = True    # with vacancies: fill and quit beliefs from published labor-market statistics
        self.censored = True        # firms learn demand from sales and sell-outs only (censored demand)
        self.cap_aware = True       # firms price for the output the labor they expect to get can make
        self.review_q = 4           # quarters between a firm's wage reviews (staggered across firms): annual
        self.anchor = 0.0           # weight of the published inflation target in firms' nominal growth beliefs (off)
        self.match_el = 0.5         # elasticity of the contact rate to labor-market tightness (matching function)
        self.m_match = None         # matching efficiency (e.g. 0.67); None = constant contact rate
        self.c_max = 0.9            # highest quarterly contact rate (search takes time)
        self.c_v = 0.5              # recruiting time per vacancy per quarter (efficiency units of the firm's labor)
        self.vacancies = False      # costly vacancies (experimental, unstable); False = free vacancies
        self.contact = 0.45         # probability an unemployed worker reaches employers in a quarter
        self.amenity_sd = 0.10      # sd of match-specific amenities (log points) in job-to-job moves
        self.g_L = 0.25             # firms' forgetting rate on own applicant and quit evidence
        self.g_W = 0.05             # firms' forgetting rate on observed growth of the going wage
        self.burn_hh = 600          # household-only burn-in for the initial distribution
        self.seed = 1
        self.__dict__.update(kw)


class Macro:
    def __init__(self, hp: HParams, mp: MParams):
        self.hp, self.mp = hp, mp
        self.rng = np.random.default_rng(mp.seed)
        N, F = hp.N, mp.F
        a_, Z = mp.alpha, mp.Z
        # ---------------- initial prices: a stationary point of the firms' rules at u = u*
        self.W = 1.0
        rL0 = mp.i_star + mp.m_L
        P0 = 1.0
        e_f = mp.eps * (1 - 1 / F)
        for _ in range(200):                     # P = eps/(eps-1) x unit cost at P (the firm optimum at certainty)
            P0 = self.unit_cost(self.W, P0 * (rL0 + mp.delta)) * e_f / (e_f - 1)
        kl = a_ / (1 - a_) * self.W / (P0 * (rL0 + mp.delta))
        # ---------------- households: stationary distribution of the household block at these prices
        e = Economy(copy.copy(hp))
        rb0, ra0 = max(0.0, mp.i_star - mp.m_b), mp.i_star + mp.m_a
        for t in range(mp.burn_hh):
            e.step(P0, self.W, rb0, ra0)
        self.age, self.z, self.b, self.a = e.age.copy(), e.z.copy(), e.b.copy(), e.a.copy()
        self.yhat, self.kappa, self.pens = e.yhat.copy(), e.kappa.copy(), e.pens.copy()
        self.G = retirement_target(hp)
        work = self.age < hp.J_W
        self.emp = np.ones(N, bool)
        self.emp[work] = self.rng.random(work.sum()) >= mp.u_star
        own = np.exp(0.8 * self.rng.standard_normal(N))
        self.own = own / own.sum()
        self.div_pending = np.zeros(N)
        # ---------------- firms: identical, at the stationary point, capital financed by loans
        L0 = (self.emp & work) @ self.z / F
        K0 = kl * L0
        self.K = np.full(F, K0)
        y0 = Z * K0 ** a_ * L0 ** (1 - a_)
        self.lnA = np.full(F, np.log(y0))        # belief: log demand at relative price 1
        self.sig2 = np.full(F, mp.taste_sd ** 2)  # belief: variance of log demand surprises
        self.inv = np.full(F, 0.1 * y0)
        self.loan = np.zeros(F) if mp.fund else np.full(F, P0 * K0)   # with the fund, firms start equity-financed
        self.Kb = np.full(F, P0 * K0)            # book value of capital (historical cost)
        self.m = np.zeros(F)
        self.p = np.full(F, P0)
        self.share_prev = np.full(F, 1 / F)
        self.share_prev_used = self.share_prev.copy()
        # ---------------- labor market state: who works where, posted wages, firms' beliefs
        self.job = np.full(N, -1)
        ew = np.nonzero(self.emp & work)[0]
        self.job[ew] = self.rng.integers(0, F, len(ew))
        self.w = np.full(F, self.W)
        self.gW_e = mp.pi_star
        self.theta = 1.0                          # last quarter's vacancies per unemployed worker
        self.q_mkt, self.Q_mkt = 0.8, 0.02        # published job-filling and quit rates
        self.q_bel, self.Q_bel = 0.8, 0.02        # firms' beliefs: discounted averages of the published rates
        self.lnphi = np.zeros(F)                  # belief: log share of planned labor actually obtained
        self.lnq = np.full(F, np.log(0.7))        # belief: hires per vacancy (fill rate) at relative pay 1
        self.Wbar_hist = [self.W] * 5
        self.lnapp = np.full(F, np.log(1.0) if mp.vacancies else np.log(0.15 * (self.z @ work) / F))   # belief: applicants (per vacancy) at the going wage
        self.lnQ = np.full(F, np.log(0.02))      # belief: quit rate at the going wage
        # ---------------- bank and government: government account is the residual that sets reserves to zero
        self.E_B = mp.cap_B * self.loan.sum()
        self.Fc = 0.0
        self.rb_lr = max(0.0, mp.i_star - mp.m_b)
        if mp.fund:                              # retirement balances are claims on the fund; scale to its initial value
            rb0 = max(0.0, mp.i_star - mp.m_b)
            E0 = P0 * y0 - self.W * L0 - mp.delta * P0 * K0
            self.Ehat = np.full(F, E0)
            nav0 = F * max(E0, 0) / (rb0 + mp.erp) + self.E_B
            self.a *= nav0 / self.a.sum()
        else:
            self.Ehat = np.zeros(F)
        self.g = 0.0
        self.g = 0.0
        self.g = -self.reserves()
        self.loss_e = 0.0
        self.pi_e = mp.pi_star
        self.u_prev = mp.u_star
        self.P_prev = P0
        GDP0 = P0 * F * y0
        self.GDP_hist = [GDP0] * 4
        self.budget_prev = None                 # last quarter's (spending net of estates, wage bill)
        self.tau = 0.25
        self.tau_s = None                       # smoothed budget share
        self.i_prev = mp.i_star
        self.pi_hist = [mp.pi_star] * 4
        self.t = 0
        self.R0 = self.reserves()

    def unit_cost(self, W, cK):
        a_ = self.mp.alpha
        return (W / (1 - a_)) ** (1 - a_) * (cK / a_) ** a_ / self.mp.Z

    # ------------------------------------------------------------------ firm problems
    def price_and_stock(self, W, rL, S_cap=None):
        """Quarter problem of each firm, from its own state (capital, inventory) and beliefs (demand level A,
        noise sigma), at the posted wage and loan rate and the last observed price index:
            max over p, S >= n:  p E[min(D, S)] + v E[(S - D)+] - C(S - n),   D = A (p/P)^(-e) xi,
        xi lognormal with mean 1. C is the labor cost of producing with current capital; v is the value of a
        unit carried into next quarter (replacement cost net of spoilage, discounted at the loan rate)."""
        mp = self.mp; a_, Z, F = mp.alpha, mp.Z, mp.F
        Pref = self.P_ref()
        self.Pref_used = Pref
        e_f = mp.eps * (1 - self.share_prev)                  # own-price elasticity buyers' choices imply
        sig = np.sqrt(self.sig2)
        uc = self.unit_cost(W, Pref * (rL + mp.delta))       # rL here is the firm's own marginal cost of funds (array)
        v = (1 - mp.inv_decay) * uc / (1 + rL)
        kfac = (Z * self.K ** a_) ** (1 / (1 - a_))
        Wc = np.broadcast_to(np.asarray(W, float), (F,))[:, None]   # each firm's own wage
        C = lambda y: Wc * (np.maximum(y, 0) ** (1 / (1 - a_))) / kfac[:, None]
        Cp = lambda y: Wc / (1 - a_) * np.maximum(y, 0) ** (a_ / (1 - a_)) / kfac[:, None]
        n = self.inv[:, None]
        A = np.exp(self.lnA)[:, None]
        s_ = sig[:, None]

        def solve(p):
            mu = A * (p / Pref) ** (-e_f[:, None])
            lo = n + 0 * p
            hi = n + 3 * mu * np.exp(4 * s_)
            for _ in range(50):                               # FOC in S: p Pr(D>S) + v Pr(D<S) = C'(S - n)
                S = 0.5 * (lo + hi)
                d2 = (np.log(mu / S) - s_ ** 2 / 2) / s_
                pr = ndtr(d2)
                g = p * pr + v[:, None] * (1 - pr) - Cp(S - n)
                lo = np.where(g > 0, S, lo); hi = np.where(g > 0, hi, S)
            S = 0.5 * (lo + hi)
            if S_cap is not None:                             # stock the firm expects it can have (labor it expects to get)
                S = np.minimum(S, np.maximum(S_cap, n[:, 0])[:, None])
            d2 = (np.log(mu / S) - s_ ** 2 / 2) / s_
            Emin = mu - (mu * ndtr(d2 + s_) - S * ndtr(d2))
            val = p * Emin + v[:, None] * (S - Emin) - C(S - n)
            return S, mu, val

        p = uc[:, None] * np.linspace(0.9, 1.8, 91)[None, :]
        S, mu, val = solve(p)
        k = val.argmax(1)
        pc = p[np.arange(F), k]
        p2 = pc[:, None] * np.linspace(0.99, 1.01, 41)[None, :]  # refine
        S2, mu2, val2 = solve(p2)
        k2 = val2.argmax(1)
        idx = np.arange(F)
        return p2[idx, k2], S2[idx, k2], mu2[idx, k2], v

    def capital(self, W, rL, s_hat):
        """Next quarter's capital: minimize expected labor cost of producing expected sales plus the user cost
        of capital plus a convex adjustment cost, at posted prices (buy at P, resell at (1-delta)P):
            min over K':  W L(s_hat, K') + P (rL + delta) K' + phi P K ((K' - K)/K)^2 / 2,  K' >= (1-delta) K."""
        mp = self.mp; a_, Z = mp.alpha, mp.Z
        P, K = self.P_ref(), self.K
        uK = P * (rL + mp.delta)
        L = lambda Kn: (s_hat / (Z * Kn ** a_)) ** (1 / (1 - a_))
        f = lambda Kn: W * a_ / (1 - a_) * L(Kn) / Kn - uK - mp.phi_K * P * (Kn - K) / K   # decreasing in K'
        lo, hi = (1 - mp.delta) * K, 3 * K
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            pos = f(mid) > 0
            lo = np.where(pos, mid, lo); hi = np.where(pos, hi, mid)
        Kn = 0.5 * (lo + hi)
        return np.where(f((1 - mp.delta) * K) <= 0, (1 - mp.delta) * K, Kn)

    def wage_choice(self, We, p, mu, v, L_prev, B, Q, floor_x):
        """Each firm's wage problem, given its price and demand belief from the quarter problem. A wage is a policy
        that persists (the firm does not cut it), so the firm evaluates it by the workforce it sustains under its
        beliefs, inflow = outflow:  B x^eta = L (sep + Q x^-eta),  i.e.  E(w) = B x^eta / (sep + Q x^-eta),  x = w / We:
            max over w >= own last wage:  p E[min(D, S)] + v E[(S - D)+] - w L,   S = n + y(L),  L = min(L_u(w), E(w)),
        where L_u(w) is the labor the stocking FOC asks for at cost w."""
        mp, hp = self.mp, self.hp; a_, Z, F, eta = mp.alpha, mp.Z, mp.F, mp.eta_L
        xg = floor_x[:, None] * (1 + np.linspace(0.0, 0.25, 51))[None, :]
        w = xg * We
        E = B[:, None] * xg ** eta / (hp.sep + Q[:, None] * xg ** (-eta))
        kfac = (Z * self.K ** a_) ** (1 / (1 - a_))
        n = self.inv[:, None]; s_ = np.sqrt(self.sig2)[:, None]; P_ = p[:, None]; M = mu[:, None]; v_ = v[:, None]
        Cp = lambda y: w / (1 - a_) * np.maximum(y, 0) ** (a_ / (1 - a_)) / kfac[:, None]
        lo = n + 0 * w; hi = n + 3 * M * np.exp(4 * s_) + 0 * w
        for _ in range(50):
            S = 0.5 * (lo + hi)
            pr = ndtr((np.log(M / S) - s_ ** 2 / 2) / s_)
            g = P_ * pr + v_ * (1 - pr) - Cp(S - n)
            lo = np.where(g > 0, S, lo); hi = np.where(g > 0, hi, S)
        S_u = 0.5 * (lo + hi)
        L_u = (np.maximum(S_u - n, 0) ** (1 / (1 - a_))) / kfac[:, None]
        L = np.minimum(L_u, E)
        S = n + Z * (self.K ** a_)[:, None] * L ** (1 - a_)
        d2 = (np.log(M / S) - s_ ** 2 / 2) / s_
        Emin = M - (M * ndtr(d2 + s_) - S * ndtr(d2))
        val = P_ * Emin + v_ * (S - Emin) - w * L
        k = val.argmax(1); idx = np.arange(F)
        return w[idx, k], E[idx, k]

    def labor_for_cost(self, cost, p, mu, v):
        """Production labor at which the stocking FOC holds, p Pr(D > S) + v Pr(D < S) = cost x dL/dy, for each
        firm (rows) and each candidate labor cost (columns)."""
        mp = self.mp; a_, Z = mp.alpha, mp.Z
        kfac = (Z * self.K ** a_) ** (1 / (1 - a_))
        n = self.inv[:, None]; s_ = np.sqrt(self.sig2)[:, None]; P_ = p[:, None]; M = mu[:, None]; v_ = v[:, None]
        Cp = lambda y: cost / (1 - a_) * np.maximum(y, 0) ** (a_ / (1 - a_)) / kfac[:, None]
        lo = n + 0 * cost; hi = n + 3 * M * np.exp(4 * s_) + 0 * cost
        for _ in range(50):
            S = 0.5 * (lo + hi)
            pr = ndtr((np.log(M / S) - s_ ** 2 / 2) / s_)
            g = P_ * pr + v_ * (1 - pr) - Cp(S - n)
            lo = np.where(g > 0, S, lo); hi = np.where(g > 0, hi, S)
        S = 0.5 * (lo + hi)
        L = (np.maximum(S - n, 0) ** (1 / (1 - a_))) / kfac[:, None]
        d2 = (np.log(M / S) - s_ ** 2 / 2) / s_
        Emin = M - (M * ndtr(d2 + s_) - S * ndtr(d2))
        val = P_ * Emin + v_ * (S - Emin) - cost * L
        return L, val

    def full_cost(self, x, We, a, Q, r):
        """Cost per worker per quarter at relative pay x: the wage plus the cost of replacing departures (recruiting
        time c_v per vacancy, vacancies filling at rate min(1, a x^eta)), amortized at the firm's cost of funds over
        expected tenure (separations, quits Q x^-eta). a, Q: the firm's beliefs at relative pay 1."""
        mp, hp = self.mp, self.hp
        q = np.minimum(1.0, a * x ** mp.eta_L)
        turnover = hp.sep + Q * x ** (-mp.eta_L)
        return x * We * (1 + (r + turnover) * mp.c_v / np.maximum(q, 1e-3))

    def pay_review(self, We, a, Q, r):
        """Annual pay review (Salop turnover model): the relative pay that minimizes cost per worker,
            min over x:  x We [1 + (r + sep + Q x^-eta) c_v / min(1, a x^eta)],
        with We the market pay expected over the coming year (last observed x last year's market growth). The
        answer does not depend on the firm's sales: those set how many it hires, not what it pays."""
        xg = np.linspace(0.7, 1.4, 281)[None, :]
        c = self.full_cost(xg, We, a[:, None], Q[:, None], r[:, None])
        return xg[0, c.argmin(1)] * We

    def wage_choice_v(self, We, p, mu, v, a, Q, floor_x, r):
        """Wage problem with costly vacancies (static beliefs, the wage persists):
            max over w >= own last wage:  p E[min(D, S)] + v E[(S - D)+] - omega(w) L,   L at the stocking FOC,
        omega(w) = full cost per worker. A higher wage costs more per worker but fills vacancies faster and cuts quits."""
        xg = floor_x[:, None] * (1 + np.linspace(0.0, 0.25, 51))[None, :]
        om = self.full_cost(xg, We, a[:, None], Q[:, None], r[:, None])
        L, val = self.labor_for_cost(om, p, mu, v)
        k = val.argmax(1); idx = np.arange(self.mp.F)
        return xg[idx, k] * We, om[idx, k]

    def anchored(self, g_obs):
        """Belief about nominal growth: Bayesian with a prior centered on the published inflation target that is
        never forgotten (weight anchor) and evidence discounted at a constant rate (the EWMA g_obs)."""
        mp = self.mp
        return mp.anchor * mp.pi_star + (1 - mp.anchor) * g_obs

    def P_ref(self):
        """Firms' expected price index this quarter: last observed x (1 + believed inflation)."""
        return self.P_prev * (1 + self.anchored(self.pi_e)) if self.mp.wage_posting else self.P_prev

    # ------------------------------------------------------------------ labor market with posted wages
    def Wbar_obs(self):
        """Going wage: employment-weighted average of posted wages (published statistic)."""
        on = self.job >= 0
        L = np.bincount(self.job[on], self.z[on], self.mp.F)
        return float(self.w @ L / L.sum()) if L.sum() > 0 else float(self.w.mean())

    def labor_market(self, w, We, L_star, work, L_target=None, q_fill=None):
        """Separations, layoffs, then search: unemployed workers and on-the-job searchers choose among hiring firms
        by random utility (logit in the relative wage, weighted by vacancies); firms take applicants in random
        order until vacancies are filled; a second round for the rejected unemployed. All draws have fixed size."""
        hp, mp, rng = self.hp, self.mp, self.rng
        N, F, z = hp.N, mp.F, self.z
        job = self.job
        u_sep, u_ojs, k_lay, u_c1, k_1, u_c2, k_2, u_ct = (rng.random(N) for _ in range(8))
        zeta = mp.amenity_sd * rng.standard_normal(N)                   # match-specific amenity of the job offered
        job[~work] = -1
        L_prev = np.bincount(job[job >= 0], z[job >= 0], F)
        job[(job >= 0) & (u_sep < hp.sep)] = -1                       # exogenous separations
        # layoffs where the workforce exceeds the plan
        def keep_upto(sel, cap, key):
            idx = np.nonzero(sel)[0]
            f = job[idx] if cap is not None else None
            return idx, f
        on = np.nonzero(job >= 0)[0]
        order = on[np.lexsort((k_lay[on], job[on]))]
        f_o = job[order]
        cz = np.cumsum(z[order])
        start = np.searchsorted(f_o, np.arange(F))
        base = np.where(start > 0, cz[np.maximum(start - 1, 0)], 0.0)
        cum_in = cz - base[f_o]
        laid = (cum_in - z[order]) >= L_star[f_o] + 1e-12
        job[order[laid]] = -1
        L_now = np.bincount(job[job >= 0], z[job >= 0], F)
        if L_target is None:
            V = np.maximum(0.0, L_star - L_now); vac = V.copy()
        else:                                                         # hires wanted, and vacancies posted for them
            V = np.maximum(0.0, L_target - L_now); vac = V.copy()   # one open position per hire wanted (requisitions)
        U_seek = z[work & (job < 0)].sum()
        quits = np.zeros(F); apps = np.zeros(F); hired = np.zeros(F)

        def round_(seek, u_c, key, V_left, count_apps):
            wt = (V_left > 1e-9) * (vac if L_target is not None else 1.0) * (w / We) ** mp.eta_L   # seekers meet hiring firms (in proportion to vacancies if posted)
            if wt.sum() <= 0 or seek.sum() == 0:
                return V_left
            cdf = np.cumsum(wt) / wt.sum()
            idx = np.nonzero(seek)[0]
            ch = np.minimum(np.searchsorted(cdf, u_c[idx]), F - 1)
            cur = job[idx]
            ok = (cur < 0) | (w[ch] * np.exp(zeta[idx]) > w[np.maximum(cur, 0)])   # on-the-job searchers move if the offer beats their job
            idx, ch, cur = idx[ok], ch[ok], cur[ok]
            if len(idx) == 0:
                return V_left
            if count_apps:
                apps[:] += np.bincount(ch, z[idx], F)
            o = np.lexsort((key[idx], ch))
            idx, ch, cur = idx[o], ch[o], cur[o]
            cz = np.cumsum(z[idx]); st = np.searchsorted(ch, np.arange(F))
            bs = np.where(st > 0, cz[np.maximum(st - 1, 0)], 0.0)
            cin = cz - bs[ch]
            acc = (cin - z[idx]) < V_left[ch] - 1e-12
            movers = acc & (cur >= 0)
            quits[:] += np.bincount(cur[movers], z[idx[movers]], F)
            job[idx[acc]] = ch[acc]
            got = np.bincount(ch[acc], z[idx[acc]], F)
            hired[:] += got
            return np.maximum(0.0, V_left - got)

        # search technology (matching function): the chance an unemployed worker reaches employers rises with
        # last quarter's vacancies per unemployed worker
        # contacts this quarter come from this quarter's vacancies (already posted) and unemployed
        th_now = vac.sum() / max(U_seek, 1e-9)
        c_rate = mp.contact if mp.m_match is None else min(mp.c_max, mp.m_match * th_now ** mp.match_el)
        contact = u_ct < c_rate
        self.c_rate_last = c_rate
        seek1 = work & (((job < 0) & contact) | ((job >= 0) & (u_ojs < mp.ojs)))
        V2 = round_(seek1, u_c1, k_1, V.copy(), True)
        V2 = np.maximum(0.0, V2 + 0.0)
        seek2 = work & (job < 0) & contact
        round_(seek2, u_c2, k_2, V2, False)
        L_f = np.bincount(job[job >= 0], z[job >= 0], F)
        emp = job >= 0
        self.theta = vac.sum() / max(U_seek, 1e-9)                      # vacancies per unemployed (efficiency units)
        # published statistics: job-filling rate (hires per vacancy) and quit rate
        self.q_mkt = min(1.0, hired.sum() / vac.sum()) if vac.sum() > 1e-9 else 1.0
        self.Q_mkt = max(quits.sum() / max(L_prev.sum(), 1e-9), 1e-4)
        g = self.mp.g_pub
        self.q_bel = (1 - g) * self.q_bel + g * self.q_mkt
        self.Q_bel = (1 - g) * self.Q_bel + g * self.Q_mkt
        self.hired_last = hired
        self.vac_last = vac.sum()
        return L_f, emp, quits, apps, vac, L_prev

    # ------------------------------------------------------------------ accounting
    def reserves(self):
        deposits = self.b.sum() + self.m.sum() + self.div_pending.sum() + self.g + self.Fc
        if not self.mp.fund:
            deposits += self.a.sum()
        return deposits + self.E_B - self.loan.sum()

    # ------------------------------------------------------------------ one quarter
    def step(self, extra_y=None, tax_shift=0.0, i_shock=0.0, record=True, R_plan=None):
        hp, mp = self.hp, self.mp
        N, F = hp.N, mp.F
        work = self.age < hp.J_W
        W = self.W
        a_, Z = mp.alpha, mp.Z

        # 1. policy rate
        pi4 = np.mean(self.pi_hist[-4:])                      # observed inflation, last four quarters
        rule = mp.i_star + mp.phi_pi * (pi4 - mp.pi_star) - mp.phi_u * (self.u_prev - mp.u_star)
        i = max(0.0, (self.i_prev if mp.fix_policy else mp.rho_i * self.i_prev + (1 - mp.rho_i) * rule) + i_shock)
        self.i_prev = i - i_shock
        # 2. bank and government post
        rb = max(0.0, i - mp.m_b)
        self.rb_lr = (1 - mp.g_lr) * self.rb_lr + mp.g_lr * rb
        ra = (self.rb_lr + mp.erp) if mp.fund else i + mp.m_a    # posted return on retirement balances
        rL = (1 + i + mp.m_L) / (1 - self.loss_e) - 1
        GDP_a = sum(self.GDP_hist[-4:])                       # last four quarters, nominal
        debt = -self.g
        if self.budget_prev is not None:
            spend_net, wb_prev = self.budget_prev
            sh = spend_net / wb_prev
            self.tau_s = sh if self.tau_s is None else (1 - mp.g_tax) * self.tau_s + mp.g_tax * sh
            self.tau = float(np.clip(self.tau_s + mp.phi_G * (debt / GDP_a - mp.d_target), 0.0, 0.7))
        tau = self.tau + tax_shift

        # 3. firms post prices and plan
        L_need = lambda y: (np.maximum(y, 0) / (Z * self.K ** a_)) ** (1 / (1 - a_))
        # marginal cost of funds: a borrower pays rL; a firm without loans is financed at the margin by its owner,
        # the fund, which requires the deposit rate plus its published equity premium
        r_own = np.where(self.loan > 0, rL, self.rb_lr + mp.erp)
        if mp.wage_posting:
            # wage posting. Expected going wage = last observed x (1 + believed growth). The firm's labor supply,
            # in its own beliefs: incumbents who stay (quits fall with its relative wage) plus applicants (rise with
            # it). It picks the wage, paid to all its workers, that maximizes the quarter's expected profit, and it
            # does not cut its own nominal wage (a cut lowers the effort of all incumbents: morale, Bewley).
            We = self.Wbar_obs() * (1 + self.anchored(self.gW_e))
            B, Q = np.exp(self.lnapp), np.exp(self.lnQ)
            L_prev = np.bincount(self.job[self.job >= 0], self.z[self.job >= 0], F)
            w0 = np.maximum(self.w, We)
            if mp.vacancies:
                # market pay growth: last year's, from published pay data
                g_mkt = (self.Wbar_hist[-1] / self.Wbar_hist[-5]) ** 0.25 - 1
                We = self.Wbar_obs() * (1 + g_mkt)
                qf = np.exp(self.lnq)
                if mp.public_labor:                           # beliefs: discounted average of published statistics
                    qf = np.full(F, self.q_bel); Q = np.full(F, self.Q_bel)
                w_post = np.maximum(self.w, self.pay_review(We * (1 + g_mkt) ** 1.5, qf, Q, r_own))   # no nominal cuts
            else:
                p0, S0, mu0, v0 = self.price_and_stock(w0, r_own)
                w_post, L_cap = self.wage_choice(We, p0, mu0, v0, L_prev, B, Q, self.w / We)
            if mp.review_q > 1:                                 # wages are reviewed once every review_q quarters (staggered)
                rev = (np.arange(F) + self.t) % mp.review_q == 0
                w_post = np.where(rev, w_post, self.w)
            if mp.vacancies:
                # price and stock at the wage; keep incumbents while their product covers the wage,
                # hire toward the labor at which it covers the full cost (wage + recruiting)
                omega = self.full_cost(w_post / We, We, qf, Q, r_own)
                self.p, S, mu_s, v = self.price_and_stock(w_post, r_own)   # price on the marginal cost of its workers
                L_keep = self.labor_for_cost(w_post[:, None], self.p, mu_s, v)[0][:, 0]
                L_tgt = self.labor_for_cost(omega[:, None], self.p, mu_s, v)[0][:, 0]
            else:
                self.p, S, mu_s, v = self.price_and_stock(w_post, r_own)
                S = np.minimum(S, self.inv + Z * self.K ** a_ * L_cap ** (1 - a_))
            W_firm = w_post
        else:
            W_firm = W
            self.p, S, mu_s, v = self.price_and_stock(W, r_own)
        if mp.wage_posting and mp.cap_aware:
            # the firm expects to get only a share of the labor it plans for (belief from its own record), so it
            # prices and stocks for the output that labor can make: a firm short of workers raises its price
            L_plan = L_need(np.maximum(0.0, S - self.inv))
            if mp.vacancies:
                # with posted vacancies the firm knows its workforce: incumbents expected to stay, plus the hires its
                # open positions are expected to bring at the published filling rate, minus time spent recruiting
                x_ = w_post / We
                stay = L_prev * np.clip(1 - hp.sep - Q * x_ ** (-mp.eta_L), 0, 1)
                mpl0 = (1 - a_) * self.p * Z * self.K ** a_
                tgt0 = np.minimum(L_need(mu_s), (mpl0 / omega) ** (1 / a_))
                V0 = np.maximum(0.0, tgt0 - stay)
                L_exp = stay + np.minimum(1.0, qf * x_ ** mp.eta_L) * V0 - mp.c_v * V0
                cap = self.inv + Z * self.K ** a_ * np.maximum(L_exp, 0.0) ** (1 - a_)
            else:
                cap = self.inv + Z * self.K ** a_ * (np.exp(self.lnphi) * L_plan) ** (1 - a_)
            self.L_plan_raw = L_plan
            self._cap, self._Spre = cap, S.copy()
            p_unc, mu_unc = self.p.copy(), mu_s.copy()           # the plan the firm would choose with the workers it wants
            self.p, S, mu_s, v = self.price_and_stock(W_firm, r_own, S_cap=cap)
        if mp.wage_posting and mp.vacancies:
            # workforce decisions look at the expected flow of sales, not this quarter's stock: the firm keeps and
            # hires the workers it needs to make what it expects to sell each quarter, as long as a worker's product
            # covers the wage (keep) or the wage plus recruiting cost (hire)
            mpl_fac = (1 - a_) * p_unc * Z * self.K ** a_
            L_flow = L_need(mu_unc)
            L_keep = np.minimum(L_flow, (mpl_fac / w_post) ** (1 / a_))
            L_tgt = np.minimum(L_flow, (mpl_fac / omega) ** (1 / a_))
        P = float(self.p @ self.share_prev)                   # posted price index (last quarter's shares)
        y_plan = np.maximum(0.0, S - self.inv)
        K_next = self.capital(W_firm, r_own, mu_s)
        I_plan = np.maximum(0.0, K_next - (1 - mp.delta) * self.K)
        Ld = L_need(y_plan)
        Ld_pre = Ld.sum()
        Ld_req = Ld.copy()                                    # production labor the firm asks for

        # 4. labor market (efficiency units)
        Lforce = np.where(work, self.z, 0.0)
        if mp.wage_posting:
            if mp.vacancies:
                Ld, emp, quits, apps, V, L_prev = self.labor_market(w_post, We, L_keep, work, L_target=L_tgt, q_fill=qf)
                obs_f = np.log(np.maximum(self.hired_last, 1e-6) / np.maximum(V, 1e-9)) - mp.eta_L * np.log(w_post / We)
                self.lnq = np.where(V > 1e-9, (1 - mp.g_L) * self.lnq + mp.g_L * obs_f, self.lnq)
                Lw = Ld.copy()                                    # workers paid
                Ld = np.maximum(Ld - mp.c_v * V, 0.0)             # production labor: recruiting takes time
            else:
                Ld, emp, quits, apps, V, L_prev = self.labor_market(w_post, We, Ld, work)
                Lw = Ld
            L_emp = emp @ self.z
            # beliefs from own evidence: applicants and quit rate, both at the going wage
            x = w_post / We
            hire = V > 1e-9
            obs_a = np.log(np.maximum(apps, 1e-6) / x ** mp.eta_L / (np.maximum(V, 1e-9) if mp.vacancies else 1.0))
            self.lnapp = np.where(hire, (1 - mp.g_L) * self.lnapp + mp.g_L * obs_a, self.lnapp)
            obs_q = np.log(np.maximum(quits / np.maximum(L_prev, 1e-9), 1e-4) * x ** mp.eta_L)
            self.lnQ = np.where(L_prev > 0, (1 - mp.g_L) * self.lnQ + mp.g_L * obs_q, self.lnQ)
            self.w = w_post
            ration = 1.0
        else:
            ration = min(1.0, Lforce.sum() / max(Ld.sum(), 1e-12))
        if not mp.wage_posting:
            Ld = Ld * ration
            churn = (self.rng.random(N) < hp.sep) & self.emp & work
            emp = self.emp & work & ~churn
            filled = emp @ self.z
            target = Ld.sum()
            order = self.rng.permutation(N)
            if filled > target:
                cand = order[emp[order]]
                k = np.searchsorted(np.cumsum(self.z[cand]), filled - target)
                emp[cand[:k + 1]] = False
            else:
                cand = order[(~emp & work)[order]]
                k = np.searchsorted(np.cumsum(self.z[cand]), target - filled)
                emp[cand[:k]] = True
            L_emp = emp @ self.z
            Ld = Ld * L_emp / max(Ld.sum(), 1e-12)
        if mp.wage_posting and mp.vacancies:
            Ld = np.minimum(Ld, L_need(y_plan))               # production follows the plan; surplus workers idle
        y = Z * self.K ** a_ * Ld ** (1 - a_)
        if getattr(self, "trace_on", False) and mp.vacancies:
            f = 0
            self.trace.append(dict(t=self.t, inv0=float(self.inv[f]), mu=float(mu_s[f]), S=float(S[f]), y_plan=float(y_plan[f]),
                                   L_flow=float(L_flow[f]), L_keep=float(L_keep[f]), L_tgt=float(L_tgt[f]), workers=float(Lw[f]),
                                   V=float(V[f]), Lprod=float(Ld[f]), y=float(Z * self.K[f] ** a_ * Ld[f] ** (1 - a_)),
                                   p=float(self.p[f]), w=float(w_post[f]), omega=float(omega[f]), q_mkt=self.q_mkt,
                                   u=1 - L_emp / Lforce.sum(), phi=float(np.exp(self.lnphi[f])), cap=float(self._cap[f]), Spre=float(self._Spre[f])))
        if mp.wage_posting and mp.cap_aware:                   # belief: share of planned labor the firm actually gets
            obs_phi = np.clip(np.log(np.maximum(Ld, 1e-9) / np.maximum(Ld_req, 1e-9)), -3.0, 0.0)
            self.lnphi = (1 - mp.g_L) * self.lnphi + mp.g_L * obs_phi
        wb_f = W_firm * (Lw if mp.wage_posting else Ld)
        # financing: borrow what cash does not cover of wages and planned investment
        new_loan = np.maximum(0.0, wb_f + P * I_plan - self.m)
        self.loan += new_loan
        self.m += new_loan - wb_f

        # 5. households
        wage_h = np.where(emp, (1 - tau) * (self.w[np.maximum(self.job, 0)] if mp.wage_posting else W) * self.z, 0.0)
        benefit = np.where(work & ~emp, hp.ui * W * self.z, 0.0)
        pension = np.where(~work, self.pens * W, 0.0)
        tax_rev = tau * wb_f.sum()
        extra = np.zeros(N) if extra_y is None else extra_y
        y_h = wage_h + benefit + pension + self.div_pending + extra
        self.div_pending = np.zeros(N)
        m_cash = self.b + y_h
        G_j = self.G[self.age]
        c = np.empty(N); db = np.empty(N); da = np.empty(N)
        w = work
        Rp = hp.R_plan if R_plan is None else R_plan(ra)
        tau_R = (hp.J_W - self.age[w]).astype(float)
        annuity = tau_R if abs(Rp - 1) < 1e-12 else (Rp ** tau_R - 1) / (Rp - 1)
        k_star = np.clip((self.G[hp.J_W] * self.yhat[w] - self.a[w] * Rp ** tau_R) / (self.yhat[w] * annuity) - hp.tau_m, 0.0, 0.5)
        go = (np.abs(k_star - self.kappa[w]) > hp.band) & emp[w]
        self.kappa[w] = np.where(go, k_star, self.kappa[w])
        d_now = (self.kappa[w] + hp.tau_m) * wage_h[w]
        cw, _, _, _, _ = solve_accounts(m_cash[w] - d_now, self.b[w], self.a[w] + d_now, self.yhat[w], G_j[w], P, rb, ra, hp, False)
        c[w], da[w] = cw, d_now
        db[w] = m_cash[w] - d_now - P * cw                    # budget closes exactly
        r = ~work
        wd = self.a[r] / (hp.J - self.age[r])
        cr, _, _, _, _ = solve_accounts(m_cash[r] + wd, self.b[r], self.a[r], self.yhat[r], G_j[r], P, rb, ra, hp, False)
        c[r], da[r] = cr, -wd
        db[r] = m_cash[r] + wd - P * cr

        # 6. goods market at posted prices; households and investing firms buy from the same firms
        S_h = P * c
        S_I = P * I_plan
        S_tot = S_h.sum() + S_I.sum()
        avail = self.inv + y
        # buyers choose among firms by random utility (logit): shares proportional to (p/P)^(-eps) x taste
        wts = (self.p / P) ** (-mp.eps) * np.exp(mp.taste_sd * self.rng.standard_normal(F))
        q_dem = (wts / wts.sum()) * S_tot / self.p
        sold = np.minimum(q_dem, avail)
        spare = avail - sold
        unmet = ((q_dem - sold) * self.p).sum()
        if unmet > 0 and spare.sum() > 0:                     # one pass of spillover to firms with stock left
            sh2 = spare * self.p / (spare * self.p).sum()
            sold = sold + np.minimum(sh2 * unmet / self.p, spare)
        revenue = sold * self.p
        spent = revenue.sum()
        fill = spent / max(S_tot, 1e-12)
        P_paid = spent / max(sold.sum(), 1e-12)
        c_real = S_h * fill / P_paid
        I_real = S_I * fill / P_paid
        self.m += revenue - S_I * fill
        S_I_f = P * I_plan
        self.K = (1 - mp.delta) * self.K + I_real
        self.inv = (avail - sold) * (1 - mp.inv_decay)
        if getattr(self, "trace_on", False) and mp.vacancies:
            self.trace[-1].update(sold=float(sold[0]), dem=float(q_dem[0]), avail=float(avail[0]))
        self.share_prev_used = self.share_prev
        self.share_prev = revenue / max(spent, 1e-12)

        # 7. settlement
        db = db + S_h * (1 - fill)                            # unspent cash stays in the buffer
        if mp.fund:
            int_hh = rb * db.sum()
            self.b = (1 + rb) * db
            a_pre = self.a + da                               # contributions and withdrawals at the fund's value
            self.Fc += da.sum()
            f_int = (rb if self.Fc > 0 else rL) * self.Fc
            self.Fc += f_int
        else:
            int_hh = rb * db.sum() + ra * (self.a + da).sum()
            self.b = (1 + rb) * db
            self.a = (1 + ra) * (self.a + da)
            f_int = 0.0
        # firms: interest; overdraft if short
        m_int = rb * np.maximum(self.m, 0)
        int_f = rL * self.loan
        self.m += m_int - int_f
        short = np.maximum(0.0, -self.m)
        self.loan += short; self.m += short
        # default: book net worth below zero -> the bank writes the loan down to restore zero net worth
        nw = P * self.K + v * self.inv + self.m - self.loan
        loss = np.where(nw < 0, -nw, 0.0)
        self.loan -= loss
        # owners are paid the quarter's accounting profit (book depreciation at historical cost); remaining cash
        # repays loans, since holding deposits at rb while owing at rL loses money
        profit = revenue - wb_f - int_f + m_int - mp.delta * self.Kb
        self.Kb = (1 - mp.delta) * self.Kb + S_I_f * fill
        div_f = np.clip(profit, 0.0, np.maximum(self.m, 0.0))
        self.m -= div_f
        self.Ehat = (1 - mp.g_earn) * self.Ehat + mp.g_earn * profit
        # government: interest on its account
        g_int = (rb if self.g > 0 else i) * self.g
        # bank
        prof_B = int_f.sum() + max(-g_int, 0.0) - int_hh - m_int.sum() - max(g_int, 0.0) - loss.sum() - f_int
        self.E_B += prof_B
        div_B = max(0.0, self.E_B - mp.cap_B * (self.loan.sum() + max(-self.g, 0.0)))
        self.E_B -= div_B
        # fund values each firm at trailing earnings capitalized at its assumed return: the long-run deposit rate
        # it has observed (discounted average of its own experience) plus its published equity premium
        r_f = self.rb_lr + mp.erp
        V = np.maximum(self.Ehat, 0.0) / r_f
        if mp.fund:
            # fund collects dividends, keeps cash for withdrawals, invests the rest in its firms (or raises cash
            # from them): holding deposits at rb while its firms borrow at rL loses money
            self.Fc += div_f.sum() + div_B
            need_c = mp.fund_cash_q * (-da[da < 0].sum())
            x = self.Fc - need_c
            wv = V / V.sum() if V.sum() > 0 else np.full(F, 1 / F)
            self.m += x * wv; self.Fc -= x
            short = np.maximum(0.0, -self.m)
            self.loan += short; self.m += short
            nav = V.sum() + self.E_B + self.Fc
            self.a = a_pre * nav / a_pre.sum()
        elif mp.div_to_fund:                                 # diagnostic: profits accrue to retirement balances
            self.a += (div_f.sum() + div_B) * self.a / self.a.sum()
        else:
            self.div_pending = self.own * (div_f.sum() + div_B)
        repay = np.minimum(self.m, self.loan)
        self.m -= repay; self.loan -= repay
        transfers = benefit.sum() + pension.sum() + extra.sum()
        self.g += tax_rev - transfers + g_int
        # aging, pensions, estates (to the government)
        retiring = self.age == hp.J_W - 1
        self.pens = np.where(retiring, hp.pension * self.yhat / W, self.pens)
        self.yhat = np.where(work, (1 - hp.g) * self.yhat + hp.g * y_h, self.yhat)
        self.age = self.age + 1
        dead = self.age >= hp.J
        estate = self.b[dead].sum() + self.a[dead].sum() + self.div_pending[dead].sum()
        self.g += estate
        if mp.fund:
            self.Fc -= self.a[dead].sum()                     # fund pays out estates' balances
        self.div_pending[dead] = 0.0
        self.age[dead] = 0; self.b[dead] = 0; self.a[dead] = 0; self.emp[dead] = True
        self.kappa[dead] = hp.kappa0; self.pens[dead] = 0
        self.emp = emp | (self.age >= hp.J_W) | dead
        self.job[(self.age >= hp.J_W) | dead] = -1                # retirees leave their firm; entrants start unemployed
        if mp.wage_posting:
            self.emp[dead] = False
        # structural budget: benefits at u*, wage bill at u* (unemployment benefits are an automatic stabilizer)
        Lf = Lforce.sum()
        self.budget_prev = (hp.ui * W * Lf * mp.u_star + pension.sum() - g_int - estate, W * Lf * (1 - mp.u_star))

        # 8. learning, observed aggregates, next quarter's wage
        loss_rate = loss.sum() / max((self.loan + loss).sum(), 1e-12)
        self.loss_e = (1 - mp.g_loss) * self.loss_e + mp.g_loss * loss_rate
        e_f = mp.eps * (1 - self.share_prev_used)
        if mp.censored:
            # the firm sees its sales and whether it sold out, not the customers it turned away or why buyers came.
            # Not sold out: sales are its demand. Sold out: demand was at least its stock, and the firm updates on
            # the mean of its belief given that (truncated lognormal), i.e. by what Bayes' rule implies.
            pred = self.lnA - e_f * np.log(self.p / self.Pref_used) - self.sig2 / 2
            sd = np.sqrt(self.sig2)
            out = sold >= avail * (1 - 1e-9)
            a_ = np.clip((np.log(np.maximum(avail, 1e-12)) - pred) / sd, -8, 8)
            mills = np.exp(-a_ ** 2 / 2) / np.sqrt(2 * np.pi) / np.maximum(1 - ndtr(a_), 1e-12)
            obs = np.where(out, pred + sd * mills, np.log(np.maximum(sold, 1e-12)))
            err = obs - pred
            self.lnA += mp.g_A * err
            self.sig2 = np.where(out, self.sig2, np.maximum((1 - mp.g_sig) * self.sig2 + mp.g_sig * err ** 2, 1e-4))
        else:
            err = np.log(np.maximum(q_dem, 1e-12)) + e_f * np.log(self.p / self.Pref_used) - self.lnA
            self.lnA += mp.g_A * err
            self.sig2 = np.maximum((1 - mp.g_sig) * self.sig2 + mp.g_sig * err ** 2, 1e-4)
        pi = P_paid / self.P_prev - 1
        self.pi_hist.append(pi)
        self.pi_e = (1 - mp.g_pi) * self.pi_e + mp.g_pi * pi
        u = 1 - L_emp / Lforce.sum()
        self.u_prev = u
        tight = Ld_pre / Lforce.sum() - (1 - mp.u_star)       # labor demand relative to the labor force
        W_new = W * (1 + mp.w_index * self.pi_e + (1 - mp.w_index) * mp.pi_star + mp.phi_W * tight)
        if mp.wage_posting:
            W_obs = self.Wbar_obs()
            self.Wbar_hist.append(W_obs)
            self.gW_e = (1 - mp.g_W) * self.gW_e + mp.g_W * (W_obs / W - 1)
            self.W = W_obs
        else:
            self.W = W * (1 + mp.pi_star) if mp.fix_wage else max(W_new, mp.lam_W * W)
        self.yhat = np.where(self.age >= hp.J_W, self.yhat * self.W / W, self.yhat)   # retirees: wage-indexed
        self.yhat[dead] = self.W * self.z[dead]
        uc_rec = float(np.mean(self.unit_cost(W, self.P_prev * (r_own + mp.delta))))
        self.P_prev = P_paid
        GDP = spent                                           # final sales (consumption + investment), nominal
        self.GDP_hist.append(GDP)
        self.t += 1
        if not record:
            return None
        return dict(t=self.t, i=i, rb=rb, ra=ra, rL=rL, tau=tau, P=P_paid, pi=pi, W=W, u=u,
                    Y=sold.sum(), C=c_real.sum(), I=I_real.sum(), GDP=GDP, fill=fill, inv=self.inv.sum(),
                    K=self.K.sum(), loans=self.loan.sum(), debt=-self.g, E_B=self.E_B, B=self.b.sum(),
                    A=self.a.sum(), M_f=self.m.sum(), defaults=int((loss > 0).sum()), loss_rate=loss_rate,
                    reserves=self.reserves() - self.R0, theta=self.theta, vac=float(self.vac_last) if mp.wage_posting else 0.0, contact=float(self.c_rate_last), y=y.sum(), Ld_pre=Ld_pre, div=div_f.sum() + div_B, profit_share=(div_f.sum()) / max(GDP, 1e-12),
                    wage_share=wb_f.sum() / max(GDP, 1e-12), wage_disp=float(np.std(np.log(W_firm * np.ones(F)))), spend_h=S_h.sum() * fill, contrib=da[da > 0].sum(),
                    withdraw=-da[da < 0].sum(), kappa_mean=float(self.kappa[work].mean()),
                    markup=float(np.mean(self.p)) / uc_rec,
                    inv_sales=self.inv.sum() / max(sold.sum(), 1e-12), sig=float(np.sqrt(self.sig2).mean()),
                    V=V.sum(), Fc=self.Fc, book=float((P * self.K + v * self.inv + self.m - self.loan).sum()))
