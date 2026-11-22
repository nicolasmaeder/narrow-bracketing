"""One Period at a Time: a macroeconomy computed quarter by quarter.

Every agent decides from its own state and prices posted this quarter; every price is posted by
some agent; every payment has a payer and a payee. Each quarter is computed once, in a fixed order:

  1. Central bank posts the policy rate (rule on observed inflation and unemployment).
  2. Bank posts deposit, retirement-account and loan rates (margins over the policy rate, loan
     premium from its own loss history). Government posts the payroll tax (rule on last quarter's
     budget and its debt).
  3. Firms post prices (markup over normal unit cost at posted wage and loan rate), plan output (expected
     sales plus an inventory target), labor and investment (cost-minimizing capital at posted
     prices), and borrow what their cash does not cover.
  4. Labor market: firms' labor demand is filled from the labor force; the rest are unemployed.
  5. Households decide spending, buffer deposits and retirement contributions (household.py).
  6. Goods market: buyers spend at posted prices, firms sell from stock; unmet demand stays in
     buyers' accounts.
  7. Settlement: interest, debt service, defaults, dividends, taxes, transfers, estates.
  8. Learning: firms' sales forecasts, bank's loss rate, observed inflation; next quarter's wage.

Accounting check: all payments are transfers between bank accounts, and the bank's balance sheet
is loans (to firms, to the government) = deposits + equity. Bank reserves, computed as the
residual, must stay exactly at zero.
"""
from __future__ import annotations
import copy
import numpy as np
from household import Params as HParams, Economy, retirement_target, solve_accounts


class MParams:
    def __init__(self, **kw):
        # firms
        self.F = 20
        self.alpha = 0.3
        self.delta = 0.02           # depreciation per quarter
        self.Z = 1.0
        self.kappa_mu = 0.1         # markup response to own inventory gap (stockouts raise prices)
        self.markup = 1.15          # price = markup x normal unit cost (labor + capital at cost-minimizing mix)
        self.eps_share = 3.33       # elasticity of buyers' allocation across firms to relative price
        self.inv_target = 0.25      # inventory target, share of expected quarterly sales
        self.inv_decay = 0.10       # inventory spoilage per quarter
        self.g_sales = 0.10         # sales-forecast gain
        self.inv_adj = 0.10         # share of the inventory gap closed per quarter
        self.taste_sd = 0.05        # idiosyncratic taste shocks in buyers' allocation across firms
        self.lam_K = 0.05           # speed of closing the gap to target capital
        self.g_inv = 0.10           # gain on the loan rate used in investment decisions
        self.g_cost = 0.02          # gain on the loan rate used in normal cost (firms price on their average financing cost)
        self.cash_target = 1.0      # firm cash kept for next quarter's payroll, quarters of wage bill
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
        for _ in range(200):                     # P = markup x normal unit cost at P
            P0 = self.unit_cost(self.W, P0 * (rL0 + mp.delta)) * mp.markup
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
        self.se = np.full(F, y0)
        self.inv = np.full(F, mp.inv_target * y0)
        self.loan = np.full(F, P0 * K0)
        self.m = np.full(F, mp.cash_target * self.W * L0)
        self.p = np.full(F, P0)
        self.share_prev = np.full(F, 1 / F)
        # ---------------- bank and government: government account is the residual that sets reserves to zero
        self.E_B = mp.cap_B * self.loan.sum()
        self.g = 0.0
        self.g = -(self.b.sum() + self.a.sum() + self.m.sum() + self.E_B - self.loan.sum())
        self.loss_e = 0.0
        self.rL_n = rL0
        self.rL_i = rL0
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

    # ------------------------------------------------------------------ accounting
    def reserves(self):
        deposits = self.b.sum() + self.a.sum() + self.m.sum() + self.div_pending.sum() + self.g
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
        i = max(0.0, mp.rho_i * self.i_prev + (1 - mp.rho_i) * rule + i_shock)
        self.i_prev = i - i_shock
        # 2. bank and government post
        rb = max(0.0, i - mp.m_b)
        ra = i + mp.m_a
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
        self.rL_n = (1 - mp.g_cost) * self.rL_n + mp.g_cost * rL
        self.rL_i = (1 - mp.g_inv) * self.rL_i + mp.g_inv * rL
        gap = (mp.inv_target * self.se - self.inv) / np.maximum(self.se, 1e-12)      # own inventory shortfall
        mu_f = mp.markup * np.clip(1 + mp.kappa_mu * gap, 0.9, 1.15)
        self.p = mu_f * self.unit_cost(W, self.P_prev * (self.rL_n + mp.delta))
        P = float(self.p @ self.share_prev)                   # posted price index (last quarter's shares)
        y_plan = np.maximum(0.0, self.se + mp.inv_adj * (mp.inv_target * self.se - self.inv))
        cK = P * (self.rL_i + mp.delta)
        K_star = self.se / (Z * ((1 - a_) * cK / (a_ * W)) ** (1 - a_))
        I_plan = np.maximum(0.0, mp.delta * self.K + mp.lam_K * (K_star - self.K))
        Ld = L_need(y_plan)
        Ld_pre = Ld.sum()

        # 4. labor market (efficiency units): separations, then layoffs or hires at random to meet demand
        Lforce = np.where(work, self.z, 0.0)
        ration = min(1.0, Lforce.sum() / max(Ld.sum(), 1e-12))
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
        y = Z * self.K ** a_ * Ld ** (1 - a_)
        wb_f = W * Ld
        # financing: borrow what cash does not cover of wages and planned investment
        new_loan = np.maximum(0.0, wb_f + P * I_plan - self.m)
        self.loan += new_loan
        self.m += new_loan - wb_f

        # 5. households
        wage_h = np.where(emp, (1 - tau) * W * self.z, 0.0)
        benefit = np.where(work & ~emp, hp.ui * W * self.z, 0.0)
        pension = np.where(~work, self.pens * W, 0.0)
        tax_rev = tau * W * L_emp
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
        wts = np.maximum(self.se, 1e-12) * (self.p / P) ** (-mp.eps_share) * np.exp(mp.taste_sd * self.rng.standard_normal(F))
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
        self.K = (1 - mp.delta) * self.K + I_real
        self.inv = (avail - sold) * (1 - mp.inv_decay)
        self.share_prev = revenue / max(spent, 1e-12)

        # 7. settlement
        db = db + S_h * (1 - fill)                            # unspent cash stays in the buffer
        int_hh = rb * db.sum() + ra * (self.a + da).sum()
        self.b = (1 + rb) * db
        self.a = (1 + ra) * (self.a + da)
        # firms: deposit interest, loan interest and amortization (as capital depreciates); overdraft if short
        m_int = rb * np.maximum(self.m, 0)
        int_f = rL * self.loan
        amort = mp.delta * self.loan
        self.m += m_int - int_f - amort
        self.loan -= amort
        short = np.maximum(0.0, -self.m)
        self.loan += short; self.m += short
        # default: book net worth below zero -> the bank writes the loan down to restore zero net worth
        nw = P * self.K + self.p * self.inv + self.m - self.loan
        loss = np.where(nw < 0, -nw, 0.0)
        self.loan -= loss
        # dividends: cash above next quarter's payroll
        div_f = np.maximum(0.0, self.m - mp.cash_target * wb_f)
        self.m -= div_f
        # government: interest on its account
        g_int = (rb if self.g > 0 else i) * self.g
        # bank
        prof_B = int_f.sum() + max(-g_int, 0.0) - int_hh - m_int.sum() - max(g_int, 0.0) - loss.sum()
        self.E_B += prof_B
        div_B = max(0.0, self.E_B - mp.cap_B * (self.loan.sum() + max(-self.g, 0.0)))
        self.E_B -= div_B
        self.div_pending = self.own * (div_f.sum() + div_B)
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
        self.div_pending[dead] = 0.0
        self.age[dead] = 0; self.b[dead] = 0; self.a[dead] = 0; self.emp[dead] = True
        self.kappa[dead] = hp.kappa0; self.pens[dead] = 0
        self.emp = emp | (self.age >= hp.J_W) | dead
        # structural budget: benefits at u*, wage bill at u* (unemployment benefits are an automatic stabilizer)
        Lf = Lforce.sum()
        self.budget_prev = (hp.ui * W * Lf * mp.u_star + pension.sum() - g_int - estate, W * Lf * (1 - mp.u_star))

        # 8. learning, observed aggregates, next quarter's wage
        loss_rate = loss.sum() / max((self.loan + loss).sum(), 1e-12)
        self.loss_e = (1 - mp.g_loss) * self.loss_e + mp.g_loss * loss_rate
        self.se = (1 - mp.g_sales) * self.se + mp.g_sales * q_dem
        pi = P_paid / self.P_prev - 1
        self.pi_hist.append(pi)
        self.pi_e = (1 - mp.g_pi) * self.pi_e + mp.g_pi * pi
        u = 1 - L_emp / Lforce.sum()
        self.u_prev = u
        tight = Ld_pre / Lforce.sum() - (1 - mp.u_star)       # labor demand relative to the labor force
        W_new = W * (1 + mp.w_index * self.pi_e + (1 - mp.w_index) * mp.pi_star + mp.phi_W * tight)
        self.W = max(W_new, mp.lam_W * W)
        self.yhat = np.where(self.age >= hp.J_W, self.yhat * self.W / W, self.yhat)   # retirees: wage-indexed
        self.yhat[dead] = self.W * self.z[dead]
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
                    reserves=self.reserves() - self.R0, y=y.sum(), Ld_pre=Ld_pre, div=div_f.sum() + div_B, profit_share=(div_f.sum()) / max(GDP, 1e-12),
                    wage_share=W * L_emp / max(GDP, 1e-12), spend_h=S_h.sum() * fill, contrib=da[da > 0].sum(),
                    withdraw=-da[da < 0].sum(), kappa_mean=float(self.kappa[work].mean()))
