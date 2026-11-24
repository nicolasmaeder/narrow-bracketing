"""Firms' quarter problems: price and stock (newsvendor), capital, pay and hiring, demand elasticity, expected inflation, published price indices, beliefs about applicant quality."""
from __future__ import annotations
import copy
import numpy as np
from scipy.special import ndtr, ndtri


class Firms:
    # ------------------------------------------------------------------ firm problems
    def price_and_stock(self, W, rL, S_cap=None, W_int=None, plan=False):
        """Quarter problem of each firm, from its own state (capital, inventory) and beliefs (demand level A,
        noise sigma), at the posted wage and loan rate and the last observed price index:
            max over p, S >= n:  p E[min(D, S)] + v E[(S - D)+] - C(S - n),   D = A (p/P)^(-e) xi,
        xi lognormal with mean 1. C is the labor cost of producing with current capital; v is the value of a
        unit carried into next quarter (replacement cost net of spoilage, discounted at the loan rate)."""
        mp = self.mp; a_, Z, F = mp.alpha, mp.Z, mp.F
        Pref = self.P_ref_sec()                               # competitors' prices: the firm's sector index
        self.Pref_used = Pref
        self.Pref_all_used = self.P_ref()
        Prc = Pref[:, None] if np.ndim(Pref) else Pref
        e_f = self.elast(short=True)                          # own-price elasticity buyers' choices imply (this quarter)
        sig = np.sqrt(self.sig2)
        uc = self.unit_cost(W, self.Pref_all_used * (rL + mp.delta))       # rL here is the firm's own marginal cost of funds (array)
        if mp.v_marginal:
            # a unit left unsold is worth what replacing it next quarter would cost: the marginal cost of producing
            # it at the firm's normal output (expected sales) with the firm's internal cost of labor, net of spoilage,
            # discounted at its cost of funds. The same cost the production decision uses.
            Wi = np.broadcast_to(np.asarray(W if W_int is None else W_int, float), (F,))
            y_n = np.exp(self.lnA)
            mc_n = self.mc_y(Wi, y_n)
            v = (1 - self.decay) * mc_n / (1 + rL)
        else:
            v = (1 - self.decay) * uc / (1 + rL)
        Wc = np.broadcast_to(np.asarray(W, float), (F,))[:, None]   # each firm's own wage
        C = lambda y: Wc * self.L_need(y)
        Cp = lambda y: self.mc_y(Wc, y)
        n = self.inv[:, None]
        A = np.exp(self.lnA)[:, None]
        s_ = sig[:, None]

        def solve(p):
            mu = A * (p / Prc) ** (-e_f[:, None])
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
            if mp.rho_c < 1:
                # customer market: the firm values the customers its price wins or loses at the margin they bring in
                # over the expected life of the relationship (static beliefs), CLV = Stot (1 - mc/p) / (r + rho)
                sh = self.cust[:, None]
                lam = self.lam_sh[:, None] if mp.share_learn else 1.0
                Lp = sh * (p / Prc) ** (-lam * mp.eps * (1 - sh))
                mc = v[:, None] * (1 + rL[:, None] if np.ndim(rL) else 1 + rL) / (1 - self.decay)[:, None]
                if mp.clv_lr:
                    # a customer is a lasting flow of orders that needs capacity built for it: its margin is over the
                    # long-run unit cost (labor and the user cost of capital), not this quarter's marginal cost
                    mc = np.maximum(mc, uc[:, None])
                clv = self._Ssec[:, None] * np.maximum(1 - mc / p, 0.0) / ((rL[:, None] if np.ndim(rL) else rL) + mp.rho_c)
                served = np.minimum(1.0, Emin / np.maximum(mu, 1e-12))      # share of demand it expects to serve
                # customers are worth their margin only up to what the firm can serve: buyers beyond its stock on
                # hand add nothing (they find it sold out), so a firm that is sold out gains nothing by cutting its price
                sh_cap = S * p / np.maximum(self._Ssec[:, None], 1e-12)   # spending share the stock on hand can serve
                won = np.minimum(mp.rho_c * Lp, np.maximum(sh_cap - (1 - mp.rho_c) * sh, 0.0))
                # a customer lost to a stock-out is worth the margin it would bring at normal prices (the long-run
                # Lerner margin implied by buyers' choices), not at today's price: otherwise pricing below cost would
                # look like a way to make lost customers worthless
                m_norm = 1.0 / self.elast(short=False)[:, None]
                clv_n = self._Ssec[:, None] * m_norm / ((rL[:, None] if np.ndim(rL) else rL) + mp.rho_c)
                val = val + won * clv - mp.stockout_loss * (1 - served) * sh * clv_n
            return S, mu, val

        p = uc[:, None] * np.exp(np.linspace(np.log(0.9), np.log(4.0), 121))[None, :]   # wide enough never to bind
        S, mu, val = solve(p)
        k = val.argmax(1)
        pc = p[np.arange(F), k]
        p2 = pc[:, None] * np.linspace(0.99, 1.01, 41)[None, :]  # refine
        S2, mu2, val2 = solve(p2)
        k2 = val2.argmax(1)
        idx = np.arange(F)
        p_o, S_o, mu_o = p2[idx, k2], S2[idx, k2], mu2[idx, k2]
        self._v_inv = np.broadcast_to(np.asarray(v, float), (F,)).copy()   # value of a unit carried forward
        if plan:
            return p_o, S_o, mu_o, v
        if self.mp.menu_cost > 0 and getattr(self, "_p_old", None) is not None:
            # state-dependent pricing: the firm changes its price when the expected gain from re-optimizing (its own
            # beliefs) exceeds a menu cost, a share of revenue at the old price (Levy et al. 1997)
            S_f, mu_f, val_f = solve(self._p_old[:, None])
            gain = val2[idx, k2] - val_f[:, 0]
            rv = gain > self.mp.menu_cost * self._p_old * mu_f[:, 0]
            self._gain_rel = gain / np.maximum(self._p_old * mu_f[:, 0], 1e-12)
            self._dp_opt = np.log(p_o / self._p_old)
            self._rev_p = rv
            p_o = np.where(rv, p_o, self._p_old); S_o = np.where(rv, S_o, S_f[:, 0]); mu_o = np.where(rv, mu_o, mu_f[:, 0])
        elif self.mp.review_p > 1 and getattr(self, "_p_old", None) is not None:
            # prices are reviewed every review_p quarters (staggered); between reviews the firm keeps its posted
            # price and chooses only its stock
            S_f, mu_f, _ = solve(self._p_old[:, None])
            rv = self._rev_p
            p_o = np.where(rv, p_o, self._p_old); S_o = np.where(rv, S_o, S_f[:, 0]); mu_o = np.where(rv, mu_o, mu_f[:, 0])
        return p_o, S_o, mu_o, v

    def capital(self, W, rL, s_hat):
        """Next quarter's capital: minimize expected labor cost of producing expected sales plus the user cost
        of capital plus a convex adjustment cost, at posted prices (buy at P, resell at (1-delta)P):
            min over K':  W L(s_hat, K') + P (rL + delta) K' + phi P K ((K' - K)/K)^2 / 2,  K' >= (1-delta) K."""
        mp = self.mp; a_, Z = mp.alpha, mp.Z
        P, K = self.P_ref(), self.K
        ub = mp.u_bar if getattr(self, "_ell", None) is not None else 1.0
        # with variable utilization, capacity is planned so that normal orders run at normal utilization u_bar
        # (spare capacity of about 20%, as in the Federal Reserve's capacity-utilization data)
        # Jorgenson user cost: interest net of the expected rise in the price of capital goods, plus depreciation
        uK = P * (rL - (self.anchored(self.pi_e) if mp.jorgenson else 0.0) + mp.delta)
        L = lambda Kn: self._L_need_ces(s_hat, ub * Kn) if ub < 1 else self.L_need(s_hat, Kn)
        # the adjustment cost is paid once, the capital earns over its life: the cost is spread over that life at the
        # user-cost rate (equivalently, the flow gain is compared with the annuity value of the one-time cost)
        amort = (rL - (self.anchored(self.pi_e) if mp.jorgenson else 0.0) + mp.delta) if mp.k_amort else 1.0
        f = lambda Kn: W * ub * self.mpk_over_mpl(ub * Kn, L(Kn)) - uK - amort * mp.phi_K * P * (Kn - K) / K   # decreasing in K'
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
        n = self.inv[:, None]; s_ = np.sqrt(self.sig2)[:, None]; P_ = p[:, None]; M = mu[:, None]; v_ = v[:, None]
        Cp = lambda y: self.mc_y(w, y)
        lo = n + 0 * w; hi = n + 3 * M * np.exp(4 * s_) + 0 * w
        for _ in range(50):
            S = 0.5 * (lo + hi)
            pr = ndtr((np.log(M / S) - s_ ** 2 / 2) / s_)
            g = P_ * pr + v_ * (1 - pr) - Cp(S - n)
            lo = np.where(g > 0, S, lo); hi = np.where(g > 0, hi, S)
        S_u = 0.5 * (lo + hi)
        L_u = self.L_need(S_u - n)
        L = np.minimum(L_u, E)
        S = n + self.prod(L)
        d2 = (np.log(M / S) - s_ ** 2 / 2) / s_
        Emin = M - (M * ndtr(d2 + s_) - S * ndtr(d2))
        val = P_ * Emin + v_ * (S - Emin) - w * L
        k = val.argmax(1); idx = np.arange(F)
        return w[idx, k], E[idx, k]

    def stock_target(self, W, p, mu, v, carry, sd):
        """Next quarter's stock: p Pr(D > S) + v Pr(D < S) = marginal cost of producing S - carry this quarter."""
        mp = self.mp; a_, Z = mp.alpha, mp.Z
        Cp = lambda y: self.mc_y(W, y)
        lo = carry.copy(); hi = carry + 3 * mu * np.exp(4 * sd)
        for _ in range(50):
            S = 0.5 * (lo + hi)
            pr = ndtr((np.log(mu / S) - sd ** 2 / 2) / sd)
            g = p * pr + v * (1 - pr) - Cp(S - carry)
            lo = np.where(g > 0, S, lo); hi = np.where(g > 0, hi, S)
        return 0.5 * (lo + hi)

    def labor_for_cost(self, cost, p, mu, v):
        """Production labor at which the stocking FOC holds, p Pr(D > S) + v Pr(D < S) = cost x dL/dy, for each
        firm (rows) and each candidate labor cost (columns)."""
        mp = self.mp; a_, Z = mp.alpha, mp.Z
        n = self.inv[:, None]; s_ = np.sqrt(self.sig2)[:, None]; P_ = p[:, None]; M = mu[:, None]; v_ = v[:, None]
        Cp = lambda y: self.mc_y(cost, y)
        lo = n + 0 * cost; hi = n + 3 * M * np.exp(4 * s_) + 0 * cost
        for _ in range(50):
            S = 0.5 * (lo + hi)
            pr = ndtr((np.log(M / S) - s_ ** 2 / 2) / s_)
            g = P_ * pr + v_ * (1 - pr) - Cp(S - n)
            lo = np.where(g > 0, S, lo); hi = np.where(g > 0, hi, S)
        S = 0.5 * (lo + hi)
        L = self.L_need(S - n)
        d2 = (np.log(M / S) - s_ ** 2 / 2) / s_
        Emin = M - (M * ndtr(d2 + s_) - S * ndtr(d2))
        val = P_ * Emin + v_ * (S - Emin) - cost * L
        return L, val

    def full_cost(self, x, We, a, Q, r):
        """Cost per worker per quarter at relative pay x: the wage plus the cost of replacing departures (recruiting
        time c_v per vacancy, vacancies filling at rate min(1, a x^eta)), amortized at the firm's cost of funds over
        expected tenure (separations, quits Q x^-eta). a, Q: the firm's beliefs at relative pay 1."""
        mp, hp = self.mp, self.hp
        turnover = hp.sep + Q * x ** (-mp.eta_L)
        if mp.hiring == "convex":                     # replacing departures at the convex recruiting cost
            T = mp.q_ref / np.minimum(1.0, np.maximum(a, 1e-3) * x ** mp.eta_L)   # better pay fills openings faster
            return x * We * (1 + (r + turnover) * mp.kappa_h * T * turnover ** mp.psi_h)
        q = np.minimum(1.0, a * x ** mp.eta_L)
        return x * We * (1 + (r + turnover) * mp.c_v / np.maximum(q, 1e-3))

    def hires_choice(self, p, mu, w, L_stay, r, Q, q_bel):
        """Hiring problem (recruiting intensity with convex costs, Blatter et al. / Gavazza-Mongey-Violante): choose
        hires H >= 0 to maximize  J-value of the hires - recruiting cost,
            marginal value of a worker  J(L) = max(0, p MPL(L) - w) / (r + sep + Q),  zero beyond the labor needed for
            expected sales;  recruiting time  R(H) = kappa T H (H / L)^psi,  T = q_ref / q_bel (published tightness).
        FOC: J(L_stay + H) = w (1 + psi) kappa T (H / L)^psi."""
        mp, hp = self.mp, self.hp; a_, Z = mp.alpha, mp.Z
        L_flow = self.L_need(mu)
        T = mp.q_ref / np.maximum(q_bel, 1e-3)
        Lb = np.maximum(L_stay, 1.0)
        # the output a hire adds is sold along the same demand belief the price and stock decision uses (short run);
        # with hire_long, along the long-run elasticity (customers come over the hire's tenure)
        e_f = self.elast(short=not mp.hire_long)
        # marginal revenue product along the firm's own demand belief: selling more output y means a lower price,
        # p(y) = p (y / mu)^(-1/e), and marginal revenue p(y) (1 - 1/e)
        # newsvendor value of a marginal unit: output joins the stock the firm expects to carry in; a unit is sold
        # (at marginal revenue) if demand exceeds the stock, and otherwise carried forward at its inventory value v
        carry0 = np.maximum(self.inv - mu, 0.0) * (1 - self.decay) if mp.mrp_stock else 0.0
        v_inv = getattr(self, "_v_inv", None)
        sd_d = np.sqrt(self.sig2)
        def mrp(L):
            L = np.maximum(L, 1e-9)
            y = self.prod(L)
            mr = p * (y / mu) ** (-1 / e_f) * (1 - 1 / e_f)
            if mp.mrp_stock and v_inv is not None:
                Stk = y + carry0
                pr_sell = ndtr((np.log(mu / np.maximum(Stk, 1e-12)) - sd_d ** 2 / 2) / sd_d)
                mr = mr * pr_sell + v_inv * (1 - pr_sell)
            val = mr * self.mpl(L)
            if getattr(self, "_ell", None) is not None:      # net of the wear on machines the worker's output needs
                val = val - np.where(L <= self._ell * self._ycap, self._cs * self._kx / self._ell, 0.0)
            return val
        self._mrp = mrp
        J = lambda L: np.maximum(0.0, mrp(L) - w) / (r + hp.sep + Q)
        g = lambda H: J(L_stay + H) - w * (1 + mp.psi_h) * mp.kappa_h * T * (H / Lb) ** mp.psi_h
        lo = np.zeros_like(L_stay); hi = np.maximum(L_stay, 1.0)
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            pos = g(mid) > 0
            lo = np.where(pos, mid, lo); hi = np.where(pos, hi, mid)
        H = 0.5 * (lo + hi)
        R = mp.kappa_h * T * H * (H / Lb) ** mp.psi_h
        return H, R, L_flow

    def pay_review(self, We, a, Q, r, We_piece=None, gL=None):
        """Annual pay review (Salop turnover model): the relative pay that minimizes cost per worker,
            min over x:  x We [1 + (r + sep + Q x^-eta) c_v / min(1, a x^eta)],
        with We the market pay expected over the coming year (last observed x last year's market growth). The
        answer does not depend on the firm's sales: those set how many it hires, not what it pays."""
        xg = np.exp(np.linspace(np.log(self.mp.x_lo), np.log(self.mp.x_hi), 281))[None, :]
        mp, hp = self.mp, self.hp
        if mp.pay_hiring and gL is not None:
            # cost per worker per quarter given the hiring the firm plans: pay plus the recruiting time needed for the
            # hires it will make, h = planned net growth + departures (quits fall with relative pay), at the convex
            # recruiting cost kappa T h^(1+psi), T falling with relative pay (openings fill faster). A firm planning to
            # shrink faster than workers leave makes no hires, so pay buys it nothing: it would like to cut, which
            # incumbents' effort does not allow (Bewley; Akerlof and Yellen)
            Wv = np.asarray(We)[:, None] if np.ndim(We) else We
            dep = hp.sep + Q[:, None] * xg ** (-mp.eta_L)
            h = np.maximum(0.0, gL[:, None] + dep)
            T = mp.q_ref / np.minimum(1.0, np.maximum(a[:, None], 1e-3) * xg ** mp.eta_L)
            c = xg * Wv * (1 + mp.kappa_h * T * h ** (1 + mp.psi_h))
        else:
            c = self.full_cost(xg, np.asarray(We)[:, None] if np.ndim(We) else We, a[:, None], Q[:, None], r[:, None])
        if self.mp.weiss and We_piece is not None:
            # pooled firms: cost per unit of effective labor = cost per head / average quality attracted (Weiss)
            We_pool = float(np.mean(np.asarray(We)[~self.piece])) if np.ndim(We) else We
            qb = self.quality_belief(xg[0], We_pool, We_piece)[None, :]
            c = np.where(self.piece[:, None], c, c * self.zbar / qb)
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

    def elast(self, short=True, share=None):
        """Own-price elasticity of a firm's demand implied by buyers' choices: nominal spending shares are logit in
        price (elasticity eps (1 - s)), so quantities add 1; in a customer market only a share rho_c of buyers re-shop
        each quarter, so the short-run elasticity is 1 + rho_c eps (1 - s)."""
        mp = self.mp
        s = self.share_prev if share is None else share
        lam = self.lam_sh if (mp.share_learn and getattr(self, "lam_sh", None) is not None) else 1.0
        if mp.rho_c < 1:
            return 1 + (mp.rho_c if short else 1.0) * lam * mp.eps * (1 - s)
        return mp.eps * (1 - s)

    def anchored(self, g_obs):
        """Belief about nominal growth: Bayesian with a prior centered on the published inflation target that is
        never forgotten (weight anchor) and evidence discounted at a constant rate (the EWMA g_obs)."""
        mp = self.mp
        return mp.anchor * mp.pi_star + (1 - mp.anchor) * g_obs

    def P_ref(self):
        """Firms' expected price index this quarter: last observed x (1 + believed inflation)."""
        return self.P_prev * (1 + self.anchored(self.pi_e)) if self.mp.wage_posting else self.P_prev

    def P_ref_sec(self):
        """Expected price index of each firm's own sector (its competitors' prices): the published sector index x
        (1 + believed inflation). Without separate sectors, the overall index."""
        if not (self.mp.services and self.mp.weiss):
            return self.P_ref()
        return self.P_prev_sec[self.sec] * (1 + self.anchored(self.pi_e))

    def quality_belief(self, x, We_pool, We_piece):
        """A pooled firm's belief of the average skill it attracts at relative pay x, from the published skill
        distribution and published pay: a worker of skill z compares x We_pool zbar (pooled, per head) with
        We_piece z (paid by output) through the same random-utility choice workers use."""
        mp = self.mp
        zs = self._zpub
        sp = max(mp.n_piece, 1) / mp.F
        lw_pool = mp.eta_L * np.log(np.maximum(x * We_pool * self.zbar, 1e-12))[..., None]
        lw_piece = mp.eta_L * np.log(np.maximum(We_piece * zs, 1e-12))
        pi = 1.0 / (1.0 + sp / (1 - sp) * np.exp(np.clip(lw_piece - lw_pool, -50, 50)))
        if mp.min_wage and self.W_min > 0:
            pi = np.where(We_piece * zs < self.W_min, 1.0, pi)       # output below the minimum: only pooled firms hire
        return (pi * zs).sum(-1) / np.maximum(pi.sum(-1), 1e-12)
