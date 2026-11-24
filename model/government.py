"""Treasury and Fed: tax schedule, term-premium pass-through, balance-sheet check."""
from __future__ import annotations
import copy
import numpy as np
from scipy.special import ndtr, ndtri


class Government:
    def tax_schedule(self, x, med):
        # liability under the bracket schedule for taxable income x (thresholds scale with median earnings)
        mp = self.mp
        tax = np.zeros_like(x)
        b = np.array(mp.brk_y) * med
        for k, rk in enumerate(mp.rate_y):
            hi = b[k + 1] if k + 1 < len(b) else np.inf
            tax += rk * np.clip(x - b[k], 0.0, hi - b[k])
        return tax

    def _tp_dev(self):
        # the term premium on long Treasuries moves required returns on other long assets (portfolio balance)
        # relative to the premium investors have come to see as normal (a discounted average of observed premia), so
        # a premium that stays low for years stops lowering required returns
        return self.TP - self.tp_norm if (self.mp.long_bonds and self.mp.tp_pass) else 0.0

    # ------------------------------------------------------------------ accounting
    def reserves(self):
        """Balance-sheet check (must be exactly zero): the bank's reserves implied by its own balance sheet
        (deposits + equity + discount loans - loans - its Treasury bills) minus the reserves on the Fed's balance
        sheet (bills held + discount loans - TGA). self.g is the Treasury's net position (TGA - bills), so the
        difference reduces to the expression below."""
        deposits = (self.b.sum() + self.m.sum() + self.div_pending.sum() + self.g + self.Fc + getattr(self, 'F_B', 0.0)
                    + getattr(self, 'q_L', 0.0) * getattr(self, 'X_fund', 0.0) + getattr(self, 'E_F', 0.0))
        if not self.mp.fund:
            deposits += self.a.sum()
        return deposits + self.E_B - self.loan.sum() - (self.L_inst.sum() if getattr(self, 'L_inst', None) is not None else 0.0)

    def public_interest(self, i):
        """Interest among Treasury, Fed, banks and fund for the quarter (positions set at the end of last quarter).
        Returns the IOR, discount rate, bill yield, and the interest received by banks, by the fund and by the
        Treasury (negative: paid)."""
        mp = self.mp
        # Treasury bills pay the target rate; the bank earns bill interest and IOR and pays the discount rate; the Fed
        # earns bill and discount-loan interest, pays IOR, and remits the rest to the Treasury (positions held during
        # the quarter, set at the end of last quarter)
        ior, dr = i + mp.ior_spread, i + mp.dr_spread
        ib = ior + mp.dealer_margin                           # auction stop-out yield: the dealer's bid
        self.i_bill = ib
        # long bonds pay their coupons (one unit per bond held). The Fed books its long bonds at amortized cost and
        # remits its net income; a loss is carried as a deferred asset and paid down from later income (2022-24)
        bank_int = ib * (self.Bills - self.SOMA - self.F_B) + ior * self.R_res - dr * self.DW + self.X_bk.sum()
        fund_int = ib * self.F_B + self.X_fund
        NI_F = (ib * self.SOMA + dr * self.DW - ior * self.R_res + self.X_F
                - self.qb_F * (1 - mp.delta_c) * self.X_F + self._real_F)
        self._real_F = 0.0
        if mp.long_bonds and not mp.fed_indemnity:
            pay_ = min(max(NI_F, 0.0), self.def_F)
            self.def_F += max(-NI_F, 0.0) - pay_
            remit = max(NI_F, 0.0) - pay_
        else:
            remit = NI_F
        self.E_F += ib * self.SOMA + dr * self.DW - ior * self.R_res + self.X_F - remit
        g_int = -ib * self.Bills - self.X_L + remit           # the balance sheets close (the Fed keeps the rest)
        # the part of the coupons that repays principal (at the going price) is not a cost in the budget the tax rule
        # sees: interest is the coupon less that repayment (the yield on the debt's market value)
        self._amort_L = self.q_L * (1 - mp.delta_c) * self.X_L
        self._remit = remit
        return ior, dr, ib, bank_int, fund_int, g_int

    def treasury_fed_close(self, i):
        """End of quarter: the Treasury issues long bonds and bills, the Fed's balance-sheet rule sets its long
        holdings and the term premium, holders mark bonds to market, the Fed absorbs the bill supply private holders
        will not take, and reserves are the result."""
        mp = self.mp
        ib = self.i_bill
        # Treasury: keeps its account at the Fed near target and issues bills for the rest of its net position;
        # Fed: open-market operations to hold reserves at target; the discount window covers what bills cannot
        GDP_a_ = sum(self.GDP_hist[-4:])
        TGA_t = mp.tga_target * GDP_a_
        if mp.long_bonds:
            d_ = mp.delta_c
            q0 = self.q_L
            XL0, XF0, Xfd0, Xbk0 = self.X_L, self.X_F, self.X_fund, self.X_bk.copy()
            # Fed's term-premium target: holds it at normal against supply; cuts it when the short-rate rule asks
            # for a rate below zero (QE); otherwise lets it return to normal gradually (QT)
            tp_back = self.TP_star + mp.g_tp * (mp.tp0 - self.TP_star)
            # QE only once the policy rate is at its lower bound (the rule's smoothing keeps the rate positive for a
            # while after the rule itself turns negative; the Fed bought long bonds only at the lower bound)
            at_lb = i <= (0.0 if mp.zlb else -1.0) + 1e-9
            cut_ = mp.kappa_tp * min(self._rule, 0.0) if at_lb else 0.0
            self.TP_star = min(max(mp.tp0 + cut_, mp.tp0 - mp.tp_cut_max), tp_back)
            # long yield: expected average short rate over the bond's duration (today's bill rate and the Fed's
            # published neutral-rate estimate) plus the term premium
            yhat = max(mp.w_now * ib + (1 - mp.w_now) * self.i_star_est, 0.0)
            self.TP_star = max(self.TP_star, -yhat)            # the Fed does not push the long yield below zero
            g_pre = self.g + q0 * XL0                          # Treasury's position before revaluing its long debt
            TP = self.TP_star
            for _ in range(100):
                q = 1.0 / max(1.0 + yhat + TP - d_, 0.01)
                g_new = g_pre - q * d_ * XL0
                Dm = max(TGA_t - g_new, 0.0)                   # marketable debt the Treasury must carry
                XL = max(d_ * XL0, mp.phi_L * Dm / q)          # issues long bonds to its target share (no buybacks)
                if mp.fed_tp_rule:
                    Xp = max((mp.h0 + (self.TP_star - mp.tp0) / mp.lam_tp) * GDP_a_ / q, 0.0)
                    if XL - Xp >= min(d_ * XF0, XL) - 1e-12 and TP == self.TP_star:
                        break                                  # the Fed buys what private holders will not, at its target
                    # the target would need sales: the Fed shrinks only by letting its bonds run off (as in 2017-19
                    # and 2022-), and the market sets the premium on what private holders must carry
                    Xp = XL - min(d_ * XF0, XL)
                else:
                    Xp = XL - min(XF0, XL)                     # passive Fed: keeps its holdings (reinvests runoff)
                TP_n = max(mp.tp0 + mp.lam_tp * (q * Xp / max(GDP_a_, 1e-12) - mp.h0), -yhat)
                if abs(TP_n - TP) < 1e-12:
                    break
                TP = 0.5 * TP + 0.5 * TP_n
            XF = XL - Xp
            self.g = g_new                                     # issuance at the market price leaves net worth unchanged
            self.q_L, self.TP, self.y_L = q, TP, yhat + TP
            self.tp_norm += mp.g_tpn * (TP - self.tp_norm)
            # Fed: market value and book value of its holdings (realized gains on sales are booked next quarter)
            self.E_F += q * d_ * XF0 - q0 * XF0
            bv = self.qb_F * d_ * XF0
            dX = XF - d_ * XF0
            if dX >= 0:
                bv += q * dX
            else:
                self._real_F += (q - self.qb_F) * (-dX); bv += self.qb_F * dX
            self.qb_F = bv / XF if XF > 1e-12 else q
            # banks mark their long bonds to market (duration risk on their equity)
            reval = q * d_ * Xbk0 - q0 * Xbk0
            if mp.multi_bank:
                self.E_k += reval; self.E_B = float(self.E_k.sum())
            else:
                self.E_B += float(reval.sum())
            Xbk_tot = mp.bank_long * Xp
            Xfd = Xp - Xbk_tot
            self.Fc -= q * (Xfd - d_ * Xfd0)                   # the fund buys (sells) at the market price
            self.X_L, self.X_F, self.X_fund = XL, XF, Xfd
            self.X_bk = np.full(mp.NB, Xbk_tot / mp.NB)
        MV_L, MV_F = self.q_L * self.X_L, self.q_L * self.X_F
        self.Bills = max(TGA_t - self.g - MV_L, 0.0)
        self.TGA = self.g + self.Bills + MV_L
        # private demand at the going yield: the fund's portfolio share and the banks' liquidity share of deposits;
        # the Fed absorbs the rest of the supply, so yields stay put however much debt is issued (2020), and reserves
        # are the result
        Dtot_ = np.maximum(self.b, 0.0).sum() + np.maximum(self.m, 0.0).sum() + self.Fc + self.div_pending.sum()
        bank_dem = mp.bank_bills * Dtot_
        self.SOMA = float(np.clip(self.Bills - self.F_B - bank_dem, 0.0, self.Bills))
        if self.F_B > self.Bills - self.SOMA:                 # the Fed's purchases also come out of the fund's bills
            dfb = self.F_B - max(self.Bills - self.SOMA, 0.0)
            self.F_B -= dfb; self.Fc += dfb
        self.DW = max(self.TGA + self.E_F - self.SOMA - MV_F, 0.0)
        self.R_res = self.SOMA + MV_F + self.DW - self.TGA - self.E_F

    def policy_rate(self, i_shock=0.0):
        """Fed: the Taylor rule on observed inflation (last four quarters) and last quarter's unemployment, with the
        neutral-rate estimate revised when inflation keeps missing the target; smoothed, floored at zero."""
        mp = self.mp
        pi4 = np.mean(self.pi_hist[-4:])                      # observed inflation, last four quarters
        if mp.rstar_learn:
            # the central bank revises its estimate of the neutral rate when inflation keeps missing the target
            # (Laubach-Williams style; FOMC long-run rate projections move slowly): a small step each quarter
            self.i_star_est += mp.g_rstar * (pi4 - mp.pi_star)
        rule = self.i_star_est + mp.phi_pi * (pi4 - mp.pi_star) - mp.phi_u * (self.u_prev - mp.u_star)
        self._rule = rule
        i = max(0.0 if mp.zlb else -1.0, (self.i_prev if mp.fix_policy else mp.rho_i * self.i_prev + (1 - mp.rho_i) * rule) + i_shock)
        self.i_prev = i - i_shock
        return i

    def budget_rule(self):
        """Treasury: sets the tax rate for the quarter from last quarter's structural budget and the debt gap."""
        mp = self.mp
        GDP_a = sum(self.GDP_hist[-4:])                       # last four quarters, nominal
        debt = -self.g
        if self.budget_prev is not None and mp.tax_split:
            spend_net, wb_prev = self.budget_prev
            need = spend_net - (mp.d_target * GDP_a * (self.gY + mp.pi_star) if mp.deficit_rule else 0.0)
            # primary-surplus response to debt, suspended while unemployment is well above normal (escape clauses:
            # consolidation waits for the recovery)
            gate = float(np.clip(1 - (self.u_prev - mp.u_star) / 0.03, 0.0, 1.0))
            need += gate * mp.bohn / 4 * (debt - mp.d_target * GDP_a)
            # the base at normal unemployment (cyclically adjusted, so a slump does not raise the rate)
            base_y = max(self._base_y_lr * GDP_a / 4 * (1 - mp.u_star) / max(1 - self.u_prev, 0.3), 1e-12)
            sh = (need - mp.tau_p * wb_prev - self._rev_c_lr * GDP_a / 4) / base_y          # structural income-tax rate
            self.tau_s = sh if self.tau_s is None else (1 - mp.g_tax) * self.tau_s + mp.g_tax * sh
            self.tau = float(np.clip(self.tau_s, mp.tau_min, 3.0))   # scale on the schedule's marginal rates
        elif self.budget_prev is not None:
            spend_net, wb_prev = self.budget_prev
            # the rule allows the deficit that keeps debt at its target share of GDP as the economy grows
            # (published trend real growth plus the inflation target, not realized nominal growth, which would feed inflation)
            sh = (spend_net - (mp.d_target * GDP_a * (self.gY + mp.pi_star) if mp.deficit_rule else 0.0)) / wb_prev
            self.tau_s = sh if self.tau_s is None else (1 - mp.g_tax) * self.tau_s + mp.g_tax * sh
            self.tau = float(np.clip(self.tau_s + mp.phi_G * (debt / GDP_a - mp.d_target), mp.tau_min, 0.7))
