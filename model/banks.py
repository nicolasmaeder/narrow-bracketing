"""Banks' posted deposit and loan rates (best responses to rivals' posted rates)."""
from __future__ import annotations
import copy
import numpy as np
from scipy.special import ndtr, ndtri


class Banks:
    def bank_rates_k(self, i):
        """Each bank's best response to its rivals' last posted rates (mean of the others), with its own depositors
        (savings by Baumol-Tobin as in bank_rates) and its own loss rate. Shares are logit in the rate (a_D, a_L)."""
        mp = self.mp; NB = mp.NB
        rd_new = np.zeros(NB); rL_new = np.zeros(NB)
        D_all = np.maximum(self.b, 0.0)
        K_all = mp.c_tr * self.yhat * (self._spend_last if getattr(self, "_spend_last", None) is not None else self.yhat) / 2
        for k in range(NB):
            riv_d = (self.rd_k.sum() - self.rd_k[k]) / (NB - 1); riv_L = (self.rL_k.sum() - self.rL_k[k]) / (NB - 1)
            mine = self.bank_h == k
            D_h, K_h = D_all[mine], K_all[mine]
            Dtot = D_h.sum()
            def prof(x):
                x = max(x, 1e-7)
                Sx = np.maximum(D_h - np.sqrt(K_h / x), 0.0).sum()
                sh = 1.0 / (1.0 + (NB - 1) * np.exp(mp.a_D * (riv_d - x)))
                return sh * (i * Dtot - x * Sx)
            grid = np.linspace(0.0, max(i, 1e-4), 101)
            vals = np.array([prof(x) for x in grid])
            j = int(vals.argmax()); lo_, hi_ = grid[max(j - 1, 0)], grid[min(j + 1, 100)]
            fine = np.linspace(lo_, hi_, 21); rd_new[k] = max(0.0, float(fine[np.argmax([prof(x) for x in fine])]))
            lam = self.loss_k[k]
            lo, hi = riv_L - 0.05, riv_L + 0.05
            for _ in range(60):
                x = 0.5 * (lo + hi)
                sh = 1.0 / (1.0 + (NB - 1) * np.exp(-mp.a_L * (riv_L - x)))
                g = -(x * (1 - lam) - i - (1 - lam) / (mp.a_L * (1 - sh)))
                lo, hi = (x, hi) if g > 0 else (lo, x)
            rL_new[k] = 0.5 * (lo + hi)
        self.rd_k, self.rL_k = rd_new, rL_new
        return rd_new, rL_new

    def bank_rates(self, i):
        """Each bank's best response to rivals' last posted rates (all banks symmetric, so one is computed):
            deposits:  max over rd  (i - rd) s_D(rd),  s_D = e^{a_D rd} / (e^{a_D rd} + (NB-1) e^{a_D rd_rivals})
                       FOC: i - rd = 1 / (a_D (1 - s_D)), and rd >= 0 (depositors can hold cash);
            loans:     max over rL  (rL (1 - loss) - i) s_L(rL),  s_L logit with -a_L rL
                       FOC: rL = i / (1 - loss) + 1 / (a_L (1 - s_L))."""
        mp = self.mp
        NB, aD, aL, lam = mp.NB, mp.a_D, mp.a_L, self.loss_e
        def br(a, x_riv, sign, cost_fn):
            lo, hi = x_riv - 0.05, x_riv + 0.05
            for _ in range(60):
                x = 0.5 * (lo + hi)
                sh = 1.0 / (1.0 + (NB - 1) * np.exp(sign * a * (x_riv - x)))
                g = cost_fn(x, sh)
                lo, hi = (x, hi) if g > 0 else (lo, x)
            return 0.5 * (lo + hi)
        if mp.two_accounts and getattr(self, "_spend_last", None) is not None:
            # savings rate with checking: a bank values every deposit it keeps at the policy rate (what replacing it
            # would cost) and pays rd only on savings; within the bank, households split balances by Baumol-Tobin,
            # S(rd) = sum max(D_h - sqrt(K_h / rd), 0), K_h = c_tr yhat_h spend_h / 2; across banks they choose by logit
            #   max over rd   s(rd) [ i D - rd S(rd) ],   given rivals' last posted savings rate
            D_h = np.maximum(self.b, 0.0); K_h = mp.c_tr * self.yhat * self._spend_last / 2
            Dtot = D_h.sum()
            def prof(x):
                x = max(x, 1e-7)
                Sx = np.maximum(D_h - np.sqrt(K_h / x), 0.0).sum()
                sh = 1.0 / (1.0 + (NB - 1) * np.exp(aD * (self.rd_post - x)))
                return sh * (i * Dtot - x * Sx)
            grid = np.linspace(0.0, max(i, 1e-4), 201)
            vals = np.array([prof(x) for x in grid])
            k = int(vals.argmax()); lo_, hi_ = grid[max(k - 1, 0)], grid[min(k + 1, 200)]
            fine = np.linspace(lo_, hi_, 41); rd = float(fine[np.argmax([prof(x) for x in fine])])
        else:
            rd = br(aD, self.rd_post, 1.0, lambda x, sh: (i - x) - 1.0 / (aD * (1 - sh)))          # >0: raise rd
        rd = max(0.0, rd)
        rL = br(aL, self.rL_post, -1.0, lambda x, sh: -(x * (1 - lam) - i - (1 - lam) / (aL * (1 - sh))))  # >0: raise rL
        self.rd_post, self.rL_post = rd, rL
        return rd, rL

    def banks_close(self):
        """End of quarter with several banks: recapitalization, customers re-shopping, and each bank's positions for
        next quarter (deposits, loans, bills, long bonds, reserves)."""
        mp = self.mp
        N, F = self.hp.N, mp.F
        NB = mp.NB
        # a bank with negative equity is recapitalized by its owner (the fund) for now
        neg = np.minimum(self.E_k, 0.0)
        if neg.sum() < 0:
            self._recap = float(-neg.sum()); self.Fc -= self._recap; self.E_k -= neg; self.E_B = float(self.E_k.sum())
        # competition: a share of customers re-shop and choose a bank by logit on posted rates (balances move
        # with them; the payment between banks settles in reserves)
        sh_ = self.rng.random(N) < mp.sw_h
        if sh_.any():
            pd_ = np.exp(mp.a_D * (self.rd_k - self.rd_k.max())); pd_ /= pd_.sum()
            self.bank_h[sh_] = self.rng.choice(NB, sh_.sum(), p=pd_)
        sf_ = self.rng.random(F) < mp.sw_f
        if sf_.any():
            pl_ = np.exp(-mp.a_L * (self.rL_k - self.rL_k.min())); pl_ /= pl_.sum()
            self.bank_f[sf_] = self.rng.choice(NB, sf_.sum(), p=pl_)
        # positions held next quarter: deposits, loans, bills (dealer holdings split by deposits), and the
        # reserve position (lent to or borrowed from other banks at IOR)
        Dk = (np.bincount(self.bank_h, np.maximum(self.b, 0.0) + self.div_pending, NB)
              + np.bincount(self.bank_f, np.maximum(self.m, 0.0), NB))
        Dk = Dk + self.Fc * Dk / max(Dk.sum(), 1e-12)
        Lk = (np.bincount(self.bank_f, self.loan, NB) + np.bincount(self.bank_h, np.maximum(-self.b, 0.0), NB)
              + (np.bincount(self.bank_h, self.L_inst, NB) if getattr(self, "L_inst", None) is not None else 0.0))
        Bb = max(self.Bills - self.SOMA - self.F_B, 0.0)
        self.Bb_k = Bb * Dk / max(Dk.sum(), 1e-12)
        self.X_bk = self.X_bk.sum() * Dk / max(Dk.sum(), 1e-12)
        # (discount-window loans add to the borrowing banks' reserves; split by deposits)
        self.res_k = Dk + self.E_k - Lk - self.Bb_k - self.q_L * self.X_bk + self.DW * Dk / max(Dk.sum(), 1e-12)
        self._Dk_prev = Dk

    def post_rates(self, i):
        """Banks post deposit and loan rates given the policy rate; returns the published average deposit and loan
        rates (each household and firm faces its own bank's)."""
        mp = self.mp
        if mp.multi_bank:
            rdk, rLk = self.bank_rates_k(i)
            self.rb_h = rdk[self.bank_h]                     # each household's deposit rate (its bank's)
            w_d = np.bincount(self.bank_h, np.maximum(self.b, 0.0), mp.NB)
            rb = float(w_d @ rdk / max(w_d.sum(), 1e-12))    # deposit-weighted average (published)
            rL = float(rLk.mean())
        elif mp.bank_objective:
            rb, rL = self.bank_rates(i)
        else:
            rb = max(0.0, i - mp.m_b)
        self.rb_lr = (1 - mp.g_lr) * self.rb_lr + mp.g_lr * rb
        if not mp.bank_objective:
            rL = (1 + i + mp.m_L) / (1 - self.loss_e) - 1
        return rb, rL
