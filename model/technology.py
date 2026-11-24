"""Production technology: unit cost, utilization, production function, marginal cost and products, technique adoption."""
from __future__ import annotations
import copy
import numpy as np
from scipy.special import ndtr, ndtri


class Technology:
    def unit_cost(self, W, cK, AK=None, AL=None):
        """Cost of a unit of output at wage W and capital rental cK with the firm's technique (CES dual)."""
        mp = self.mp; a_ = mp.alpha
        if not mp.tech or getattr(self, "AK", None) is None:
            return (W / (1 - a_)) ** (1 - a_) * (cK / a_) ** a_ / mp.Z
        AK = self.AK if AK is None else AK; AL = self.AL if AL is None else AL
        s = mp.sigma_f
        return (a_ ** s * (cK / AK) ** (1 - s) + (1 - a_) ** s * (W / AL) ** (1 - s)) ** (1 / (1 - s))

    # ---------------------------------------------------------------- the firm's technology
    def util_setup(self, W):
        """Short-run cost structure for the quarter at the firm's wage W: with utilization variable, capital services
        cost c_s = delta1 P per unit (wear), so below capacity output costs the CES unit cost c(W, c_s) per unit, with
        labor ell and capital services kx per unit of output; capacity is u_max K / kx."""
        mp = self.mp
        if not (mp.utilization and mp.tech):
            self._ell = None; return
        s_ = mp.sigma_f; om = mp.alpha
        W = np.broadcast_to(np.asarray(W, float), (mp.F,)).copy()
        P = self.P_ref()
        cs = mp.delta1 * P
        # putty-clay: machines are built for a capital-labor ratio chosen at long-run factor prices (the long-run
        # rental of capital); in the short run labor and machine-hours are used in that ratio
        uK_lr = P * np.maximum(self.r_lr_f - self.anchored(self.pi_e) + mp.delta, 1e-3)
        Wd = self.wp_lr * P if mp.invest_lr else W                # the long-run wage the machines were planned for
        c_lr = self.unit_cost(Wd, uK_lr)
        self._cs = cs
        self._ell = (1 - om) ** s_ * (Wd / self.AL) ** (-s_) / self.AL * c_lr ** s_
        self._kx = om ** s_ * (uK_lr / self.AK) ** (-s_) / self.AK * c_lr ** s_   # capital services (u K) per unit
        self._ycap = mp.u_max * self.K / self._kx

    def _uparams(self, nd):
        sh = (-1,) + (1,) * (nd - 1)
        return self._ell.reshape(sh), self._ycap.reshape(sh)

    def _tk(self, nd, K=None):
        sh = (-1,) + (1,) * (nd - 1)
        K = self.K if K is None else K
        if not self.mp.tech:
            return K.reshape(sh), None
        return (self.AK * K).reshape(sh), self.AL.reshape(sh)

    def prod(self, L, K=None):
        """Output from labor L (and capital K) with the firm's technique."""
        L = np.asarray(L, float); mp = self.mp
        if K is None and getattr(self, "_ell", None) is not None:
            ell, ycap = self._uparams(L.ndim)
            above = self._prod_ces(L, mp.u_max * self.K)
            return np.where(L <= ell * ycap, np.maximum(L, 0) / ell, np.maximum(above, ycap))
        return self._prod_ces(L, K)

    def _prod_ces(self, L, K=None):
        L = np.asarray(L, float); mp = self.mp
        ek, AL = self._tk(L.ndim, K)
        if not mp.tech:
            return mp.Z * ek ** mp.alpha * np.maximum(L, 0) ** (1 - mp.alpha)
        om, r = mp.alpha, self.rho
        return (om * ek ** r + (1 - om) * (AL * np.maximum(L, 1e-12)) ** r) ** (1 / r)

    def L_need(self, y, K=None):
        """Labor needed to make y. With low substitution capital limits capacity; output is capped below it."""
        y = np.asarray(y, float); mp = self.mp
        if K is None and getattr(self, "_ell", None) is not None:
            ell, ycap = self._uparams(y.ndim)
            return np.where(y <= ycap, np.maximum(y, 0) * ell, self._L_need_ces(y, mp.u_max * self.K))
        return self._L_need_ces(y, K)

    def _L_need_ces(self, y, K=None):
        y = np.asarray(y, float); mp = self.mp
        ek, AL = self._tk(y.ndim, K)
        if not mp.tech:
            return (np.maximum(y, 0) / (mp.Z * ek ** mp.alpha)) ** (1 / (1 - mp.alpha))
        om, r = mp.alpha, self.rho
        ycap = 0.98 * om ** (1 / r) * ek
        yy = np.clip(y, 1e-12, ycap)
        L = ((yy ** r - om * ek ** r) / (1 - om)) ** (1 / r) / AL
        return np.where(y <= 1e-12, 0.0, L)

    def mc_y(self, W, y, K=None):
        """Marginal labor cost of output, W / MPL at the labor that makes y (steeply rising beyond capacity)."""
        y = np.asarray(y, float); mp = self.mp
        if K is None and getattr(self, "_ell", None) is not None:
            # below capacity: the unit cost at the wage and the wear cost of capital services (flat); above: labor
            # added to machines at full utilization
            ell, ycap = self._uparams(y.ndim)
            Wb = np.asarray(W, float)
            sh = (-1,) + (1,) * (y.ndim - 1)
            c = Wb * ell + self._cs * self._kx.reshape(sh)       # short-run marginal cost: labor + machine wear
            return np.where(y <= ycap, c + 0 * y, np.maximum(self._mc_ces(Wb, y, mp.u_max * self.K), c))
        return self._mc_ces(W, y, K)

    def _mc_ces(self, W, y, K=None):
        y = np.asarray(y, float); mp = self.mp
        ek, AL = self._tk(y.ndim, K)
        if not mp.tech:
            return W / (1 - mp.alpha) * np.maximum(y, 0) ** (mp.alpha / (1 - mp.alpha)) / (mp.Z * ek ** mp.alpha) ** (1 / (1 - mp.alpha))
        om, r = mp.alpha, self.rho
        ycap = 0.98 * om ** (1 / r) * ek
        yy = np.clip(y, 1e-12, ycap)
        L = np.maximum(((yy ** r - om * ek ** r) / (1 - om)) ** (1 / r) / AL, 1e-12)
        mpl = (1 - om) * AL ** r * L ** (r - 1) * yy ** (1 - r)
        return W / mpl * np.exp(np.minimum(20 * np.maximum(y / ycap - 1, 0), 30.0))

    def mpl(self, L, K=None):
        L = np.maximum(np.asarray(L, float), 1e-12); mp = self.mp
        if K is None and getattr(self, "_ell", None) is not None:
            ell, ycap = self._uparams(L.ndim)
            return np.where(L <= ell * ycap, 1.0 / ell + 0 * L, self._mpl_ces(L, mp.u_max * self.K))
        return self._mpl_ces(L, K)

    def _mpl_ces(self, L, K=None):
        L = np.maximum(np.asarray(L, float), 1e-12); mp = self.mp
        y = self._prod_ces(L, K)
        if not mp.tech:
            return (1 - mp.alpha) * y / L
        ek, AL = self._tk(L.ndim, K)
        return (1 - mp.alpha) * AL ** self.rho * L ** (self.rho - 1) * y ** (1 - self.rho)

    def mpk_over_mpl(self, K, L):
        mp = self.mp
        if not mp.tech:
            return mp.alpha / (1 - mp.alpha) * L / K
        r = self.rho
        return mp.alpha / (1 - mp.alpha) * (self.AK / self.AL) ** r * (K / np.maximum(L, 1e-12)) ** (r - 1)

    def adopt(self, w, uK, p, r_own, W_agg, uK_agg):
        """Ideas and technique choice. Ideas are public (Jones 2005): new ideas, pairs (a, b) drawn from the Pareto idea
        distribution, arrive at a rate that grows at g_N and join the stock of known ideas. Each firm evaluates the
        stock at its own factor prices and adopts the cheapest technique if the unit-cost saving, valued over the
        technique's expected life (its cost of funds plus the rate at which it has seen its techniques superseded),
        exceeds the adoption cost:   (c_old - c_new) yhat / (r + turnover) > adopt_cost x p yhat."""
        mp, rng, F = self.mp, self.rng, self.mp.F
        th = mp.theta_I
        # new ideas near the current frontier (ideas far below it are never used, so they are not drawn)
        cA = self.unit_cost(W_agg, uK_agg, self.poolK, self.poolL)
        j = int(cA.argmin()); aR, bR = self.poolK[j], self.poolL[j]
        lnR = self.lnlam0 + mp.g_N * self.t - mp.pK * np.log(th * aR) - mp.pL * np.log(th * bR)
        n = int(min(rng.poisson(min(np.exp(lnR), 500.0)), 500))
        if n > 0:
            a = th * aR * rng.random(n) ** (-1 / mp.pK); b = th * bR * rng.random(n) ** (-1 / mp.pL)
            self.poolK = np.concatenate([self.poolK, a]); self.poolL = np.concatenate([self.poolL, b])
            cA = self.unit_cost(W_agg, uK_agg, self.poolK, self.poolL)
            cA = np.where(np.isfinite(cA), cA, np.inf)
            keep = cA <= 1.10 * cA.min()
            keep[int(cA.argmin())] = True                      # ideas far from useful at any nearby factor prices are dropped
            if keep.sum() > 300:
                keep = cA <= np.sort(cA)[299]
            self.poolK, self.poolL = self.poolK[keep], self.poolL[keep]
        self.n_ideas = n
        wv = np.asarray(w, float)[:, None]; uv = np.asarray(uK, float)[:, None] if np.ndim(uK) else uK
        hf = np.exp(self.a_eff)[:, None]                       # the firm's own efficiency applies to any technique
        c = self.unit_cost(wv, uv, self.poolK[None, :] * hf, self.poolL[None, :] * hf)
        k = c.argmin(1); idx = np.arange(F); hf = hf[:, 0]
        c_new = c[idx, k]
        if mp.embodied:
            # embodied: the choice is the design of the machines the firm buys from now on; installed machines keep
            # theirs (the installed technique moves toward it as new capital is installed, see the capital update)
            c_old = self.unit_cost(w, uK, self.AKn, self.ALn)
            go = (c_old - c_new) / (r_own + self.turn) > mp.adopt_cost * p
            self.AKn = np.where(go, self.poolK[k] * hf, self.AKn); self.ALn = np.where(go, self.poolL[k] * hf, self.ALn)
        else:
            c_old = self.unit_cost(w, uK)
            go = (c_old - c_new) / (r_own + self.turn) > mp.adopt_cost * p
            self.AK = np.where(go, self.poolK[k] * hf, self.AK); self.AL = np.where(go, self.poolL[k] * hf, self.AL)
        self.turn = (1 - mp.g_T) * self.turn + mp.g_T * go
        self.adopt_last = float(go.mean())
        # the adoption cost is the time of the firm's own workers (installation, retraining) this quarter
        self._adopt_L = np.where(go, mp.adopt_cost * p * np.exp(self.lnA) / w, 0.0)
