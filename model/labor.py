"""Labor market: earnings by job, published average pay, matching of workers to firms."""
from __future__ import annotations
import copy
import numpy as np
from scipy.special import ndtr, ndtri


class Labor:
    # ------------------------------------------------------------------ labor market with posted wages
    def earn_fac(self, job, z=None):
        """What a worker earns per unit of posted pay: own skill at output-paying firms, the average skill (one
        wage per head) at pooled firms."""
        z = self.z if z is None else z
        jf = np.maximum(job, 0)
        return np.where(self.piece[jf], z, self.zbar)

    def Wbar_obs(self, sel=None):
        """Going wage: employment-weighted average of posted wages (published statistic), optionally for a subset
        of firms."""
        on = self.job >= 0
        L = np.bincount(self.job[on], self.z[on], self.mp.F)
        if sel is not None:
            L = L * sel
            return float(self.w @ L / L.sum()) if L.sum() > 0 else float(self.w[sel].mean())
        return float(self.w @ L / L.sum()) if L.sum() > 0 else float(self.w.mean())

    def labor_market(self, w, We, L_star, work, L_target=None, q_fill=None, hires=None):
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
        if mp.weiss and mp.min_wage and self.W_min > 0:
            jf_ = np.maximum(job, 0)
            job[(job >= 0) & self.piece[jf_] & (w[jf_] * z < self.W_min)] = -1   # output worth less than the minimum
        # layoffs where the workforce exceeds the plan
        def keep_upto(sel, cap, key):
            idx = np.nonzero(sel)[0]
            f = job[idx] if cap is not None else None
            return idx, f
        on = np.nonzero(job >= 0)[0]
        if len(on) == 0:
            on = np.zeros(0, int)
        order = on[np.lexsort((k_lay[on], job[on]))] if len(on) else on
        if len(order):
            f_o = job[order]
            cz = np.cumsum(z[order])
            start = np.searchsorted(f_o, np.arange(F))
            base = np.where(start > 0, cz[np.maximum(start - 1, 0)], 0.0)
            cum_in = cz - base[f_o]
            laid = (cum_in - z[order]) >= L_star[f_o] + 1e-12
            job[order[laid]] = -1
            self._laid_z = float(z[order[laid]].sum())
        L_now = np.bincount(job[job >= 0], z[job >= 0], F)
        if hires is not None:                                         # hires the firm wants; openings implied at the believed fill rate
            V = np.minimum(hires, np.maximum(0.0, L_target - L_now) + 1e-12); vac = V / max(self.q_bel, 0.05)
        elif L_target is None:
            V = np.maximum(0.0, L_star - L_now); vac = V.copy()
        else:                                                         # hires wanted, and vacancies posted for them
            V = np.maximum(0.0, L_target - L_now); vac = V.copy()   # one open position per hire wanted (requisitions)
        U_seek = z[work & (job < 0)].sum()
        quits = np.zeros(F); apps = np.zeros(F); hired = np.zeros(F)
        We_ref = float(np.mean(We)) if np.ndim(We) else We

        def round_(seek, u_c, key, V_left, count_apps):
            wt = (V_left > 1e-9) * (vac if L_target is not None else 1.0) * (w / We) ** mp.eta_L   # seekers meet hiring firms (in proportion to vacancies if posted)
            if wt.sum() <= 0 or seek.sum() == 0:
                return V_left
            idx = np.nonzero(seek)[0]
            cur = job[idx]
            if mp.weiss:
                # each worker compares what it would earn: own skill x pay at output-paying firms, pay per head at
                # pooled firms
                fac = np.where(self.piece[None, :], z[idx, None], self.zbar)
                wti = (V_left > 1e-9)[None, :] * (vac if L_target is not None else 1.0)[None, :] * (w[None, :] * fac / We_ref) ** mp.eta_L
                if mp.min_wage and self.W_min > 0:
                    wti = wti * ~(self.piece[None, :] & (w[None, :] * fac < self.W_min))   # not hired below the minimum
                cdfi = np.cumsum(wti, 1); tot = cdfi[:, -1]
                good = tot > 0
                ch = np.where(good, (cdfi < (u_c[idx] * tot)[:, None]).sum(1), 0)
                ch = np.minimum(ch, F - 1)
                e_new = w[ch] * fac[np.arange(len(idx)), ch]
                e_cur = w[np.maximum(cur, 0)] * fac[np.arange(len(idx)), np.maximum(cur, 0)]
                ok = good & ((cur < 0) | (e_new * np.exp(zeta[idx]) > e_cur))
            else:
                cdf = np.cumsum(wt) / wt.sum()
                ch = np.minimum(np.searchsorted(cdf, u_c[idx]), F - 1)
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
        # further rounds within the quarter: unemployed contacts who were turned away apply again (a quarter has many
        # weeks of search); each round draws fresh choices (fixed-size draws)
        for rr in range(mp.search_rounds - 1):
            seek2 = work & (job < 0) & contact
            if rr == 0:
                V2 = round_(seek2, u_c2, k_2, V2, False)
            else:
                V2 = round_(seek2, rng.random(N), rng.random(N), V2, False)
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
