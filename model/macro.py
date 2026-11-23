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
from scipy.special import ndtr, ndtri
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
        self.inv_decay_sv = 0.99    # services: produced to order within the quarter, unused capacity is lost
        self.mrp_stock = True       # a hire's output adds to the stock carried in (not to zero stock)
        self.phi_K = 2.0            # convex capital adjustment cost (x P K x (dK/K)^2 / 2)
        self.g_A = 0.10             # firms' forgetting rate on own-demand evidence (level)
        self.g_sig = 0.05           # forgetting rate on own forecast errors (noise)
        # bank
        self.m_b = 0.005            # deposit rate = policy - m_b (floor 0)
        self.m_a = 0.0              # retirement-account rate = policy + m_a
        self.m_L = 0.005            # loan rate margin (used only if bank_objective is False)
        # banks (Monti-Klein in monopolistic competition): NB symmetric banks; depositors and borrowers choose banks by
        # random utility in the rate (logit semi-elasticities a_D, a_L); each bank sets its rates to maximize expected
        # profit given rivals' last posted rates, its funding cost (policy rate) and its own loss history
        self.two_accounts = True    # households split liquid balances between checking (no interest) and savings
        self.c_tr = 0.001           # cost of a transfer between savings and checking, share of quarterly income (Baumol-Tobin)
        self.bank_objective = True
        self.NB = 5
        self.a_D = 150.0            # depositors' semi-elasticity of bank choice (per unit of quarterly rate): savings spread about 2%/yr
        self.a_L = 250.0            # borrowers' semi-elasticity: loan spread about 2%/yr
        self.g_loss = 0.05          # loss-rate gain
        self.cap_B = 0.08           # bank capital target, share of loans
        # government
        self.d_target = 1.0         # government debt target, share of annual GDP
        self.phi_G = 0.02           # tax response to debt gap
        self.g_tax = 0.05           # gain on the observed budget share used to set the tax rate
        # central bank
        self.i_star = 0.0125        # neutral policy rate per quarter (5% a year)
        self.pi_star = 0.005        # inflation target per quarter (2% a year)
        self.u_star = 0.05
        self.phi_pi, self.phi_u = 1.5, 0.25   # Taylor (1993): 0.5 on the output gap, about 1 on the unemployment gap (Okun), 0.25 per quarter
        self.zlb = True             # lower bound on the policy rate (off only in diagnostics)
        self.rstar_learn = True     # neutral-rate estimate revised slowly
        self.g_rstar = 0.01
        self.rho_i = 0.85           # interest-rate smoothing (quarterly; Clarida, Gali and Gertler 2000 estimate about 0.8)
        # wages (placeholder rule): indexation to observed inflation + response to unemployment, downward rigid
        self.phi_W = 0.05
        self.w_index = 0.5          # indexation to observed inflation; the rest anchored at the target
        self.lam_W = 0.995
        self.g_pi = 0.1             # gain on observed inflation (firms' expectations move slowly; Candia, Coibion and Gorodnichenko)
        self.fix_policy = False     # diagnostics: policy rate frozen (shocks still pass through)
        self.fix_wage = False       # diagnostics: wage grows at the inflation target
        self.div_to_fund = False    # diagnostics: firm and bank profits credited to retirement balances
        # retirement fund (owns firms and bank)
        self.fund = True
        self.erp = 0.0075           # fund's published equity premium per quarter (3% a year)
        self.g_lr = 0.02            # fund's forgetting rate on observed deposit rates (long-run rate belief)
        self.g_earn = 0.025         # forgetting rate on firms' earnings: about ten years, as in normalized-earnings valuation (Graham-Dodd; Shiller CAPE)
        self.fund_cash_q = 0.5      # cash kept for withdrawals, in quarters of withdrawals
        # wage posting (replaces the wage rule)
        self.wage_posting = True    # False = placeholder wage rule (v3 results)
        self.eta_L = 8.0            # workers' elasticity of job choice to the relative wage (logit)
        self.ojs = 0.10             # share of employed workers searching on the job each quarter
        self.hiring = "convex"      # vacancy mode: firms choose hires with convex recruiting costs ("convex") or post openings for the whole gap ("gap")
        self.kappa_h = 5.0          # recruiting time per hire scale: about half a quarter's wage per hire at normal hiring (Blatter, Muehlemann and Schenker: 10-17 weeks of wages for skilled hires; lower for others)
        self.psi_h = 1.0            # convexity: marginal cost per hire = (1 + psi_h) x average (Blatter et al.: marginal about 2x average)
        self.q_ref = 0.8            # job-filling rate at which tightness adds nothing to recruiting cost
        self.g_pub = 0.25           # forgetting rate on published labor statistics
        self.public_labor = True    # with vacancies: fill and quit beliefs from published labor-market statistics
        self.censored = False       # True: firms see only sales and sell-outs (censored); False: they also see backorders (orders placed with them that they could not fill), not spillover from rivals
        self.cap_aware = True       # firms price for the output the labor they expect to get can make
        self.stockout_loss = 0.3    # share of customers turned away by a stockout who switch sellers (Anderson, Fitzsimons and Simester 2006)
        self.share_learn = False    # firms learn how much share their relative price actually wins (rivals co-move;
                                    # Blinder et al. 1998 coordination failure): regression of own share changes on
                                    # the logit prediction from own price relative to the expected index
        self.g_sh = 0.02            # gain of that regression (discounted least squares)
        self.w_sh = 1.0             # prior weight on the logit slope, in quarters of typical data
        self.rho_c = 0.5            # customer market: share of buyers who re-shop each quarter (1 = all re-shop, no customer base)
        self.menu_cost = 0.03       # state-dependent pricing: change price when the expected gain exceeds this share of revenue; calibrated to the frequency of price changes (about 40% a quarter), as in Golosov-Lucas and Nakamura-Steinsson (direct measures, Zbaracki et al. 2004: about 1.2%); 0: fixed review_p
        self.k_full = False         # capital decision uses the wage-based labor cost (as inventory)
        self.v_full = False         # inventory valued at production cost at the wage (recruiting costs are not inventoriable), not including marginal hiring cost
        self.v_marginal = True      # unsold stock valued at the marginal cost of replacing it (internal labor cost), not at a price-level-based unit cost
        self.flow_price = False      # with production lags: a reviewed price is set for the flow at the marginal cost of supply
        self.prod_lag = True        # production takes a quarter: goods made now are sold from next quarter on
        self.review_p = 3           # quarters between a firm's price reviews (staggered); 1 = every quarter
        self.wage_forecast = True   # firms forecast market pay growth from published inflation and unemployment
        self.g_F = 0.02             # forgetting rate of that forecast (discounted least squares)
        self.review_q = 4           # quarters between a firm's wage reviews (staggered across firms): annual
        self.anchor = 0.5           # weight of the published inflation target in firms' expected inflation (as in their pay forecasts)
        self.match_el = 0.5         # elasticity of the contact rate to labor-market tightness (matching function)
        self.m_match = 0.67         # matching efficiency; None = constant contact rate
        self.c_max = 0.6            # highest quarterly contact rate (search takes time)
        self.c_v = 0.5              # recruiting time per vacancy per quarter (efficiency units of the firm's labor)
        self.vacancies = True       # hiring with recruiting costs and matching; False = free vacancies, constant contact
        self.contact = 0.45         # probability an unemployed worker reaches employers in a quarter
        self.amenity_sd = 0.10      # sd of match-specific amenities (log points) in job-to-job moves
        self.g_L = 0.25             # firms' forgetting rate on own applicant and quit evidence
        self.g_W = 0.05             # firms' forgetting rate on observed growth of the going wage
        # technology: each firm uses one technique, a pair of capital and labor productivities (A_K, A_L), in a CES
        # technology with low substitution (plant-level elasticity about 0.5: Oberfield and Raval 2021). Ideas for new
        # techniques arrive at random (Jones 2005; Kortum 1997): an idea is a pair (a, b) with independent Pareto tails,
        # Pr(a > x, b > y) = x^-pK y^-pL, arriving at a rate that grows at g_N (research effort is a primitive). A firm
        # adopts its best new idea when the cost saving at its own factor prices, over the technique's expected life,
        # exceeds an adoption cost. No aggregate production function is assumed; in Jones's result the aggregate is
        # Cobb-Douglas with capital share pK / (pK + pL) and labor-augmenting growth g_N / pL, which the model can test.
        self.tech = True            # False: Cobb-Douglas K^alpha L^(1-alpha), no technical change
        self.sigma_f = 0.5          # firm-level elasticity of substitution between capital and labor
        self.pK, self.pL = 50.0, 70.0  # Pareto tail exponents of ideas (capital, labor); high exponents = many small improvements. With the CES above, techniques chosen along the growth path have a capital cost share of about 0.3
        self.g_N = 0.35             # growth rate of the arrival rate of ideas per quarter (Jones: labor productivity g_N / pL = 2% a year)
        self.theta_I = 0.95         # ideas below this share of the frontier's productivities are never useful and are not drawn
        self.R0 = 31.0              # ideas per quarter above that threshold, at the start (about the stationary value)
        self.adopt_cost = 0.02      # cost of adopting a technique (workers' time), share of a quarter's revenue
        self.g_T = 0.02             # forgetting rate on the firm's own record of technique turnover
        self.g_Y = 0.02             # forgetting rate on published real growth (final sales, labor productivity)
        self.deficit_rule = False   # tax rule allows the deficit that keeps debt at its target share of growing GDP
        self.invest_lr = True       # capital evaluated at the long-run cost of funds plus a hurdle premium
        self.fund_flow = False      # fund's valuation multiple moves with members' net flows (inelastic markets)
        self.k_rev = 0.02           # quarterly reversion of the multiple toward its fundamental value
        self.M_flow = 5.0           # price impact of flows: Gabaix and Koijen (2021) estimate about 5
        self.g_s = 0.05             # forgetting rate on own sales used to plan capacity
        self.hurdle = 0.0           # premium of firms' investment discount rate over their long-run cost of funds
        self.tau_min = 0.0          # lowest payroll tax rate (negative = wage subsidy)
        self.retiree_income = False # retirees' reference income tracks their own income (pension + drawdown)
        self.anchor_W = 0.5         # weight of (inflation target + published productivity growth) in firms' pay-growth forecast
        self.pay_prod = False        # pay forecast uses published productivity growth as a regressor
        self.bequest = False        # estates go to heirs (the entrant replacing the deceased), not to the government
        self.r_assumed = 0.0        # > 0: fund values firms at this published assumed return instead of the long-run deposit rate + premium
        self.market_belief = True   # firms' demand belief = own market position x published market size
        # Weiss (1980) adverse selection: some firms measure each worker's output and pay by it (piece rates, Lazear
        # 2000); the rest cannot see skill and pay one wage per head. Workers know their skill; the better ones sort to
        # output-based pay, so a pooled firm's average quality rises with its wage and it will not cut pay below the
        # level where quality falls faster than the wage; it lays off (at random) instead
        self.weiss = True 
        # minimum wage (policy rule): earnings per worker may not fall below W_min; revised every mw_review quarters to
        # kaitz times the published median earnings of employed workers (indexed state minimums; Kaitz index 0.37-0.59
        # across states, 0.28 for the federal minimum alone)
        self.x_lo, self.x_hi = 0.7, 1.4   # range of relative pay a pay review considers
        # two sectors: services (sold only to households, produced by pooled-pay firms: output per worker is hard to
        # measure, pay is posted per head) and goods (bought by households and as investment, produced by firms that
        # pay by measured output). Households spend a share s_sv of consumption on services (Cobb-Douglas upper tier)
        # firm-specific efficiency (know-how, management, organization; Syverson 2011): Hicks-neutral, AR(1) in logs,
        # quarterly persistence rho_a, stationary s.d. sd_a (within-industry TFP: 90/10 ratio about 2, s.d. of log
        # about 0.25-0.3; persistence about 0.8 a year, Foster, Haltiwanger and Syverson 2008)
        # participation: a working-age person without a job searches if UI (paid only to searchers, for ui_quarters
        # after a job loss) plus the published job-finding rate x a quarter's earnings covers a personal cost of
        # searching, kappa x earnings, with kappa persistent per person (caregiving, health, alternatives); measured
        # unemployment counts searchers only (CPS)
        self.olf = True
        self.spread_c = 0.02        # household borrowing rate over the bank loan rate, per quarter (card and HELOC mix)
        self.cred_shock_rho = 0.95  # persistence of a credit-standards shock (experiments)
        self.ui_quarters = 2
        self.kappa_med = 0.1
        self.kappa_sd = 1.0
        self.firm_eff = False
        self.rho_a = 0.95
        self.sd_a = 0.25
        self.services = True
        self.s_sv = 0.25
        self.min_wage = False
        self.kaitz = 0.45
        self.mw_review = 4
        self.n_piece = 14           # firms paying by measured output
        self.pool_outside = True       # pooled firms benchmark against their workers' pay at output-paying firms
        self.pool_own_market = False   # pooled firms compare their pay with other pooled firms (service jobs with service jobs)
        self.piece_own_market = True   # output-paying firms compare their pay with other output-paying firms
        self.commission = 0.0       # > 0: output-paying firms pay this share of the value of output per unit of effective labor
                                    # (commission contract, like the JMP's capital sector paying the value of output); quarterly
        self.search_rounds = 6      # application rounds per quarter for unemployed contacts (a quarter has many weeks of search)
        # variable capital utilization (Greenwood, Hercowitz and Huffman): running machines harder wears them out.
        # Depreciation per quarter = delta0 + delta1 u, u in [0, u_max]. Below capacity, extra output needs labor
        # plus more intensive use of the existing machines, so short-run marginal cost is flat (Blinder et al. 1998:
        # most firms report constant or falling marginal cost); capacity is chosen for normal orders under uncertainty
        self.embodied = True       # techniques embodied in capital (Solow 1960 vintages; Jones and Liu 2024): a chosen
                                   # technique applies only to capital installed after it is chosen
        self.k_amort = False       # capital adjustment cost spread over the capital's life (user-cost rate)
        self.hire_long = False     # hiring values output along the long-run elasticity (False: same belief as pricing)
        # external finance premium (Bernanke-Gertler): the bank prices each firm's loan for its expected loss, with a
        # default probability that rises as the firm's interest coverage (operating cash flow / interest, observed
        # through the lending relationship) weakens: PD = pd0 (cov_ref / cov)^pd_el, rate = (1 + rL)/(1 - PD LGD) - 1
        self.efp = True
        # consumer durables: a stock that depreciates dur_delta a quarter; the household upgrades to its target
        # (dur_theta quarters of reference income at goods prices) when the service value of closing the gap, psi x
        # gap share x income, covers the user cost (borrowing rate + depreciation) of the gap plus a fixed cost
        # dur_F x income: (S,s) under fixed adjustment costs (Grossman and Laroque 1990). Higher rates widen the band
        # scarring: skill decays while unemployed beyond scar_q quarters and recovers on the job up to its original level
        # (displaced workers' earnings losses, Davis and von Wachter 2011)
        self.scar = False
        self.scar_q = 2
        self.scar_d = 0.01
        self.scar_r = 0.02
        self.durables = False
        self.dur_theta = 3.0
        self.dur_delta = 0.05
        self.dur_psi = 0.34
        self.dur_F = 0.064         # replacement about every 16 quarters
        self.inst_n = 20            # installment (auto) loans: quarters to repay
        self.inst_spread = 0.005    # over the bank loan rate, per quarter
        self.inst_down = 0.1        # down payment share
        self.inst_dti = 0.15        # payments on installment debt at most this share of income (lenders' DTI rule)
        self.pd0 = 0.004            # quarterly default probability at the reference coverage (about 1.6% a year)
        self.cov_ref = 4.0          # reference interest coverage (investment-grade firms run about 4-6)
        self.pd_el = 2.0            # halving coverage quadruples the default probability
        self.lgd = 0.5              # loss given default
        self.hoard = 0.0           # labor hoarding: a worker is kept while MRP covers the wage net of the expected cost
                                   # of rehiring later (share of the wage; hiring costs about half a quarter's wage)
        self.clv_lr = False        # customers valued at the margin over long-run unit cost (capacity they need)
        self.cap_unc = False       # capacity planned for orders at the price without a capacity limit (see invest)
        self.utilization = False   # on: see README (destabilizes for now)
        self.u_bar = 0.8            # normal (design) utilization
        self.k_speed = 0.25         # share of the gap to desired capacity closed per quarter (time to build)
        self.u_max = 1.0
        self.delta1 = 0.0125        # wear per unit of utilization (at u = 0.8: 0.01, half of depreciation)
        self.cap_sd = 0.15          # uncertainty about normal orders over capital's horizon (log s.d.)
        self.plan_price = True      # production planned at next quarter's planned price, not this quarter's rationing price
        self.jorgenson = True       # user cost of capital nets out expected capital-goods inflation
        self.pay_hiring = True      # pay review prices the hiring the firm plans (demand-dependent pay)
        self.vshock_rho = 0.95      # persistence of an asset-value shock
        self.exo_ramp = 0           # diagnostic: quarters over which exogenous growth ramps up
        self.exo_g = 0.0            # diagnostic: exogenous labor-augmenting growth per quarter instead of ideas
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
        # technique: CES normalized at the starting point (capital share alpha), so the start is the same economy
        self.rho = 1 - 1 / mp.sigma_f
        # (with variable utilization the starting point is at normal utilization 0.8)
        self.AK = np.full(F, y0 / K0 / (0.8 if mp.utilization else 1.0)) if mp.tech else None
        self.AL = np.full(F, y0 / L0) if mp.tech else None
        self.a_eff = np.zeros(F)                     # log firm efficiency
        self.kappa_s = mp.kappa_med * np.exp(mp.kappa_sd * self.rng.standard_normal(hp.N))   # personal search cost
        self.ui_q = np.zeros(hp.N, int)              # quarters of UI left
        self.z0 = self.z.copy(); self.uq = np.zeros(hp.N, int)   # original skill; quarters without a job
        self.search = np.ones(hp.N, bool)
        self.f_pub = 0.5                             # published job-finding rate of searchers
        self.cred_fac = 1.0                          # banks' credit standards: share of normal credit lines offered
        self.cred_shock = 0.0
        self.cred_target = 0.0
        self.cov = np.full(F, 4.0)                   # firms' interest coverage as the bank sees it (smoothed)
        if mp.firm_eff and mp.tech:
            self.a_eff = mp.sd_a * self.rng.standard_normal(F)
            self.AK = self.AK * np.exp(self.a_eff); self.AL = self.AL * np.exp(self.a_eff)
        self.AKn = None if self.AK is None else self.AK.copy()   # technique of the machines the firm now buys
        self.ALn = None if self.AL is None else self.AL.copy()
        self.lnlam0 = np.log(mp.R0) + mp.pK * np.log(mp.theta_I * y0 / K0) + mp.pL * np.log(mp.theta_I * y0 / L0)
        self.poolK = np.array([self.AK[0] if mp.tech else y0 / K0]); self.poolL = np.array([y0 / L0])   # stock of known ideas (useful part)
        self.n_ideas = 0
        self.wp_lr = np.full(F, 1.0 / P0)        # firm's long-run real wage (discounted average)
        self.s_lr = np.full(F, y0)               # firm's normal sales (discounted average of its own)
        self.piece = (np.arange(F) < mp.n_piece) if mp.weiss else np.zeros(F, bool)
        self.sec = ((~self.piece) if (mp.services and mp.weiss) else np.zeros(F, bool)).astype(int)   # 1 = services
        self.n_sec = np.bincount(self.sec, minlength=2).astype(float)
        # services cannot be stored: unsold service capacity is mostly lost (decay inv_decay_sv)
        self.decay = np.where(self.sec == 1, mp.inv_decay_sv, mp.inv_decay)
        self.to_order = (self.sec == 1) & mp.prod_lag    # services: staffed and sold within the quarter
        self.zbar = float(self.z[self.age < self.hp.J_W].mean())      # published average skill of the labor force
        self.W_min = 0.0                                            # minimum earnings per worker (0: none yet)
        self.med_earn = None                                         # published median earnings of the employed
        self.qbar = np.full(F, self.zbar)                              # each firm's average worker quality (observed)
        self._rev_pe = np.full(F, P0 * y0 / L0)
        zw = np.sort(self.z[self.age < self.hp.J_W]); self._zpub = zw[:: max(1, len(zw) // 400)]   # published skill distribution
        self.gL_plan = np.zeros(F)               # firm's planned net workforce growth per quarter
        self.vshock = 1.0                        # asset-value shock multiplier (experiments)
        self.r_lr_f = np.full(F, mp.i_star + 0.005)   # firm's long-run cost of funds (discounted average of its own)
        self.turn = np.full(F, 0.05)             # belief: rate at which the firm's technique is superseded (own record)
        self.gY = mp.g_N / mp.pL if mp.tech else 0.0   # published real growth of final sales (discounted average)
        self.Y_prev = None
        self.adopt_last = 0.0
        self.lnA = np.full(F, np.log(y0))        # belief: log demand at relative price 1 (= share belief x expected market)
        # demand belief in two parts: the firm's own position in its market (learned from its own orders) and the size of
        # the market (published final sales, extrapolated at their published growth). The position is stationary even
        # when the economy grows.
        self.ln_a = np.full(F, np.log(1.0 / F))  # belief: log of demand at relative price 1 per unit of real final sales
        if mp.services and mp.weiss:                # start at the sector's share of final sales split evenly
            sh0 = np.where(self.sec == 1, 0.8 * mp.s_sv, 1 - 0.8 * mp.s_sv) / self.n_sec[self.sec]
            self.ln_a = np.log(sh0)
        self.S_last = P0 * F * y0                 # last published nominal final sales
        self.gS = mp.pi_star + (mp.g_N / mp.pL if mp.tech else 0.0)   # published growth of nominal final sales (discounted average)
        self.sig2 = np.full(F, mp.taste_sd ** 2)  # belief: variance of log demand surprises
        self.inv = np.full(F, (1.1 if mp.prod_lag else 0.1) * y0)
        self.inv[self.to_order] = 0.1 * y0
        self.loan = np.zeros(F) if mp.fund else np.full(F, P0 * K0)   # with the fund, firms start equity-financed
        self.Kb = np.full(F, P0 * K0)            # book value of capital (historical cost)
        self.m = np.zeros(F)
        self.p = np.full(F, P0)
        self.share_prev = 1.0 / self.n_sec[self.sec]
        self.share_prev_used = self.share_prev.copy()
        self.cust = 1.0 / self.n_sec[self.sec]           # customer base (share of its sector's buyers who buy from the firm)
        self._Ssec = np.zeros(F)                       # spending on the firm's sector last quarter
        self._Stot = 0.0
        self.lam_sh = np.ones(F)
        self.sh_v0 = 0.01                     # typical quarterly variance of the logit prediction (set from the baseline)
        self.sh_m = np.zeros((4, F)); self.sh_m[2] = self.sh_v0; self.sh_m[3] = self.sh_v0   # moments start at the prior (slope 1)
        # ---------------- labor market state: who works where, posted wages, firms' beliefs
        self.job = np.full(N, -1)
        ew = np.nonzero(self.emp & work)[0]
        self.job[ew] = self.rng.integers(0, F, len(ew))
        self.w = np.full(F, self.W)
        self.gW_e = mp.pi_star
        # pay-growth forecast: prior that market pay follows price inflation one for one
        # pay-growth forecast on published inflation, unemployment and labor productivity growth; prior: pay follows
        # prices and productivity one for one
        g0 = mp.g_N / mp.pL if mp.tech else 0.0               # the economy starts as if it had been growing at trend
        self.gProd = g0                                       # published labor productivity growth (discounted average)
        if mp.pay_prod:
            self.thW = np.array([0.0, 1.0, 0.0, 1.0])
            self.RW = np.diag([1.0, 1e-4, 1e-3, 1e-4])
            self.xW = np.array([1.0, mp.pi_star, mp.u_star, g0])
        else:                                                 # intercept absorbs trend productivity growth
            self.thW = np.array([g0, 1.0, 0.0])
            self.RW = np.diag([1.0, 1e-4, 1e-3])
            self.xW = np.array([1.0, mp.pi_star, mp.u_star])
        self.g_fore = mp.pi_star + g0
        self.theta = 1.0                          # last quarter's vacancies per unemployed worker
        self.q_mkt, self.Q_mkt = 0.8, 0.02        # published job-filling and quit rates
        self.q_bel, self.Q_bel = 0.8, 0.02        # firms' beliefs: discounted averages of the published rates
        self.hfill = np.ones(F)                   # belief: share of wanted hires that actually arrive
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
            nav0 = F * max(E0, 0) / (mp.r_assumed if mp.r_assumed else rb0 + mp.erp) + self.E_B
            self.a *= nav0 / self.a.sum()
            self.mult = 1.0 / (mp.r_assumed if mp.r_assumed else rb0 + mp.erp)   # earnings multiple (quarters)
        else:
            self.Ehat = np.zeros(F)
        self.g = 0.0
        self.g = 0.0
        self.g = -self.reserves()
        self.loss_e = 0.0
        self.rd_post, self.rL_post = max(0.0, mp.i_star - mp.m_b), mp.i_star + mp.m_L   # banks' last posted rates
        self.pi_e = mp.pi_star
        self.u_prev = mp.u_star
        self.P_prev = P0
        self.P_prev_sec = np.array([P0, P0], float)     # published price index of each sector (goods, services)
        GDP0 = P0 * F * y0
        self.GDP_hist = [GDP0] * 4
        self.budget_prev = None                 # last quarter's (spending net of estates, wage bill)
        self.tau = 0.25
        self.tau_s = None                       # smoothed budget share
        self.i_prev = mp.i_star
        self.pi_hist = [mp.pi_star] * 4
        self.i_star_est = mp.i_star
        self.t = 0
        self.R0 = self.reserves()

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

    # ------------------------------------------------------------------ labor market with posted wages
    def earn_fac(self, job, z=None):
        """What a worker earns per unit of posted pay: own skill at output-paying firms, the average skill (one
        wage per head) at pooled firms."""
        z = self.z if z is None else z
        jf = np.maximum(job, 0)
        return np.where(self.piece[jf], z, self.zbar)

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

    # ------------------------------------------------------------------ accounting
    def reserves(self):
        deposits = self.b.sum() + self.m.sum() + self.div_pending.sum() + self.g + self.Fc
        if not self.mp.fund:
            deposits += self.a.sum()
        return deposits + self.E_B - self.loan.sum() - (self.L_inst.sum() if getattr(self, 'L_inst', None) is not None else 0.0)

    # ------------------------------------------------------------------ one quarter
    def step(self, extra_y=None, tax_shift=0.0, i_shock=0.0, record=True, R_plan=None):
        hp, mp = self.hp, self.mp
        N, F = hp.N, mp.F
        work = self.age < hp.J_W
        W = self.W
        a_, Z = mp.alpha, mp.Z
        job0 = self.job.copy()
        if mp.olf:
            nonemp = work & (self.job < 0)
            thr = self.f_pub + hp.ui * (self.ui_q > 0)          # per unit of a quarter's earnings
            self.search = np.where(nonemp, self.kappa_s <= thr, True)
            part = work & ((self.job >= 0) | self.search)       # labor force
        else:
            part = work

        # 1. policy rate
        pi4 = np.mean(self.pi_hist[-4:])                      # observed inflation, last four quarters
        if mp.rstar_learn:
            # the central bank revises its estimate of the neutral rate when inflation keeps missing the target
            # (Laubach-Williams style; FOMC long-run rate projections move slowly): a small step each quarter
            self.i_star_est += mp.g_rstar * (pi4 - mp.pi_star)
        rule = self.i_star_est + mp.phi_pi * (pi4 - mp.pi_star) - mp.phi_u * (self.u_prev - mp.u_star)
        i = max(0.0 if mp.zlb else -1.0, (self.i_prev if mp.fix_policy else mp.rho_i * self.i_prev + (1 - mp.rho_i) * rule) + i_shock)
        self.i_prev = i - i_shock
        # 2. bank and government post
        if mp.bank_objective:
            rb, rL = self.bank_rates(i)
        else:
            rb = max(0.0, i - mp.m_b)
        self.rb_lr = (1 - mp.g_lr) * self.rb_lr + mp.g_lr * rb
        ra = (self.rb_lr + mp.erp) if mp.fund else i + mp.m_a    # posted return on retirement balances
        if not mp.bank_objective:
            rL = (1 + i + mp.m_L) / (1 - self.loss_e) - 1
        GDP_a = sum(self.GDP_hist[-4:])                       # last four quarters, nominal
        debt = -self.g
        if self.budget_prev is not None:
            spend_net, wb_prev = self.budget_prev
            # the rule allows the deficit that keeps debt at its target share of GDP as the economy grows
            # (published trend real growth plus the inflation target, not realized nominal growth, which would feed inflation)
            sh = (spend_net - (mp.d_target * GDP_a * (self.gY + mp.pi_star) if mp.deficit_rule else 0.0)) / wb_prev
            self.tau_s = sh if self.tau_s is None else (1 - mp.g_tax) * self.tau_s + mp.g_tax * sh
            self.tau = float(np.clip(self.tau_s + mp.phi_G * (debt / GDP_a - mp.d_target), mp.tau_min, 0.7))
        tau = self.tau + tax_shift

        # 3. firms post prices and plan
        L_need = self.L_need
        # marginal cost of funds: a borrower pays rL; a firm without loans is financed at the margin by its owner,
        # the fund, which requires the deposit rate plus its published equity premium
        if mp.efp:
            pd_f = np.clip(mp.pd0 * (mp.cov_ref / np.maximum(self.cov, 1e-3)) ** mp.pd_el, 0.0, 0.25)
            self.rL_f = (1 + rL) / (1 - pd_f * mp.lgd) - 1
        else:
            self.rL_f = np.full(F, rL)
        r_own = np.where(self.loan > 0, self.rL_f, self.rb_lr + mp.erp)
        if mp.market_belief:                                  # expected demand = own position x expected size of the market
            self.lnA = self.ln_a + np.log(self.S_last * np.exp(self.gS) / self.P_ref())
        self._adopt_L = np.zeros(F)
        if mp.firm_eff and mp.tech:                           # firm efficiency shock (applies to all its machines)
            da = (mp.rho_a - 1) * self.a_eff + mp.sd_a * np.sqrt(1 - mp.rho_a ** 2) * self.rng.standard_normal(F)
            self.a_eff = self.a_eff + da
            self.AK = self.AK * np.exp(da); self.AL = self.AL * np.exp(da)
            self.AKn = self.AKn * np.exp(da); self.ALn = self.ALn * np.exp(da)
        if mp.tech and mp.exo_g > 0:                          # diagnostic: smooth exogenous labor-augmenting growth
            ge = mp.exo_g * min(1.0, self.t / mp.exo_ramp) if mp.exo_ramp else mp.exo_g
            self.AL = self.AL * np.exp(ge); self.poolL = self.poolL * np.exp(ge)
            if self.ALn is not None: self.ALn = self.ALn * np.exp(ge)
        elif mp.tech:                                         # technique choice at last quarter's own wage and price
            # a technique is long-lived: it is evaluated at the same discount rate as capital (long-run cost of funds
            # plus the hurdle premium when invest_lr)
            r_T = self.r_lr_f + mp.hurdle if mp.invest_lr else r_own
            self.adopt(self.w if mp.wage_posting else np.full(F, W), self.P_ref() * (r_T + mp.delta), self.p, r_T,
                       W, self.P_ref() * (float(np.mean(r_T)) + mp.delta))
        if mp.review_p > 1 or mp.menu_cost > 0:
            self._p_old = self.p.copy()
            self._rev_p = (np.arange(F) + self.t + 1) % mp.review_p == 0
        if mp.wage_posting:
            # wage posting. Expected going wage = last observed x (1 + believed growth). The firm's labor supply,
            # in its own beliefs: incumbents who stay (quits fall with its relative wage) plus applicants (rise with
            # it). It picks the wage, paid to all its workers, that maximizes the quarter's expected profit, and it
            # does not cut its own nominal wage (a cut lowers the effort of all incumbents: morale, Bewley).
            if mp.wage_forecast:
                # expected market pay growth: forecast from published data (last year's inflation, unemployment) by
                # discounted least squares on the published pay series
                self.xW = np.array([1.0, np.mean(self.pi_hist[-4:]), self.u_prev] + ([self.gProd] if mp.pay_prod else []))
                # partly anchored on the published inflation target plus published productivity growth (anchor weight)
                self.g_fore = mp.anchor_W * (mp.pi_star + self.gProd) + (1 - mp.anchor_W) * float(self.xW @ self.thW)
                We = self.Wbar_obs() * (1 + self.g_fore)
            else:
                We = self.Wbar_obs() * (1 + self.anchored(self.gW_e))
            B, Q = np.exp(self.lnapp), np.exp(self.lnQ)
            L_prev = np.bincount(self.job[self.job >= 0], self.z[self.job >= 0], F)
            w0 = np.maximum(self.w, We)
            if mp.vacancies:
                # market pay growth: last year's, from published pay data
                g_mkt = self.g_fore if mp.wage_forecast else (self.Wbar_hist[-1] / self.Wbar_hist[-5]) ** 0.25 - 1
                We = self.Wbar_obs() * (1 + g_mkt)
                qf = np.exp(self.lnq)
                if mp.public_labor:                           # beliefs: discounted average of published statistics
                    qf = np.full(F, self.q_bel); Q = np.full(F, self.Q_bel)
                We_piece = None
                if mp.weiss:                                  # pooled firms also look at published pay at output-paying firms
                    We_piece = self.Wbar_obs(self.piece) * (1 + g_mkt) ** 2.5
                    if mp.piece_own_market:                   # output-paying firms benchmark against each other only
                        if mp.pool_outside:
                            # pooled firms benchmark against their workers' outside option: what workers of their
                            # average skill (observed output per head) would earn paid by output, per head
                            We_pool_ = self.Wbar_obs(self.piece) * self.qbar / self.zbar
                        else:
                            We_pool_ = self.Wbar_obs(~self.piece) if mp.pool_own_market else self.Wbar_obs()
                        We = np.where(self.piece, self.Wbar_obs(self.piece), We_pool_) * (1 + g_mkt)
                w_des = self.pay_review(We * (1 + g_mkt) ** 1.5, qf, Q, r_own, We_piece, self.gL_plan)
                self._cut_wanted = w_des < self.w                      # the firm would cut pay if workers accepted it
                w_post = np.maximum(self.w, w_des)                     # no nominal cuts
                if mp.weiss and mp.commission > 0:
                    # output-paying firms: commission on last quarter's revenue per unit of effective labor; it moves
                    # with sales, down as well as up (flexible pay)
                    w_com = mp.commission * self._rev_pe * (1 + g_mkt)
                    w_post = np.where(self.piece, w_com, w_post)
                    We_piece = float(np.mean(w_com[self.piece])) * (1 + g_mkt) ** 1.5
                    w_post = np.where(self.piece, w_post, np.maximum(self.w, self.pay_review(We * (1 + g_mkt) ** 1.5, qf, Q, r_own, We_piece)))
            else:
                p0, S0, mu0, v0 = self.price_and_stock(w0, r_own)
                w_post, L_cap = self.wage_choice(We, p0, mu0, v0, L_prev, B, Q, self.w / We)
            if mp.review_q > 1:                                 # wages are reviewed once every review_q quarters (staggered)
                rev = ((np.arange(F) + self.t) % mp.review_q == 0) | (self.piece & (mp.commission > 0))
                w_post = np.where(rev, w_post, self.w)
            if mp.min_wage and self.W_min > 0:
                # pooled firms pay per head: pay per head at least the minimum; output-paying firms pay z w, and do
                # not employ workers whose output is worth less than the minimum (see labor_market)
                w_post = np.where(self.piece, w_post, np.maximum(w_post, self.W_min / self.zbar))
            # cost per unit of effective labor: pooled firms pay per head, so divide by the average quality they have
            qfac = np.where(self.piece, 1.0, self.zbar / np.maximum(self.qbar, 1e-6)) if mp.weiss else np.ones(F)
            w_eff = w_post * qfac
            self.util_setup(w_eff)
            if mp.vacancies:
                # price and stock at the wage; keep incumbents while their product covers the wage,
                # hire toward the labor at which it covers the full cost (wage + recruiting)
                omega = self.full_cost(w_post / We, We, qf, Q, r_own) * qfac
                self.p, S, mu_s, v = self.price_and_stock(w_eff, r_own)   # price on the marginal cost of its workers
                L_keep = self.labor_for_cost(w_post[:, None], self.p, mu_s, v)[0][:, 0]
                L_tgt = self.labor_for_cost(omega[:, None], self.p, mu_s, v)[0][:, 0]
                if mp.hiring == "convex":
                    x_ = w_post / We
                    stay = L_prev * np.clip(1 - hp.sep - Q * x_ ** (-mp.eta_L), 0, 1)
                    H_want, R_rec, L_flow = self.hires_choice(self.p, mu_s, w_eff, stay, r_own, Q, np.minimum(1.0, qf * x_ ** mp.eta_L))
                    # keep a worker while marginal revenue product covers the wage
                    lo_, hi_ = np.zeros(F), 10 * np.maximum(L_prev, 1.0)
                    for _ in range(50):
                        mid_ = 0.5 * (lo_ + hi_); pos_ = self._mrp(mid_) > w_eff * (1 - mp.hoard)
                        lo_ = np.where(pos_, mid_, lo_); hi_ = np.where(pos_, hi_, mid_)
                    L_keep = 0.5 * (lo_ + hi_)
                    L_tgt = stay + H_want
                    # planned net growth of the workforce (discounted average), used at the next pay review
                    self.gL_plan = 0.75 * self.gL_plan + 0.25 * np.clip(np.minimum(L_tgt, L_keep) / np.maximum(L_prev, 1e-9) - 1, -0.5, 0.5)   # incl. planned layoffs
            else:
                self.p, S, mu_s, v = self.price_and_stock(w_post, r_own)
                S = np.minimum(S, self.inv + self.prod(L_cap))
            W_firm = w_eff
        else:
            W_firm = W
            self.p, S, mu_s, v = self.price_and_stock(W, r_own)
        if mp.wage_posting and mp.cap_aware:
            # the firm expects to get only a share of the labor it plans for (belief from its own record), so it
            # prices and stocks for the output that labor can make: a firm short of workers raises its price
            L_plan = L_need(np.maximum(0.0, S - self.inv))
            if mp.vacancies and mp.hiring == "convex":
                Hx = H_want * self.hfill                      # hires it expects to actually get (own record)
                L_exp = stay + Hx - np.minimum(R_rec * self.hfill, Hx)   # incumbents who stay + new hires net of start-up cost
                cap = self.inv + self.prod(np.maximum(L_exp, 0.0))
            elif mp.vacancies:
                # with posted vacancies the firm knows its workforce: incumbents expected to stay, plus the hires its
                # open positions are expected to bring at the published filling rate, minus time spent recruiting
                x_ = w_post / We
                stay = L_prev * np.clip(1 - hp.sep - Q * x_ ** (-mp.eta_L), 0, 1)
                mpl0 = (1 - a_) * self.p * Z * self.K ** a_
                tgt0 = np.minimum(L_need(mu_s), (mpl0 / omega) ** (1 / a_))
                V0 = np.maximum(0.0, tgt0 - stay)
                L_exp = stay + np.minimum(1.0, qf * x_ ** mp.eta_L) * V0 - mp.c_v * V0
                cap = self.inv + self.prod(np.maximum(L_exp, 0.0))
            else:
                cap = self.inv + self.prod(np.exp(self.lnphi) * L_plan)
            self.L_plan_raw = L_plan
            self._cap, self._Spre = cap, S.copy()
            p_unc, mu_unc = self.p.copy(), mu_s.copy()           # the plan the firm would choose with the workers it wants
            self.p, S, mu_s, v = self.price_and_stock(W_firm, r_own, S_cap=cap)
        if mp.wage_posting and mp.vacancies and mp.hiring != "convex":
            # workforce decisions look at the expected flow of sales, not this quarter's stock: the firm keeps and
            # hires the workers it needs to make what it expects to sell each quarter, as long as a worker's product
            # covers the wage (keep) or the wage plus recruiting cost (hire)
            mpl_fac = (1 - a_) * p_unc * Z * self.K ** a_
            L_flow = L_need(mu_unc)
            L_keep = np.minimum(L_flow, (mpl_fac / w_post) ** (1 / a_))
            L_tgt = np.minimum(L_flow, (mpl_fac / omega) ** (1 / a_))
        if mp.prod_lag:
            # production takes a quarter: this quarter's sales come from stock on hand, which the price rations;
            # production now is for next quarter's stock, chosen against next quarter's demand (two quarters of
            # uncertainty) given the stock the firm expects to carry over
            if mp.wage_posting and mp.vacancies and mp.hiring == "convex":
                x_ = w_post / We
                Tq = mp.q_ref / np.maximum(np.minimum(1.0, qf * x_ ** mp.eta_L), 1e-3)
                mch = (1 + mp.psi_h) * mp.kappa_h * Tq * (H_want / np.maximum(stay, 1.0)) ** mp.psi_h
                tenure_r = r_own + hp.sep + Q * x_ ** (-mp.eta_L)
                om_e = w_eff * (1 + np.where(H_want > 1e-9, mch * tenure_r, 0.0))
            else:
                om_e = W_firm
            # a price kept for several quarters is set for the flow: jointly with the stock, at the marginal cost of
            # supplying it (firms not reviewing keep their posted price); this quarter's sales come from stock on hand
            if mp.flow_price:
                self.p, _, mu_s, v = self.price_and_stock(om_e, r_own)
            else:
                Scap_ = self.inv
                if self.to_order.any():
                    # services are made this quarter: what can be served is the output of the workforce it expects
                    cap_to = cap if (mp.wage_posting and mp.cap_aware) else self.inv + self.prod(np.maximum(L_prev, 0.0))
                    Scap_ = np.where(self.to_order, cap_to, self.inv)
                self.p, S, mu_s, v = self.price_and_stock(W_firm, r_own, S_cap=Scap_, W_int=om_e if mp.v_full else W_firm)
            sd = np.sqrt(self.sig2)
            self._mu_lag, self._inv_lag = mu_s.copy(), self.inv.copy()
            d2 = (np.log(mu_s / np.maximum(self.inv, 1e-12)) - sd ** 2 / 2) / sd
            Esales = mu_s - (mu_s * ndtr(d2 + sd) - self.inv * ndtr(d2))
            carry = np.maximum(self.inv - Esales, 0.0) * (1 - self.decay)
            if mp.wage_posting and mp.vacancies and mp.hiring == "convex":
                # production is planned at the marginal cost of the labor that makes it, the same cost the hiring
                # decision uses: the wage, plus (if the firm is adding workers) the marginal hiring cost spread over
                # expected tenure
                if mp.plan_price:
                    # production for next quarter is planned with next quarter's price: the price the firm would
                    # choose jointly with its stock (no stock constraint, no adjustment cost), not this quarter's
                    # rationing price; otherwise a high price that rations scarce stock would plan scarce stock again
                    p_pl, _, mu_pl, v_pl = self.price_and_stock(W_firm, r_own, plan=True)
                    S_next = self.stock_target(om_e, p_pl, mu_pl, v_pl, carry, sd * np.sqrt(2))
                else:
                    S_next = self.stock_target(om_e, self.p, mu_s, v, carry, sd * np.sqrt(2))
                y_plan = np.maximum(0.0, S_next - carry)
                self._ycap_ratio = self.prod(np.maximum(L_exp, 0.0)) / np.maximum(y_plan, 1e-9)
                y_plan = np.minimum(y_plan, self.prod(np.maximum(L_exp, 0.0)))
            else:
                S_next = self.stock_target(W_firm, self.p, mu_s, v, carry, sd * np.sqrt(2))
                y_plan = np.maximum(0.0, S_next - carry)
        if mp.services and mp.weiss:
            # posted price index of each sector (last quarter's within-sector shares): machines are goods; households
            # buy both, so their price index is the Cobb-Douglas (expenditure-share) index of the two
            g_ = self.sec == 0
            P_g = float(self.p[g_] @ self.share_prev[g_]); P_s = float(self.p[~g_] @ self.share_prev[~g_])
            P = P_g ** (1 - mp.s_sv) * P_s ** mp.s_sv
        else:
            P = P_g = float(self.p @ self.share_prev)         # posted price index (last quarter's shares)
        if not mp.prod_lag:
            y_plan = np.maximum(0.0, S - self.inv)
        elif self.to_order.any():
            # services: the capacity staffed this quarter is the newsvendor stock for this quarter's demand
            y_plan = np.where(self.to_order, np.maximum(0.0, S - self.inv), y_plan)
        # capital is evaluated at the firm's discount rate for investment: its long-run cost of funds (discounted average
        # of its own) plus a hurdle premium. Firms' discount rates are high and move little with interest rates
        # (Gormsen and Huber 2023; Sharpe and Suarez 2021)
        self.r_lr_f = (1 - mp.g_lr) * self.r_lr_f + mp.g_lr * r_own
        # capacity is planned for normal sales: a discounted average of the firm's own sales, grown at published trend
        # growth over the coming year (a one-quarter surge in orders does not justify new capacity: no accelerator loop)
        s_cap = self.s_lr * np.exp(4 * self.gY) if mp.invest_lr else mu_s
        if mp.invest_lr and getattr(self, "_ell", None) is not None and mp.cap_unc:
            # sales are capped by capacity, so a firm at capacity does not see the orders it would get with more of
            # it: capacity is planned for the orders it expects at the price it would charge with marginal cost flat
            # (no capacity limit), relative to the orders it expects at its plan price with current capacity
            ycap0 = self._ycap
            self._ycap = np.full_like(ycap0, 1e300)
            _, _, mu_u, _ = self.price_and_stock(W_firm, r_own, plan=True)
            self._ycap = ycap0
            _, _, mu_c, _ = self.price_and_stock(W_firm, r_own, plan=True)
            s_cap = s_cap * np.clip(mu_u / np.maximum(mu_c, 1e-12), 1.0, 1.5)
        W_K = om_e if (mp.prod_lag and mp.v_marginal and mp.k_full) else W_firm
        if mp.invest_lr:
            # capital lasts many years, so the firm weighs it against the wage relative to prices it expects over that
            # horizon: a discounted average of its own real wage, not this quarter's
            self.wp_lr = (1 - mp.g_lr) * self.wp_lr + mp.g_lr * W_K / self.P_ref()
            W_K = self.wp_lr * self.P_ref()
        K_next = self.capital(W_K, self.r_lr_f + mp.hurdle if mp.invest_lr else r_own, s_cap)
        I_plan = np.maximum(0.0, K_next - (1 - getattr(self, "_dep", mp.delta)) * self.K)
        Ld = L_need(y_plan)
        Ld_pre = Ld.sum()
        Ld_req = Ld.copy()                                    # production labor the firm asks for

        # 4. labor market (efficiency units)
        Lforce = np.where(part, self.z, 0.0)
        if mp.wage_posting:
            if mp.vacancies:
                Ld, emp, quits, apps, V, L_prev = self.labor_market(w_post, We, L_keep, part, L_target=L_tgt, q_fill=qf,
                                                                    hires=H_want if mp.hiring == "convex" else None)
                if mp.weiss:                                      # each firm observes its workers' average output
                    heads = np.bincount(self.job[self.job >= 0], minlength=F).astype(float)
                    self.qbar = np.where(heads > 0, Ld / np.maximum(heads, 1e-9), self.qbar)
                if mp.hiring == "convex":                         # belief: share of wanted hires that arrive
                    hw = H_want > 1e-6
                    obs_h = np.clip(self.hired_last / np.maximum(H_want, 1e-9), 0.0, 1.0)
                    self.hfill = np.where(hw, (1 - mp.g_L) * self.hfill + mp.g_L * obs_h, self.hfill)
                obs_f = np.log(np.maximum(self.hired_last, 1e-6) / np.maximum(V, 1e-9)) - mp.eta_L * np.log(w_post / We)
                self.lnq = np.where(V > 1e-9, (1 - mp.g_L) * self.lnq + mp.g_L * obs_f, self.lnq)
                Lw = Ld.copy()                                    # workers paid
                if mp.hiring == "convex":                         # recruiting time is spent on the hires actually made
                    Tq = mp.q_ref / max(self.q_bel, 1e-3)
                    hr = self.hired_last
                    R_rec = np.minimum(mp.kappa_h * Tq * hr * (hr / np.maximum(L_prev, 1.0)) ** mp.psi_h, hr)
                    Ld = np.maximum(Ld - R_rec, 0.0)                 # new hires' start-up cost (adaptation, training)
                else:
                    Ld = np.maximum(Ld - mp.c_v * V, 0.0)         # production labor: recruiting takes time
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
            Ld = np.maximum(Ld - self._adopt_L, 0.0)          # time spent installing a new technique
            Ld = np.minimum(Ld, L_need(y_plan))               # production follows the plan; surplus workers idle
        y = self.prod(Ld)
        self._yprod = y
        if getattr(self, "trace_on", False) and mp.vacancies:
            f = 0
            self.trace.append(dict(t=self.t, inv0=float(self.inv[f]), mu=float(mu_s[f]), S=float(S[f]), y_plan=float(y_plan[f]),
                                   L_flow=float(L_flow[f]), L_keep=float(L_keep[f]), L_tgt=float(L_tgt[f]), workers=float(Lw[f]),
                                   V=float(V[f]), Lprod=float(Ld[f]), y=float(self.prod(Ld)[f]),
                                   p=float(self.p[f]), w=float(w_post[f]), omega=float(omega[f]), q_mkt=self.q_mkt,
                                   u=1 - L_emp / Lforce.sum(), phi=float(np.exp(self.lnphi[f])), cap=float(self._cap[f]), Spre=float(self._Spre[f])))
        if mp.wage_posting and mp.cap_aware:                   # belief: share of planned labor the firm actually gets
            obs_phi = np.clip(np.log(np.maximum(Ld, 1e-9) / np.maximum(Ld_req, 1e-9)), -3.0, 0.0)
            self.lnphi = (1 - mp.g_L) * self.lnphi + mp.g_L * obs_phi
        wb_f = W_firm * (Lw if mp.wage_posting else Ld)
        if mp.weiss:                                          # exact pay: by output at piece firms, per head at pooled firms
            heads = np.bincount(self.job[self.job >= 0], minlength=F).astype(float)
            wb_f = w_post * np.where(self.piece, Lw, self.zbar * heads)
        # financing: borrow what cash does not cover of wages and planned investment
        new_loan = np.maximum(0.0, wb_f + P_g * I_plan - self.m)
        self.loan += new_loan
        self.m += new_loan - wb_f

        # 5. households
        wage_h = np.where(emp, (1 - tau) * (self.w[np.maximum(self.job, 0)] if mp.wage_posting else W)
                          * (self.earn_fac(self.job) if mp.weiss else self.z), 0.0)
        if mp.scar:
            ne_ = work & (self.job < 0)
            self.uq = np.where(ne_, self.uq + 1, 0)
            self.z = np.where(ne_ & (self.uq > mp.scar_q), self.z * (1 - mp.scar_d),
                              np.where(~ne_, np.minimum(self.z0, self.z * (1 + mp.scar_r)), self.z))
        if mp.olf:
            lost = (job0 >= 0) & (self.job < 0) & work
            self.ui_q = np.where(emp, 0, np.where(lost, mp.ui_quarters, np.maximum(self.ui_q - 1, 0)))
            sk = part & (job0 < 0)
            if sk.any():
                self.f_pub = 0.75 * self.f_pub + 0.25 * float(np.mean(self.job[sk] >= 0))
            benefit = np.where(part & ~emp & ((self.ui_q > 0) | lost), hp.ui * W * self.z, 0.0)
            self._lfpr = float(part.sum() / max(work.sum(), 1))
        else:
            benefit = np.where(work & ~emp, hp.ui * W * self.z, 0.0)
        pension = np.where(~work, self.pens * W, 0.0)
        if emp.any():
            self.med_earn = float(np.median((self.w[self.job[emp]] if mp.wage_posting else W) * (self.earn_fac(self.job[emp], self.z[emp]) if mp.weiss else self.z[emp])))
        if mp.min_wage and self.med_earn is not None and self.t % mp.mw_review == 0:
            self.W_min = mp.kaitz * self.med_earn                      # annual revision to a share of the published median
        tax_rev = tau * wb_f.sum()
        extra = np.zeros(N) if extra_y is None else extra_y
        y_h = wage_h + benefit + pension + self.div_pending + extra
        self.div_pending = np.zeros(N)
        S_D = np.zeros(N); gap_buy = np.zeros(N); cash_D = np.zeros(N); self._int_inst = 0.0
        if mp.durables:
            r_a = float(np.mean(rL)) + mp.inst_spread
            den = max(mp.dur_psi - (r_a + mp.dur_delta) * mp.dur_theta, 1e-6)
            thr = mp.dur_F / den                               # (S,s) band: upgrade when the gap share exceeds thr
            if getattr(self, "dur", None) is None:
                self.dur = mp.dur_theta * self.yhat / P_g * (1 - min(thr, 0.9) * self.rng.random(N))   # spread over the band
                self.L_inst = np.zeros(N); self.pay_inst = np.zeros(N); self.r_inst = np.zeros(N)
                self.n_inst = np.zeros(N, int)
            # scheduled payments on installment loans (fixed nominal annuity at the origination rate)
            on_ = self.n_inst > 0
            int_i = self.r_inst * self.L_inst
            pay = np.where(on_, np.where(self.n_inst == 1, self.L_inst + int_i, self.pay_inst), 0.0)   # last payment clears it
            self.L_inst = np.where(on_, np.maximum(self.L_inst + int_i - pay, 0.0), 0.0)
            self.n_inst = np.maximum(self.n_inst - 1, 0)
            self.L_inst = np.where(self.n_inst > 0, self.L_inst, 0.0)
            self._int_inst = float(int_i[on_].sum())
            cash_D += pay
            # purchases
            self.dur = self.dur * (1 - mp.dur_delta)
            dstar = mp.dur_theta * self.yhat / P_g
            gap = np.maximum(dstar - self.dur, 0.0)
            ann = r_a / (1 - (1 + r_a) ** (-mp.inst_n))
            # pay cash out of the buffer above its target (cash earns less than the loan costs); finance the rest
            # (at least the down payment share of the financed part in cash)
            excess = np.maximum(self.b + y_h - pay - hp.beta * self.yhat, 0.0)
            price_ = P_g * gap
            fin = np.maximum(price_ - excess, 0.0)
            loan_amt = (1 - mp.inst_down) * fin
            new_pay = loan_amt * ann
            dti_ok = (loan_amt <= 0) | ((np.where(self.n_inst > 0, self.pay_inst, 0.0) + new_pay) <= mp.inst_dti * self.yhat)
            down_ok = self.b + y_h - pay >= price_ - loan_amt
            buy = (gap / np.maximum(dstar, 1e-12) >= thr) & dti_ok & down_ok & (gap > 0)
            gap_buy = np.where(buy, gap, 0.0)
            S_D = P_g * gap_buy
            loan_new = np.where(buy, loan_amt, 0.0)            # the bank's loan creates the deposit that pays the seller
            # the new loan is consolidated with any old balance into one new annuity
            Lnew = self.L_inst + loan_new
            nb = buy & (loan_new > 0)
            self.L_inst = np.where(nb, Lnew, self.L_inst)
            self.r_inst = np.where(nb, r_a, self.r_inst)
            self.n_inst = np.where(nb, mp.inst_n, self.n_inst)
            self.pay_inst = np.where(nb, Lnew * ann, self.pay_inst)
            cash_D += S_D - loan_new                           # cash part (down payment, or all of it)
            self._loan_new_D = loan_new
        m_cash = self.b + y_h - cash_D
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
        r_c = (float(np.mean(rL)) + mp.spread_c) if hp.cred_lim > 0 else None
        if hp.cred_lim > 0:
            # credit standards (a bank policy): normal lines, scaled down after a credit-standards shock, which decays
            # a tightening is phased in (banks cut lines over about a year: 35% of the remaining gap each quarter)
            # while the target itself decays
            self.cred_shock += 0.35 * (self.cred_target - self.cred_shock)
            self.cred_target *= mp.cred_shock_rho
            self.cred_fac = 1.0 - self.cred_shock
        lim_h = hp.cred_lim * self.cred_fac
        cw, _, _, _, _ = solve_accounts(m_cash[w] - d_now, self.b[w], self.a[w] + d_now, self.yhat[w], G_j[w], P, rb, ra, hp, False, r_c=r_c, lim=lim_h)
        c[w], da[w] = cw, d_now
        db[w] = m_cash[w] - d_now - P * cw                    # budget closes exactly
        r = ~work
        wd = self.a[r] / (hp.J - self.age[r])
        cr, _, _, _, _ = solve_accounts(m_cash[r] + wd, self.b[r], self.a[r], self.yhat[r], G_j[r], P, rb, ra, hp, False, r_c=r_c, lim=lim_h)
        c[r], da[r] = cr, -wd
        db[r] = m_cash[r] + wd - P * cr
        self._chargeoff = 0.0
        if r_c is not None:
            # a household that cannot get back within its credit line defaults on the excess; the bank charges it off
            fl_h = -lim_h * self.yhat / (1 + r_c)
            exc = np.maximum(fl_h - db, 0.0)
            db = db + exc
            self._chargeoff = float(exc.sum())
            self.E_B -= self._chargeoff

        # 6. goods market at posted prices; households and investing firms buy from the same firms
        S_h = P * c
        S_I = P_g * I_plan
        S_tot = S_h.sum() + S_I.sum() + S_D.sum()
        sec, nsec = self.sec, 2
        if mp.services and mp.weiss:
            S_sec = np.array([(1 - mp.s_sv) * S_h.sum() + S_I.sum() + S_D.sum(), mp.s_sv * S_h.sum()])
        else:
            S_sec = np.array([S_tot, 0.0])
        Ssec_f = S_sec[sec]
        secsum = lambda x: np.bincount(sec, x, minlength=nsec)[sec]  # sum over the firm's sector
        avail = (self.inv + np.where(self.to_order, y, 0.0)) if mp.prod_lag else self.inv + y
        # buyers choose among firms by random utility (logit): shares proportional to (p/P)^(-eps) x taste
        wts = (self.p / P) ** (-mp.eps) * np.exp(mp.taste_sd * self.rng.standard_normal(F))
        if mp.rho_c < 1:
            # customer market: a share rho_c of buyers shop around (logit); the rest stay with their seller
            cust_old = self.cust.copy()
            self.cust = (1 - mp.rho_c) * self.cust + mp.rho_c * wts / secsum(wts)
            if mp.share_learn:
                # what the firm sees: its price relative to the index it expected, and the change in its customer
                # share; the logit slope with rivals' prices as expected predicts z; the firm estimates how much of
                # z it actually gets (lam), by discounted least squares with a prior at 1
                zz = -mp.rho_c * mp.eps * (1 - cust_old) * np.log(self.p / self.Pref_used)
                yy = np.log(self.cust / np.maximum(cust_old, 1e-12))
                g = mp.g_sh
                self.sh_m = (1 - g) * self.sh_m + g * np.stack([zz, yy, zz * zz, zz * yy])
                mz, my, szz, szy = self.sh_m
                vz = np.maximum(szz - mz * mz, 0.0); cz = szy - mz * my
                w0 = mp.w_sh * self.sh_v0
                self.lam_sh = np.clip((w0 + cz) / (w0 + vz), 0.0, 1.5)
            q_dem = self.cust * Ssec_f / self.p
        else:
            q_dem = (wts / secsum(wts)) * Ssec_f / self.p
        self._Stot = S_tot
        self._Ssec = Ssec_f.copy()
        sold = np.minimum(q_dem, avail)
        spare = avail - sold
        unmet = secsum((q_dem - sold) * self.p)
        spv = secsum(spare * self.p)
        sh2 = np.where(spv > 0, spare * self.p / np.maximum(spv, 1e-300), 0.0)   # one pass of spillover within the sector
        sold = sold + np.minimum(sh2 * unmet / self.p, spare)
        revenue = sold * self.p
        spent = revenue.sum()
        fill = spent / max(S_tot, 1e-12)
        rev_s = np.bincount(sec, revenue, minlength=nsec); q_s = np.bincount(sec, sold, minlength=nsec)
        fill_s = np.where(S_sec > 0, rev_s / np.maximum(S_sec, 1e-12), 1.0)
        Pp_s = np.where(q_s > 0, rev_s / np.maximum(q_s, 1e-12), 1.0)
        if mp.services and mp.weiss:
            fill_g, fill_h = fill_s[0], (1 - mp.s_sv) * fill_s[0] + mp.s_sv * fill_s[1]
            c_real = S_h * ((1 - mp.s_sv) * fill_s[0] / Pp_s[0] + mp.s_sv * fill_s[1] / Pp_s[1])
        else:
            fill_g = fill_h = fill
            c_real = S_h * fill / Pp_s[0]
        P_paid = spent / max(sold.sum(), 1e-12)
        I_real = S_I * fill_g / Pp_s[0]
        if mp.durables:
            self.dur = self.dur + gap_buy * fill_g             # durables delivered (rationed with other goods)
            self._D_real = float(S_D.sum() * fill_g / Pp_s[0])
        self.m += revenue - S_I * fill_g
        S_I_f = P_g * I_plan
        if getattr(self, "_ell", None) is not None:          # depreciation rises with utilization
            u_f = np.clip(self._kx * self._yprod / np.maximum(self.K, 1e-12), 0.0, mp.u_max)
            self._dep = (mp.delta - 0.8 * mp.delta1) + mp.delta1 * u_f
            self._util = u_f
        else:
            self._dep = np.full(F, mp.delta)
        K_old = (1 - self._dep) * self.K
        self.K = K_old + I_real
        if mp.embodied and mp.tech:
            # vintages: the installed technique is the capital-weighted (geometric) mix of old machines and new ones
            ph = I_real / np.maximum(self.K, 1e-12)
            self.AK = np.exp((1 - ph) * np.log(self.AK) + ph * np.log(self.AKn))
            self.AL = np.exp((1 - ph) * np.log(self.AL) + ph * np.log(self.ALn))
        self.inv = (avail - sold) * (1 - self.decay) + (np.where(self.to_order, 0.0, y) if mp.prod_lag else 0.0)
        if getattr(self, "trace_on", False) and mp.vacancies:
            self.trace[-1].update(sold=float(sold[0]), dem=float(q_dem[0]), avail=float(avail[0]))
        self._qd, self._sold, self._avail = q_dem.copy(), sold.copy(), avail.copy()
        # value of output per unit of effective labor (this quarter's production at the posted price), smoothed
        ok_ = Ld > 1e-6 * max(Ld.max(), 1e-9)
        vpe = np.where(ok_, self.p * self.prod(Ld) / np.maximum(Ld, 1e-9), self._rev_pe)
        self._rev_pe = 0.5 * self._rev_pe + 0.5 * vpe
        if mp.rho_c < 1:
            # customers who found the firm sold out leave with probability stockout_loss and re-shop next quarter
            unserved = np.maximum(q_dem - np.minimum(q_dem, avail), 0.0) / np.maximum(q_dem, 1e-12)
            lost = mp.stockout_loss * unserved * self.cust
            self.cust = self.cust - lost + secsum(lost) * wts / secsum(wts)
        self.share_prev_used = self.share_prev
        self.share_prev = revenue / np.maximum(secsum(revenue), 1e-12)   # share of its sector

        # 7. settlement
        db = db + S_h * (1 - fill_h) + (S_D * (1 - fill_g) if mp.durables else 0.0)                            # unspent cash stays in the buffer
        if mp.fund:
            if mp.two_accounts:
                # checking pays nothing; savings pay rb. Baumol-Tobin: the checking balance that minimizes transfer
                # costs plus forgone interest is sqrt(cost x spending / (2 rb)); at rb = 0 everything sits in checking
                spend = np.maximum(S_h * fill_h, 1e-12)
                self._spend_last = spend
                M_chk = np.minimum(db, np.sqrt(mp.c_tr * self.yhat * spend / (2 * max(rb, 1e-9)))) if rb > 0 else db
                sav = np.maximum(db - M_chk, 0.0)
                debt_h = np.maximum(-db, 0.0) if r_c is not None else np.zeros_like(db)   # credit-line balances
                int_hh = rb * sav.sum() - (r_c or 0.0) * debt_h.sum()
                self.b = db + rb * sav - (r_c or 0.0) * debt_h
                self._hh_debt = float(debt_h.sum())
                self._chk_share = float(M_chk.sum() / max(db.sum(), 1e-12))
            else:
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
        m_int = (0.0 if mp.two_accounts else rb) * np.maximum(self.m, 0)   # firms' cash sits in checking
        int_f = self.rL_f * self.loan
        if mp.efp:                                            # the bank's view of coverage: operating cash flow / interest
            cf_ = revenue - wb_f
            cov_obs = np.clip(cf_ / np.maximum(int_f, 1e-9), 0.05, 50.0)
            self.cov = np.where(self.loan > 0, 0.75 * self.cov + 0.25 * cov_obs, self.cov)
        self.m += m_int - int_f
        short = np.maximum(0.0, -self.m)
        self.loan += short; self.m += short
        # default: book net worth below zero -> the bank writes the loan down to restore zero net worth
        nw = P_g * self.K + v * self.inv + self.m - self.loan
        loss = np.where(nw < 0, -nw, 0.0)
        self.loan -= loss
        # owners are paid the quarter's accounting profit (book depreciation at historical cost); remaining cash
        # repays loans, since holding deposits at rb while owing at rL loses money
        profit = revenue - wb_f - int_f + m_int - self._dep * self.Kb
        self.Kb = (1 - self._dep) * self.Kb + S_I_f * fill_g
        div_f = np.clip(profit, 0.0, np.maximum(self.m, 0.0))
        self.m -= div_f
        self.Ehat = (1 - mp.g_earn) * self.Ehat + mp.g_earn * profit
        # government: interest on its account
        g_int = (rb if self.g > 0 else i) * self.g
        # bank
        prof_B = int_f.sum() + max(-g_int, 0.0) - int_hh - m_int.sum() - max(g_int, 0.0) - loss.sum() - f_int + self._int_inst
        self.E_B += prof_B
        div_B = max(0.0, self.E_B - mp.cap_B * (self.loan.sum() + max(-self.g, 0.0)))
        self.E_B -= div_B
        # fund values each firm at trailing earnings capitalized at its assumed return: the long-run deposit rate
        # it has observed (discounted average of its own experience) plus its published equity premium
        if mp.fund_flow:
            # the fund posts the value of its firms as a multiple of their trailing earnings; the multiple moves with
            # last quarter's net flow into the fund (inelastic markets, Gabaix and Koijen 2021: a flow of 1% of the
            # market's value raises prices by about M = 5%)
            V = np.maximum(self.Ehat, 0.0) * self.mult
        else:
            r_f = mp.r_assumed if mp.r_assumed else self.rb_lr + mp.erp
            V = np.maximum(self.Ehat, 0.0) / r_f
        V = V * self.vshock                                   # experiments: asset-value bust (decays back)
        self.vshock = 1.0 + mp.vshock_rho * (self.vshock - 1.0)
        if mp.fund:
            # fund collects dividends, keeps cash for withdrawals, invests the rest in its firms (or raises cash
            # from them): holding deposits at rb while its firms borrow at rL loses money
            self.Fc += div_f.sum() + div_B
            need_c = mp.fund_cash_q * (-da[da < 0].sum())
            x = self.Fc - need_c
            if mp.fund_flow:
                # members' net purchases of fund units (contributions less withdrawals) move the posted multiple
                # (inelastic markets); slow-moving value investors pull it back toward the fundamental multiple
                # (trailing earnings at the long-run deposit rate plus the published premium)
                nav_pre = V.sum() + self.E_B + self.Fc
                flow = da.sum() / max(nav_pre, 1e-9)
                m_fund = 1.0 / (self.rb_lr + mp.erp)
                self.mult = float(np.clip(self.mult * np.exp(mp.M_flow * flow - mp.k_rev * np.log(self.mult / m_fund)), 4.0, 400.0))
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
        if mp.retiree_income:
            # reference income is a discounted average of the household's own income, for retirees including what they
            # draw from their retirement account
            self.yhat = (1 - hp.g) * self.yhat + hp.g * (y_h + np.where(work, 0.0, -da))
        else:
            self.yhat = np.where(work, (1 - hp.g) * self.yhat + hp.g * y_h, self.yhat)
        self.age = self.age + 1
        dead = self.age >= hp.J
        if mp.bequest:
            # estates go to heirs: the entrant who replaces the deceased inherits the estate into a retirement account
            # (an inherited account stays with the fund; the bank balance is rolled into it), so it is spent through
            # lower contributions and in retirement
            estate = 0.0
            inh = self.b[dead] + self.div_pending[dead]
            self.a[dead] += inh
            if mp.fund:
                self.Fc += inh.sum()
            else:
                pass
            self.b[dead] = 0.0; self.div_pending[dead] = 0.0
        else:
            estate = self.b[dead].sum() + self.a[dead].sum() + self.div_pending[dead].sum()
            self.g += estate
            if mp.fund:
                self.Fc -= self.a[dead].sum()                 # fund pays out estates' balances
            self.div_pending[dead] = 0.0; self.b[dead] = 0; self.a[dead] = 0
        if mp.durables and getattr(self, "L_inst", None) is not None:
            self.E_B -= self.L_inst[dead].sum()                # installment balances of the deceased are charged off
            self.L_inst[dead] = 0.0; self.n_inst[dead] = 0; self.pay_inst[dead] = 0.0
            self.dur[dead] = 0.5 * mp.dur_theta * self.yhat[dead] / max(P_g, 1e-12)
        self.age[dead] = 0; self.emp[dead] = True
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
        e_f = self.elast(short=True, share=self.share_prev_used)
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
            self.ln_a += mp.g_A * (obs + e_f * np.log(self.p / self.Pref_used) - np.log(S_tot / self.Pref_all_used) - self.ln_a)
            self.sig2 = np.where(out, self.sig2, np.maximum((1 - mp.g_sig) * self.sig2 + mp.g_sig * err ** 2, 1e-4))
        else:
            err = np.log(np.maximum(q_dem, 1e-12)) + e_f * np.log(self.p / self.Pref_used) - self.lnA
            self.lnA += mp.g_A * err
            # own position: the part of the surprise not explained by the size of the market (published afterwards)
            err_a = np.log(np.maximum(q_dem, 1e-12)) + e_f * np.log(self.p / self.Pref_used) - np.log(S_tot / self.Pref_all_used) - self.ln_a
            self.ln_a += mp.g_A * err_a
            self.sig2 = np.maximum((1 - mp.g_sig) * self.sig2 + mp.g_sig * err ** 2, 1e-4)
        # published real growth of final sales (a national-accounts statistic); firms expect their own demand to grow
        # with it, so the level belief carries a trend (discounted least squares on a level plus a published trend)
        Y_real = sold.sum()
        if self.Y_prev is not None:
            self.gY = (1 - mp.g_Y) * self.gY + mp.g_Y * np.log(Y_real / self.Y_prev)
            self.gProd = (1 - mp.g_Y) * self.gProd + mp.g_Y * np.log(Y_real / L_emp / (self.Y_prev / self.L_prev_emp))
        self.Y_prev, self.L_prev_emp = Y_real, L_emp
        self.s_lr = (1 - mp.g_s) * self.s_lr * np.exp(self.gY) + mp.g_s * np.maximum(q_dem, 1e-9)   # orders, incl. backorders
        self.gS = (1 - mp.g_Y) * self.gS + mp.g_Y * np.log(S_tot / self.S_last)
        self.S_last = S_tot
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
            if mp.wage_forecast:                              # discounted least squares (constant-gain RLS)
                xw, yw, gF = self.xW, W_obs / W - 1, mp.g_F
                self.RW = self.RW + gF * (np.outer(xw, xw) - self.RW)
                self.thW = self.thW + gF * np.linalg.solve(self.RW, xw) * (yw - xw @ self.thW)
                # projection facility (Evans and Honkapohja): estimates that would make pay respond more than one for
                # one to inflation, or rise with unemployment, are not used
                self.thW[1] = np.clip(self.thW[1], 0.0, 1.0); self.thW[2] = min(self.thW[2], 0.0)
            self.W = W_obs
        else:
            self.W = W * (1 + mp.pi_star) if mp.fix_wage else max(W_new, mp.lam_W * W)
        if not mp.retiree_income:
            self.yhat = np.where(self.age >= hp.J_W, self.yhat * self.W / W, self.yhat)   # retirees: wage-indexed
        if getattr(self, 'z0', None) is not None:
            self.z[dead] = self.z0[dead]; self.uq[dead] = 0      # a new entrant starts with the original skill
        self.yhat[dead] = self.W * self.z[dead]
        uc_rec = float(np.mean(self.unit_cost(W, self.P_prev * (r_own + mp.delta))))
        self.P_prev = P_paid
        self.P_prev_sec = np.where(q_s > 0, Pp_s, P_paid)
        GDP = spent                                           # final sales (consumption + investment), nominal
        self.GDP_hist.append(GDP)
        self.t += 1
        if not record:
            return None
        return dict(t=self.t, i=i, rb=rb, ra=ra, rL=rL, tau=tau, P=P_paid, pi=pi, W=W, u=u,
                    Y=sold.sum(), C=c_real.sum(), I=I_real.sum(), GDP=GDP, fill=fill, inv=self.inv.sum(),
                    K=self.K.sum(), loans=self.loan.sum(), debt=-self.g, E_B=self.E_B, B=self.b.sum(),
                    A=self.a.sum(), M_f=self.m.sum(), defaults=int((loss > 0).sum()), loss_rate=loss_rate,
                    reserves=self.reserves() - self.R0, theta=self.theta, chk_share=getattr(self, "_chk_share", 1.0), price_changes=float(np.mean(np.abs(self.p / self._p_old - 1) > 1e-9)) if getattr(self, "_p_old", None) is not None else 1.0, vac=float(self.vac_last) if mp.wage_posting else 0.0, contact=float(self.c_rate_last), y=y.sum(), Ld_pre=Ld_pre, div=div_f.sum() + div_B, profit_share=(div_f.sum()) / max(GDP, 1e-12),
                    wage_share=wb_f.sum() / max(GDP, 1e-12), wage_disp=float(np.std(np.log(W_firm * np.ones(F)))), spend_h=S_h.sum() * fill_h, contrib=da[da > 0].sum(),
                    withdraw=-da[da < 0].sum(), kappa_mean=float(self.kappa[work].mean()),
                    markup=float(np.mean(self.p)) / uc_rec,
                    inv_sales=self.inv.sum() / max(sold.sum(), 1e-12), sig=float(np.sqrt(self.sig2).mean()),
                    V=V.sum(), Fc=self.Fc, mult=getattr(self, 'mult', 0.0),
                    util=float(np.average(getattr(self, '_util', np.full(F, 0.8)), weights=self.K)),
                    nocut_bind=float(np.mean(self._cut_wanted[((np.arange(F) + self.t) % mp.review_q == 0)])) if hasattr(self, '_cut_wanted') else 0.0, adopt=self.adopt_last, gY=self.gY, n_ideas=self.n_ideas, pool=len(self.poolK) if mp.tech else 0,
                    AK=float(np.exp(np.mean(np.log(self.AK)))) if mp.tech else 1.0, AL=float(np.exp(np.mean(np.log(self.AL)))) if mp.tech else 1.0,
                    Lprod=float(Ld.sum()), Kuse=float(self.K.sum()), book=float((P * self.K + v * self.inv + self.m - self.loan).sum()))
