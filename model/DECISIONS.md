# Decisions, objectives and information

The paper's three tests are:

- **Lucas critique.** No decision rule is fitted to past policy. Each decision below solves a stated objective using:
  - the decision-maker's own state, including its beliefs;
  - beliefs updated from its own observations or from published statistics;
  - prices posted this quarter or observed last quarter.
- **State-wise computation.** Each quarter is computed once, in a fixed order, as a function of the state. No market is solved for a fixed point within the quarter.
- **Werner accounting.** Loans create deposits and every payment moves money between bank accounts. Bank reserves stay at zero (`check_principles.py`).

Policy rules (the central bank's interest-rate rule and the government's tax rule) are the policy being varied, so they are primitives by design.

| Agent | Decision | Objective | Beliefs / information | Updating |
|---|---|---|---|---|
| Household (worker) | consumption, liquid buffer | one-quarter utility of consumption relative to reference income (homothetic, so balanced growth is possible) plus value of buffer relative to a published target (1.5 quarters of income); entry cost for drawing the buffer below its starting balance (mental accounts, Shefrin–Thaler) | own income (EWMA), posted prices and rates | own income |
| Household | checking vs savings | Baumol–Tobin: checking balance minimizes transfer costs plus interest forgone on checking, √(transfer cost × spending / (2 × savings rate)); checking pays nothing | own spending and income, posted savings rate | — |
| Household (worker) | retirement contribution rate | calculator: contribution that reaches the published retirement target at a published planning return; re-planned only outside a band | own balance, own income | — |
| Household (retiree) | drawdown | balance ÷ quarters left (log utility with zero perceived return) | own balance, age | — |
| Buyer | which firm to buy from | customer market: each quarter half of buyers re-shop by random utility (logit in relative price), the rest stay with their seller (switching costs); buyers turned away by a stockout switch with probability 0.3 (Anderson, Fitzsimons and Simester) | posted prices, own experience of stockouts | — |
| Worker | which job | random utility: relative wage plus a match-specific amenity; job-to-job only if the offer beats the current job | posted wages | — |
| Firm | price (changed when the expected gain under own beliefs exceeds a price-adjustment cost of 1.2% of revenue, physical plus managerial, Zbaracki et al. 2004) | expected profit from selling the stock on hand, p E[min(D, S)] + v E[(S − D)+], plus the value of customers won or lost (margin over the expected relationship, static beliefs; customers beyond what the stock on hand can serve are worth nothing), net of customers lost to stockouts; unsold stock valued at its production cost at the wage | expected demand = own position in the market × published market size (last published nominal final sales grown at their published growth, over the expected price index); demand noise | discounted least squares on own orders (sales plus backorders) relative to the market |
| Firm | technique of the machines it buys (embodied: installed machines keep theirs, so the firm's technique is the capital-weighted mix of vintages) | choose the cheapest known technique at own wage and capital cost if the unit-cost saving, valued over the technique's expected life (cost of funds plus the rate at which it has seen its techniques superseded), exceeds the adoption cost (workers' time, 2% of a quarter's revenue) | stock of public ideas | own record of technique turnover (discounted average) |
| Firm | production (made now, sold from next quarter) | stocking newsvendor for next quarter at the marginal cost of the labor that makes it (wage, plus marginal hiring cost spread over tenure when adding workers) | same, plus expected carry-over | — |
| Firm | hires and layoffs | marginal revenue product along own demand belief, minus wage, over expected tenure, net of convex hiring costs (Blatter et al.; Gavazza–Mongey–Violante); layoffs when marginal revenue product falls below the wage | published job-filling and quit rates; own record of the share of wanted hires that arrive | discounted averages |
| Firm | pay (reviewed yearly, staggered; no nominal cuts) | cost per worker including replacing departures (Salop turnover model); a nominal cut lowers incumbents' effort (Bewley) | expected market pay growth: half the published inflation target plus published productivity growth (as in salary-budget surveys), half a forecast from published data | discounted least squares on published pay, inflation, unemployment, with a projection facility (Evans and Honkapohja) |
| Firm | capital | expected labor cost of expected sales plus user cost plus convex adjustment cost (Jorgenson) | own cost of funds | — |
| Firm | dividends, borrowing | pay out accounting profit; use spare cash to repay loans (cash sits in checking at zero interest, loans cost the loan rate) | own accounts | — |
| Bank (5 symmetric; consolidated balance sheet) | savings and loan rates (checking pays nothing) | expected profit given rivals' last posted rates; depositors and borrowers choose banks by logit in the rate (Monti–Klein; Drechsler–Savov–Schnabl) | own loss history, policy rate | discounted average of loss rates |
| Retirement fund | valuation, investment of cash | values firms at trailing earnings over its assumed return (long-run deposit rate it has observed plus a published premium); keeps cash for withdrawals, invests the rest in its firms | own earnings records, deposit-rate history | discounted averages |
| Central bank | policy rate | smoothed Taylor rule (policy primitive) | published inflation, unemployment | — |
| Government | payroll tax | structural budget plus debt correction (policy primitive) | own budget | — |

## Institutions taken as given (with evidence)

- **Pay review timing:** once a year (Grigsby, Hurst and Yildirmaz).
- **Matching function:** contacts depend on openings per unemployed (Petrongolo and Pissarides).
- **Production:** takes a quarter. Each firm's technology is CES in capital and labor with elasticity 0.5 (Oberfield and Raval 2021), given its technique.
- **Embodiment:** a technique is built into the machines that carry it (Solow 1960; Greenwood, Hercowitz and Krusell 1997; Jones and Liu 2024). The firm's installed technique moves toward the one it now buys at the rate new capital replaces old (capital-weighted geometric average of vintages). In the task framework of Jones and Liu (2024), a firm CES in capital and labor with elasticity 0.5 is exactly what a continuum of tasks with automated share β and capital quality Z gives (capital-augmenting term (Z/β)², labor-augmenting term (1−β)⁻²), so a technique here is an (automation, quality) pair. The investment decision still values new capital at the installed technique (a known approximation).
- **Ideas:** pairs of capital and labor productivity with independent Pareto tails, public, arriving at a rate that grows at a fixed rate (research effort; Jones 2005; Kortum 1997). No aggregate production function is assumed; its long-run form and the direction of technical change are outcomes.

Pay review timing is fixed. Price timing is state-dependent: whether to change price is itself a decision (menu cost), so the frequency of price changes responds to policy.

## Open: variable utilization (Nov 2026)

Diagnosis of why the economy breaks with utilization on (u about 20%):
- Not a customer price war. An option where firms learn how much share their relative price actually wins (`share_learn`, discounted least squares against the logit prediction, prior at 1) gives a learned slope of about 1 in the baseline and does not help with utilization on.
- The demand elasticity eps = 7.7 was calibrated under a steep short-run marginal cost. With flat marginal cost the markup over variable cost alone must cover capital, which needs eps of about 4.5.
- The main amplifier is the fund's valuation: firms are valued at one year of trailing earnings (g_earn = 0.25). With lower margins, earnings and household wealth fall 98% within 30 quarters, consumption falls 45%, and the policy rate sits at zero.
- With a ten-year memory (g_earn = 0.025, the horizon of Shiller's CAPE), the baseline unemployment s.d. falls from 2–3 points to 0.5 and the 20% asset bust raises unemployment only 2 points. With utilization on, u is still about 10% with capacity nearly full: capacity is planned on normal sales, and sales are capped by capacity, so capacity can stay too small.
- Default changed to the ten-year memory (g_earn = 0.025).
- Two further options tried with utilization on, both off by default because they did not help. `cap_unc` plans capacity for orders at the price the firm would charge without a capacity limit; it overbuilds capital and lowers profits, and u rises to about 30%. `clv_lr` values customers at the margin over long-run unit cost; it changes little. A ZLB-off diagnostic (`zlb`) shows the policy-rate floor is not the problem. The remaining mechanism: after the switch, the real wage rises about 30% with no productivity gain (markups over short-run cost fall). Labor-saving techniques are then adopted in a wave (labor productivity +60% in 25 years), and labor capacity falls to about 75% of the labor force.
- **Found and fixed (default):** hiring valued extra output along the long-run demand elasticity (about 8), while the price and stock decision used the short-run one (about 4.6). With flat marginal cost every firm wanted to hire up to capacity, and together they bid wages up: that was the real-wage drift. Hiring now sells extra output along the same demand belief as pricing (`hire_long = False`). Baseline (4 seeds): u 2.6-2.7% (s.d. 0.5-0.7), inflation 1.4-1.6%, price over marginal cost 1.52, wage share 0.42-0.44, K/Y 1.3. With utilization on (eps 6): u 5-6% (s.d. 3-4), utilization 0.93-0.94 against 0.8 planned. Capital stays about 15% below the putty-clay target because the convex adjustment cost (phi_K = 2) is weighed against one quarter's gain rather than the capital's life. Spreading it over the life (`k_amort`) with phi_K 20-35 makes investment self-driven (accelerator) volatility, so it is off.
- 20% asset bust with the new defaults: unemployment +1.5 points at the peak, the price level -13% (still too much deflation), output about -1%.

## Why the price level fell too much in the asset bust (Nov 23)

`price_decomp.py` splits firms' price changes in the bust (relative to the no-shock path, common random numbers). In the first year the fall comes from the demand belief (published market size drops about 5-8%). After that prices follow unit cost, which follows posted pay: pay ended 14% below the no-shock path because essentially every reviewing firm froze pay for three and a half years. A firm that plans to shrink faster than attrition gets nothing from pay, so it wants a cut and freezes. In the data raises slowed but continued (job-stayer nominal wage growth fell from about 4% to about 2% in 2009-11; Grigsby, Hurst and Yildirmaz 2021).

The channels tested and ruled out: no customer market (rho_c = 1; deflation smaller but a deep, persistent recession, u +5 points for years), learned share response, full-cost customer value, and a higher firm elasticity (sigma_f = 0.8).

Fix (default): Weiss (1980) adverse selection in hiring and quits, the user's JMP mechanism and Campbell and Kamlani's (1997) leading reason firms do not cut pay. Pay then buys worker quality even for a shrinking firm. Bust (4 seeds): u +2.5 points for about three years with slow recovery, output -3%, investment -7%, consumption -3%, price level -8% (data: core prices about -4% relative to trend in 2008-11). Long run (4 seeds): u 3.2-3.6% (s.d. 1.3-1.6), inflation 1.4-1.7%, growth 1.5-1.65%, wage share 0.47, K/Y 1.2-1.3, wealth/GDP 6.7, price changes 27-30% a quarter (below the 35-40% target; the menu cost needs recalibrating), measured price over marginal cost about 1.9 (measured at posted pay per head, not per unit of quality).

### Robustness of the Weiss result (16 seeds) and the skill distribution

16 seeds, 20% asset bust. Weiss on (default): u +2.5 points for about three years, then back to baseline. Output -2 to -3% for two to three years, price level -5 to -7%. Weiss off: u +2.5 points at the peak but gone within a year, then below baseline; output barely moves; price level -10.5%. The default stands.

Caveat (the user's JMP, Fire_Sale.tex, appendix A.4, following Weiss 1980): a true Weiss wage floor exists only if output per dollar, E[q | q <= q*] / q*, has an interior maximum. Under the lognormal skill distribution (the default, s.d. 0.5) this ratio falls monotonically in q* at every dispersion, so there is no floor. Under lognormal, quality enters pay setting as a reason pay is worth something to a shrinking firm, not as a floor. With a mass at low skill (rectified normal, z_min 0.05) the ratio peaks at about half of workers, so a floor exists. On 4 seeds that version gives a smaller recession and more deflation; it is not yet calibrated (the floor binds in normal times).

