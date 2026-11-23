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

