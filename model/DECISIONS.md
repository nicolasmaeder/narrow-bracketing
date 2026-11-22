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
| Household (worker) | consumption, liquid buffer | one-quarter utility of consumption plus value of buffer relative to a published target (1.5 quarters of income); entry cost for drawing the buffer below its starting balance (mental accounts, Shefrin–Thaler) | own income (EWMA), posted prices and rates | own income |
| Household | checking vs savings | Baumol–Tobin: checking balance minimizes transfer costs plus interest forgone on checking, √(transfer cost × spending / (2 × savings rate)); checking pays nothing | own spending and income, posted savings rate | — |
| Household (worker) | retirement contribution rate | calculator: contribution that reaches the published retirement target at a published planning return; re-planned only outside a band | own balance, own income | — |
| Household (retiree) | drawdown | balance ÷ quarters left (log utility with zero perceived return) | own balance, age | — |
| Buyer | which firm to buy from | customer market: each quarter half of buyers re-shop by random utility (logit in relative price), the rest stay with their seller (switching costs); buyers turned away by a stockout switch with probability 0.3 (Anderson, Fitzsimons and Simester) | posted prices, own experience of stockouts | — |
| Worker | which job | random utility: relative wage plus a match-specific amenity; job-to-job only if the offer beats the current job | posted wages | — |
| Firm | price (reviewed every 3 quarters, staggered) | expected profit from selling the stock on hand, p E[min(D, S)] + v E[(S − D)+], plus the value of customers won or lost (margin over the expected relationship, static beliefs), net of customers lost to stockouts; unsold stock valued at its production cost at the wage | own demand level and noise | discounted least squares on own sales plus backorders |
| Firm | production (made now, sold from next quarter) | stocking newsvendor for next quarter at the marginal cost of the labor that makes it (wage, plus marginal hiring cost spread over tenure when adding workers) | same, plus expected carry-over | — |
| Firm | hires and layoffs | marginal revenue product along own demand belief, minus wage, over expected tenure, net of convex hiring costs (Blatter et al.; Gavazza–Mongey–Violante); layoffs when marginal revenue product falls below the wage | published job-filling and quit rates; own record of the share of wanted hires that arrive | discounted averages |
| Firm | pay (reviewed yearly, staggered; no nominal cuts) | cost per worker including replacing departures (Salop turnover model); a nominal cut lowers incumbents' effort (Bewley) | expected market pay growth | discounted least squares on published pay, inflation, unemployment |
| Firm | capital | expected labor cost of expected sales plus user cost plus convex adjustment cost (Jorgenson) | own cost of funds | — |
| Firm | dividends, borrowing | pay out accounting profit; use spare cash to repay loans (cash sits in checking at zero interest, loans cost the loan rate) | own accounts | — |
| Bank (5 symmetric; consolidated balance sheet) | savings and loan rates (checking pays nothing) | expected profit given rivals' last posted rates; depositors and borrowers choose banks by logit in the rate (Monti–Klein; Drechsler–Savov–Schnabl) | own loss history, policy rate | discounted average of loss rates |
| Retirement fund | valuation, investment of cash | values firms at trailing earnings over its assumed return (long-run deposit rate it has observed plus a published premium); keeps cash for withdrawals, invests the rest in its firms | own earnings records, deposit-rate history | discounted averages |
| Central bank | policy rate | smoothed Taylor rule (policy primitive) | published inflation, unemployment | — |
| Government | payroll tax | structural budget plus debt correction (policy primitive) | own budget | — |

## Institutions taken as given (with evidence)

- **Pay review timing:** once a year (Grigsby, Hurst and Yildirmaz).
- **Price review timing:** every 2–4 quarters (Nakamura and Steinsson; Bils and Klenow).
- **Matching function:** contacts depend on openings per unemployed (Petrongolo and Pissarides).
- **Production:** takes a quarter.

These are fixed. Review timing that responds to policy (state-dependent pricing) would be the natural next step for the Lucas-critique argument.
