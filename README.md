# One Period at a Time

Code for a macro model that is computed one quarter at a time. Every agent decides from its own state (including its beliefs) and the prices posted that quarter; every price is posted by some agent; beliefs are updated from the agent's own observations. There is no fixed point within a period and no equilibrium law of motion that agents must know, so the economy evolves as X_{t+1} = Φ(X_t, ε_t).

## Layout

- `model/household.py`: household block. One-quarter objective over consumption, a liquid buffer and a retirement balance with targets from published advice; retirement contributions set by a calculator at a fixed planning return (re-planned only outside a band); retirees draw down balance ÷ quarters left.
- `model/macro.py`: the economy.
  - Firms: pricing-newsvendor problem for price and stock; capital from expected cost plus user cost plus adjustment cost; demand level and noise learned by discounted least squares.
  - Buyers: logit choice across firms.
  - Bank: lending creates deposits.
  - Retirement fund: owns firms and bank; values firms at trailing earnings over its assumed return.
  - Government: payroll tax from the structural budget and a debt correction; estates.
  - Central bank: smoothed Taylor rule.
  - Accounting check: bank reserves stay at zero.
- `model/run_macro.py`: build and run the baseline (`python run_macro.py 800`).
- `model/experiments_macro.py`: stimulus check vs withholding cut; policy-rate shock under fixed vs current planning return.
- `model/diag.py`: shock responses with one block switched off at a time.
- `household_tests/`: household block checks against micro evidence (MPCs by account and liquidity, pension offset, windfall size) and calibration grids.
- `archive/`: earlier versions (rule-based firms; dividends spent by households; fund before wage posting).
- `figures/`: current figures.

## Status

`model/DECISIONS.md` lists every decision, its objective and its information. `model/check_principles.py` checks the accounting (reserves stay at zero to about 1e-15 of deposits) and that a quarter depends only on the state.

Since the last version:
- **Production takes a quarter.** Price rations stock on hand, and production targets next quarter's stock.
- **One marginal labor cost.** Production is planned at the marginal cost of the labor that makes it, the same cost the hiring decision uses.
- **Banks optimize.** Deposit and loan rates come from Monti–Klein profit maximization by 5 symmetric banks, with customers choosing banks by logit in the rate. Spreads are about 2% a year.

- **Two kinds of deposit.** Checking pays nothing; savings pay a rate set by the banks' profit problem. Households split their liquid balances by Baumol–Tobin, keeping about 31% in checking. Firms' cash sits in checking.

Long run over 2 seeds: unemployment about 6.2% (sd about 2.6 points), inflation about 2.5%, savings rate about 1.9 points below the policy rate, labor share about 0.51.

Open:
- About 11% of goods demand goes unmet in a quarter. Unmet orders effectively become backlog, since households keep the cash and spend it later.
- Unemployment is somewhat volatile.
- Review timing is fixed rather than state-dependent. State-dependent pricing (`menu_cost`, the price changes when the gain under own beliefs exceeds a share of revenue) is implemented but destabilizes the economy. Prices then become far more flexible than non-cuttable wages, and real-wage swings of ±40% drive mass layoffs and rehiring. The next ingredient is buyers' aversion to price changes (customer markets; Blinder et al.'s top reason for sticky prices).
