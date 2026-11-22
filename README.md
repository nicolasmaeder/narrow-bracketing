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
- **Real growth, with the aggregate technology derived rather than assumed.** Each firm uses one technique (a capital and a labor productivity) in a CES technology with elasticity 0.5 (Oberfield and Raval). Ideas are public and arrive at a rate that grows over time (Jones 2005); each firm adopts the cheapest known technique at its own factor prices when the saving covers an adoption cost (its workers' time). No aggregate production function appears anywhere.
- **Demand beliefs that survive growth.** A firm's expected demand is its own position in its market (learned from its orders) times the published size of the market, so the learned part is stationary.
- **Balanced-growth household objective.** Consumption is valued relative to reference income, like the buffer and retirement terms, so the objective is homothetic (King, Plosser and Rebelo).
- **Pay forecasts partly anchored** on the inflation target plus published productivity growth (as salary-budget surveys are), with a projection facility on the learned part.
- **State-dependent pricing is now the default.** A firm changes its price when the expected gain under its own beliefs exceeds a menu cost of 0.7% of revenue (Levy et al. 1997). About 38% of firms change price each quarter, close to the micro evidence, and the economy is stable (it was not before the grid fix below). Price timing is no longer an institution taken as given.
- **Bug fix:** the price grid stopped at 1.8 times unit cost; once productivity growth lowered unit costs, firms hit the cap and could not price away shortages, which drove boom-bust cycles.

Long run over 4 seeds (800 quarters): unemployment 3.6–4.5% (sd 0.9–1.6), inflation 2.4–2.6%, real growth about 2.0% a year, 95% of goods demand met.

Derived aggregate technology (2 seeds, 700 quarters): technical change is labor-augmenting (labor productivity of techniques +2.0% a year, capital productivity +0.0–0.1%); the capital share is stable while the wage-rental ratio rises 2% a year, so the long-run aggregate elasticity of substitution is 0.96–1.00 (Cobb-Douglas) although each firm's is 0.5; over 5-year windows it is about zero. This is Jones's result, produced by firms' decisions.

Open:
- About 11% of goods demand goes unmet in a quarter. Unmet orders effectively become backlog, since households keep the cash and spend it later.
- Unemployment is somewhat volatile.
- Review timing is fixed rather than state-dependent. State-dependent pricing (`menu_cost`, the price changes when the gain under own beliefs exceeds a share of revenue) is implemented but destabilizes the economy. Prices then become far more flexible than non-cuttable wages, and real-wage swings of ±40% drive mass layoffs and rehiring. The next ingredient is buyers' aversion to price changes (customer markets; Blinder et al.'s top reason for sticky prices).
