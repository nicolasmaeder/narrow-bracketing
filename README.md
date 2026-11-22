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

Current default economy (all decisions from objectives, beliefs from own records or published statistics):

- **Pay**: reviewed once a year (staggered). Each firm chooses relative pay to minimize cost per worker including replacing departures (Salop turnover model). Market pay growth is forecast from published inflation and unemployment by discounted least squares. No nominal cuts (morale).
- **Hiring**: each quarter a firm chooses hires. A worker's value is marginal revenue product minus wage over expected tenure, with marginal revenue along the firm's own demand belief. Hiring costs are convex in the firm's hiring rate (Blatter, Muehlemann & Schenker; Gavazza, Mongey & Violante) and rise with published tightness. They are paid as new hires' start-up output loss. Layoffs happen when marginal revenue product falls below the wage. A matching function with this quarter's openings and unemployed sets job contacts. Fill and quit beliefs are discounted averages of the published rates.
- **Prices**: reviewed every three quarters (staggered). Between reviews firms choose only stock. Firms learn demand from their own sales and sell-outs (censored).
- **Long run (3 seeds)**: unemployment 3.3–4.4% (sd 1.4–4.4 points), inflation about 2%, policy rate about 5.5%, unmet goods demand 6–8%.
- **Policy responses (4 seeds)**:
  - A stimulus check of 1% of GDP raises output 0.9% on impact; unemployment falls by up to 0.2 points.
  - A 1-point rate rise lowers output by up to 0.7% and raises unemployment by 0.1–0.2 points. Inflation first rises about 1 point through firms' financing costs, then falls about 1 point.
- **Open**:
  - The labor share is 0.41–0.45 (US about 0.57).
  - Openings per unemployed run 3–6 (US 0.5–2).
  - One seed is much noisier.
  - Bank rate margins are not yet derived from an objective.
