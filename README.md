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

- **Pay**: reviewed once a year (staggered). Each firm chooses relative pay to minimize cost per worker including replacing departures (Salop). Market pay growth is forecast from published inflation and unemployment (discounted least squares). No nominal cuts.
- **Hiring**: each quarter a firm chooses hires. A worker's value is marginal revenue product minus wage over expected tenure. Hiring costs are convex in the firm's hiring rate (about 18 weeks of wages per hire) and rise with published tightness. They are paid as the new hire's lost output in the first quarter. Layoffs happen when marginal revenue product falls below the wage. A matching function with this quarter's openings and unemployed sets contacts.
- **Prices**: reviewed every three quarters (staggered). Firms learn demand from their sales plus backorders (orders placed with them that they could not fill), not from spillover from rivals.
- **Calibration**: workers' response to relative pay 8; recruiting cost scale 20; neutral rate 5%.
- **Long run (3 seeds)**:
  - Unemployment 3.7–4.8% (sd 0.7–1.2 points); inflation about 2% (sd 0.6–1.0).
  - Openings per unemployed about 0.8; labor share about 0.49.
- **Open**: a shortage economy. About 8% of goods demand goes unmet every quarter and inventories are zero, so a stimulus check raises prices but not output. With censored demand learning (`censored=True`, `g_A=0.03`), inventories exist and demand is met (98%), but firms' slow learning produces an 8-year boom-bust cycle (inflation sd about 3). The planned fix is production lags: output made this quarter is sold next quarter, so firms hold stock as a buffer.
