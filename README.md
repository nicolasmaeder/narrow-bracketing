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

- Wage posting (firms post wages, workers choose employers by logit, firms learn applicants and quits) is in `macro.py` behind `wage_posting`, off by default: it is not yet stable. With it off, wages follow a placeholder rule.
- Still to replace with objectives: wage setting, bank rate margins. Planned: production lags.
