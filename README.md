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

- Wages are posted by firms (`wage_posting=True`). Each firm picks the wage that maximizes the quarter's expected profit, evaluating it by the workforce it sustains under its own beliefs about applicants and quits; it does not cut its own nominal wage (morale); it reviews wages once a year (staggered). Workers choose among hiring firms by logit in the relative wage; on-the-job searchers move if an offer plus a match-specific amenity beats their job.
- Baseline is stable over 1,000 quarters and across seeds: unemployment ≈ 4.8%, inflation ≈ 2.3%, policy rate ≈ 3.8%, labor share ≈ 0.57.
- Open problem: with a constant contact rate, unemployment is almost purely frictional, so demand shocks barely move it for two years. A matching function (`m_match`, contact rising with vacancies per unemployed) makes unemployment cyclical but produces a regular boom-bust cycle in which unemployment swings between 0% and 10%.
- Firms learn demand only from their own sales and whether they sold out (`censored`): if stock was left, sales are demand; if they sold out, they update on the mean of their belief given demand ≥ stock (Bayes with a truncated lognormal). Unmet goods demand falls to about 1%.
- Firms price for the output that the labor they expect to get can make (`cap_aware`): each firm learns from its own record what share of the labor it asks for it actually gets. Before this, firms priced as if they could always hire, so 6–12% of goods demand went unmet every quarter without raising prices.
- Experimental, off by default: annual pay reviews as a turnover-cost problem (Salop) with (`vacancies=True`, recruiting takes `c_v` of a worker's time per vacancy; firms hire toward the labor whose product covers wage plus amortized recruiting cost, lay off when product falls below the wage) together with a matching function (`m_match`, `c_max`). Still unstable: vacancies explode when firms see spilled-over unmet demand as their own, and when firms stop hiring their fill-rate beliefs freeze at crisis levels, which drives a wage spiral during a slump.
- Firms forecast market pay growth from published inflation and unemployment by discounted least squares (`wage_forecast`); without it, flat pay was self-confirming and nominal wages moved only in upward bursts. The central bank's neutral rate is now 5% (`i_star=0.0125`): at 3% the economy ran hot (unemployment 4.3%, inflation 3.2%). Baseline: unemployment about 5% (sd 1.6–2.5 points), inflation about 2%.
- `wage_posting=False` reproduces the earlier placeholder wage rule.
- Still to replace with objectives: bank rate margins. Planned: production lags.
