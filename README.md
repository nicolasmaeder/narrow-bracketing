# One Period at a Time

Code for a macro model that is computed one quarter at a time. Every agent decides from its own state (including its beliefs) and the prices posted that quarter; every price is posted by some agent; beliefs are updated from the agent's own observations. There is no fixed point within a period and no equilibrium law of motion that agents must know, so the economy evolves as X_{t+1} = Φ(X_t, ε_t).

## Layout

- `model/household.py`: household block. One-quarter objective over consumption, a liquid buffer and a retirement balance with targets from published advice; retirement contributions set by a calculator at a fixed planning return (re-planned only outside a band); retirees draw down balance ÷ quarters left.
- `model/macro.py`: the economy.
  - Firms: pricing-newsvendor problem for price and stock; capital from expected cost plus user cost plus adjustment cost; demand level and noise learned by discounted least squares.
  - Buyers: logit choice across firms.
  - Bank: lending creates deposits.
  - Retirement fund: owns firms and bank; values firms at normalized earnings (a discounted average with a memory of about ten years, as in Graham-Dodd normalized earnings and Shiller's CAPE) over its assumed return.
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

Calibration pass (Nov 22): hiring cost about half a quarter's wage per hire (Blatter et al.); six application rounds a quarter; price-adjustment cost calibrated to the frequency of price changes (about 35-40% a quarter, as in Golosov-Lucas and Nakamura-Steinsson); Jorgenson user cost; production planned at next quarter's planned price; firms' expected inflation half anchored on the target, slow to move. Long run (4 seeds): unemployment about 2.7-4%, inflation 2.0%, growth 2%, price over marginal cost 1.42, wage share 0.48, K/Y 1.3, wealth/GDP 6.3. In a 20% asset-value bust the no-cut rule binds for about half of the firms reviewing pay; but prices still fall too much (about 14% over four years against roughly 4% for core prices in 2008-11), investment rises, and output barely falls.

Since the last version:
- **Real growth, with the aggregate technology derived rather than assumed.** Each firm uses one technique (a capital and a labor productivity) in a CES technology with elasticity 0.5 (Oberfield and Raval). Ideas are public and arrive at a rate that grows over time (Jones 2005); each firm chooses the cheapest known technique at its own factor prices when the saving covers an adoption cost (its workers' time). Techniques are embodied in machines (Solow 1960; Jones and Liu 2024): the chosen technique is that of the machines the firm buys, and its installed technique is the capital-weighted mix of vintages, so new techniques spread at the pace of investment. No aggregate production function appears anywhere.

  Embodiment matters for cycles. Disembodied (retooling all capital at once), technique switches are lumpy and drive part of the endogenous fluctuation; embodied, the unemployment s.d. falls from 3.0 to 1.3 and from 2.9 to 1.9 points in two of three seeds (2.0% growth, markups and shares unchanged). After a 20% asset bust (4 seeds, common random numbers) unemployment peaks at +8 points rather than +6.5, output falls 4.5% rather than 3%, and the recovery takes about 7 rather than 5 years, because the investment slump also slows the spread of new techniques. (These numbers are with the earlier one-year earnings memory; see below.)

- **Valuation memory (Nov 22).** With one year of trailing earnings, household wealth moved one-for-one with profits, and that was the model's main amplifier, of the endogenous cycle and of the asset bust alike. With a ten-year memory (4 seeds): unemployment 2.6% (s.d. 0.5 points), inflation 1.3-1.4% (below the 2% target), growth 1.5-1.8%, price over marginal cost 1.42, wage share 0.47, K/Y 1.3, wealth/GDP 6.6. A 20% asset bust now raises unemployment only 2 points, so large recessions have to come from shocks with real-world channels (credit, housing, borrowers), not from valuation myopia.
- **Same demand belief in pricing and hiring (Nov 22).** Hiring had valued output along the long-run elasticity, pricing along the short-run one; hiring now uses the pricing belief. This fixed the real-wage drift under utilization (u falls from about 20% to 5-6%). Baseline: price over marginal cost 1.52, wage share 0.43, u 2.6%.
- **Variable utilization is still off.** See DECISIONS.md: under flat marginal cost the real wage drifts up, then a wave of labor-saving technique adoption follows and causes capacity-driven unemployment.
- **Demand beliefs that survive growth.** A firm's expected demand is its own position in its market (learned from its orders) times the published size of the market, so the learned part is stationary.
- **Balanced-growth household objective.** Consumption is valued relative to reference income, like the buffer and retirement terms, so the objective is homothetic (King, Plosser and Rebelo).
- **Pay forecasts partly anchored** on the inflation target plus published productivity growth (as salary-budget surveys are), with a projection facility on the learned part.
- **State-dependent pricing is now the default.** A firm changes its price when the expected gain under its own beliefs exceeds a price-adjustment cost of 1.2% of revenue (Zbaracki et al. 2004). Firms no longer value customers they cannot serve: previously a sold-out firm could gain by slashing its price to win customers it then turned away, which caused rare hyperinflation episodes (2 of 16 seeds). About 74% of firms change price each quarter, which is too frequent (evidence: 30–50%). Price timing is no longer an institution taken as given.
- **Bug fix:** the price grid stopped at 1.8 times unit cost; once productivity growth lowered unit costs, firms hit the cap and could not price away shortages, which drove boom-bust cycles.

Long run over 4 seeds (800 quarters): unemployment 3.6–4.5% (sd 0.9–1.6), inflation 2.4–2.6%, real growth about 2.0% a year, 95% of goods demand met.

Derived aggregate technology (2 seeds, 700 quarters): technical change is labor-augmenting (labor productivity of techniques +2.0% a year, capital productivity +0.0–0.1%); the capital share is stable while the wage-rental ratio rises 2% a year, so the long-run aggregate elasticity of substitution is 0.96–1.00 (Cobb-Douglas) although each firm's is 0.5; over 5-year windows it is about zero. This is Jones's result, produced by firms' decisions.


Experiments, 16 seeds, common random numbers (`model/experiments_many.py`, `model/analyze_many.py`; mean and 5–95% range across seeds), no crises in any baseline:
- Policy-rate shock (+1 pp, decaying 0.8 a quarter): output −0.2% after 3–4 quarters, unemployment +0.13 pp, inflation −0.3 pp on impact and −0.15 to −0.2 pp for two years. Fixed vs current planning return makes almost no difference. Consumption rises 0.3% on impact (households are all net savers, so higher rates raise their interest income) and turns negative after a year; investment falls 3.7%.
- Stimulus check (1% of annual GDP): output +0.2% at quarters 4–7, unemployment −0.07 pp, inflation +1.4 pp in the first quarter.
- Withholding cut (same total, 4 quarters): similar output, smaller and more drawn-out inflation. With state-dependent pricing, cumulative nominal household spending is not lower than after the check (4.3% vs 3.7% of quarterly GDP over 8 quarters), unlike with fixed price reviews (1.5% vs 3.2%); this needs a partial-equilibrium MPC measure to separate household behavior from wage and price responses.

Open:
- About 11% of goods demand goes unmet in a quarter. Unmet orders effectively become backlog, since households keep the cash and spend it later.
- Unemployment is somewhat volatile.
- Review timing is fixed rather than state-dependent. State-dependent pricing (`menu_cost`, the price changes when the gain under own beliefs exceeds a share of revenue) is implemented but destabilizes the economy. Prices then become far more flexible than non-cuttable wages, and real-wage swings of ±40% drive mass layoffs and rehiring. The next ingredient is buyers' aversion to price changes (customer markets; Blinder et al.'s top reason for sticky prices).
