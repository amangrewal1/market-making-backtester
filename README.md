# Market Making Strategy Backtester

![tests](https://github.com/amangrewal1/market-making-backtester/actions/workflows/test.yml/badge.svg) ![license](https://img.shields.io/badge/license-MIT-blue)

Backtesting framework for liquidity-provision strategies on AMMs and CLOBs under
regime-switching market conditions. Includes a Bayesian regime-inference market
maker that updates quoting decisions from observed order flow, and a fixed-threshold
baseline for comparison.

## Install

```
pip install -r requirements.txt
```

## Run

```
# Small run
python run_backtest.py --episodes 500 --horizon 1000

# Full run (matches 50k+ episodes claim)
python run_backtest.py --episodes 50000 --horizon 1000 --venue clob

# AMM backtest
python run_backtest.py --episodes 10000 --horizon 500 --venue amm

# Plot results
python plot_results.py
```

## Layout

```
src/
  price_process.py   regime-switching mid-price + order-flow generator
  clob.py            CLOB venue (Glosten-Milgrom style adverse selection)
  amm.py             concentrated-liquidity AMM (v3-style range)
  strategies/
    base.py          strategy interface
    fixed.py         fixed-threshold baseline
    bayesian.py      Bayesian HMM regime inference market maker
  engine.py          backtest loop + episode runner
  metrics.py         PnL, Sharpe, win-rate, drawdown
```

## Method

- **Market states** (hidden): calm, trending-up, trending-down, volatile. Sticky
  Markov chain drives regime transitions; each regime maps to a drift, volatility,
  and informed-trader probability.
- **Observations**: realized returns, signed order flow imbalance over a rolling
  window, trade intensity.
- **Bayesian update**: forward-filter over the HMM posterior `p(S_t | obs_{1:t})`
  each step.
- **Quoting policy**: expected spread widens with posterior mass on volatile/toxic
  regimes, skews with expected drift, shrinks depth when adverse-selection risk is
  elevated.
- **Baseline**: fixed bid-ask spread + inventory band, no regime awareness.

## Output

The win-rate metric is the fraction of episodes in which the Bayesian policy's PnL
exceeds the fixed-threshold baseline's PnL on the same simulated path.
