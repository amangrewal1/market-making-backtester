# Bayesian Regime Inference

The Bayesian market-maker runs a forward-filter over the regime HMM:

  `p(S_t | obs_{1:t}) ∝ p(obs_t | S_t) ∑_{S_{t-1}} p(S_t | S_{t-1}) p(S_{t-1} | obs_{1:t-1})`

Observations are realized returns, signed order-flow imbalance, and trade
intensity.

Quoting responds to the posterior:
- Expected spread widens with posterior mass on volatile regimes
- Quote midpoint skews with posterior expected drift
- Depth shrinks when adverse-selection risk (informed-trader prob) is elevated
