# CLOB vs AMM

| Venue | Price Discovery | Adverse Selection | Inventory Management |
|---|---|---|---|
| CLOB | Continuous, quoted bid/ask | Glosten-Milgrom style | Explicit via quote sides |
| AMM | Constant-product / concentrated-range | Slippage curve | Implicit via range bounds |

The Bayesian strategy works on both venues; its spread widening translates
to wider AMM range bounds under volatile regimes.
