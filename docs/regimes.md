# Regime Parameters

| Regime | Drift | Volatility | P(informed trader) |
|---|---|---|---|
| calm | 0 | low | 0.05 |
| trending-up | + | low | 0.10 |
| trending-down | − | low | 0.10 |
| volatile | 0 | high | 0.40 |

The informed-trader probability drives the Glosten-Milgrom adverse-selection
loop on the CLOB venue.
