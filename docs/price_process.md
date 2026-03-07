# Regime-Switching Price Process

Mid-price evolves under a hidden 4-state Markov chain:
- **calm** — low volatility, zero drift
- **trending-up** — low volatility, positive drift
- **trending-down** — low volatility, negative drift
- **volatile** — high volatility, zero drift, high informed-trader probability

Transitions are sticky (P[stay] ≈ 0.95 per step) to match realistic regime
persistence.
