"""rolling_sharpe handles degenerate cases"""

from src.metrics import rolling_sharpe


def test_sharpe_flat_pnl_returns_zero():
    flat = [0.0] * 300
    s = rolling_sharpe(flat, window=252)
    assert s == 0.0


def test_sharpe_short_input_returns_nan():
    import math
    s = rolling_sharpe([1.0, 2.0, 3.0], window=252)
    assert math.isnan(s)

