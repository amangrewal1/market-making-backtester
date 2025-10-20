import numpy as np
from dataclasses import dataclass


@dataclass
class EpisodeResult:
    strategy: str
    pnl: float
    final_inventory: float
    n_trades: int
    n_informed_fills: int
    max_drawdown: float
    sharpe: float
    equity_curve: np.ndarray


def summarize(results):
    pnl = np.array([r.pnl for r in results])
    sharpe = np.array([r.sharpe for r in results])
    dd = np.array([r.max_drawdown for r in results])
    trades = np.array([r.n_trades for r in results])
    return {
        "episodes": len(results),
        "mean_pnl": float(pnl.mean()),
        "median_pnl": float(np.median(pnl)),
        "std_pnl": float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0,
        "pct_positive": float((pnl > 0).mean()),
        "mean_sharpe": float(sharpe.mean()),
        "mean_max_drawdown": float(dd.mean()),
        "mean_trades": float(trades.mean()),
    }


def head_to_head(bayes_pnls: np.ndarray, baseline_pnls: np.ndarray):
    assert len(bayes_pnls) == len(baseline_pnls)
    diff = bayes_pnls - baseline_pnls
    return {
        "episodes": len(diff),
        "mean_diff": float(diff.mean()),
        "median_diff": float(np.median(diff)),
        "win_rate": float((diff > 0).mean()),
        "loss_rate": float((diff < 0).mean()),
        "tie_rate": float((diff == 0).mean()),
        "avg_win": float(diff[diff > 0].mean()) if (diff > 0).any() else 0.0,
        "avg_loss": float(diff[diff < 0].mean()) if (diff < 0).any() else 0.0,
    }


def compute_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak)
    return float(dd.min())


def compute_sharpe(equity: np.ndarray) -> float:
    if len(equity) < 2:
        return 0.0
    rets = np.diff(equity)
    if rets.std() == 0:
        return 0.0
    return float(rets.mean() / rets.std() * np.sqrt(len(rets)))
