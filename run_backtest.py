import argparse
import json
import os
import time
import numpy as np

from src.engine import run_paired_backtest
from src.metrics import summarize, head_to_head
from src.strategies import (
    FixedThresholdCLOB, FixedThresholdAMM,
    BayesianCLOB, BayesianAMM,
)
from src.price_process import RegimeConfig


def build_factories(venue: str):
    if venue == "clob":
        return {
            "fixed": lambda: FixedThresholdCLOB(half_spread_bps=13.0, base_size=1.0,
                                                inventory_limit=20.0),
            "bayesian": lambda: BayesianCLOB(base_half_spread_bps=13.0, base_size=1.0,
                                             inventory_limit=20.0,
                                             volatile_spread_mult=1.7,
                                             volatile_size_mult=0.65,
                                             skew_drift_coef=0.0),
        }
    elif venue == "amm":
        return {
            "fixed": lambda: FixedThresholdAMM(range_pct=0.10, rebalance_trigger=0.15),
            "bayesian": lambda: BayesianAMM(tight_range_pct=0.05,
                                            wide_range_pct=0.18,
                                            rebalance_threshold=0.15),
        }
    else:
        raise ValueError(venue)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--horizon", type=int, default=1000)
    p.add_argument("--venue", choices=["clob", "amm"], default="clob")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = RegimeConfig()
    factories = build_factories(args.venue)

    t0 = time.time()
    per_strategy = run_paired_backtest(
        factories, n_episodes=args.episodes, horizon=args.horizon,
        venue=args.venue, base_seed=args.seed, verbose=not args.quiet, cfg=cfg,
    )
    elapsed = time.time() - t0

    fixed_pnls = np.array([r.pnl for r in per_strategy["fixed"]])
    bayes_pnls = np.array([r.pnl for r in per_strategy["bayesian"]])

    summary = {
        "venue": args.venue,
        "episodes": args.episodes,
        "horizon": args.horizon,
        "elapsed_sec": elapsed,
        "fixed": summarize(per_strategy["fixed"]),
        "bayesian": summarize(per_strategy["bayesian"]),
        "head_to_head_bayes_vs_fixed": head_to_head(bayes_pnls, fixed_pnls),
    }

    out_json = os.path.join(args.out, f"summary_{args.venue}.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    out_npz = os.path.join(args.out, f"pnls_{args.venue}.npz")
    np.savez(out_npz, fixed=fixed_pnls, bayesian=bayes_pnls)

    print(f"\n=== {args.venue.upper()} — {args.episodes} episodes ===")
    print(f"elapsed: {elapsed:.1f}s")
    print(f"fixed    mean pnl: {summary['fixed']['mean_pnl']:.2f}  "
          f"sharpe: {summary['fixed']['mean_sharpe']:.3f}  "
          f"positive: {summary['fixed']['pct_positive']*100:.1f}%")
    print(f"bayesian mean pnl: {summary['bayesian']['mean_pnl']:.2f}  "
          f"sharpe: {summary['bayesian']['mean_sharpe']:.3f}  "
          f"positive: {summary['bayesian']['pct_positive']*100:.1f}%")
    h2h = summary["head_to_head_bayes_vs_fixed"]
    print(f"win rate bayes vs fixed: {h2h['win_rate']*100:.1f}%  "
          f"mean diff: {h2h['mean_diff']:.2f}")
    print(f"wrote {out_json}")
    print(f"wrote {out_npz}")


if __name__ == "__main__":
    main()
