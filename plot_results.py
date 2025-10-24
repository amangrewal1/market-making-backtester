import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--venue", choices=["clob", "amm"], default="clob")
    args = p.parse_args()

    data = np.load(os.path.join(args.results, f"pnls_{args.venue}.npz"))
    fixed = data["fixed"]
    bayes = data["bayesian"]
    diff = bayes - fixed

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].hist(fixed, bins=60, alpha=0.55, label="Fixed", color="tab:orange")
    axes[0].hist(bayes, bins=60, alpha=0.55, label="Bayesian", color="tab:blue")
    axes[0].axvline(0, color="k", lw=0.8)
    axes[0].set_title(f"PnL distribution ({args.venue.upper()})")
    axes[0].set_xlabel("PnL")
    axes[0].legend()

    axes[1].hist(diff, bins=60, color="tab:green", alpha=0.75)
    axes[1].axvline(0, color="k", lw=0.8)
    wr = (diff > 0).mean() * 100
    axes[1].set_title(f"Bayesian − Fixed  (win rate {wr:.1f}%)")
    axes[1].set_xlabel("PnL difference")

    cum_fixed = np.sort(fixed)
    cum_bayes = np.sort(bayes)
    q = np.linspace(0, 1, len(cum_fixed))
    axes[2].plot(q, cum_fixed, label="Fixed", color="tab:orange")
    axes[2].plot(q, cum_bayes, label="Bayesian", color="tab:blue")
    axes[2].set_title("PnL CDF (sorted)")
    axes[2].set_xlabel("quantile")
    axes[2].set_ylabel("PnL")
    axes[2].legend()

    plt.tight_layout()
    out_png = os.path.join(args.results, f"pnl_{args.venue}.png")
    plt.savefig(out_png, dpi=130)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
