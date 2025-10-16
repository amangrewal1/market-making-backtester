import numpy as np
from collections import deque
from .base import Strategy, Quote, AMMAction
from ..price_process import N_REGIMES, RegimeConfig, build_transition_matrix


class RegimeFilter:
    def __init__(self, cfg: RegimeConfig = None, flow_window: int = 20):
        self.cfg = cfg or RegimeConfig()
        self.P = build_transition_matrix(self.cfg)
        self.belief = np.ones(N_REGIMES) / N_REGIMES
        self.flow_buf = deque(maxlen=flow_window)
        self.arrival_buf = deque(maxlen=flow_window)
        self.flow_window = flow_window

    def reset(self):
        self.belief = np.ones(N_REGIMES) / N_REGIMES
        self.flow_buf.clear()
        self.arrival_buf.clear()

    def _log_emission(self, ret: float, flow_imb: float, arr_rate: float):
        logp = np.zeros(N_REGIMES)
        for s in range(N_REGIMES):
            mu = self.cfg.drifts[s]
            sig = max(self.cfg.vols[s], 1e-6)
            logp[s] += -0.5 * ((ret - mu) / sig) ** 2 - np.log(sig)

            tox = self.cfg.toxicity[s]
            expected_imb = 0.0
            if s == 1:
                expected_imb = +tox
            elif s == 2:
                expected_imb = -tox
            sig_imb = 0.4 + 0.3 * (1 - tox)
            logp[s] += -0.5 * ((flow_imb - expected_imb) / sig_imb) ** 2

            lam = max(self.cfg.arrival_rate[s], 1e-3)
            logp[s] += arr_rate * np.log(lam) - lam
        return logp

    def update(self, ret: float, signed_flow: float, n_arrivals: int):
        self.flow_buf.append(signed_flow)
        self.arrival_buf.append(n_arrivals)

        total_flow = sum(abs(f) for f in self.flow_buf) + 1e-9
        flow_imb = sum(self.flow_buf) / total_flow
        arr_rate = np.mean(self.arrival_buf) if self.arrival_buf else 1.0

        prior = self.P.T @ self.belief
        logl = self._log_emission(ret, flow_imb, arr_rate)
        logp = np.log(prior + 1e-12) + logl
        logp -= logp.max()
        post = np.exp(logp)
        self.belief = post / post.sum()
        return self.belief


class BayesianCLOB(Strategy):
    name = "bayesian_clob"

    def __init__(self, cfg: RegimeConfig = None, base_half_spread_bps: float = 13.0,
                 base_size: float = 1.0, inventory_limit: float = 20.0,
                 volatile_spread_mult: float = 4.0,
                 volatile_size_mult: float = 0.2,
                 skew_drift_coef: float = 250.0,
                 inventory_skew_coef: float = 0.6):
        self.cfg = cfg or RegimeConfig()
        self.filter = RegimeFilter(self.cfg)
        self.base_half = base_half_spread_bps * 1e-4
        self.base_size = base_size
        self.inv_lim = inventory_limit
        self.vol_spread_mult = volatile_spread_mult
        self.vol_size_mult = volatile_size_mult
        self.skew_drift_coef = skew_drift_coef
        self.inv_skew_coef = inventory_skew_coef

    def reset(self):
        self.filter.reset()

    def observe(self, mid: float, ret: float, fills, n_arrivals: int):
        signed_flow = sum(f.side * f.size for f in fills) if fills else 0.0
        self.filter.update(ret, signed_flow, n_arrivals)

    def quote(self, mid: float, inventory: float) -> Quote:
        b = self.filter.belief
        p_vol = float(b[3])
        expected_drift = float(np.dot(b, self.cfg.drifts))

        spread_mult = 1.0 + (self.vol_spread_mult - 1.0) * p_vol
        size_scale = 1.0 - (1.0 - self.vol_size_mult) * p_vol
        half_spread = self.base_half * spread_mult

        drift_skew = self.skew_drift_coef * expected_drift
        inv_frac = inventory / self.inv_lim
        inv_skew = self.inv_skew_coef * half_spread * inv_frac

        bid = mid * (1.0 - half_spread + drift_skew - inv_skew)
        ask = mid * (1.0 + half_spread + drift_skew - inv_skew)

        bid_size = self.base_size * size_scale * max(0.0, 1.0 - inv_frac)
        ask_size = self.base_size * size_scale * max(0.0, 1.0 + inv_frac)
        return Quote(bid=bid, ask=ask, bid_size=bid_size, ask_size=ask_size)


class BayesianAMM(Strategy):
    name = "bayesian_amm"

    def __init__(self, cfg: RegimeConfig = None,
                 tight_range_pct: float = 0.05,
                 wide_range_pct: float = 0.20,
                 rebalance_threshold: float = 0.15):
        self.cfg = cfg or RegimeConfig()
        self.filter = RegimeFilter(self.cfg)
        self.tight = tight_range_pct
        self.wide = wide_range_pct
        self.reb_thresh = rebalance_threshold

    def reset(self):
        self.filter.reset()

    def observe(self, mid: float, ret: float, fills, n_arrivals: int):
        signed_flow = sum(f.side * f.size for f in fills) if fills else 0.0
        self.filter.update(ret, signed_flow, n_arrivals)

    def amm_action(self, mid, current_range_low, current_range_high):
        b = self.filter.belief
        p_vol = float(b[3])
        p_trend = float(b[1] + b[2])
        r = self.tight + (self.wide - self.tight) * (0.8 * p_vol + 0.3 * p_trend)

        if current_range_low == 0.0:
            return AMMAction(range_low=mid * (1 - r), range_high=mid * (1 + r),
                             rebalance=True)
        if mid < current_range_low or mid > current_range_high:
            return AMMAction(range_low=mid * (1 - r), range_high=mid * (1 + r),
                             rebalance=True)

        center = 0.5 * (current_range_low + current_range_high)
        if abs(mid / center - 1.0) > self.reb_thresh:
            return AMMAction(range_low=mid * (1 - r), range_high=mid * (1 + r),
                             rebalance=True)
        return None
