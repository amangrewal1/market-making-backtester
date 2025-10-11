from .base import Strategy, Quote, AMMAction


class FixedThresholdCLOB(Strategy):
    name = "fixed_threshold_clob"

    def __init__(self, half_spread_bps: float = 8.0, base_size: float = 1.0,
                 inventory_limit: float = 20.0, skew_coef: float = 0.5):
        self.half_spread = half_spread_bps * 1e-4
        self.base_size = base_size
        self.inv_lim = inventory_limit
        self.skew_coef = skew_coef

    def quote(self, mid: float, inventory: float) -> Quote:
        inv_frac = inventory / self.inv_lim
        skew = self.skew_coef * self.half_spread * inv_frac
        bid = mid * (1.0 - self.half_spread - skew)
        ask = mid * (1.0 + self.half_spread - skew)

        bid_size = self.base_size * max(0.0, 1.0 - inv_frac)
        ask_size = self.base_size * max(0.0, 1.0 + inv_frac)
        return Quote(bid=bid, ask=ask, bid_size=bid_size, ask_size=ask_size)


class FixedThresholdAMM(Strategy):
    name = "fixed_threshold_amm"

    def __init__(self, range_pct: float = 0.10, rebalance_trigger: float = 0.20):
        self.range_pct = range_pct
        self.rebalance_trigger = rebalance_trigger

    def amm_action(self, mid, current_range_low, current_range_high):
        if current_range_low == 0.0 or mid < current_range_low or mid > current_range_high:
            return AMMAction(range_low=mid * (1 - self.range_pct),
                             range_high=mid * (1 + self.range_pct),
                             rebalance=True)
        mid_range = 0.5 * (current_range_low + current_range_high)
        if abs(mid / mid_range - 1.0) > self.rebalance_trigger:
            return AMMAction(range_low=mid * (1 - self.range_pct),
                             range_high=mid * (1 + self.range_pct),
                             rebalance=True)
        return None
