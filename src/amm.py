import numpy as np
from dataclasses import dataclass


@dataclass
class AMMState:
    capital: float
    anchor_price: float
    range_low: float
    range_high: float
    fees_earned: float = 0.0
    il_realized: float = 0.0
    active: bool = True

    @classmethod
    def initialize(cls, price: float, usd: float,
                   range_low: float, range_high: float):
        return cls(capital=usd, anchor_price=price,
                   range_low=range_low, range_high=range_high, active=True)

    def width_pct(self) -> float:
        return (self.range_high - self.range_low) / self.anchor_price


class AMM:
    def __init__(self, fee: float = 3e-3, rng: np.random.Generator = None):
        self.fee = fee
        self.rng = rng or np.random.default_rng()

    def _in_range(self, state: AMMState, mid: float) -> bool:
        return state.range_low <= mid <= state.range_high

    def step(self, state: AMMState, mid: float, next_mid: float,
             external_flow: float, toxicity: float):
        if not state.active or not self._in_range(state, mid):
            return 0.0

        informed = self.rng.random() < toxicity
        if informed:
            signal = np.sign(next_mid - mid)
        else:
            signal = np.sign(self.rng.standard_normal())
        flow = signal * abs(external_flow)

        concentration = 0.20 / max(state.width_pct(), 0.01)
        fee_earn = abs(flow) * self.fee * concentration
        state.fees_earned += fee_earn

        if informed:
            il_hit = abs(next_mid - mid) / mid * abs(flow) * 0.3 * concentration
            state.il_realized += il_hit
        return flow

    def liquidity_value(self, state: AMMState, mid: float) -> float:
        if not state.active:
            return 0.0
        price_change = (mid - state.anchor_price) / state.anchor_price
        il_passive = -state.capital * 0.25 * (price_change ** 2) / \
                     max(state.width_pct(), 0.01)
        return state.capital + state.fees_earned - state.il_realized + il_passive

    def impermanent_pnl(self, state: AMMState, mid: float) -> float:
        if not state.active:
            return 0.0
        return self.liquidity_value(state, mid) - state.capital
