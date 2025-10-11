from dataclasses import dataclass
from typing import Optional


@dataclass
class Quote:
    bid: float
    ask: float
    bid_size: float
    ask_size: float


@dataclass
class AMMAction:
    range_low: float
    range_high: float
    rebalance: bool = False


class Strategy:
    name: str = "base"

    def reset(self):
        pass

    def observe(self, mid: float, ret: float, fills, n_arrivals: int):
        pass

    def quote(self, mid: float, inventory: float) -> Quote:
        raise NotImplementedError

    def amm_action(self, mid: float, current_range_low: float,
                   current_range_high: float) -> Optional[AMMAction]:
        return None
