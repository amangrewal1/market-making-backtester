import numpy as np
from dataclasses import dataclass


@dataclass
class Fill:
    side: int          # +1 = MM sold (trader bought at ask), -1 = MM bought
    size: float
    price: float
    informed: bool


@dataclass
class CLOBState:
    inventory: float = 0.0
    cash: float = 0.0
    n_trades: int = 0
    n_informed_fills: int = 0
    realized_pnl: float = 0.0

    def mark_to_market(self, mid: float) -> float:
        return self.cash + self.inventory * mid


class CLOB:
    def __init__(self, tick: float = 1e-4, inventory_cap: float = 50.0,
                 informed_skill: float = 0.75,
                 uninformed_elasticity: float = 12.0,
                 rng: np.random.Generator = None):
        self.tick = tick
        self.inventory_cap = inventory_cap
        self.skill = informed_skill
        self.elasticity = uninformed_elasticity
        self.rng = rng or np.random.default_rng()

    def step(self, state: CLOBState, mid: float, next_mid: float,
             bid: float, ask: float, bid_size: float, ask_size: float,
             n_arrivals: int, toxicity: float):
        fills = []
        true_dir = np.sign(next_mid - mid)
        for _ in range(n_arrivals):
            informed = self.rng.random() < toxicity

            if informed and self.rng.random() < self.skill:
                if true_dir > 0 and next_mid > ask:
                    side = +1
                elif true_dir < 0 and next_mid < bid:
                    side = -1
                else:
                    continue
            else:
                side = +1 if self.rng.random() < 0.5 else -1
                quote = ask if side == +1 else bid
                spread_cost = abs(quote - mid) / mid
                if self.rng.random() < 1.0 - np.exp(-self.elasticity * spread_cost):
                    continue

            if side == +1:
                size = min(ask_size, max(0.0, self.inventory_cap + state.inventory))
                if size <= 0:
                    continue
                state.cash += size * ask
                state.inventory -= size
                fills.append(Fill(+1, size, ask, informed))
            else:
                size = min(bid_size, max(0.0, self.inventory_cap - state.inventory))
                if size <= 0:
                    continue
                state.cash -= size * bid
                state.inventory += size
                fills.append(Fill(-1, size, bid, informed))

            state.n_trades += 1
            state.n_informed_fills += int(informed)

        return fills
