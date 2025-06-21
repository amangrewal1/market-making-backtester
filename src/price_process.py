import numpy as np
from dataclasses import dataclass, field


REGIMES = ["calm", "trend_up", "trend_down", "volatile"]
N_REGIMES = len(REGIMES)


@dataclass
class RegimeConfig:
    drifts: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 3e-4, -3e-4, 0.0])
    )
    vols: np.ndarray = field(
        default_factory=lambda: np.array([2e-3, 4e-3, 4e-3, 1.2e-2])
    )
    # prob that an arriving trade is informed in each regime
    toxicity: np.ndarray = field(
        default_factory=lambda: np.array([0.05, 0.22, 0.22, 0.40])
    )
    # Poisson arrival rate of trades per step
    arrival_rate: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.3, 1.3, 2.0])
    )
    stickiness: float = 0.97  # diagonal prob of transition matrix


def build_transition_matrix(cfg: RegimeConfig) -> np.ndarray:
    P = np.eye(N_REGIMES) * cfg.stickiness
    off = (1.0 - cfg.stickiness) / (N_REGIMES - 1)
    P += (1.0 - np.eye(N_REGIMES)) * off
    return P


class RegimeSwitchingMarket:
    def __init__(self, cfg: RegimeConfig = None, rng: np.random.Generator = None):
        self.cfg = cfg or RegimeConfig()
        self.P = build_transition_matrix(self.cfg)
        self.rng = rng or np.random.default_rng()

    def simulate(self, T: int, s0: float = 100.0, regime0: int = 0):
        regimes = np.zeros(T, dtype=np.int32)
        mids = np.zeros(T)
        returns = np.zeros(T)
        arrivals = np.zeros(T, dtype=np.int32)
        mids[0] = s0
        regimes[0] = regime0
        for t in range(1, T):
            regimes[t] = self.rng.choice(N_REGIMES, p=self.P[regimes[t - 1]])
            mu = self.cfg.drifts[regimes[t]]
            sig = self.cfg.vols[regimes[t]]
            r = mu + sig * self.rng.standard_normal()
            returns[t] = r
            mids[t] = mids[t - 1] * np.exp(r)
            arrivals[t] = self.rng.poisson(self.cfg.arrival_rate[regimes[t]])
        return mids, returns, regimes, arrivals
