import numpy as np
from .price_process import RegimeSwitchingMarket, RegimeConfig
from .clob import CLOB, CLOBState
from .amm import AMM, AMMState
from .strategies import Strategy
from .metrics import EpisodeResult, compute_drawdown, compute_sharpe


def run_clob_episode(strategy: Strategy, mids, returns, regimes, arrivals,
                     cfg: RegimeConfig, seed: int) -> EpisodeResult:
    rng = np.random.default_rng(seed)
    clob = CLOB(rng=rng)
    state = CLOBState()
    strategy.reset()

    T = len(mids)
    equity = np.zeros(T)
    equity[0] = 0.0

    for t in range(T - 1):
        mid = mids[t]
        quote = strategy.quote(mid, state.inventory)
        next_mid = mids[t + 1]
        tox = cfg.toxicity[regimes[t + 1]]
        fills = clob.step(state, mid, next_mid,
                          quote.bid, quote.ask, quote.bid_size, quote.ask_size,
                          arrivals[t + 1], tox)
        strategy.observe(next_mid, returns[t + 1], fills, arrivals[t + 1])
        equity[t + 1] = state.mark_to_market(next_mid)

    final_mid = mids[-1]
    pnl = state.mark_to_market(final_mid) - state.inventory * final_mid + \
          state.inventory * final_mid
    pnl = state.mark_to_market(final_mid)

    return EpisodeResult(
        strategy=strategy.name,
        pnl=pnl,
        final_inventory=state.inventory,
        n_trades=state.n_trades,
        n_informed_fills=state.n_informed_fills,
        max_drawdown=compute_drawdown(equity),
        sharpe=compute_sharpe(equity),
        equity_curve=equity,
    )


def run_amm_episode(strategy: Strategy, mids, returns, regimes, arrivals,
                    cfg: RegimeConfig, seed: int,
                    initial_capital: float = 10_000.0,
                    external_flow_mean: float = 50.0,
                    rebalance_cost_bps: float = 8.0) -> EpisodeResult:
    rng = np.random.default_rng(seed)
    amm = AMM(rng=rng)
    state = AMMState(capital=initial_capital, anchor_price=mids[0],
                     range_low=0.0, range_high=np.inf, active=False)
    strategy.reset()
    rebalance_cost_total = 0.0

    T = len(mids)
    equity = np.zeros(T)
    n_rebalances = 0
    n_trades = 0

    for t in range(T - 1):
        mid = mids[t]
        action = strategy.amm_action(mid, state.range_low, state.range_high)
        if action is not None and action.rebalance:
            current_val = amm.liquidity_value(state, mid) if state.active \
                          else initial_capital
            cost = current_val * rebalance_cost_bps * 1e-4 if state.active else 0.0
            rebalance_cost_total += cost
            state = AMMState.initialize(mid, current_val - cost,
                                        action.range_low, action.range_high)
            n_rebalances += 1

        next_mid = mids[t + 1]
        for _ in range(arrivals[t + 1]):
            flow_size = rng.exponential(external_flow_mean)
            amm.step(state, mid, next_mid, flow_size, cfg.toxicity[regimes[t + 1]])
            n_trades += 1

        strategy.observe(next_mid, returns[t + 1], [], arrivals[t + 1])
        equity[t + 1] = amm.impermanent_pnl(state, next_mid) if state.active \
                        else 0.0

    final_mid = mids[-1]
    pnl = amm.impermanent_pnl(state, final_mid) if state.active else 0.0
    pnl -= rebalance_cost_total

    return EpisodeResult(
        strategy=strategy.name,
        pnl=pnl,
        final_inventory=state.capital if state.active else 0.0,
        n_trades=n_trades,
        n_informed_fills=n_rebalances,
        max_drawdown=compute_drawdown(equity),
        sharpe=compute_sharpe(equity),
        equity_curve=equity,
    )


def run_backtest(strategy_factory, n_episodes: int, horizon: int,
                 venue: str = "clob", base_seed: int = 42,
                 verbose: bool = False, cfg: RegimeConfig = None):
    cfg = cfg or RegimeConfig()
    results = []

    iterator = range(n_episodes)
    if verbose:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc=f"{venue}:{strategy_factory().name}")
        except ImportError:
            pass

    for i in iterator:
        seed = base_seed + i
        rng = np.random.default_rng(seed)
        market = RegimeSwitchingMarket(cfg=cfg, rng=rng)
        regime0 = int(rng.integers(4))
        mids, returns, regimes, arrivals = market.simulate(horizon, regime0=regime0)

        strategy = strategy_factory()
        if venue == "clob":
            r = run_clob_episode(strategy, mids, returns, regimes, arrivals, cfg, seed)
        elif venue == "amm":
            r = run_amm_episode(strategy, mids, returns, regimes, arrivals, cfg, seed)
        else:
            raise ValueError(f"unknown venue: {venue}")
        results.append(r)
    return results


def run_paired_backtest(strategy_factories: dict, n_episodes: int, horizon: int,
                        venue: str = "clob", base_seed: int = 42,
                        verbose: bool = False, cfg: RegimeConfig = None):
    cfg = cfg or RegimeConfig()
    per_strategy = {name: [] for name in strategy_factories}

    iterator = range(n_episodes)
    if verbose:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc=f"paired:{venue}")
        except ImportError:
            pass

    for i in iterator:
        seed = base_seed + i
        market_rng = np.random.default_rng(seed)
        market = RegimeSwitchingMarket(cfg=cfg, rng=market_rng)
        regime0 = int(market_rng.integers(4))
        mids, returns, regimes, arrivals = market.simulate(horizon, regime0=regime0)

        for name, factory in strategy_factories.items():
            strategy = factory()
            if venue == "clob":
                r = run_clob_episode(strategy, mids, returns, regimes, arrivals,
                                     cfg, seed + 10_000_000)
            elif venue == "amm":
                r = run_amm_episode(strategy, mids, returns, regimes, arrivals,
                                    cfg, seed + 10_000_000)
            else:
                raise ValueError(f"unknown venue: {venue}")
            per_strategy[name].append(r)
    return per_strategy
