
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CASH = "_cash"


def _rebalance(pre, desired, executable, rates):
    """Compute self-financing close fills at the supplied execution prices."""
    fixed = pre[:-1][~executable].sum()
    target_live = desired[executable]
    cost = 0.0
    for _ in range(20):
        budget = max(0.0, 1 - cost - fixed)
        live = target_live * (1 - cost)
        if live.sum() > budget:
            live *= budget / live.sum()
        delta = live - pre[:-1][executable]
        buy = np.maximum(delta, 0).sum()
        sell = -np.minimum(delta, 0).sum()
        new_cost = buy * rates[0] + sell * rates[1]
        if abs(new_cost - cost) <= 1e-14:
            post = pre.copy()
            post[:-1][executable] = live
            post[-1] = max(0.0, 1 - new_cost - post[:-1].sum())
            return post, buy, sell, new_cost
        cost = new_cost
    raise RuntimeError("Self-financing transaction cost calculation did not converge")


@dataclass
class Portfolio:
    """Close-to-close marks, then close execution using orders formed on prior sessions."""
    assets: pd.Index
    weights: np.ndarray = field(init=False)
    nav: float = 1.0

    def __post_init__(self):
        self.weights = np.zeros(len(self.assets) + 1)
        self.weights[-1] = 1.0

    @property
    def positions(self) -> pd.Series:
        return pd.Series(self.weights[:-1], index=self.assets)

    def step(self, returns, tradable, target=None, rates=(0.0, 0.0), *, repair=None):
        start = self.weights.copy()
        observed = np.asarray(returns, dtype=float)
        executable = np.asarray(tradable, dtype=bool) & np.isfinite(observed)
        marks = np.append(np.where(np.isfinite(observed), observed, 0.0), 0.0)
        if (marks < -1.0).any():
            raise ValueError("Daily returns below -100%")
        marked = start * (1.0 + marks)
        wealth = marked.sum()
        if wealth <= 0:
            raise ValueError("Portfolio insolvent")
        pre = marked / wealth
        post = pre.copy()
        buy = sell = cost = blocked = 0.0
        repaired = False
        if target is not None:
            desired = np.asarray(target, dtype=float)
            if desired.shape != executable.shape or not np.isfinite(desired).all():
                raise ValueError("Invalid order vector")
            if desired.min() < -1e-10 or desired.sum() > 1 + 1e-9:
                raise ValueError("Expected a long-only target with at most 100% invested")
            desired = desired.clip(0)
            blocked = desired[~executable].sum()
            post, buy, sell, cost = _rebalance(pre, desired, executable, rates)
            if repair is not None:
                # Frozen close weights depend slightly on fees. Recheck the actual ledger after repair.
                for _ in range(4):
                    revised = repair(post / (1 - cost), executable)
                    if revised is None:
                        break
                    post, buy, sell, cost = _rebalance(pre, revised, executable, rates)
                    repaired = True
                else:
                    if repair(post / (1 - cost), executable) is not None:
                        raise RuntimeError("Execution constraint repair did not converge")
        self.weights = post / (1 - cost)
        net = wealth * (1 - cost) - 1
        self.nav *= 1 + net
        stats = {
            "net_return": net,
            "gross_return": wealth - 1,
            "cost": wealth * cost,
            "cost_on_trade_notional": cost,
            "turnover": (buy + sell) / 2,
            "buy_turnover": buy,
            "sell_turnover": sell,
            "stale_weight": start[:-1][~np.isfinite(observed)].sum(),
            "blocked_target_weight": blocked,
            "execution_repaired": repaired,
            "nav": self.nav,
        }
        return stats, start, self.weights.copy()


def stock_rates(day: pd.Timestamp, costs: dict, multiplier=1.0) -> tuple[float, float]:
    buy = costs["stock_commission_bps"] + costs["stock_slippage_bps"]
    tax = costs["stock_sell_tax_old_bps"] if day < pd.Timestamp(costs["stock_sell_tax_change"]) else costs["stock_sell_tax_new_bps"]
    return buy * multiplier / 10000, (buy + tax) * multiplier / 10000


def etf_rates(costs: dict, multiplier=1.0) -> tuple[float, float]:
    rate = costs["etf_bps"] * multiplier / 10000
    return rate, rate


def result_frames(dates, assets, rows, starts, ends):
    columns = [*assets, CASH]
    return (
        pd.DataFrame(rows, index=dates).rename_axis("date"),
        pd.DataFrame(starts, index=dates, columns=columns).rename_axis("date"),
        pd.DataFrame(ends, index=dates, columns=columns).rename_axis("date"),
    )
