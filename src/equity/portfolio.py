
import logging

import cvxpy as cp
import numpy as np
import pandas as pd

from src.backtest.portfolio import Portfolio, result_frames, stock_rates
from src.utils.calendar import execution_map
from src.utils.optimization import solve

LOG = logging.getLogger(__name__)


def optimize_equity(day, previous, strategy, cfg, sector_weights):
    score = day[strategy].to_numpy()
    current = previous.reindex(day["code"]).fillna(0).to_numpy()
    w = cp.Variable(len(day))
    base_constraints = [
        cp.sum(w) == 1, w >= 0, w <= cfg["max_position"],
        cp.sum_squares(w) <= 1 / cfg["min_effective_holdings"],
    ]
    sector_col = "model_industry" if "model_industry" in day else "industry"
    sectors = day[sector_col]
    for sector, target in sector_weights.items():
        indicator = sectors.eq(sector).to_numpy(dtype=float)
        base_constraints.append(cp.abs(indicator @ w - target) <= cfg["industry_band"])

    change = cp.norm1(w - current)
    legacy = max(0.0, previous.sum() - current.sum())
    turnover = (change + legacy) / 2
    utility = score @ w - cfg["optimizer_turnover_penalty"] * change
    budget = cfg["monthly_turnover_budget"]
    relaxed = False
    if previous.sum() > 1e-8:
        weights = solve(cp.Problem(cp.Maximize(utility), base_constraints + [turnover <= budget]), w, allow_infeasible=True)
        if weights is None:
            minimum = solve(cp.Problem(cp.Minimize(turnover), base_constraints), w)
            lowest_turnover = (np.abs(minimum - current).sum() + legacy) / 2
            LOG.info("Forced turnover of %.2f%% exceeds budget on %s", 100 * lowest_turnover, day["date"].iloc[0].date())
            weights = solve(cp.Problem(cp.Maximize(utility), base_constraints + [turnover <= lowest_turnover + 1e-6]), w)
            relaxed = True
    else:
        weights = solve(cp.Problem(cp.Maximize(utility), base_constraints), w)

    result = pd.Series(weights, index=day["code"])
    observed = result.groupby(sectors.to_numpy()).sum().reindex(sector_weights.index, fill_value=0)
    if (observed - sector_weights).abs().max() > cfg["industry_band"] + 1e-6:
        raise RuntimeError("Equity target violates the hard industry band")
    result.attrs["turnover_budget_relaxed"] = relaxed
    result.attrs["decision_turnover"] = float((np.abs(weights - current).sum() + legacy) / 2)
    return result


def benchmark_industry_weights(reference):
    sector_col = "model_industry" if "model_industry" in reference else "industry"
    market_caps = reference.set_index("code")["total_mv_100m"].astype(float)
    market_caps = market_caps.where(market_caps.gt(0))
    market_caps = market_caps.fillna(market_caps.median()).fillna(1)
    sector_weights = market_caps.groupby(reference.set_index("code")[sector_col]).sum()
    return sector_weights / sector_weights.sum()


def industry_fill_repair(assets, planned, candidates, reference, benchmark, cfg):
    """Repair only blocked-fill risk, using the decision basket and current executability."""
    sector_codes = reference.set_index("code")["model_industry"].reindex(assets).fillna("OUTSIDE")
    buckets = benchmark.index.union(pd.Index(["OUTSIDE"]))
    benchmark_vector = benchmark.reindex(buckets, fill_value=0).to_numpy()
    groups = buckets.get_indexer(sector_codes)
    permitted = assets.isin(candidates)
    grouping = np.equal.outer(np.arange(len(buckets)), groups).astype(float)
    outside = buckets.get_loc("OUTSIDE")
    band = cfg["industry_band"]

    def reconcile(close_weights, executable):
        actual = grouping @ close_weights[:-1]
        actual[outside] += close_weights[-1]
        if np.abs(actual - benchmark_vector).max() <= band + 1e-6:
            return None

        fixed = np.where(executable, 0.0, close_weights[:-1])
        live = permitted & executable
        matrix = grouping[:, live]
        fixed_exposure = grouping @ fixed
        w = cp.Variable(int(live.sum()))
        interior = max(0.0, band - 1e-5)
        constraints = [
            w >= 0, w <= cfg["max_position"], cp.sum(w) == 1 - fixed.sum(),
            cp.sum_squares(w) + (fixed @ fixed) <= 1 / cfg["min_effective_holdings"] + 1e-8,
            cp.abs(matrix @ w + fixed_exposure - benchmark_vector) <= interior,
        ]
        problem = cp.Problem(cp.Minimize(cp.sum_squares(w - planned[live])), constraints)
        problem.solve(solver=cp.CLARABEL)
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or w.value is None:
            raise RuntimeError("Blocked stock fills make the hard execution industry band infeasible")
        revised = fixed.copy()
        revised[live] = np.maximum(np.asarray(w.value).ravel(), 0)
        return revised / revised.sum()

    return reconcile


def run_equity_portfolio(returns, tradable, scores, memberships, config, strategy, *,
                         top_n=None, frequency=1, lag=None, cost_multiplier=1.0):
    cfg = config["equity"]
    start = pd.Timestamp(config["dates"]["start"])
    dates = returns.loc[start:config["dates"]["end"]].index
    position = returns.index.get_loc(dates[0])
    history_dates = returns.index[position - (lag or cfg["execution_lag"]):]
    decisions = pd.DatetimeIndex(scores["date"].unique()).sort_values()
    decisions = decisions[decisions >= history_dates[0]][::frequency]
    orders = execution_map(returns.index, decisions, lag or cfg["execution_lag"])
    strategy_days = {date: group for date, group in scores.groupby("date", sort=False) if date in orders.index}
    pit_groups = {date: group for date, group in memberships.groupby("date", sort=False) if date in orders.index}
    account = Portfolio(returns.columns)
    batches = {}
    rows, starts, ends, book = [], [], [], []

    def order_on(day):
        current = strategy_days[day]
        if top_n is not None:
            selections = current.nlargest(top_n, strategy)["code"]
            weights = pd.Series(1 / len(selections), index=selections)
        else:
            reference = pit_groups[day]
            sector = benchmark_industry_weights(reference)
            weights = optimize_equity(current, account.positions, strategy, cfg, sector)
        execution = orders.at[day]
        target = weights.reindex(returns.columns, fill_value=0).to_numpy()
        repair = None if top_n is not None else industry_fill_repair(
            returns.columns, target, current["code"], reference, sector, cfg,
        )
        batches[execution] = (target, repair)
        record = weights.loc[weights.gt(0)].rename("target_weight").reset_index()
        book.append(record.assign(
            decision_date=day, execution_date=execution, industry_constrained=top_n is None,
            turnover_budget_relaxed=weights.attrs.get("turnover_budget_relaxed", False),
            decision_turnover=weights.attrs.get("decision_turnover", np.nan),
        ))

    for decision in strategy_days:
        if decision < start:
            order_on(decision)
    ret_array = returns.loc[dates].to_numpy()
    trade_array = tradable.loc[dates].to_numpy()
    for i, day in enumerate(dates):
        target, repair = batches.pop(day, (None, None))
        stats, begin, close = account.step(
            ret_array[i], trade_array[i], target, stock_rates(day, config["costs"], cost_multiplier), repair=repair,
        )
        rows.append(stats)
        starts.append(begin)
        ends.append(close)
        if day in strategy_days:
            order_on(day)
    return (*result_frames(dates, returns.columns, rows, starts, ends), pd.concat(book, ignore_index=True))
