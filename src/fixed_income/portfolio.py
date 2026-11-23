
import cvxpy as cp
import numpy as np
import pandas as pd

from src.backtest.portfolio import Portfolio, etf_rates, result_frames
from src.utils.calendar import month_end_decisions
from src.utils.optimization import solve


def optimize_bond(signal, eligible, previous, cfg, *, allow_credit=True):
    assets = pd.Index(cfg["durations"])
    duration = np.array(list(cfg["durations"].values()))
    gate = bool(signal["credit_gate"])
    caps = pd.Series(cfg["caps"], dtype=float).reindex(assets).where(eligible, 0)
    credit_cap = cfg["credit_gate_cap"] if gate else cfg["credit_cap"]
    caps.loc["credit"] = min(caps.loc["credit"], credit_cap) if allow_credit else 0
    if gate and eligible["cash"]:
        caps.loc["cash"] = cfg["gated_cash_cap"]
    credit_score = signal["credit_score"] if pd.notna(signal["credit_score"]) else 0.5
    scores = pd.Series({
        "cash": 0.0, "cgb5": 0.15, "cgb10": 0.10, "cgb30": 0.04,
        "credit": 2 * (credit_score - 0.46) - 1.0 * gate,
        "short_cp": 0.2 * (credit_score - 0.34),
    })
    for asset, tenor in (("cgb5", 5), ("cgb10", 10), ("cgb30", 30)):
        rank = signal[f"carry_roll_{tenor}y_pctl"]
        scores.loc[asset] += 0.4 * (rank - 0.5) if pd.notna(rank) else 0

    w = cp.Variable(len(assets))
    realized_duration = duration @ w
    target = float(signal["duration_target"])
    duration_ceiling = cfg["gated_duration_ceiling"] if gate else cfg["offensive_duration"][1]
    target = np.clip(target, cfg["defensive_duration"][0], duration_ceiling)
    duration_lower = max(cfg["defensive_duration"][0], target - cfg["duration_tolerance_years"])
    duration_upper = min(duration_ceiling, target + cfg["duration_tolerance_years"])
    credit_indices = [assets.get_loc(asset) for asset in ("credit", "short_cp")]
    aggregate_credit_cap = cfg["credit_cap"] if allow_credit else 0
    core = [w >= 0, cp.sum(w) == 1, w <= caps.to_numpy(), cp.sum(w[credit_indices]) <= aggregate_credit_cap]
    constraints = core + [realized_duration >= duration_lower, realized_duration <= duration_upper]
    utility = 0.01 * (scores.reindex(assets).to_numpy() @ w)
    utility -= cfg["duration_penalty"] * cp.square(realized_duration - target)
    utility -= cfg["turnover_penalty"] * cp.norm1(w - previous.reindex(assets).to_numpy())
    weights = solve(cp.Problem(cp.Maximize(utility), constraints), w, allow_infeasible=True)
    if weights is None:
        # Allow a target-band miss when eligible holdings are scarce, keeping the tail ceiling hard.
        hard = core + [realized_duration <= duration_ceiling]
        weights = solve(cp.Problem(cp.Maximize(utility), hard), w)
    return pd.Series(weights, index=assets)


def simulate_bond(returns, tradable, eligible, signals, config, *, static=None, allow_credit=True, cost_multiplier=1.0):
    cfg = config["bond"]
    dates = returns.loc[config["dates"]["start"]:config["dates"]["end"]].index
    account = Portfolio(returns.columns)
    calendar = returns.index
    initial = calendar[calendar.get_loc(dates[0]) - 1]
    monthly = set(month_end_decisions(calendar))
    rows, starts, ends, order_rows = [], [], [], []

    def get_order(date):
        if static is not None:
            target = pd.Series(static, dtype=float).reindex(returns.columns, fill_value=0)
        else:
            target = optimize_bond(signals.loc[date], eligible.loc[date], account.positions, cfg, allow_credit=allow_credit)
        execution_date = calendar[calendar.get_loc(date) + 1]
        order_rows.append({
            "decision_date": date, "execution_date": execution_date,
            "credit_gate": bool(signals.at[date, "credit_gate"]),
            "target_duration": float(target @ pd.Series(cfg["durations"])),
            **target.to_dict(),
        })
        return target.to_numpy()

    order = get_order(initial)
    previous_gate = bool(signals.at[initial, "credit_gate"])
    rates = etf_rates(config["costs"], cost_multiplier)
    duration = pd.Series(cfg["durations"]).reindex(returns.columns).to_numpy()
    for day in dates:
        stats, begin, end = account.step(returns.loc[day], tradable.loc[day], order, rates)
        gate = bool(signals.at[day, "credit_gate"])
        stats["modified_duration"] = end[:-1] @ duration
        stats["lgfv_weight"] = end[returns.columns.get_loc("credit")]
        stats["short_cp_weight"] = end[returns.columns.get_loc("short_cp")]
        stats["credit_weight"] = stats["lgfv_weight"] + stats["short_cp_weight"]
        stats["cash_weight"] = end[returns.columns.get_loc("cash")] + end[-1]
        stats["credit_gate"] = gate
        rows.append(stats)
        starts.append(begin)
        ends.append(end)
        order = None
        if day < dates[-1] and (day in monthly or (static is None and gate != previous_gate)):
            order = get_order(day)
        previous_gate = gate

    return (*result_frames(dates, returns.columns, rows, starts, ends), pd.DataFrame(order_rows))
