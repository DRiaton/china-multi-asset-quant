
import numpy as np
import pandas as pd

from src.backtest.portfolio import CASH
from src.allocation.portfolio import classify_contributions

CLASSES = {
    "equity": ("equity_etf", "equity_active"),
    "bond": ("bond_passive", "bond_active"),
    "cash": ("cash", CASH),
}


def bhb_daily(strategy, benchmark, strategy_weights, benchmark_weights, leg_gross):
    p_weight, p_contrib = classify_contributions(strategy_weights, leg_gross, CLASSES)
    b_weight, b_contrib = classify_contributions(benchmark_weights, leg_gross, CLASSES)
    # An unheld benchmark sleeve keeps its ETF reference return. The internal cash wallet
    # earns zero when actually held, independently of the portfolio's cash-ETF opportunity.
    reference = leg_gross[["equity_etf", "bond_passive", "cash"]].rename(
        columns={"equity_etf": "equity", "bond_passive": "bond"},
    ).fillna(0)
    b_return = b_contrib.div(b_weight.where(b_weight.ne(0))).combine_first(reference)
    p_return = p_contrib.div(p_weight.where(p_weight.ne(0))).combine_first(b_return)
    diff_weight = p_weight - b_weight
    diff_return = p_return - b_return
    selection = b_weight * diff_return
    result = pd.DataFrame({
        "asset_allocation": (diff_weight * b_return).sum(axis=1),
        "equity_selection": selection["equity"],
        "fixed_income_selection": selection["bond"],
        "cash_implementation": selection["cash"],
        "interaction": (diff_weight * diff_return).sum(axis=1),
        "transaction_cost_drag": benchmark["cost"] - strategy["cost"],
    })
    active = strategy["net_return"] - benchmark["net_return"]
    if not np.allclose(result.sum(axis=1), active, atol=1e-12):
        raise ValueError("Daily Brinson effects do not reconcile")
    return result


def log_slope(strategy, benchmark):
    diff = np.asarray(strategy) - np.asarray(benchmark)
    log_diff = np.log1p(strategy) - np.log1p(benchmark)
    equal = np.abs(diff) < 1e-12
    return np.divide(log_diff, diff, out=np.asarray(1 / (1 + benchmark)).copy(), where=~equal)


def carino_link(effects, strategy_returns, benchmark_returns, sessions):
    """Carino-link daily active effects to cumulative and annualized arithmetic excess."""
    factor = log_slope(strategy_returns, benchmark_returns)
    numerator = effects.mul(factor, axis=0).sum()
    strategy_log, benchmark_log = np.log1p(strategy_returns).sum(), np.log1p(benchmark_returns).sum()
    cumulative = np.expm1([strategy_log, benchmark_log])
    multiplier = sessions / len(effects)
    annual = np.expm1(np.array([strategy_log, benchmark_log]) * multiplier)
    rows = pd.DataFrame({
        "annualized": numerator * multiplier / log_slope(*annual),
        "cumulative": numerator / log_slope(*cumulative),
    }).rename_axis("effect")
    active = pd.Series({"annualized": annual[0] - annual[1], "cumulative": cumulative[0] - cumulative[1]})
    residual = active - rows.sum()
    if abs(residual["annualized"]) >= 0.0005:
        raise ValueError("Annualized Brinson attribution residual exceeds 5 bps")
    rows.loc["active_total"] = active
    rows.loc["reconciliation_residual"] = residual
    return rows
