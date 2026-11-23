
import logging

import pandas as pd

from src.utils.calendar import execution_map
from .factors import build_factor_inputs, neutralize
from .learning import MODELS, monthly_rank_ic, rank_normalize_labels, realized_labels, walk_forward
from .portfolio import benchmark_industry_weights, run_equity_portfolio
from .research import evaluate_factors

LOG = logging.getLogger(__name__)


def industry_deviations(books, weights, inputs):
    """Compare constrained targets and realized close exposures to the same decision benchmark."""
    references = {date: group for date, group in inputs.groupby("date", sort=False)}
    end_books = {name: data for name, data in weights.groupby("strategy", sort=False)}
    rows = []
    for (strategy, decision, execution), order in books.loc[books["industry_constrained"]].groupby(
        ["strategy", "decision_date", "execution_date"], sort=False,
    ):
        reference = references[decision]
        sectors = reference.set_index("code")["model_industry"]
        benchmark = benchmark_industry_weights(reference)
        target = order.set_index("code")["target_weight"]
        target = target.groupby(sectors).sum()
        close = end_books[strategy].loc[execution].drop(labels="strategy")
        actual_sectors = sectors.reindex(close.index).fillna("OUTSIDE")
        actual = close.groupby(actual_sectors).sum()
        for sector in benchmark.index.union(actual.loc[actual.abs().gt(1e-12)].index):
            bm, planned, filled = benchmark.get(sector, 0.0), target.get(sector, 0.0), actual.get(sector, 0.0)
            rows.append({
                "strategy": strategy, "decision_date": decision, "execution_date": execution, "industry": sector,
                "benchmark_weight": bm, "target_weight": planned, "abs_deviation": abs(planned - bm),
                "executed_weight": filled, "executed_abs_deviation": abs(filled - bm),
            })
    return pd.DataFrame(rows)


def equity_stage(raw, processed, outputs, config):
    cfg = config["equity"]
    benchmark = pd.read_parquet(processed / "csi300_close.parquet")["csi300_close"]
    memberships = pd.read_parquet(processed / "memberships.parquet")
    inputs, returns, tradable, prices = build_factor_inputs(raw, memberships, benchmark, config)
    LOG.info(
        "Industry coverage across constituent observations: PIT %.2f%%, latest-snapshot proxy %.2f%%, enabled=%s",
        100 * inputs["industry_pit"].ne("Unknown").mean(), 100 * inputs["industry_proxy"].ne("Unknown").mean(),
        cfg.get("use_static_industry_proxy", False),
    )
    cleaned, coverage = [], []
    for date, data in inputs.loc[inputs["decision_tradable"]].groupby("date", sort=True):
        x, missing = neutralize(data, cfg["factors"], cfg)
        cleaned.append(x)
        coverage.append(missing.rename(date))
    coverage = pd.DataFrame(coverage).rename_axis("date")
    count = int(coverage.gt(cfg["missing_warning_fraction"]).sum().sum())
    if count:
        LOG.warning("%s factor/month cells exceed the missing-data warning threshold", count)
    coverage.to_csv(outputs / "equity_factor_missingness.csv")

    panel = pd.concat(cleaned, ignore_index=True)
    decisions = pd.DatetimeIndex(panel["date"].unique()).sort_values()
    executions = execution_map(returns.index, decisions, cfg["execution_lag"])
    panel = panel.join(realized_labels(panel, prices, tradable, executions, benchmark))
    panel["label_cs_rank"] = rank_normalize_labels(panel)
    panel.to_parquet(processed / "equity_features.parquet", index=False)
    ic = monthly_rank_ic(panel, cfg["factors"])
    scores, train_log = walk_forward(panel, ic, cfg, config["analytics"]["seed"], oos_start=config["dates"]["start"])
    scores.to_parquet(processed / "equity_scores.parquet", index=False)
    train_log.to_csv(outputs / "equity_walk_forward_log.csv")
    ensemble_cols = train_log.columns.str.startswith(("weight_", "ensemble_"))
    train_log.loc[:, ensemble_cols].to_csv(outputs / "equity_ensemble_weights.csv")
    summary, groups = evaluate_factors(panel, ic, cfg, config["dates"]["start"])
    summary.to_csv(outputs / "equity_factor_ic_summary.csv", index=False)
    groups.to_csv(outputs / "equity_factor_groups.csv", index=False)
    pd.DataFrame({
        "eligible_members": panel.groupby("date").size(),
        "industry_known_members": panel["industry"].ne("Unknown").groupby(panel["date"]).sum(),
        "industry_pit_known_members": panel["industry_pit"].ne("Unknown").groupby(panel["date"]).sum(),
        "industry_proxy_known_members": panel["industry_proxy"].ne("Unknown").groupby(panel["date"]).sum(),
        "industry_proxy_used_members": panel["industry_proxy_used"].groupby(panel["date"]).sum(),
    }).to_csv(outputs / "equity_point_in_time_coverage.csv")

    results, weights, orders = [], [], []

    def run(name, model, **overrides):
        log, _, end, book = run_equity_portfolio(returns, tradable, scores, inputs, config, model, **overrides)
        results.append(log.assign(strategy=name))
        weights.append(end.assign(strategy=name))
        orders.append(book.assign(strategy=name))
        return log

    for model in MODELS:
        run("Equity_Alpha_Sleeve" if model == "Ensemble" else model, model)
    run("Equity_Top50", "Ensemble", top_n=cfg["top_n"])
    run("Equity_Execution_Delay", "Ensemble", lag=cfg["execution_lag"] + 1)
    for label, cycle in cfg["frequency_scenarios"].items():
        for names in cfg["holdings_scenarios"]:
            run(f"Equity_{label}_{names}", "Ensemble", frequency=cycle, top_n=names)
    for multiplier in config["analytics"]["cost_multipliers"]:
        if multiplier != 1:
            run(f"Equity_Cost_{multiplier:g}x", "Ensemble", cost_multiplier=multiplier)

    daily = pd.concat(results).reset_index()
    daily.to_parquet(outputs / "equity_daily.parquet", index=False)
    close_weights = pd.concat(weights)
    close_weights.reset_index().to_parquet(outputs / "equity_weights.parquet", index=False)
    books = pd.concat(orders, ignore_index=True)
    books.to_parquet(outputs / "equity_orders.parquet", index=False)
    deviations = industry_deviations(books, close_weights, inputs)
    if deviations[["abs_deviation", "executed_abs_deviation"]].max().max() > cfg["industry_band"] + 1e-4:
        raise RuntimeError("Historical target or execution-close industry deviation exceeds tolerance")
    deviations.to_csv(outputs / "equity_industry_deviations.csv", index=False)
    return daily
