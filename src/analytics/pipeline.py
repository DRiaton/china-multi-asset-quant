
import hashlib
import json

import numpy as np
import pandas as pd

from src.allocation.pipeline import load_allocation_market
from src.allocation.portfolio import ABLATION, CORE_STRATEGIES
from src.fixed_income.pipeline import load_market_assets
from .attribution import bhb_daily, carino_link
from .performance import expected_best_daily_sharpe, metrics, sharpe_ratio, stress_table

FLAGSHIP = "Macro_BL_FixedIncomePlus"
BENCHMARK = "Static_20_80"


def split_daily(path):
    frame = pd.read_parquet(path)
    return {name: group.drop(columns="strategy").set_index("date").sort_index() for name, group in frame.groupby("strategy", sort=False)}


def score_group(strategies, candidates, cash, config, path, benchmark=None):
    sessions = config["dates"]["periods_per_year"]
    sharpes = [
        sharpe_ratio(daily["net_return"] - cash.reindex(daily.index).fillna(0), sessions) / np.sqrt(sessions)
        for name, daily in strategies.items() if name in candidates
    ]
    null_best = expected_best_daily_sharpe(sharpes)
    table = pd.DataFrame([
        {"strategy": name, "dsr_trials": len(sharpes), **metrics(daily, cash, config, null_best, benchmark)}
        for name, daily in strategies.items()
    ]).set_index("strategy")
    table.to_csv(path)
    return table


def sensitivity_table(outputs, allocation, equity, bond, config):
    params = pd.read_csv(outputs / "bl_grid_parameters.csv").set_index("strategy")
    params.join(allocation).to_csv(outputs / "bl_grid_sensitivity.csv")
    rows = []
    for layer, table, base, prefix in (
        ("allocation", allocation, FLAGSHIP, "Allocation_Cost"),
        ("equity", equity, "Equity_Alpha_Sleeve", "Equity_Cost"),
        ("bond", bond, "Bond_Active_Sleeve", "Bond_Active_Cost"),
    ):
        for multiplier in config["analytics"]["cost_multipliers"]:
            name = base if multiplier == 1 else f"{prefix}_{multiplier:g}x"
            row = table.loc[name]
            rows.append({
                "layer": layer, "cost_multiplier": multiplier, "strategy": name,
                **row[["cagr", "sharpe", "max_drawdown", "annualized_cost_bps"]].to_dict(),
            })
    pd.DataFrame(rows).to_csv(outputs / "cost_sensitivity.csv", index=False)
    equity_cases = [
        {"strategy": f"Equity_{frequency}_{size}", "frequency": frequency, "holding_count": size}
        for frequency in config["equity"]["frequency_scenarios"]
        for size in config["equity"]["holdings_scenarios"]
    ]
    pd.DataFrame(equity_cases).set_index("strategy").join(equity).to_csv(outputs / "equity_sensitivity.csv")


def flagship_exposure_summary(outputs, config):
    weights = split_daily(outputs / "allocation_weights_end.parquet")[FLAGSHIP]
    stock_weights = split_daily(outputs / "equity_weights.parquet")["Equity_Alpha_Sleeve"]
    bond_weights = split_daily(outputs / "bond_weights.parquet")["Bond_Active_Sleeve"]
    top = pd.DataFrame({
        "equity": weights["equity_active"],
        "bond": weights["bond_active"],
        "cash": weights[["cash", "_cash"]].sum(axis=1),
    })
    equity_cash = top["equity"] * stock_weights["_cash"]
    debt_cash = top["bond"] * bond_weights[["cash", "_cash"]].sum(axis=1)
    lookthrough = pd.DataFrame({
        "equity": top["equity"] - equity_cash,
        "bond": top["bond"] - debt_cash,
        "cash": top["cash"] + equity_cash + debt_cash,
    })
    rows = []
    for layer, exposures in (("top_level", top), ("lookthrough", lookthrough)):
        if exposures.isna().any().any() or not np.allclose(exposures.sum(axis=1), 1, atol=1e-10):
            raise RuntimeError(f"Flagship {layer} exposures do not reconcile")
        mean = exposures.mean()
        if mean["cash"] >= config["allocation"]["flagship_max_mean_cash"]:
            raise RuntimeError(f"Flagship {layer} mean cash exceeds its acceptance threshold")
        rows.extend({"layer": layer, "asset": asset, "mean_weight": weight} for asset, weight in mean.items())
    table = pd.DataFrame(rows)
    table.to_csv(outputs / "flagship_exposure_summary.csv", index=False)
    return table


def analytics_stage(processed, outputs, config, config_path):
    etfs, _, _ = load_market_assets(processed, config["data"]["etf_codes"])
    cash = etfs["cash"]
    annualization = config["dates"]["periods_per_year"]
    allocation = split_daily(outputs / "allocation_daily.parquet")
    equity = split_daily(outputs / "equity_daily.parquet")
    bond = split_daily(outputs / "bond_daily.parquet")
    close = pd.read_parquet(processed / "csi300_close.parquet")["csi300_close"]
    index = close.loc[config["dates"]["start"]:config["dates"]["end"]].pct_change().fillna(0)
    allocation_names = {name for name in allocation if not name.startswith("Allocation_Cost")}
    ap = score_group(allocation, allocation_names, cash, config, outputs / "performance_summary.csv")
    ep = score_group(equity, equity.keys(), cash, config, outputs / "equity_performance.csv", benchmark=index)
    bp = score_group(bond, {n for n in bond if "Cost" not in n}, cash, config, outputs / "bond_performance.csv")
    stress = stress_table({**allocation, **equity, **bond}, config["analytics"]["stress_windows"])
    stress.to_csv(outputs / "stress_tests.csv", index=False)
    sensitivity_table(outputs, ap, ep, bp, config)

    ablation = ap.loc[list(ABLATION)].reset_index()
    ablation.insert(0, "step", np.arange(len(ablation)))
    ablation["delta_cagr"] = ablation["cagr"].diff()
    ablation["delta_sharpe"] = ablation["sharpe"].diff()
    ablation.to_csv(outputs / "ablation_summary.csv", index=False)

    market = load_allocation_market(processed, outputs, config)
    weights = split_daily(outputs / "allocation_weights_start.parquet")
    result, baseline = allocation[FLAGSHIP], allocation[BENCHMARK]
    daily = bhb_daily(result, baseline, weights[FLAGSHIP], weights[BENCHMARK], market["gross"].loc[result.index])
    daily.to_csv(outputs / "brinson_daily.csv")
    attribution = carino_link(daily, result["net_return"], baseline["net_return"], annualization)
    attribution.to_csv(outputs / "brinson_attribution.csv")

    cgb_pca = pd.read_csv(outputs / "bond_pca_loadings.csv")
    dns = pd.read_csv(outputs / "bond_dns_factors.csv", parse_dates=["date"])
    exposure = flagship_exposure_summary(outputs, config)
    sectors = pd.read_csv(outputs / "equity_industry_deviations.csv")
    ensemble = pd.read_csv(outputs / "equity_ensemble_weights.csv", parse_dates=["date"])
    start_pos = close.index.get_loc(result.index[0])
    initial_decision = close.index[start_pos - config["equity"]["execution_lag"]]
    coverage = pd.read_csv(outputs / "equity_point_in_time_coverage.csv", parse_dates=["date"])
    coverage = coverage.loc[coverage["date"].ge(initial_decision)]
    members = coverage["eligible_members"].sum()
    metadata = {
        "version": "V2",
        "first_session": result.index[0].date().isoformat(),
        "last_session": result.index[-1].date().isoformat(),
        "sessions": len(result),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "annualization_sessions": annualization,
        "allocation_trials_for_dsr": len(allocation_names),
        "bond_curve_pca_fit_share_first_3": float(cgb_pca["fit_variance_share"].sum()),
        "bond_curve_pca_oos_projected_share_first_3": float(cgb_pca["oos_projected_variance_share"].sum()),
        "dns_median_rmse_bps_oos": float(dns.loc[dns["date"].ge(config["dates"]["start"]), "rmse_bps"].median()),
        "brinson_annualized_residual_bps": float(attribution.at["reconciliation_residual", "annualized"] * 10000),
        "core_strategies": list(CORE_STRATEGIES),
        "industry_static_proxy_enabled": config["equity"].get("use_static_industry_proxy", False),
        "industry_pit_coverage": float(coverage["industry_pit_known_members"].sum() / members),
        "industry_proxy_coverage": float(coverage["industry_proxy_known_members"].sum() / members),
        "industry_max_target_deviation": float(sectors["abs_deviation"].max()),
        "industry_max_executed_deviation": float(sectors["executed_abs_deviation"].max()),
        "flagship_mean_exposure": {key: block.set_index("asset")["mean_weight"].to_dict() for key, block in exposure.groupby("layer")},
        "ensemble_mean_oos_weights": ensemble.loc[ensemble["date"].ge(initial_decision)].filter(regex="^weight_").mean().to_dict(),
    }
    (outputs / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
