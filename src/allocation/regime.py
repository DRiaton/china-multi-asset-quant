
import numpy as np
import pandas as pd
from scipy.special import softmax

from src.fixed_income.signals import rolling_z
from src.utils.calendar import month_end_decisions

REGIMES = ("Recovery", "Overheat", "Stagflation", "Recession")
CENTERS = np.array([[1., -1.], [1., 1.], [-1., 1.], [-1., -1.]])


def filtered_regimes(features: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Forward filter with anchored soft clusters and an expanding transition estimate."""
    counts = np.ones((len(REGIMES), len(REGIMES)))
    probabilities = np.ones(len(REGIMES)) / len(REGIMES)
    previous = None
    observations = 0
    rows = []
    for day, row in features[["growth_z", "inflation_z"]].iterrows():
        z = row.fillna(0).to_numpy().clip(-3, 3)
        emissions = softmax(CENTERS @ z / cfg["regime_temperature"])
        state = int(np.argmax(emissions))
        if row.notna().all():
            observations += 1
            if observations >= cfg["regime_model_min_months"]:
                transition = counts / counts.sum(axis=1, keepdims=True)
                prediction = probabilities @ transition
                mixed = cfg["regime_memory"] * prediction + (1 - cfg["regime_memory"]) / len(REGIMES)
                posterior = emissions * mixed
                probabilities = posterior / posterior.sum()
            else:
                probabilities = emissions
            if previous is not None:
                counts[previous, state] += 1
            previous = state
        rows.append({
            "date": day, "rule_regime": REGIMES[state],
            "filtered_regime": REGIMES[np.argmax(probabilities)],
            "regime_training_observations": observations,
            **{f"p_{name}": p for name, p in zip(REGIMES, probabilities)},
        })
    return pd.DataFrame(rows).set_index("date")


def macro_features(macro, bond, config):
    cfg = config["allocation"]
    calendar = macro.index
    decisions = month_end_decisions(calendar)
    # GDP is stored for audit. The growth score uses PMI, PMI momentum and trailing financing flow.
    monthly = macro.reindex(decisions).copy()
    age_days = (monthly.index.to_series() - monthly["social_period_end"]).dt.days
    monthly["social_age_days"] = age_days
    monthly["social_flow_yoy"] = monthly["social_flow_yoy"].where(age_days.le(cfg["social_max_age_days"]))
    monthly["funding_change_3m"] = -monthly["shibor_3m"].diff(cfg["funding_change_months"])
    monthly["term_spread"] = bond["spread_10y_1y"].reindex(decisions)
    for feature in (
        "pmi", "pmi_mom_3m", "social_flow_yoy", "cpi_yoy", "ppi_yoy",
        "m1_m2_change", "funding_change_3m", "term_spread",
    ):
        monthly[f"z_{feature}"] = rolling_z(monthly[feature], cfg["macro_window_months"], cfg["macro_min_months"])
    monthly["growth_z"] = monthly[["z_pmi", "z_pmi_mom_3m", "z_social_flow_yoy"]].mean(axis=1)
    monthly["inflation_z"] = monthly[["z_cpi_yoy", "z_ppi_yoy"]].mean(axis=1)
    monthly["liquidity_z"] = monthly[["z_m1_m2_change", "z_funding_change_3m", "z_term_spread"]].mean(axis=1)
    features = monthly.join(filtered_regimes(monthly, cfg))
    features["risk_appetite"] = (
        features["p_Recovery"] + 0.5 * features["p_Overheat"]
        - 0.85 * features["p_Stagflation"] - 0.55 * features["p_Recession"]
    )

    erp = 1 / macro["index_pe_ttm"].where(macro["index_pe_ttm"].gt(0)) - bond["cgb_y_10y"]
    daily = erp.rename("erp").to_frame()
    for years in cfg["erp_windows_years"]:
        window = years * config["dates"]["periods_per_year"]
        daily[f"erp_z_{years}y"] = rolling_z(erp, window, cfg["erp_min"])
        daily[f"erp_pct_{years}y"] = erp.rolling(window, min_periods=cfg["erp_min"]).rank(pct=True)
    return features.join(daily.reindex(features.index)), daily


def regime_performance(features, asset_returns, executions, config):
    portfolio_assets = ["equity", "cgb30", "cgb5", "short_cp", "credit", "convertible", "gold"]
    start = pd.Timestamp(config["dates"]["start"])
    result = []
    holding = executions.shift(-1)
    for regime_column in ("rule_regime", "filtered_regime"):
        for regime in REGIMES:
            dates = features.index[(features[regime_column] == regime) & executions.reindex(features.index).ge(start)]
            for asset in portfolio_assets:
                period_returns, cash_returns = [], []
                for decision in dates:
                    entry, end = executions.get(decision), holding.get(decision)
                    if pd.isna(entry) or pd.isna(end):
                        continue
                    sub = asset_returns.loc[entry:end, [asset, "cash"]].iloc[1:]
                    if len(sub) and sub.notna().all().all():
                        result_window = (1 + sub).prod() - 1
                        period_returns.append(result_window[asset])
                        cash_returns.append(result_window["cash"])
                values = np.asarray(period_returns)
                excess = values - np.asarray(cash_returns)
                sharpe = np.sqrt(12) * excess.mean() / excess.std(ddof=1) if len(values) > 1 and excess.std(ddof=1) > 0 else np.nan
                result.append({
                    "engine": regime_column, "regime": regime, "asset": asset, "months": len(values),
                    "annualized_return": np.expm1(np.log1p(values).mean() * 12) if len(values) else np.nan,
                    "sharpe": sharpe,
                })
    return pd.DataFrame(result)
