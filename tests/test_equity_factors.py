import json

import numpy as np
import pandas as pd
import pytest

from src.equity.factors import fundamental_factors, neutralize
from src.equity.learning import realized_labels
from src.equity.portfolio import optimize_equity


def test_fundamental_factors_require_positive_denominators():
    values = pd.DataFrame({"pe_ttm": [10., -8., 0.], "pb": [2., -3., 0.], "pcf": [5., 0., -7.]})
    actual = fundamental_factors(values)
    factors = ["ep_ttm", "bp", "cfp", "implied_roe", "cash_earnings_quality"]
    np.testing.assert_allclose(actual.loc[0, factors], [.1, .5, .2, .2, 2.])
    assert actual.loc[1:, factors].isna().all().all()


def test_residualized_factor_is_industry_and_size_neutral(config):
    rng = np.random.default_rng(23)
    industries = np.repeat(["C", "J", "H"], [16, 16, 2])
    log_size = rng.normal(4, .45, len(industries))
    raw = 3 * log_size + np.where(industries == "C", -4, 4) + rng.normal(size=len(industries))
    raw[3] = np.nan
    day = pd.DataFrame({"industry": industries, "total_mv_100m": np.exp(log_size), "score": raw})
    cleaned, missing = neutralize(day, ["score"], config["equity"])
    assert cleaned["score"].notna().all()
    assert cleaned["model_industry"].tail(2).eq("OTHER").all()
    assert missing["score"] == pytest.approx(1 / len(industries))
    assert cleaned["score"].std(ddof=0) == pytest.approx(1.)
    np.testing.assert_allclose(cleaned.groupby("model_industry")["score"].mean(), 0., atol=1e-10)
    assert cleaned["score"] @ log_size == pytest.approx(0, abs=1e-10)


def test_realized_labels_begin_at_entry_close_and_finish_at_next_entry_close():
    days = pd.to_datetime(["2022-11-30", "2022-12-01", "2022-12-30", "2023-01-03"])
    panel = pd.DataFrame({"date": days[[0, 0, 2]], "code": ["A", "B", "A"]})
    execution = pd.Series(days[[1, 3]], index=days[[0, 2]])
    prices = pd.DataFrame({"A": [6., 10., 11., 12.], "B": [10., 11., 12., 13.]}, index=days)
    benchmark = pd.Series([90., 100., 105., 110.], index=days)
    valid = prices.notna()
    valid.loc[days[1], "B"] = False
    labels = realized_labels(panel, prices, valid, execution, benchmark)
    assert labels.at[0, "label_start"] == days[1]
    assert labels.at[0, "label_end"] == days[3]
    assert labels.at[0, "label"] == pytest.approx(.10)
    assert labels.loc[1:, "label"].isna().all()


def test_equity_optimizer_satisfies_position_industry_and_concentration_limits(config):
    cfg = config["equity"]
    codes = [f"{i:06}" for i in range(60)]
    day = pd.DataFrame({"date": pd.Timestamp("2022-12-30"), "code": codes,
                        "industry": ["C"] * 30 + ["J"] * 30, "score": np.linspace(-1, 1, 60)})
    sectors = pd.Series({"C": .50, "J": .50})
    weights = optimize_equity(day, pd.Series(1 / 60, index=codes), "score", cfg, sectors)
    assert weights.sum() == pytest.approx(1.)
    assert weights.min() >= 0
    assert weights.max() <= cfg["max_position"] + 1e-6
    assert 1 / weights.pow(2).sum() >= cfg["min_effective_holdings"] - 1e-3
    observed = weights.groupby(day.set_index("code")["industry"]).sum()
    assert observed.sub(sectors).abs().le(cfg["industry_band"] + 1e-6).all()


def test_saved_optimized_equity_orders_respect_industry_band(project_root, config):
    outputs = project_root / "outputs"
    if not (outputs / "equity_orders.parquet").is_file():
        pytest.skip("Run the equity stage to generate historical orders and industry audit")
    orders = pd.read_parquet(outputs / "equity_orders.parquet")
    audit = pd.read_csv(outputs / "equity_industry_deviations.csv",
                        parse_dates=["decision_date", "execution_date"])
    optimized = {"E1_ICIR", "E2_Ridge", "E3_LightGBM", "Equity_Alpha_Sleeve", "Equity_Execution_Delay"}
    optimized.update(name for name in orders["strategy"].unique() if name.startswith("Equity_Cost_"))
    keys = ["strategy", "decision_date", "execution_date"]
    required = orders.loc[orders["strategy"].isin(optimized), keys].drop_duplicates()
    assert set(required["strategy"]) == optimized
    assert orders.loc[orders["strategy"].isin(optimized), "industry_constrained"].all()
    equity_daily = pd.read_parquet(outputs / "equity_daily.parquet", columns=["date"])
    sessions = pd.DatetimeIndex(equity_daily["date"].drop_duplicates().sort_values())
    t_plus_one = required.loc[required["strategy"].ne("Equity_Execution_Delay")]
    positions = sessions.searchsorted(t_plus_one["decision_date"], side="right")
    np.testing.assert_array_equal(t_plus_one["execution_date"].to_numpy(), sessions[positions].to_numpy())
    expected = pd.MultiIndex.from_frame(required)
    actual = pd.MultiIndex.from_frame(audit[keys].drop_duplicates())
    assert expected.difference(actual).empty
    audit = audit.loc[audit["strategy"].isin(optimized)]
    assert not audit.duplicated([*keys, "industry"]).any()
    np.testing.assert_allclose(audit["abs_deviation"],
                               (audit["target_weight"] - audit["benchmark_weight"]).abs(), atol=1e-10)
    assert audit[["abs_deviation", "executed_abs_deviation"]].max().max() <= .04 + 1e-4
    assert config["equity"]["industry_band"] == .04
    coverage = pd.read_csv(outputs / "equity_point_in_time_coverage.csv", parse_dates=["date"])
    oos = coverage.loc[coverage["date"].between(config["dates"]["start"], config["dates"]["end"])]
    assert not oos.empty
    assert oos["industry_proxy_known_members"].div(oos["eligible_members"]).min() > .95
    raw_audit = json.loads((outputs / "raw_audit.json").read_text(encoding="utf-8"))
    provenance = raw_audit["industry_model_provenance"]
    assert provenance["static_proxy_enabled"] is True
    assert provenance["static_proxy_is_point_in_time"] is False
    np.testing.assert_allclose(audit["executed_abs_deviation"],
                               (audit["executed_weight"] - audit["benchmark_weight"]).abs(), atol=1e-10)
    for column in ("target_weight", "benchmark_weight", "executed_weight"):
        np.testing.assert_allclose(audit.groupby(keys)[column].sum(), 1, atol=1e-8)

def test_training_label_ranks_are_cross_sectional_and_outlier_invariant():
    from src.equity.learning import rank_normalize_labels

    days = pd.to_datetime(["2022-11-30"] * 4 + ["2022-12-30"] * 4)
    panel = pd.DataFrame({"date": days, "label": [-2., 0., 10., np.nan, -100., -100., -99., -98.]})
    ranks = rank_normalize_labels(panel)
    assert ranks.name == "label_cs_rank"
    assert pd.isna(ranks.iloc[3])
    assert ranks.dropna().abs().max() <= 1
    np.testing.assert_allclose(ranks.groupby(days).mean(), 0, atol=1e-12)
    assert ranks.iloc[0] < ranks.iloc[1] < ranks.iloc[2]
    assert ranks.iloc[4] == ranks.iloc[5]
    larger_outlier = panel.copy()
    larger_outlier.loc[2, "label"] = 1e9
    pd.testing.assert_series_equal(ranks, rank_normalize_labels(larger_outlier))

def test_equity_execution_retains_industry_band_when_buys_are_blocked(config):
    from src.equity.portfolio import run_equity_portfolio

    dates = pd.to_datetime(["2022-12-29", "2022-12-30", "2023-01-03", "2023-01-04"])
    decision, execution = dates[1:3]
    codes = [f"{i:06}" for i in range(100)]
    sectors = np.repeat(["C", "J"], 50)
    config["dates"].update(start=str(decision.date()), end=str(dates[-1].date()))
    frame = pd.DataFrame({"date": decision, "code": codes, "industry": sectors, "model_industry": sectors,
                          "total_mv_100m": 1., "score": np.linspace(10., 0., len(codes))})
    returns = pd.DataFrame(0., index=dates, columns=codes)
    tradable = returns.notna()
    blocked = codes[:10]
    tradable.loc[execution, blocked] = False
    daily, opening, closing, orders = run_equity_portfolio(
        returns, tradable, frame, frame, config, "score",
    )
    target = orders.set_index("code")["target_weight"].reindex(codes, fill_value=0.)
    assert target.loc[blocked].sum() > 2 * config["equity"]["industry_band"]
    assert bool(daily.loc[execution, "execution_repaired"])
    assert opening.loc[execution, codes].sum() == 0
    assert closing.loc[execution, blocked].sum() == 0
    realized = closing.loc[execution, codes].groupby(sectors).sum()
    assert realized.sum() >= .9999
    assert realized.sub(.50).abs().le(config["equity"]["industry_band"] + 1e-4).all()
