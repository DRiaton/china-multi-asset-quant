from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.allocation.optimizers import shrinkage_covariance
from src.allocation.regime import REGIMES, macro_features
from src.data.macro import align_daily_close, align_publications
from src.data.market import etf_market
from src.data.point_in_time import backward_by_code, constituent_memberships, point_in_time_industry
from src.equity.factors import price_factors
from src.equity.learning import MODELS, monthly_rank_ic, walk_forward
from src.equity.portfolio import run_equity_portfolio
from src.fixed_income.signals import bond_signals
from src.utils.calendar import execution_map, month_end_decisions


@pytest.mark.parametrize(
    "period,lag,effective",
    [("2022-12", 0, "2023-01-03"), ("2022-12", 12, "2023-01-13"),
     ("2022-12", 15, "2023-01-16"), ("2022Q4", 20, "2023-01-23")],
)
def test_macro_release_rules(period, lag, effective):
    calendar = pd.bdate_range("2022-12-30", "2023-02-01").difference(
        pd.DatetimeIndex(["2023-01-02"])
    )
    obs = pd.DataFrame({"value": [37.]}, index=pd.PeriodIndex([period], freq="Q" if "Q" in period else "M"))
    aligned = align_publications(obs, calendar, lag)
    assert aligned["value"].first_valid_index() == pd.Timestamp(effective)
    assert aligned.at[pd.Timestamp(effective), "effective_date"] == pd.Timestamp(effective)
    assert aligned.loc[aligned.index < effective, "value"].isna().all()


def test_daily_closes_are_not_visible_same_day():
    calendar = pd.bdate_range("2023-01-05", periods=4)
    close = pd.DataFrame({"close": [2.1, 2.4, 80.]}, index=calendar[[0, 1, 3]])
    aligned = align_daily_close(close, calendar)
    assert np.isnan(aligned.iloc[0]["close"])
    np.testing.assert_allclose(aligned["close"].iloc[1:], [2.1, 2.4, 2.4])
    assert (aligned.dropna()["source_date"] < aligned.dropna().index).all()


def test_constituents_follow_effective_date_and_preserve_removed_names():
    days = pd.to_datetime(["2022-12-29", "2022-12-30", "2023-01-03"])
    codes = [f"{i:06}" for i in range(301)]
    snapshots = pd.concat([
        pd.DataFrame({"code": codes[:300], "update_date": days[0], "query_date": days[1]}),
        pd.DataFrame({"code": codes[1:], "update_date": days[2], "query_date": days[2]}),
    ], ignore_index=True)
    actual = constituent_memberships(snapshots, days)
    assert actual.groupby("date").size().eq(300).all()
    assert (actual["update_date"] <= actual["date"]).all()
    assert set(actual.loc[actual["date"].eq(days[1]), "code"]) == set(codes[:300])
    assert set(actual.loc[actual["date"].eq(days[2]), "code"]) == set(codes[1:])


def test_backward_valuation_and_sector_join_by_code():
    early, change, late = pd.to_datetime(["2022-12-30", "2023-01-05", "2023-01-20"])
    bars = pd.DataFrame({"date": [late, early, change, early], "code": ["A", "B", "A", "A"]})
    value = pd.DataFrame({"date": [change, early], "code": ["A", "B"], "pe_ttm": [10, 99]})
    merged = backward_by_code(bars, value, max_age_days=7)
    actual = merged.set_index(["date", "code"])["pe_ttm"]
    assert pd.isna(actual.at[(early, "A")])
    assert actual.at[(early, "B")] == 99
    assert actual.at[(change, "A")] == 10
    assert pd.isna(actual.at[(late, "A")])
    assert (merged["valuation_date"].dropna() <= merged.dropna(subset=["valuation_date"])["date"]).all()
    industry = pd.DataFrame({"code": ["A", "A", "B"], "updateDate": [early, change, change],
                             "industry": ["C Manufacturing", "J Finance", "B Mining"]})
    assert point_in_time_industry(bars, industry).tolist() == ["J", "Unknown", "J", "C"]


def test_etf_seasoning_uses_only_past_traded_sessions(monkeypatch):
    calendar = pd.bdate_range("2023-01-03", periods=66)
    listing = calendar[3]
    live = calendar[3:]
    halted = live[10]
    nav_gap = live[-1]
    nav = pd.DataFrame({"code": "A", "date": live, "acc_nav": 1.0, "daily_ret_pct": 0.2})
    nav.loc[nav["date"].eq(nav_gap), "daily_ret_pct"] = np.nan
    bars = pd.DataFrame({"code": "A", "date": live, "amount": 20., "volume": 10., "close": 100.})
    bars.loc[bars["date"].eq(halted), "volume"] = 0
    bars.loc[bars["date"].ge(live[20]), "close"] = 10.0
    tables = {"etf_nav_daily": nav, "etf_daily": bars}

    def read_fixture(_root, name, columns=None):
        frame = tables[name]
        return frame.copy() if columns is None else frame[columns].copy()

    monkeypatch.setattr("src.data.market.read_table", read_fixture)
    cfg = {"etf_codes": {"alpha": "A"}, "etf_acc_nav_codes": [], "etf_seasoning_sessions": 60}
    full = etf_market(Path("."), calendar, cfg)
    first_eligible = live.delete(10)[59]
    assert full["eligible"]["A"].idxmax() == first_eligible
    assert not full["eligible"].at[listing, "A"]
    assert not full["tradable"].at[halted, "A"]
    assert not full["tradable"].at[nav_gap, "A"]
    assert full["returns"].at[live[20], "A"] == pytest.approx(.002)
    for table in ("eligible", "tradable", "returns"):
        prefix = calendar[:40]
        assert_frame_equal(full[table].loc[prefix], etf_market(Path("."), prefix, cfg)[table])


def test_monthly_macro_erp_and_bond_signals_are_invariant_to_future_data(config):
    calendar = pd.bdate_range("2018-01-01", "2024-03-29", name="date")
    cutoff = pd.Timestamp("2022-12-31")
    i = np.arange(len(calendar))
    macro = pd.DataFrame({
        "pmi": 50 + np.sin(i / 40), "pmi_mom_3m": np.cos(i / 60),
        "social_flow_yoy": np.sin(i / 150), "social_period_end": calendar - pd.Timedelta(days=30),
        "cpi_yoy": np.cos(i / 90), "ppi_yoy": np.sin(i / 53),
        "m1_m2_change": np.cos(i / 100), "shibor_3m": .025 + .004 * np.sin(i / 20),
        "shibor_1w": .02 + .004 * np.cos(i / 30), "index_pe_ttm": 12 + 3 * np.cos(i / 70),
    }, index=calendar)
    futures = i > calendar.searchsorted(cutoff) - 1
    macro.loc[futures, ["pmi", "index_pe_ttm"]] *= 20
    cfg = config["bond"]
    yields = .025 + .003 * np.cos(i[:, None] / 30 + np.asarray(cfg["maturities"])[None, :])
    curves = pd.concat([
        pd.DataFrame(yields + shift, columns=cfg["yield_columns"]).assign(date=calendar, curve_id=curve)
        for shift, curve in [(0, "CGB"), (.006, "MTN_AAA"), (.004, "BANK_AAA")]
    ])
    future_rows = curves["date"].gt(cutoff)
    curves.loc[future_rows, cfg["yield_columns"]] += 0.20
    full_bond, _ = bond_signals(curves, calendar, macro, config)
    past_bond, _ = bond_signals(curves.loc[~future_rows], calendar[~futures], macro.loc[:cutoff], config)
    assert_frame_equal(full_bond.loc[:cutoff], past_bond, atol=1e-10, rtol=0)
    audit = full_bond.dropna(subset=["cgb_source_date"])
    assert (audit["cgb_source_date"] < audit.index).all()
    full_features, full_erp = macro_features(macro, full_bond, config)
    past_features, past_erp = macro_features(macro.loc[:cutoff], past_bond, config)
    assert pd.Timestamp("2022-11-30") in past_features.index
    assert_frame_equal(full_features.loc[past_features.index], past_features, atol=1e-10, rtol=0)
    assert_frame_equal(full_erp.loc[:cutoff], past_erp, atol=1e-10, rtol=0)
    np.testing.assert_allclose(full_features[[f"p_{name}" for name in REGIMES]].sum(axis=1), 1., atol=1e-12)


def test_covariance_uses_only_complete_trailing_data(config):
    rng = np.random.default_rng(187)
    index = pd.bdate_range("2022-01-03", "2023-04-28")
    returns = pd.DataFrame(rng.normal(size=(len(index), 3)), index=index, columns=list("ABC"))
    decision = pd.Timestamp("2022-12-30")
    returns.loc[returns.index > decision] *= 1_000
    cfg = config["allocation"]
    future = shrinkage_covariance(returns, decision, cfg, 252)
    historical = shrinkage_covariance(returns.loc[:decision], decision, cfg, 252)
    np.testing.assert_allclose(future, historical, atol=1e-10, rtol=0)
    assert (np.linalg.eigvalsh(future) > 0).all()


def test_walk_forward_uses_only_observed_labels_and_respects_purge(config):
    rng = np.random.default_rng(42)
    decisions = pd.date_range("2020-01-31", "2023-01-31", freq="ME")
    cutoff = pd.Timestamp("2022-12-31")
    panel = pd.DataFrame([
        (date, f"{i:04}", "C", "C", 100.0 + i, x, y, .1 * x - .25 * y + noise)
        for date in decisions
        for i, (x, y, noise) in enumerate(rng.normal(size=(90, 3)))
    ], columns=["date", "code", "model_industry", "industry", "total_mv_100m", "x", "y", "label"])
    panel["label_end"] = panel["date"] + pd.offsets.MonthEnd() + pd.Timedelta(days=2)
    cfg = {**config["equity"], "factors": ["x", "y"]}
    cfg["lgbm_params"] = {**cfg["lgbm_params"], "n_estimators": 8}
    full, full_log = walk_forward(panel, monthly_rank_ic(panel, cfg["factors"]), cfg, 101)
    prefix = panel.loc[panel["date"].le(cutoff)].copy()
    prefix.loc[prefix["label_end"].gt(cutoff), "label"] = np.nan
    past, past_log = walk_forward(prefix, monthly_rank_ic(prefix, cfg["factors"]), cfg, 101)
    assert_frame_equal(full.loc[full["date"].le(cutoff), list(MODELS)], past[list(MODELS)],
                       atol=1e-10, rtol=0)
    assert_frame_equal(full_log.loc[:cutoff], past_log, atol=1e-11, rtol=0)
    weights = [f"weight_{model}" for model in MODELS[:-1]]
    np.testing.assert_allclose(full_log[weights].sum(axis=1), 1, atol=1e-12)
    assert (full_log[weights] >= 0).all().all()
    assert full_log[weights].nunique().max() > 1
    np.testing.assert_allclose(full_log.loc[:cutoff, weights], past_log[weights], atol=1e-11, rtol=0)
    for model in MODELS[:-1]:
        matured_oos = full_log[f"ensemble_latest_label_end_{model}"].dropna()
        assert not matured_oos.empty and (matured_oos <= matured_oos.index).all()
    observed_ic = full_log["ic_latest_label_end"].dropna()
    assert not observed_ic.empty and (observed_ic <= observed_ic.index).all()
    for name in ("ridge_latest_label_end", "lgbm_latest_label_end"):
        trained = full_log.dropna(subset=[name])
        assert not trained.empty and trained[name].le(trained["purge_cutoff"]).all()


def test_t_plus_one_fills_after_execution_session_mark(config):
    dates = pd.to_datetime(["2023-04-27", "2023-04-28", "2023-05-04", "2023-05-05"])
    decisions = month_end_decisions(dates)
    assert execution_map(dates, decisions).at[dates[1]] == dates[2]
    returns = pd.DataFrame({"A": [0., .7, .6, .1]}, index=dates)
    scores = pd.DataFrame({"date": [dates[1]], "code": ["A"], "alpha": [1.]})
    config["dates"].update(start="2023-04-28", end="2023-05-05")
    result, opening, closing, orders = run_equity_portfolio(
        returns, returns.notna(), scores, scores, config, "alpha", top_n=1
    )
    assert orders["execution_date"].iloc[0] == dates[2]
    assert opening.at[dates[2], "A"] == 0
    assert closing.at[dates[2], "A"] == pytest.approx(1)
    assert result.at[dates[1], "net_return"] == 0
    assert result.at[dates[2], "gross_return"] == 0
    assert result.at[dates[3], "gross_return"] == pytest.approx(.1)


def test_trailing_price_factors_and_formula(config):
    rng = np.random.default_rng(842)
    calendar = pd.bdate_range("2021-12-01", "2023-04-01")
    market_returns = rng.normal(0, .01, size=len(calendar))
    benchmark = pd.Series(100 * np.cumprod(1 + market_returns), index=calendar)
    close = pd.DataFrame(
        20 * np.cumprod(1 + market_returns[:, None] + rng.normal(0, .01, size=(len(calendar), 2)), axis=0),
        index=calendar, columns=["A", "B"],
    )
    high = close * 1.02
    volume = close * 900 + 100
    amount = volume * close
    turnover = volume / 1e5
    full = price_factors(close, high, volume, amount, turnover, benchmark, config["equity"])
    historical = [obj.loc[:"2022-12-31"] for obj in (close, high, volume, amount, turnover, benchmark)]
    past = price_factors(*historical, config["equity"])
    assert len(full) == 11
    for factor, values in full.items():
        pd.testing.assert_frame_equal(values.loc[past[factor].index], past[factor], atol=1e-10, rtol=0)
    date = past["mom_12_1"].index[-1]
    location = calendar.get_loc(date)
    assert full["mom_12_1"].at[date, "A"] == pytest.approx(
        close["A"].iloc[location - 21] / close["A"].iloc[location - 252] - 1
    )
    assert full["rev_20d"].at[date, "B"] == pytest.approx(
        1 - close["B"].iloc[location] / close["B"].iloc[location - 20]
    )

def test_saved_historical_inputs_preserve_asof_contract_and_signal_prefix(project_root, config):
    processed = project_root / "data" / "processed"
    files = [processed / f"{name}.parquet" for name in ("macro_asof", "curves", "memberships", "equity_features")]
    if not all(file.is_file() for file in files):
        pytest.skip("Run the data, equity and bond stages to validate historical signal inputs")
    macro, curves, membership, factors = map(pd.read_parquet, files)
    assert membership.groupby("date").size().eq(300).all()
    assert (membership["update_date"] <= membership["date"]).all()
    matched = factors.dropna(subset=["valuation_date"])
    assert (matched["valuation_date"] <= matched["date"]).all()
    assert (matched["date"] - matched["valuation_date"]).dt.days.le(config["data"]["valuation_max_age_days"]).all()
    for key, days in config["data"]["macro_lag_days"].items():
        window = macro[[f"{key}_period_end", f"{key}_effective_date"]].dropna()
        assert (window.iloc[:, 1] <= window.index).all()
        assert (window.iloc[:, 1] > window.iloc[:, 0] + pd.Timedelta(days=days)).all()

    cutoff = pd.Timestamp("2022-12-31")
    # One future session in the calendar establishes the last month boundary.
    # Its market observations and already-aligned macro metrics are withheld.
    calendar = macro.index[: macro.index.searchsorted(cutoff, side="right") + 1]
    observed_macro = macro.loc[:cutoff].reindex(calendar)
    prior_curves = curves.loc[curves["date"].le(cutoff)]
    full_bond, _ = bond_signals(curves, macro.index, macro, config)
    prior_bond, _ = bond_signals(prior_curves, calendar, observed_macro, config)
    assert_frame_equal(full_bond.loc[:cutoff], prior_bond.loc[:cutoff], atol=1e-10, rtol=0)
    full_features, full_erp = macro_features(macro, full_bond, config)
    prior_features, prior_erp = macro_features(observed_macro, prior_bond, config)
    assert prior_features.index[-1] == pd.Timestamp("2022-12-30")
    signal_cols = ["risk_appetite", "rule_regime", "filtered_regime", "erp_z_3y", "erp_pct_3y",
                   *[f"p_{name}" for name in REGIMES]]
    assert_frame_equal(full_features.loc[:cutoff, signal_cols], prior_features[signal_cols], atol=1e-10, rtol=0)
    assert_frame_equal(full_erp.loc[:cutoff], prior_erp.loc[:cutoff], atol=1e-10, rtol=0)

