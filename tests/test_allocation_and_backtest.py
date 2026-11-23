import numpy as np
import pandas as pd
import pytest

from src.allocation.optimizers import equal_risk_contribution, hrp, optimize_bl
from src.allocation.portfolio import simulate_allocation
from src.analytics.attribution import bhb_daily, carino_link
from src.backtest.portfolio import CASH, Portfolio, etf_rates, stock_rates


@pytest.mark.parametrize("day,sell", [("2023-08-25", .00175), ("2023-08-28", .00125)])
def test_stock_and_etf_fee_rates(day, sell, config):
    assert stock_rates(pd.Timestamp(day), config["costs"]) == pytest.approx((.00075, sell))
    assert etf_rates(config["costs"]) == pytest.approx((.00030, .00030))


def test_cost_and_turnover_are_self_financing_with_blocked_trade():
    portfolio = Portfolio(pd.Index(["A", "B"]))
    rates = (.00075, .00125)
    first, _, after_first = portfolio.step([.05, .10], [True, True], [1., 0.], rates)
    assert first["gross_return"] == 0
    assert first["buy_turnover"] == pytest.approx(1 / (1 + rates[0]))
    assert first["sell_turnover"] == 0
    np.testing.assert_allclose(after_first, [1, 0, 0], atol=1e-12)
    blocked, _, after_blocked = portfolio.step([np.nan, .20], [False, True], [0., 1.], rates)
    assert blocked["stale_weight"] == pytest.approx(1.)
    assert blocked["turnover"] == pytest.approx(0.)
    np.testing.assert_allclose(after_blocked, after_first, atol=1e-12)
    stats, _, after = portfolio.step([.02, .10], [True, True], [0., 1.], rates)
    assert stats["sell_turnover"] == pytest.approx(1.)
    assert stats["cost_on_trade_notional"] == pytest.approx(
        stats["buy_turnover"] * rates[0] + stats["sell_turnover"] * rates[1]
    )
    assert stats["turnover"] == pytest.approx((stats["buy_turnover"] + stats["sell_turnover"]) / 2)
    assert stats["cost"] == pytest.approx((1 + stats["gross_return"]) * stats["cost_on_trade_notional"])
    assert stats["net_return"] == pytest.approx(stats["gross_return"] - stats["cost"])
    assert portfolio.nav == pytest.approx((1 + first["net_return"]) * (1 + stats["net_return"]))
    np.testing.assert_allclose(after, [0, 1, 0], atol=1e-12)


def test_risk_parity_equalizes_risk_and_hrp_is_fully_invested():
    cov = np.array([[.04, .001, .008], [.001, .01, -.003], [.008, -.003, .08]])
    erc = equal_risk_contribution(cov)
    contributions = erc * (cov @ erc) / (erc @ cov @ erc)
    np.testing.assert_allclose(contributions, 1 / 3, atol=2e-5)
    for weights in (erc, hrp(cov)):
        assert weights.sum() == pytest.approx(1.)
        assert np.all(weights > 0)


@pytest.mark.parametrize("corridor", ["tactical_bounds", "balanced_bounds"])
def test_bl_optimizer_respects_equity_corridor_and_asset_caps(config, corridor):
    cfg = config["allocation"]
    assets = ["equity_etf", "bond_passive", "gold", "convertible"]
    covariance = np.diag([.04, .004, .03, .02])
    previous = pd.Series([.2, .8, 0., 0.], index=assets)
    weights, detail = optimize_bl(
        assets, covariance, previous,
        {"risk_appetite": 1.5, "erp_z_3y": 2., "liquidity_z": 1.},
        {"credit_gate": False, "duration_score": .75, "credit_score": .9},
        cfg, cfg[corridor], delta=cfg["risk_aversion"], tau=cfg["tau"],
    )
    low, high = cfg[corridor]
    assert low - 1e-6 <= weights["equity_etf"] <= high + 1e-6
    assert weights.min() >= 0
    assert weights.sum() == pytest.approx(1)
    assert "cash" not in weights.index
    assert weights["bond_passive"] >= 1 - high - sum(cfg["satellite_caps"].values()) - 1e-6
    for asset, cap in cfg["satellite_caps"].items():
        assert weights[asset] <= cap + 1e-6
    assert detail["core_cash_target"] == 0
    assert detail["core_weight_sum"] == pytest.approx(1)
    assert detail["view_count"] == 3
    assert detail["relative_view_max_row_sum"] == 0
    assert "view_equity_etf_minus_bond" in detail
    assert any(0 < value < 1 for key, value in detail.items() if key.startswith("confidence_"))


def test_nested_costs_and_brinson_link_reconcile(config):
    calendar = pd.to_datetime(["2022-12-30", "2023-01-03", "2023-01-04", "2023-01-05"])
    gross = pd.DataFrame({
        "equity_etf": [0., .12, .02, -.005],
        "bond_passive": [0., .02, -.001, .005],
        "equity_active": [0., .10, .03, -.006],
        "bond_active": [0., .03, .002, .006],
        "cash": [0., .0001, .0001, .0001],
    }, index=calendar)
    cost = pd.DataFrame(0., index=gross.index, columns=gross.columns)
    cost.loc[:, ["equity_active", "bond_active"]] = [.0003, .0001]
    net = gross - cost
    market = {"gross": gross, "net": net, "cost": cost, "turnover": cost * 0,
              "tradable": gross.notna(), "eligible": gross.notna(), "covariance": net}
    config["dates"].update(start="2023-01-03", end="2023-01-05")
    features = pd.DataFrame(index=calendar[:1])
    passive, base, closing, orders = simulate_allocation(market, features, features, config, kind="static20")
    active, active_weights, _, _ = simulate_allocation(
        market, features, features, config, kind="static60", equity_asset="equity_active", bond_asset="bond_active"
    )
    assert orders.iloc[0]["decision_date"] == calendar[0]
    assert orders.iloc[0]["execution_date"] == calendar[1]
    assert base.at[calendar[1], CASH] == 1.
    assert closing.at[calendar[1], "equity_etf"] == pytest.approx(.2)
    assert active["nested_cost"].iloc[1] > 0
    np.testing.assert_allclose(active["gross_return"] - active["cost"], active["net_return"], atol=1e-12)
    effects = bhb_daily(active, passive, active_weights, base, gross.loc[active.index])
    np.testing.assert_allclose(effects.sum(axis=1), active["net_return"] - passive["net_return"], atol=1e-12)
    linked = carino_link(effects, active["net_return"], passive["net_return"], config["dates"]["periods_per_year"])
    assert linked.loc["active_total", "cumulative"] == pytest.approx(
        active["nav"].iloc[-1] - passive["nav"].iloc[-1]
    )
    assert linked.loc["reconciliation_residual", "annualized"] == pytest.approx(0., abs=1e-12)


def test_saved_market_results_reconcile_if_pipeline_has_run(project_root, config):
    outputs = project_root / "outputs"
    files = [outputs / name for name in
             ("allocation_daily.parquet", "allocation_weights_end.parquet", "allocation_orders.csv")]
    if not all(file.is_file() for file in files):
        pytest.skip("Run the allocation stage to validate the generated historical results")
    daily = pd.read_parquet(files[0])
    weights = pd.read_parquet(files[1]).drop(columns=["date", "strategy"])
    orders = pd.read_csv(files[2], parse_dates=["decision_date", "execution_date"])
    np.testing.assert_allclose(weights.sum(axis=1), 1., atol=1e-9)
    assert (weights.fillna(0).to_numpy() >= -1e-9).all()
    np.testing.assert_allclose(daily["gross_return"] - daily["cost"], daily["net_return"], atol=1e-12)
    assert (orders["decision_date"] < orders["execution_date"]).all()
    for _, strategy in daily.sort_values("date").groupby("strategy"):
        assert strategy["date"].iloc[0] == pd.Timestamp(config["dates"]["start"])
        assert strategy["date"].iloc[-1] == pd.Timestamp(config["dates"]["end"])
        np.testing.assert_allclose((1 + strategy["net_return"]).cumprod(), strategy["nav"], atol=1e-10)


    brinson_file = outputs / "brinson_attribution.csv"
    if brinson_file.is_file():
        attribution = pd.read_csv(brinson_file, index_col="effect")
        annual = attribution["annualized"]
        component_sum = annual.drop(["active_total", "reconciliation_residual"]).sum()
        assert abs(annual["active_total"] - component_sum) < .0001
        assert annual["reconciliation_residual"] == pytest.approx(annual["active_total"] - component_sum)
        end_nav = daily.sort_values("date").groupby("strategy").agg(nav=("nav", "last"), n=("nav", "size"))
        cagr = end_nav["nav"].pow(config["dates"]["periods_per_year"] / end_nav["n"]) - 1
        assert annual["active_total"] == pytest.approx(cagr["Macro_BL_FixedIncomePlus"] - cagr["Static_20_80"])


def test_overlay_cuts_immediately_and_restores_equity_gradually(config):
    from src.allocation.portfolio import overlay

    base = pd.Series({"equity_etf": 0.30, "bond_passive": 0.7, "cash": 0.0})
    history = pd.DataFrame({"equity_etf": [0.001] * 20, "bond_passive": [0.0002] * 20})
    cut, vol_scale, _, cut_scale = overlay(base, history, "equity_etf", config["allocation"]["drawdown_thresholds"][1], 1.0, config)
    restoration, _, _, restored_scale = overlay(
        base, history, "equity_etf", 0.01, vol_scale, config, prev_equity_scale=cut_scale,
    )
    assert cut_scale == config["allocation"]["drawdown_scale"][1]
    assert restored_scale == pytest.approx(cut_scale + config["allocation"]["overlay_recovery_step"])
    assert cut["bond_passive"] >= base["bond_passive"]
    assert restoration["bond_passive"] >= base["bond_passive"]
    assert cut["equity_etf"] < restoration["equity_etf"] < base["equity_etf"]
    assert restoration.sum() == pytest.approx(1)

def test_saved_flagship_stays_invested_and_executes_next_session(project_root):
    outputs = project_root / "outputs"
    file = outputs / "allocation_weights_end.parquet"
    if not file.is_file():
        pytest.skip("Run the allocation stage to generate flagship weights")
    weights = pd.read_parquet(file)
    flagship = weights.loc[weights["strategy"].eq("Macro_BL_FixedIncomePlus")].sort_values("date")
    assert not flagship.empty
    top_cash = flagship[["cash", CASH]].fillna(0).sum(axis=1).mean()
    assert top_cash < .50
    exposure = pd.read_csv(outputs / "flagship_exposure_summary.csv")
    cash_by_layer = exposure.loc[exposure["asset"].eq("cash")].set_index("layer")["mean_weight"]
    assert cash_by_layer["top_level"] == pytest.approx(top_cash)
    assert cash_by_layer["lookthrough"] < .50
    np.testing.assert_allclose(exposure.groupby("layer")["mean_weight"].sum(), 1, atol=1e-10)
    orders = pd.read_csv(outputs / "allocation_orders.csv", parse_dates=["decision_date", "execution_date"])
    core = orders.loc[orders["strategy"].eq("Macro_BL_FixedIncomePlus")]
    sessions = pd.DatetimeIndex(flagship["date"])
    expected = sessions[sessions.searchsorted(core["decision_date"], side="right")]
    np.testing.assert_array_equal(core["execution_date"].to_numpy(), expected.to_numpy())


def test_overlay_preserves_bond_and_gold_during_joint_risk_triggers(config):
    from src.allocation.portfolio import overlay

    base = pd.Series({"equity_active": .25, "bond_active": .55, "convertible": .10, "gold": .10, "cash": 0.})
    history = pd.DataFrame(
        {"equity_active": [.05, -.05] * 20, "bond_active": .0002,
         "convertible": [.03, -.03] * 20, "gold": .0001}
    )
    desired, volatility_scale, _, drawdown_scale = overlay(
        base, history, "equity_active", .10, 1., config, bond_asset="bond_active",
    )
    assert 0 < volatility_scale < 1
    assert drawdown_scale == config["allocation"]["drawdown_scale"][1]
    assert (desired[["equity_active", "convertible"]] < base[["equity_active", "convertible"]]).all()
    assert desired["gold"] == pytest.approx(base["gold"])
    assert desired["bond_active"] >= base["bond_active"]
    assert desired["cash"] >= 0
    assert desired.sum() == pytest.approx(1)

def test_overlay_trailing_peak_expiry_unwinds_cash_without_nav_recovery(config):
    calendar = pd.bdate_range("2020-01-01", periods=285)
    config["dates"].update(start=str(calendar[1].date()), end=str(calendar[-1].date()))
    gross = pd.DataFrame(0., index=calendar, columns=["equity_etf", "bond_passive", "cash"])
    shock = 7
    gross.loc[calendar[shock], "equity_etf"] = -.50
    market = {"gross": gross, "net": gross, "cost": gross * 0, "turnover": gross * 0,
              "tradable": gross.notna(), "eligible": gross.notna(), "covariance": gross}
    features = pd.DataFrame(index=calendar[[0]])
    daily, _, closing, orders = simulate_allocation(
        market, features, features, config, kind="static20", use_overlay=True,
    )
    after_shock = calendar[shock + 1]
    assert orders.loc[orders["decision_date"].eq(calendar[shock]), "execution_date"].item() == after_shock
    assert closing.loc[after_shock, ["cash", CASH]].sum() > 0
    assert daily["nav"].iloc[-1] < daily["nav"].iloc[0]
    assert daily.loc[calendar[shock], "trailing_drawdown"] >= config["allocation"]["drawdown_thresholds"][1]
    assert daily["trailing_drawdown"].iloc[-1] < config["allocation"]["drawdown_thresholds"][0]
    assert closing.iloc[-1][["cash", CASH]].sum() <= config["allocation"]["overlay_trade_band"]
    assert closing.iloc[-1]["equity_etf"] > closing.loc[after_shock, "equity_etf"]

def test_bl_uses_strategic_prior_and_only_bond_relative_views(config, monkeypatch):
    from src.allocation import optimizers

    assets = ["equity_etf", "bond_passive", "convertible", "gold"]
    covariance = np.diag([.04, .004, .02, .03])
    cfg = config["allocation"]
    original = optimizers.black_litterman
    observed = []

    def capture(prior, covariance, views, returns, omega, tau, delta):
        observed.append((prior.copy(), views.copy()))
        return original(prior, covariance, views, returns, omega, tau, delta)

    monkeypatch.setattr(optimizers, "black_litterman", capture)
    common = (
        {"risk_appetite": .3, "erp_z_3y": .4, "liquidity_z": -.1},
        {"credit_gate": False, "duration_score": .5, "credit_score": .5},
        cfg, cfg["balanced_bounds"],
    )
    strategic = cfg["balanced_strategic_equity_weight"]
    weights, _ = optimize_bl(assets, covariance, pd.Series(0., index=assets), *common,
                             delta=cfg["risk_aversion"], tau=cfg["tau"], strategic_equity_weight=strategic)
    prior, views = observed[0]
    np.testing.assert_allclose(prior, [strategic, 1 - strategic, 0, 0], atol=1e-12)
    expected_views = {(1, -1, 0, 0), (0, -1, 1, 0), (0, -1, 0, 1)}
    assert {tuple(row) for row in views} == expected_views
    assert len(views) == len(expected_views)
    assert set(weights.index) == set(assets)

    cash_assets = [*assets, "cash"]
    with pytest.raises(ValueError, match="accepts only equity, bond"):
        optimize_bl(cash_assets, np.diag([.04, .004, .02, .03, .000002]),
                    pd.Series(0., index=cash_assets), *common,
                    delta=cfg["risk_aversion"], tau=cfg["tau"], strategic_equity_weight=strategic)
