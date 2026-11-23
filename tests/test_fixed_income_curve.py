import numpy as np
import pandas as pd
import pytest

from src.fixed_income.curve import carry_roll, curve_pca, nelson_siegel, par_bond_risk, theoretical_return
from src.fixed_income.portfolio import optimize_bond


def test_nelson_siegel_recovers_coefficients_and_reports_bp(config):
    cfg = config["bond"]
    tenors = np.asarray(cfg["maturities"])
    x = cfg["ns_lambda_per_year"] * tenors
    slope = (1 - np.exp(-x)) / x
    known = np.array([[.04, -.02, .005], [.02, .005, -.009]])
    curve = known[:, :1] + known[:, 1:2] * slope + known[:, 2:] * (slope - np.exp(-x))
    raw = pd.DataFrame(curve, columns=cfg["yield_columns"])
    coefficients, fitted = nelson_siegel(raw, tenors, cfg["ns_lambda_per_year"])
    np.testing.assert_allclose(coefficients.iloc[:, :3], known, atol=1e-12)
    np.testing.assert_allclose(fitted, raw, atol=1e-12)
    assert coefficients["rmse_bps"].max() < 1e-8
    raw.iloc[0, 1] += .001
    coefficients, fitted = nelson_siegel(raw, tenors, cfg["ns_lambda_per_year"])
    assert coefficients.at[0, "rmse_bps"] == pytest.approx(np.sqrt(np.mean((fitted.iloc[0] - raw.iloc[0]) ** 2)) * 10000)


def test_frozen_pca_never_refits_to_oos_future(config):
    rng = np.random.default_rng(65)
    columns = config["bond"]["yield_columns"]
    days = pd.bdate_range("2020-01-01", periods=300)
    curves = pd.DataFrame(.03 + .001 * (rng.normal(size=(300, 3)) @ rng.normal(size=(3, 8))),
                          index=days, columns=columns)
    curves.iloc[250:] += .01 * rng.normal(size=(50, 8))
    full_scores, full = curve_pca(curves, 175, days[200])
    prior_scores, prior = curve_pca(curves.iloc[:250], 175, days[200])
    pd.testing.assert_frame_equal(prior_scores, full_scores.loc[prior_scores.index], atol=1e-10, rtol=0)
    pd.testing.assert_frame_equal(prior[columns], full[columns])
    assert (full["fit_end"] < days[200]).all()
    assert full["fit_variance_share"].sum() == pytest.approx(1.)


def test_par_bond_risk_and_carry_on_flat_curve(config):
    cfg = config["bond"]
    yield_ = .03
    eps = 1e-5
    periods = np.arange(1, 11)
    flows = np.full(10, yield_ / 2)
    flows[-1] += 1

    def cash_flow_price(discount_yield):
        return (flows / (1 + discount_yield / 2) ** periods).sum()

    duration, convexity = par_bond_risk(yield_, 5.)
    up, down, base = (cash_flow_price(y) for y in (yield_ + eps, yield_ - eps, yield_))
    assert base == pytest.approx(1.)
    assert duration == pytest.approx((down - up) / (2 * eps), rel=1e-6)
    assert convexity == pytest.approx((up - 2 * base + down) / eps ** 2, rel=1e-5)
    curve = pd.DataFrame(yield_, index=pd.bdate_range("2023-01-03", periods=4), columns=cfg["yield_columns"])
    np.testing.assert_allclose(carry_roll(curve, cfg["maturities"], 5, 1), yield_, atol=1e-12)
    daily = theoretical_return(curve, cfg["maturities"], 5, 252)
    assert pd.isna(daily.iloc[0])
    np.testing.assert_allclose(daily.iloc[1:], yield_ / 252, atol=1e-12)
    curve.iloc[-1] += .002
    daily = theoretical_return(curve, cfg["maturities"], 5, 252)
    assert daily.iloc[-1] == pytest.approx(yield_ / 252 - duration * .002 + convexity * .002 ** 2 / 2)


@pytest.mark.parametrize("gate,target", [(False, 6.0), (True, 6.0)])
def test_bond_optimizer_enforces_duration_credit_and_eligibility(config, gate, target):
    cfg = config["bond"]
    duration = pd.Series(cfg["durations"])
    eligible = pd.Series(True, index=duration.index)
    eligible.loc["cgb30"] = False
    previous = pd.Series(0., index=duration.index)
    previous.loc["cash"] = 1
    signal = {"credit_gate": gate, "credit_score": .987, "duration_target": target,
              **{f"carry_roll_{tenor}y_pctl": .75 for tenor in (5, 10, 30)}}
    weights = optimize_bond(signal, eligible, previous, cfg)
    assert weights.min() >= 0
    assert weights.sum() == pytest.approx(1.)
    assert weights["cgb30"] <= 1e-7
    assert weights[["credit", "short_cp"]].sum() <= cfg["credit_cap"] + 1e-6
    assert weights["credit"] <= cfg["credit_gate_cap" if gate else "credit_cap"] + 1e-6
    cash_cap = cfg["gated_cash_cap"] if gate else cfg["caps"]["cash"]
    assert weights["cash"] <= cash_cap + 1e-6
    expected_target = min(target, cfg["gated_duration_ceiling"]) if gate else target
    assert abs(weights @ duration - expected_target) <= cfg["duration_tolerance_years"] + 1e-6
    if gate:
        assert weights @ duration <= cfg["gated_duration_ceiling"] + 1e-6
