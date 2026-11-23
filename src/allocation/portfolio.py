
from collections import deque

import numpy as np
import pandas as pd

from src.backtest.portfolio import CASH, Portfolio, etf_rates, result_frames
from src.utils.calendar import execution_map
from .optimizers import equal_risk_contribution, hrp, optimize_bl, shrinkage_covariance

CORE_STRATEGIES = ("Static_20_80", "Static_60_40", "Risk_Parity_ERC", "HRP", "ERP_Tactical_20_80", "Macro_BL_FixedIncomePlus")
ABLATION = ("Static_20_80", "Macro_BL_Passive", "Macro_BL_Bond_Active", "Macro_BL_Equity_Bond_Active",
            "Macro_BL_FixedIncomePlus", "Macro_BL_Satellites")


def allocation_order(kind, previous, day, macro, bond, covariance_returns, eligible, config, *,
                     equity_asset, bond_asset, satellites, balanced, delta, tau):
    cfg = config["allocation"]
    base = [equity_asset, bond_asset]
    bounds = cfg["balanced_bounds"] if balanced else cfg["tactical_bounds"]
    diagnostics = {}
    if kind == "static20":
        equity = cfg["strategic_equity_weight"]
    elif kind == "static60":
        equity = cfg["static_balanced_equity_weight"]
    elif kind == "erp":
        percentile = np.nan_to_num(macro.get("erp_pct_3y", np.nan), nan=0.5)
        equity = bounds[0] + percentile * (bounds[1] - bounds[0])
    else:
        assets = base.copy()
        if kind == "bl" and satellites:
            for satellite in cfg["satellite_caps"]:
                history = covariance_returns.loc[:day, satellite].tail(cfg["cov_window"])
                if eligible.at[day, satellite] and history.notna().sum() >= cfg["cov_min"]:
                    assets.append(satellite)
        cov = shrinkage_covariance(covariance_returns[assets], day, cfg, config["dates"]["periods_per_year"])
        if kind == "bl":
            return optimize_bl(
                assets, cov, previous, macro, bond, cfg, bounds, delta=delta, tau=tau,
                strategic_equity_weight=cfg["balanced_strategic_equity_weight"] if balanced else cfg["strategic_equity_weight"],
            )
        weights = equal_risk_contribution(cov) if kind == "erc" else hrp(cov)
        diagnostics["equity_risk_contribution"] = weights[0] * (cov @ weights)[0] / (weights @ cov @ weights)
        return pd.Series(weights, index=assets), diagnostics
    return pd.Series([equity, 1 - equity], index=base), diagnostics


class TrailingDrawdown:
    """Rolling high-water mark on realized close NAV, including the initial NAV until it expires."""

    def __init__(self, window, initial_nav=1.0):
        self.levels = deque([initial_nav], maxlen=window)

    def update(self, nav):
        self.levels.append(nav)
        return 1 - nav / max(self.levels)


def overlay(base, risk_history, equity_asset, drawdown, prev_scale, config, *,
            prev_equity_scale=1.0, bond_asset=None, target_vol=None):
    cfg = config["allocation"]
    bond_asset = bond_asset or next(asset for asset in base.index if asset.startswith("bond"))
    target_vol = cfg["target_vol"] if target_vol is None else target_vol
    exposure = risk_history.reindex(columns=base.index).fillna(0).to_numpy() @ base.to_numpy()
    volatility = np.std(exposure, ddof=1) * np.sqrt(config["dates"]["periods_per_year"]) if len(exposure) > 1 else 0.0
    scale = min(1, target_vol / volatility) if volatility > 0 else 1
    scale = min(scale, prev_scale + cfg["overlay_recovery_step"])
    first, second = cfg["drawdown_thresholds"]
    mild, severe = cfg["drawdown_scale"]
    equity_factor = severe if drawdown >= second else mild if drawdown >= first else 1.0
    equity_factor = min(equity_factor, prev_equity_scale + cfg["overlay_recovery_step"])

    risk_assets = [asset for asset in (equity_asset, "convertible") if asset in base.index]
    desired = base.copy()
    desired.loc[risk_assets] *= scale * equity_factor
    released = base.loc[risk_assets].sum() - desired.loc[risk_assets].sum()
    cash_fraction = cfg["overlay_release_to_cash"]
    desired["cash"] = base.get("cash", 0.0) + cash_fraction * released
    desired[bond_asset] += (1 - cash_fraction) * released
    return desired, scale, volatility, equity_factor


def simulate_allocation(market, features, bond, config, *, kind, equity_asset="equity_etf", bond_asset="bond_passive",
                        use_overlay=False, satellites=False, balanced=False, delta=None, tau=None, cost_multiplier=1.0):
    cfg = config["allocation"]
    delta = cfg["risk_aversion"] if delta is None else delta
    tau = cfg["tau"] if tau is None else tau
    ret, gross, child_cost, child_turnover, tradable, eligible, covariance_returns = (
        market[key] for key in ("net", "gross", "cost", "turnover", "tradable", "eligible", "covariance")
    )
    assets = pd.Index([equity_asset, bond_asset, "cash", *(cfg["satellite_caps"] if satellites else [])])
    ret, gross, child_cost, child_turnover, tradable = (
        frame.reindex(columns=assets) for frame in (ret, gross, child_cost, child_turnover, tradable)
    )
    dates = ret.loc[config["dates"]["start"]:config["dates"]["end"]].index
    initial = ret.index[ret.index.get_loc(dates[0]) - 1]
    monthly = set(features.index)
    next_date = execution_map(ret.index, ret.index)
    account = Portfolio(assets)
    peak, scaling, equity_scale = 1.0, 1.0, 1.0
    trailing = TrailingDrawdown(cfg["drawdown_window"])
    target_vol = cfg["balanced_target_vol"] if balanced else cfg["target_vol"]
    entries, starts, ends, orders = [], [], [], []

    def decide(day):
        base, detail = allocation_order(
            kind, account.positions, day, features.loc[:day].iloc[-1], bond.loc[day], covariance_returns, eligible, config,
            equity_asset=equity_asset, bond_asset=bond_asset, satellites=satellites, balanced=balanced, delta=delta, tau=tau,
        )
        return base.reindex(assets, fill_value=0), detail

    def send(day, order, reason, detail):
        orders.append({
            "decision_date": day, "execution_date": next_date.at[day], "reason": reason, **detail,
            **order.to_dict(),
        })
        return order

    def risk_history(day):
        return covariance_returns.loc[:day].tail(cfg["vol_window"])

    base, detail = decide(initial)
    desired = base
    if use_overlay:
        desired, scaling, volatility, equity_scale = overlay(
            base, risk_history(initial), equity_asset, 0, scaling, config, bond_asset=bond_asset, target_vol=target_vol,
        )
        detail.update(scale=scaling, forecast_volatility=volatility, drawdown_equity_scale=equity_scale, trailing_drawdown=0.0)
    pending = last_sent = send(initial, desired, "monthly", detail)
    sent_scale, sent_equity_scale = scaling, equity_scale

    rates = etf_rates(config["costs"], cost_multiplier)
    net_values = ret.loc[dates].to_numpy()
    gross_values = gross.loc[dates].fillna(0).to_numpy()
    costs = child_cost.loc[dates].fillna(0).to_numpy()
    turnover = child_turnover.loc[dates].fillna(0).to_numpy()
    trade_values = tradable.loc[dates].to_numpy()
    for i, day in enumerate(dates):
        stats, beginning, ending = account.step(net_values[i], trade_values[i], pending, rates)
        stats["top_cost"] = stats["cost"]
        stats["nested_cost"] = beginning[:-1] @ costs[i]
        stats["cost"] += stats["nested_cost"]
        stats["nested_turnover"] = beginning[:-1] @ turnover[i]
        stats["gross_return"] = beginning[:-1] @ gross_values[i]
        peak = max(peak, account.nav)
        drawdown = 1 - account.nav / peak
        recent_drawdown = trailing.update(account.nav)
        stats.update(
            drawdown=drawdown, trailing_drawdown=recent_drawdown,
            overlay_scale_ordered=sent_scale, overlay_equity_scale_ordered=sent_equity_scale,
        )
        entries.append(stats)
        starts.append(beginning)
        ends.append(ending)
        pending = None
        if day not in next_date.index:
            continue
        refresh = day in monthly
        detail = {}
        if refresh:
            base, detail = decide(day)
        if use_overlay:
            desired, scaling, volatility, equity_scale = overlay(
                base, risk_history(day), equity_asset, recent_drawdown, scaling, config, prev_equity_scale=equity_scale,
                bond_asset=bond_asset, target_vol=target_vol,
            )
            detail.update(
                scale=scaling, forecast_volatility=volatility, drawdown_equity_scale=equity_scale,
                trailing_drawdown=recent_drawdown,
            )
        else:
            desired = base
        if refresh or (use_overlay and (desired - last_sent).abs().sum() / 2 > cfg["overlay_trade_band"]):
            pending = last_sent = send(day, desired, "monthly" if refresh else "risk_overlay", detail)
            sent_scale, sent_equity_scale = scaling, equity_scale

    daily, start, end = result_frames(dates, assets, entries, starts, ends)
    close_fees = daily["cost"]
    if not np.allclose(daily["net_return"], daily["gross_return"] - close_fees, atol=1e-12):
        raise RuntimeError("Nested sleeve and allocation costs did not reconcile")
    return daily, start, end, pd.DataFrame(orders)


def classify_contributions(weights, leg_gross, classes):
    out_weights, out_contribution = {}, {}
    for category, assets in classes.items():
        existing = [asset for asset in assets if asset in weights.columns]
        out_weights[category] = weights[existing].sum(axis=1)
        moved = [asset for asset in existing if asset != CASH]
        out_contribution[category] = (weights[moved] * leg_gross[moved].fillna(0)).sum(axis=1)
    return pd.DataFrame(out_weights), pd.DataFrame(out_contribution)
