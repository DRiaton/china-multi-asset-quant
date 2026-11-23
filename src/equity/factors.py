
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.market import read_table, require_unique
from src.data.point_in_time import backward_by_code, group_industries, point_in_time_industry, static_industry_proxy


def price_factors(close, high, volume, amount, turnover, benchmark, cfg):
    w = cfg["windows"]
    long, medium, skip = w["mom_long"], w["mom_medium"], w["mom_skip"]
    short, volatility, downside = w["short"], w["volatility"], w["downside"]
    returns = close.pct_change(fill_method=None)
    market = benchmark.pct_change(fill_method=None)
    variance = returns.rolling(volatility).var()
    beta_cov = returns.rolling(volatility).cov(market)
    market_var = market.rolling(volatility).var().replace(0, np.nan)
    residual_var = (variance - beta_cov.pow(2).div(market_var, axis=0)).clip(lower=0)
    bear = market.where(market.lt(0))
    down_cov = returns.rolling(downside, min_periods=cfg["min_down_days"]).cov(bear)
    down_var = bear.rolling(downside, min_periods=cfg["min_down_days"]).var().replace(0, np.nan)
    turn_short = turnover.where(turnover.gt(0)).rolling(short).mean()
    turn_long = turnover.where(turnover.gt(0)).rolling(downside).mean()
    return {
        "mom_12_1": close.shift(skip) / close.shift(long) - 1,
        "mom_6_1": close.shift(skip) / close.shift(medium) - 1,
        "rev_20d": -(close / close.shift(short) - 1),
        "dist_52w_high": close / high.rolling(long, min_periods=long * 4 // 5).max(),
        "real_vol_60d": -np.sqrt(variance),
        "idio_vol_60d": -np.sqrt(residual_var),
        "downside_beta_120d": -down_cov.div(down_var, axis=0),
        "turnover_20d": -np.log(turn_short),
        "abn_turnover": -turn_short / turn_long,
        "amihud_illiq_20d": returns.abs().div(amount.where(amount.gt(0))).rolling(short).mean() * 1e8,
        "vol_price_corr_20d": -close.rolling(short).corr(np.log(volume.where(volume.gt(0)))),
    }


def fundamental_factors(frame: pd.DataFrame) -> pd.DataFrame:
    pe = frame["pe_ttm"].astype(float).where(frame["pe_ttm"].gt(0))
    pb = frame["pb"].astype(float).where(frame["pb"].gt(0))
    pcf = frame["pcf"].astype(float).where(frame["pcf"].gt(0))
    return frame.assign(
        ep_ttm=1 / pe,
        bp=1 / pb,
        cfp=1 / pcf,
        implied_roe=pb / pe,
        cash_earnings_quality=pe / pcf,
    )


def build_factor_inputs(raw: Path, memberships: pd.DataFrame, benchmark: pd.Series, config: dict):
    bars = read_table(raw, "equity_daily_hfq")
    require_unique(bars, ["date", "code"], "equity bars")
    bars = bars.loc[bars["date"].le(benchmark.index[-1])].copy()
    calendar = benchmark.index[benchmark.index >= bars["date"].min()]
    codes = pd.Index(sorted(bars["code"].unique()), name="code")

    def wide(name: str) -> pd.DataFrame:
        values = bars.pivot(index="date", columns="code", values=name).reindex(index=calendar, columns=codes)
        return values.astype(float).where(values.gt(0))

    open_price, raw_close, high = map(wide, ("open", "close", "high"))
    volume, amount, turnover = map(wide, ("volume", "amount", "turnover"))
    trade_at_close = raw_close.notna() & open_price.notna() & volume.notna()
    # Carry the last observable mark over absent sessions. There is no synthetic exit fill.
    marked_close = raw_close.ffill()
    returns = marked_close.pct_change(fill_method=None).where(raw_close.notna())
    valuations = read_table(raw, "equity_valuation_baidu")
    inputs = fundamental_factors(backward_by_code(
        memberships, valuations, config["data"]["valuation_max_age_days"],
    ))
    dates = calendar.get_indexer(inputs["date"])
    stocks = codes.get_indexer(inputs["code"])
    if min(dates.min(), stocks.min()) < 0:
        raise ValueError("An index member has no associated equity calendar/assets")
    inputs["decision_tradable"] = trade_at_close.to_numpy()[dates, stocks]
    industry = read_table(raw, "stock_industry")
    inputs["industry_pit"] = point_in_time_industry(inputs, industry)
    inputs["industry_proxy"] = static_industry_proxy(inputs, industry)
    cfg = config["equity"]
    use_proxy = cfg.get("use_static_industry_proxy", False)
    inputs["industry"] = inputs["industry_proxy"] if use_proxy else inputs["industry_pit"]
    inputs["industry_proxy_used"] = use_proxy & inputs["industry_proxy"].ne("Unknown")
    inputs["model_industry"] = inputs.groupby("date")["industry"].transform(
        lambda sectors: group_industries(sectors, cfg["min_industry_count"]),
    )
    for name, values in price_factors(
        marked_close, high, volume, amount, turnover, benchmark.reindex(calendar), config["equity"],
    ).items():
        inputs[name] = values.to_numpy()[dates, stocks]
    return inputs, returns, trade_at_close, marked_close


def neutralize(day: pd.DataFrame, factors: list[str], cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    result = day.copy()
    if "model_industry" not in result:
        result["model_industry"] = group_industries(result["industry"], cfg["min_industry_count"])
    sectors = result["model_industry"]
    log_size = np.log(result["total_mv_100m"].astype(float).where(result["total_mv_100m"].gt(0)))
    log_size = log_size.fillna(log_size.groupby(sectors).transform("median")).fillna(log_size.median()).fillna(0.0)
    log_size = (log_size - log_size.mean()) / max(float(log_size.std(ddof=0)), 1e-10)
    design = np.column_stack([np.ones(len(result)), log_size, pd.get_dummies(sectors, drop_first=True, dtype=float)])
    missing = result[factors].isna().mean()
    for name in factors:
        value = result[name].astype(float).replace([np.inf, -np.inf], np.nan)
        center = value.median()
        dispersion = (value - center).abs().median()
        if np.isfinite(dispersion) and dispersion > 0:
            cutoff = cfg["mad_multiple"] * dispersion
            value = value.clip(center - cutoff, center + cutoff)
        value = value.fillna(value.groupby(sectors).transform("median")).fillna(value.median()).fillna(0.0)
        residual = value.to_numpy() - design @ np.linalg.lstsq(design, value, rcond=None)[0]
        std = np.std(residual)
        result[name] = (residual - np.mean(residual)) / std if std > 1e-10 else 0.0
    return result, missing
