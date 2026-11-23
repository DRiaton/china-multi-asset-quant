
from pathlib import Path

import numpy as np
import pandas as pd

YIELD_CURVES = ("CGB", "MTN_AAA", "BANK_AAA")


def read_table(raw: Path, name: str, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(raw / f"{name}.parquet", columns=columns)


def require_unique(frame: pd.DataFrame, keys: list[str], name: str) -> None:
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f"{name} has missing or duplicate keys: {keys}")


def exchange_calendar(raw: Path, config: dict) -> tuple[pd.DatetimeIndex, pd.Series]:
    index = read_table(raw, "indices_daily", ["date", "code", "close"])
    index = index.loc[index["code"].eq(config["data"]["index_code"])].sort_values("date")
    require_unique(index, ["date"], "CSI 300 reference")
    close = index.set_index("date")["close"].astype(float)
    close = close.loc[:config["dates"]["end"]].rename("csi300_close")
    if (~np.isfinite(close) | close.le(0)).any():
        raise ValueError("Bad CSI 300 reference closes")
    close.index = pd.DatetimeIndex(close.index, name="date").as_unit("ns")
    return close.index, close


def load_curves(raw: Path, columns: list[str]) -> pd.DataFrame:
    curves = read_table(raw, "chinabond_yields", ["date", "curve_id", *columns])
    require_unique(curves, ["date", "curve_id"], "ChinaBond curves")
    missing = set(YIELD_CURVES) - set(curves["curve_id"])
    if missing:
        raise ValueError(f"Missing required curves: {sorted(missing)}")
    # Raw yields are percentage points. Downstream models operate in decimals.
    curves[columns] = curves[columns].div(100).replace([np.inf, -np.inf], np.nan)
    return curves.sort_values(["date", "curve_id"]).reset_index(drop=True)


def etf_market(raw: Path, calendar: pd.DatetimeIndex, cfg: dict) -> dict[str, pd.DataFrame]:
    """Return observed NAV returns, trading availability, eligibility and audit fields."""
    nav = read_table(raw, "etf_nav_daily").sort_values(["code", "date"])
    bars = read_table(raw, "etf_daily", ["code", "date", "amount", "volume"])
    require_unique(nav, ["date", "code"], "ETF NAV")
    require_unique(bars, ["date", "code"], "ETF bars")
    nav["acc_return"] = nav["acc_nav"].where(nav["acc_nav"].gt(0)).groupby(nav["code"]).pct_change(fill_method=None)
    nav["source_return"] = nav["daily_ret_pct"].div(100)
    use_acc = nav["code"].isin(cfg["etf_acc_nav_codes"])
    nav["return"] = nav["source_return"].mask(use_acc, nav["acc_return"])
    # Missing official rates remain unobserved; no reconstruction from exchange prices.
    nav["return"] = nav["return"].where(np.isfinite(nav["return"]) & nav["return"].gt(-1))

    codes = list(cfg["etf_codes"].values())
    returns = nav.pivot(index="date", columns="code", values="return").reindex(index=calendar, columns=codes)
    latest_nav = nav.pivot(index="date", columns="code", values="acc_nav").reindex(index=calendar, columns=codes)
    liquid = bars.assign(tradable=bars["amount"].gt(0) & bars["volume"].gt(0))
    liquid = liquid.pivot(index="date", columns="code", values="tradable").reindex(index=calendar, columns=codes).eq(True)
    observed = latest_nav.gt(0)
    eligible = liquid.cumsum().ge(cfg["etf_seasoning_sessions"])
    return {
        "returns": returns,
        "tradable": liquid & observed & returns.notna(),
        "eligible": eligible,
        "nav_audit": nav[["date", "code", "return", "source_return", "acc_return"]],
    }
