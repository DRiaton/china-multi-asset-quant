
from pathlib import Path

import pandas as pd

from src.utils.calendar import next_session
from .market import read_table, require_unique

MONTHLY = {
    "pmi": ("macro_pmi", {"制造业-指数": "pmi"}),
    "cpi": ("macro_cpi_monthly", {"全国-同比增长": "cpi_yoy"}),
    "ppi": ("macro_ppi_monthly", {"当月同比增长": "ppi_yoy"}),
    "money": (
        "macro_money_supply",
        {"货币(M1)-同比增长": "m1_yoy", "货币和准货币(M2)-同比增长": "m2_yoy"},
    ),
    "social": ("macro_social_financing", {"社会融资规模增量": "social_flow"}),
}


def month_period(labels: pd.Series) -> pd.PeriodIndex:
    labels = labels.astype(str).str.strip()
    dates = labels.str.replace(r"^(\d{4})年(\d{1,2})月份?$", r"\1-\2", regex=True)
    dates = dates.str.replace(r"^(\d{4})(\d{2})$", r"\1-\2", regex=True)
    if not dates.str.fullmatch(r"\d{4}-(0?[1-9]|1[0-2])").all():
        raise ValueError("Unrecognized monthly labels")
    return pd.PeriodIndex(dates, freq="M")


def quarter_period(labels: pd.Series) -> pd.PeriodIndex:
    parts = labels.astype(str).str.extract(r"^(\d{4})年第(?:1-)?([1-4])季度$")
    if parts.isna().any().any():
        raise ValueError("Unrecognized GDP quarter labels")
    return pd.PeriodIndex(parts[0] + "Q" + parts[1], freq="Q-DEC")


def align_publications(observations: pd.DataFrame, calendar: pd.DatetimeIndex, offset_days: int) -> pd.DataFrame:
    """Make observations usable at the first exchange session after period-end plus the lag."""
    if observations.index.has_duplicates:
        raise ValueError("Duplicate macro observation period")
    obs = observations.sort_index().copy()
    obs.insert(0, "period_end", obs.index.to_timestamp(how="end").normalize())
    obs.index = next_session(calendar, obs["period_end"] + pd.to_timedelta(offset_days, unit="D"))
    obs = obs.loc[obs.index.notna()].loc[lambda x: ~x.index.duplicated(keep="last")]
    obs.insert(1, "effective_date", obs.index)
    return obs.reindex(calendar, method="ffill")


def align_daily_close(observations: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """At a decision date, show the newest observation strictly before that session."""
    obs = observations.sort_index().copy()
    if obs.index.has_duplicates:
        raise ValueError("Duplicate daily observation date")
    obs.insert(0, "source_date", obs.index)
    obs.index = next_session(calendar, obs.index)
    obs = obs.loc[obs.index.notna()].loc[lambda x: ~x.index.duplicated(keep="last")]
    return obs.reindex(calendar, method="ffill")


def macro_market(raw: Path, calendar: pd.DatetimeIndex, config: dict) -> pd.DataFrame:
    cfg = config["data"]
    lags = cfg["macro_lag_days"]
    daily = pd.DataFrame(index=calendar)
    for key, (table, renames) in MONTHLY.items():
        frame = read_table(raw, table, ["月份", *renames])
        monthly = frame.rename(columns=renames).drop(columns="月份")
        monthly.index = month_period(frame["月份"])
        monthly = monthly.sort_index()
        if key == "pmi":
            monthly["pmi_mom_3m"] = monthly["pmi"] - monthly["pmi"].reindex(monthly.index - 3).to_numpy()
        if key == "money":
            monthly["m1_m2_gap"] = monthly["m1_yoy"] - monthly["m2_yoy"]
            monthly["m1_m2_change"] = monthly["m1_m2_gap"] - monthly["m1_m2_gap"].reindex(monthly.index - 3).to_numpy()
        if key == "social":
            flow = monthly["social_flow"]
            trailing_flow = flow.rolling(12, min_periods=12).sum()
            monthly["social_flow_yoy"] = trailing_flow / trailing_flow.reindex(monthly.index - 12).to_numpy() - 1
        publication = align_publications(monthly, calendar, lags[key])
        daily = daily.join(publication.rename(columns={
            "period_end": f"{key}_period_end", "effective_date": f"{key}_effective_date",
        }))

    gdp = read_table(raw, "macro_gdp", ["季度", "国内生产总值-同比增长"])
    gdp_values = gdp[["国内生产总值-同比增长"]].rename(columns={"国内生产总值-同比增长": "gdp_ytd_yoy"})
    gdp_values.index = quarter_period(gdp["季度"])
    gdp_aligned = align_publications(gdp_values, calendar, lags["gdp"])
    daily = daily.join(gdp_aligned.add_prefix("gdp_").rename(columns={"gdp_gdp_ytd_yoy": "gdp_ytd_yoy"}))

    shibor = read_table(raw, "macro_shibor", ["日期", "1W-定价", "3M-定价"])
    shibor["日期"] = pd.to_datetime(shibor["日期"], format="%Y-%m-%d")
    shibor = shibor.set_index("日期").rename(columns={"1W-定价": "shibor_1w", "3M-定价": "shibor_3m"}).div(100)
    daily = daily.join(align_daily_close(shibor, calendar).rename(columns={"source_date": "shibor_source_date"}))
    daily["shibor_age_days"] = (daily.index.to_series() - daily["shibor_source_date"]).dt.days

    valuation = read_table(raw, "hs300_valuation", ["date", "pe_ttm", "pb"]).sort_values("date")
    require_unique(valuation, ["date"], "CSI 300 PE")
    published = pd.merge_asof(
        pd.DataFrame({"date": calendar}),
        valuation.rename(columns={"date": "index_valuation_date"}).astype({"index_valuation_date": "datetime64[ns]"}),
        left_on="date",
        right_on="index_valuation_date",
        direction="backward",
        tolerance=pd.Timedelta(days=cfg["index_pe_max_age_days"]),
    ).set_index("date")
    return daily.join(published.rename(columns={"pe_ttm": "index_pe_ttm", "pb": "index_pb"}))
