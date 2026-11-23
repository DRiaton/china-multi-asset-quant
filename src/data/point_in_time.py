
import numpy as np
import pandas as pd

from .market import require_unique


def backward_by_code(bars: pd.DataFrame, valuations: pd.DataFrame, max_age_days: int) -> pd.DataFrame:
    """Backward as-of join with date-first ordering required by pandas."""
    require_unique(valuations, ["date", "code"], "equity valuation")
    valuations = valuations.rename(columns={"date": "valuation_date"}).copy()
    left = bars.copy()
    left["date"] = left["date"].astype("datetime64[ns]")
    valuations["valuation_date"] = valuations["valuation_date"].astype("datetime64[ns]")
    return pd.merge_asof(
        left.sort_values(["date", "code"]),
        valuations.sort_values(["valuation_date", "code"]),
        left_on="date",
        right_on="valuation_date",
        by="code",
        direction="backward",
        tolerance=pd.Timedelta(days=max_age_days),
    )


def constituent_memberships(snapshots: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Use the latest effective membership snapshot, retaining validation/provenance dates."""
    require_unique(snapshots, ["query_date", "code"], "CSI 300 constituent snapshots")
    groups = snapshots.groupby("query_date", sort=True)
    if not groups["code"].size().eq(300).all() or not groups["update_date"].nunique().eq(1).all():
        raise ValueError("Expected 300 distinct constituents and one effective date per query")
    history = {}
    for query, snapshot in groups:
        effective = pd.Timestamp(snapshot["update_date"].iloc[0])
        members = frozenset(snapshot["code"])
        if effective > query:
            raise ValueError("Constituent update is after its historical query")
        if effective in history:
            if history[effective][1] != members:
                raise ValueError(f"Conflicting CSI 300 membership for {effective:%Y-%m-%d}")
            continue
        history[effective] = (query, members)

    effective_dates = pd.DatetimeIndex(sorted(history))
    rows = []
    for day, position in zip(dates, effective_dates.searchsorted(dates, side="right") - 1):
        if position < 0:
            continue
        update = effective_dates[position]
        query, members = history[update]
        rows.extend((day, code, update, query) for code in sorted(members))
    return pd.DataFrame(rows, columns=["date", "code", "update_date", "source_query_date"])


def point_in_time_industry(rows: pd.DataFrame, industry: pd.DataFrame) -> pd.Series:
    """Attach only historically dated labels; unsupported classifications are unknown."""
    labels = industry[["code", "updateDate", "industry"]].copy()
    labels["updateDate"] = pd.to_datetime(labels["updateDate"], errors="raise").astype("datetime64[ns]")
    labels = labels.dropna(subset=["updateDate"])
    require_unique(labels, ["code", "updateDate"], "industry labels")
    order = rows[["date", "code"]].copy()
    order["date"] = order["date"].astype("datetime64[ns]")
    order["_row"] = np.arange(len(order))
    merged = pd.merge_asof(
        order.sort_values(["date", "code"]),
        labels.sort_values(["updateDate", "code"]),
        left_on="date",
        right_on="updateDate",
        by="code",
        direction="backward",
    ).sort_values("_row")
    sectors = merged["industry"].str.extract(r"^([A-Z])", expand=False).fillna("Unknown")
    return pd.Series(sectors.to_numpy(), index=rows.index, name="industry")


def static_industry_proxy(rows: pd.DataFrame, industry: pd.DataFrame) -> pd.Series:
    """Map the latest CSRC label as a fixed research taxonomy with no effective-date claim."""
    snapshot = industry[["code", "updateDate", "industry"]].copy()
    snapshot["updateDate"] = pd.to_datetime(snapshot["updateDate"], errors="raise")
    snapshot = snapshot.dropna(subset=["updateDate"])
    require_unique(snapshot, ["code", "updateDate"], "static industry proxy")
    latest = snapshot.sort_values("updateDate").drop_duplicates("code", keep="last").set_index("code")
    sectors = latest["industry"].str.extract(r"^([A-Z])", expand=False)
    return rows["code"].map(sectors).fillna("Unknown").rename("industry")


def group_industries(sectors: pd.Series, minimum: int) -> pd.Series:
    """Group sparse classified sectors while preserving an explicit unknown bucket."""
    counts = sectors.value_counts()
    small = sectors.ne("Unknown") & sectors.map(counts).lt(minimum)
    return sectors.mask(small, "OTHER")
