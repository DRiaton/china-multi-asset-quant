
import numpy as np
import pandas as pd


def next_session(calendar: pd.DatetimeIndex, dates, *, strict: bool = True) -> pd.DatetimeIndex:
    """Map dates to the first observed exchange session strictly after or on/after them."""
    dates = pd.DatetimeIndex(dates)
    positions = calendar.searchsorted(dates, side="right" if strict else "left")
    result = np.full(len(positions), np.datetime64("NaT"), dtype="datetime64[ns]")
    valid = (positions < len(calendar)) & dates.notna()
    result[valid] = calendar.to_numpy()[positions[valid]].astype("datetime64[ns]")
    return pd.DatetimeIndex(result, name="effective_date")


def month_end_decisions(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Only return ends of months whose next exchange session is observed."""
    months = calendar.to_period("M")
    return calendar[:-1][months[:-1] != months[1:]]


def execution_map(calendar: pd.DatetimeIndex, decisions: pd.DatetimeIndex, lag: int = 1) -> pd.Series:
    if lag < 1:
        raise ValueError("Close signals must execute at least one session later")
    positions = calendar.get_indexer(decisions)
    if (positions < 0).any():
        raise ValueError("Decision dates must be trading sessions")
    valid = positions + lag < len(calendar)
    return pd.Series(calendar[positions[valid] + lag], index=decisions[valid], name="execution_date")
