
from pathlib import Path

from .macro import macro_market
from .market import etf_market, exchange_calendar, load_curves, read_table
from .point_in_time import constituent_memberships
from src.utils.calendar import month_end_decisions


def prepare_data(raw: Path, processed: Path, config: dict) -> None:
    processed.mkdir(parents=True, exist_ok=True)
    calendar, index_close = exchange_calendar(raw, config)
    index_close.to_frame().to_parquet(processed / "csi300_close.parquet")
    for key, frame in etf_market(raw, calendar, config["data"]).items():
        frame.to_parquet(processed / f"etf_{key}.parquet")

    yields = load_curves(raw, config["bond"]["yield_columns"])
    yields.loc[yields["date"].le(config["dates"]["end"])].to_parquet(processed / "curves.parquet", index=False)
    macro_market(raw, calendar, config).to_parquet(processed / "macro_asof.parquet")
    decisions = month_end_decisions(calendar)
    membership = constituent_memberships(read_table(raw, "hs300_constituents_pit"), decisions)
    membership.loc[membership["date"].ge(config["dates"]["warmup_start"])].to_parquet(
        processed / "memberships.parquet", index=False,
    )
