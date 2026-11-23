"""Read-only audit of the locally supplied research raw data."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


KEYS = {
    "chinabond_yields": ["curve_id", "date"],
    "equity_daily_hfq": ["code", "date"],
    "equity_valuation_baidu": ["code", "date"],
    "etf_daily": ["code", "date"],
    "etf_nav_daily": ["code", "date"],
    "hs300_constituents_pit": ["update_date", "code"],
    "hs300_valuation": ["date"],
    "indices_daily": ["code", "date"],
    "macro_cpi_monthly": ["月份"],
    "macro_gdp": ["季度"],
    "macro_money_supply": ["月份"],
    "macro_pmi": ["月份"],
    "macro_ppi_monthly": ["月份"],
    "macro_shibor": ["日期"],
    "macro_social_financing": ["月份"],
    "stock_industry": ["code"],
}


def date_stats(values, column, table):
    raw = values.astype("string")
    freq = None
    if column == "季度":
        groups = raw.str.extract(r"^(\d{4})年第(?:1-)?([1-4])季度$")
        parsed = groups[0] + "Q" + groups[1]
        periods = pd.PeriodIndex(parsed, freq="Q")
        freq = "Q"
    elif column == "月份":
        parts = (raw.str.extract(r"^(\d{4})(\d{2})$")
                 if table == "macro_social_financing"
                 else raw.str.extract(r"^(\d{4})年(\d{2})月份$"))
        parsed = pd.to_datetime(parts[0] + parts[1], format="%Y%m", errors="coerce")
        periods = pd.PeriodIndex(parsed, freq="M")
        freq = "M"
    else:
        periods = pd.to_datetime(values, format="%Y-%m-%d", errors="coerce")
    info = {
        "min": str(periods.min()),
        "max": str(periods.max()),
        "unique": int(pd.Series(periods).nunique()),
        "invalid_or_missing": int(pd.isna(periods).sum()),
        "raw_examples": list(raw.dropna().drop_duplicates().iloc[[0, -1]].unique()),
        "monotonically_increasing": bool(periods.is_monotonic_increasing),
    }
    if freq:
        idx = pd.PeriodIndex(periods).dropna()
        info["missing_periods"] = list(
            pd.period_range(idx.min(), idx.max(), freq=freq).difference(idx).astype(str)
        )
    return info


def stats(frame, parquet, expected, name):
    info = {
        "rows": len(frame),
        "arrow_schema": {field.name: str(field.type) for field in parquet.schema_arrow},
        "manifest_rows_match": len(frame) == expected["rows"],
        "manifest_columns_match": list(frame.columns) == expected["cols"],
        "key": KEYS[name],
        "duplicate_key_extra_rows": int(frame.duplicated(KEYS[name]).sum()),
        "missing": {col: int(value) for col, value in frame.isna().sum().items()},
        "blank": {},
        "numeric": {},
        "dates": {},
    }
    for col, values in frame.items():
        if col in {"date", "日期", "月份", "季度", "query_date", "update_date", "updateDate"}:
            info["dates"][col] = date_stats(values, col, name)
        if pd.api.types.is_numeric_dtype(values):
            shifted = values.to_numpy(dtype="float64", na_value=np.nan)
            finite = shifted[np.isfinite(shifted)]
            info["numeric"][col] = {
                "nonfinite_excluding_null": int(np.isinf(shifted).sum()),
                "zero": int((finite == 0).sum()),
                "negative": int((finite < 0).sum()),
                "min": float(finite.min()) if len(finite) else None,
                "p01": float(np.quantile(finite, .01)) if len(finite) else None,
                "p99": float(np.quantile(finite, .99)) if len(finite) else None,
                "max": float(finite.max()) if len(finite) else None,
            }
        elif pd.api.types.is_string_dtype(values.dtype):
            info["blank"][col] = int(values.str.strip().eq("").sum())
    if "code" in frame:
        info["unique_codes"] = int(frame.code.nunique())
        info["manifest_codes_match"] = info["unique_codes"] == expected.get("n_unique_codes")
    if "date_min" in expected:
        bounds = info["dates"]["date"]
        info["manifest_dates_match"] = all(
            str(pd.Timestamp(bounds[key]).date()) == expected["date_" + key]
            for key in ("min", "max")
        )
    if {"open", "high", "low", "close"} <= set(frame):
        maximum = frame[["open", "low", "close"]].max(axis=1)
        minimum = frame[["open", "high", "close"]].min(axis=1)
        info["ohlc_inconsistency"] = {
            "high_below_other": int(frame.high.lt(maximum).sum()),
            "low_above_other": int(frame.low.gt(minimum).sum()),
        }
    return info


def frame_records(frame):
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def nav_details(frame, tape, calendar, oos_start):
    nav = frame.sort_values(["code", "date"]).copy()
    for col in ("acc_nav", "unit_nav"):
        nav[col + "_ret_pct"] = nav.groupby("code")[col].pct_change(fill_method=None).mul(100)
    sets = {}
    for code, g in nav.groupby("code"):
        price = tape[tape.code.eq(code)].sort_values("date")
        sessions = calendar[(calendar >= g.date.min()) & (calendar <= g.date.max())]
        discrepancy = g.acc_nav_ret_pct.sub(g.daily_ret_pct).abs()
        large = g.daily_ret_pct.abs().gt(10) | g.acc_nav_ret_pct.abs().gt(10) | g.unit_nav_ret_pct.abs().gt(10)
        taped = pd.DatetimeIndex(price.date)
        after_start = taped[taped >= oos_start]
        sets[code] = {
            "tape_trade_60": str(taped[59].date()),
            "tape_trade_61": str(taped[60].date()),
            "oos_tape_dates_missing_nav": len(after_start.difference(g.date)),
            "oos_tape_dates_missing_provider_return": int(
                (g.date.isin(after_start) & g.daily_ret_pct.isna()).sum()
            ),
            "nav_first": str(g.date.min().date()),
            "tape_first": str(price.date.min().date()),
            "nav_last": str(g.date.max().date()),
            "nav_non_calendar_rows": len(pd.DatetimeIndex(g.date).difference(calendar)),
            "missing_calendar_rows_from_first_nav": len(sessions.difference(g.date)),
            "missing_calendar_examples": list(sessions.difference(g.date).astype(str)[:10]),
            "tape_dates_without_nav_since_first_nav": len(pd.DatetimeIndex(price.date).difference(g.date).intersection(sessions)),
            "discrepancy_gt_0_1_percentage_point": int(discrepancy.gt(.1).sum()),
            "discrepancy_gt_1_percentage_point": int(discrepancy.gt(1).sum()),
            "max_discrepancy_percentage_points": float(discrepancy.max()),
            "large_nav_move_examples": frame_records(g.loc[large].head(20)),
            "missing_return_examples": frame_records(g.loc[g.daily_ret_pct.isna()].head(10)),
        }
    return sets


def inspect(raw):
    manifest = json.loads((raw / "manifest.json").read_text(encoding="utf-8"))
    actual_files = sorted(path.name for path in raw.glob("*.parquet"))
    report = {
        "manifest_generated_at": manifest["generated_at"],
        "manifest_files_match": sorted(manifest["files"]) == actual_files,
        "tables": {},
    }
    frames = {}
    for filename in actual_files:
        name = Path(filename).stem
        parquet = pq.ParquetFile(raw / filename)
        frame = parquet.read().to_pandas()
        report["tables"][name] = stats(frame, parquet, manifest["files"][filename], name)
        frames[name] = frame
    calendar = pd.DatetimeIndex(frames["indices_daily"].loc[lambda x: x.code.eq("000300"), "date"]).sort_values()
    oos_start = pd.Timestamp("2018-01-02")
    report["nav_checks"] = nav_details(frames["etf_nav_daily"], frames["etf_daily"], calendar, oos_start)
    report["bond_30y_missing_by_curve"] = {
        name: int(curve.y_30y.isna().sum())
        for name, curve in frames["chinabond_yields"].groupby("curve_id")
    }
    report["index_coverage"] = frame_records(
        frames["indices_daily"].groupby(["code", "alias"]).date.agg(["min", "max", "size"]).reset_index()
    )
    pit = frames["hs300_constituents_pit"]
    counts = pit.groupby(["query_date", "update_date"]).size()
    memberships = pit.groupby(["update_date", "query_date"]).code.agg(frozenset)
    repeated = memberships.groupby(level="update_date").agg(["size", "nunique"])
    ind = frames["stock_industry"]
    report["pit_checks"] = {
        "unique_effective_dates": int(pit.update_date.nunique()),
        "repeated_effective_dates": repeated.index[repeated["size"].gt(1)].strftime("%Y-%m-%d").tolist(),
        "conflicting_sets_for_same_effective_date": int(repeated["nunique"].gt(1).sum()),
        "conflicting_names_for_same_effective_date_and_code": int(
            pit.groupby(["update_date", "code"]).code_name.nunique().gt(1).sum()
        ),
        "snapshot_sizes": frame_records(counts.rename("size").reset_index()),
        "query_before_update": int(pit.query_date.lt(pit.update_date).sum()),
        "query_minus_update_days": sorted((pit.query_date - pit.update_date).dt.days.unique().tolist()),
        "industry_missing_for_pit_codes": len(set(pit.code).difference(ind.code)),
        "industry_blank_for_pit_codes": int(ind.loc[ind.code.isin(pit.code), "industry"].str.strip().eq("").sum()),
        "equity_missing_for_pit_codes": len(set(pit.code).difference(frames["equity_daily_hfq"].code)),
        "valuation_missing_for_pit_codes": len(set(pit.code).difference(frames["equity_valuation_baidu"].code)),
    }
    for name, date_key in [("equity_daily_hfq", "date"), ("equity_valuation_baidu", "date"),
                           ("macro_shibor", "日期"), ("chinabond_yields", "date")]:
        frame = frames[name]
        days = pd.DatetimeIndex(pd.to_datetime(frame[date_key]).unique())
        within = calendar[(calendar >= days.min()) & (calendar <= days.max())]
        report["tables"][name]["calendar_days_missing"] = len(within.difference(days))
        report["tables"][name]["calendar_days_missing_oos"] = len(
            within[within >= pd.Timestamp("2018-01-02")].difference(days)
        )
        report["tables"][name]["off_calendar_dates"] = len(days.difference(calendar))
        report["tables"][name]["calendar_dates_after_last_observation"] = (
            calendar[calendar > days.max()].strftime("%Y-%m-%d").tolist()
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = inspect(args.raw)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

