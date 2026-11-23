"""
Reproducible open-source data acquisition script for the China Multi-Asset
(Equity Multi-Factor + Fixed Income Yield Curve/Duration/Credit + Macro Regime)
quantitative research project.

All data sources are 100% free and publicly accessible (AkShare + ChinaBond +
Sina + Baidu Valuation + Baostock Point-in-Time CSI 300 constituents), enabling
full zero-cost reproducibility on GitHub.

Usage:
    python scripts/fetch_data.py --macro-bond
    python scripts/fetch_data.py --equity --workers 8
    python scripts/fetch_data.py --all --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Clear proxy env vars so domestic Chinese financial endpoints connect directly
for _k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
    os.environ.pop(_k, None)

import akshare as ak
import baostock as bs
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


ETF_UNIVERSE = {
    "511010": ("sh511010", "CGB_5Y_ETF", "bond_rate_mid"),
    "511260": ("sh511260", "CGB_10Y_ETF", "bond_rate_long"),
    "511090": ("sh511090", "CGB_30Y_ETF", "bond_rate_ultra"),
    "511220": ("sh511220", "CREDIT_LGFV_ETF", "bond_credit"),
    "511360": ("sh511360", "SHORT_CP_ETF", "bond_short"),
    "511380": ("sh511380", "CONV_BOND_ETF", "convertible"),
    "511880": ("sh511880", "CASH_MMF_ETF", "cash"),
    "510300": ("sh510300", "CSI300_ETF", "equity_large"),
    "510500": ("sh510500", "CSI500_ETF", "equity_mid"),
    "510880": ("sh510880", "DIVIDEND_ETF", "equity_dividend"),
    "518880": ("sh518880", "GOLD_ETF", "gold"),
}

INDEX_UNIVERSE = {
    "000300": ("sh000300", "CSI300"),
    "000905": ("sh000905", "CSI500"),
    "399006": ("sz399006", "CHINEXT"),
    "000015": ("sh000015", "SSE_DIVIDEND"),
    "000922": ("sh000922", "CSI_DIVIDEND"),
    "000906": ("sh000906", "CSI800"),
}


def retry_call(fn, retries: int = 3, sleep_sec: float = 1.0, desc: str = ""):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_err = exc
            if attempt == retries:
                raise RuntimeError(f"Failed {desc} after {retries} attempts: {last_err}") from exc
            time.sleep(sleep_sec * attempt)
    raise RuntimeError(f"Failed {desc}: {last_err}")


def fetch_chinabond_yields(start_year: int = 2015, end_year: int = 2026, force: bool = False) -> pd.DataFrame:
    """Fetch daily ChinaBond yield curves (CGB, AAA MTN, AAA Bank Bond)."""
    out_path = RAW_DIR / "chinabond_yields.parquet"
    if out_path.exists() and not force:
        df = pd.read_parquet(out_path)
        print(f"[CACHE] chinabond_yields.parquet already exists: shape={df.shape}", flush=True)
        return df

    frames = []
    for yr in range(start_year, end_year + 1):
        s_dt = f"{yr}0101"
        e_dt = f"{yr}1231"
        try:
            df = retry_call(
                lambda s=s_dt, e=e_dt: ak.bond_china_yield(start_date=s, end_date=e),
                desc=f"bond_china_yield {yr}",
            )
            if df is not None and len(df) > 0:
                frames.append(df)
                print(f"  ChinaBond {yr}: {len(df)} rows", flush=True)
        except Exception as exc:
            print(f"  [SKIP] ChinaBond {yr}: {exc}", flush=True)
        time.sleep(0.25)

    out = pd.concat(frames, ignore_index=True)
    rename_map = {
        "曲线名称": "curve_name",
        "日期": "date",
        "3月": "y_3m",
        "6月": "y_6m",
        "1年": "y_1y",
        "3年": "y_3y",
        "5年": "y_5y",
        "7年": "y_7y",
        "10年": "y_10y",
        "30年": "y_30y",
    }
    out = out.rename(columns=rename_map)
    out["date"] = pd.to_datetime(out["date"])
    for col in ["y_3m", "y_6m", "y_1y", "y_3y", "y_5y", "y_7y", "y_10y", "y_30y"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    curve_id_map = {
        "中债国债收益率曲线": "CGB",
        "中债中短期票据收益率曲线(AAA)": "MTN_AAA",
        "中债商业银行普通债收益率曲线(AAA)": "BANK_AAA",
    }
    out["curve_id"] = out["curve_name"].map(curve_id_map).fillna("OTHER")
    out = (
        out.drop_duplicates(subset=["curve_name", "date"])
        .sort_values(["curve_id", "date"])
        .reset_index(drop=True)
    )
    out.to_parquet(out_path, index=False)
    print(f"[SAVED] chinabond_yields.parquet: shape={out.shape}", flush=True)
    return out


def fetch_macro_data(force: bool = False) -> None:
    """Fetch China macroeconomic datasets (PMI, CPI, PPI, M1/M2, Social Financing, Shibor, GDP)."""
    tasks = {
        "macro_pmi": ak.macro_china_pmi,
        "macro_cpi_monthly": ak.macro_china_cpi,
        "macro_ppi_monthly": ak.macro_china_ppi,
        "macro_money_supply": ak.macro_china_money_supply_em,
        "macro_social_financing": ak.macro_china_social_financing,
        "macro_shibor": ak.interest_rate_shibor_js,
        "macro_gdp": ak.macro_china_gdp,
    }
    for name, fn in tasks.items():
        path = RAW_DIR / f"{name}.parquet"
        if path.exists() and not force:
            df = pd.read_parquet(path)
            print(f"[CACHE] {name}.parquet: shape={df.shape}", flush=True)
            continue
        try:
            df = retry_call(fn, desc=name)
            for col in df.columns:
                if df[col].dtype == "object" or str(df[col].dtype) == "string":
                    df[col] = df[col].astype(str)
            df.to_parquet(path, index=False)
            print(f"[SAVED] {name}.parquet: shape={df.shape}", flush=True)
        except Exception as exc:
            print(f"[WARN] {name} failed: {exc}", flush=True)


def fetch_etf_and_indices(force: bool = False) -> None:
    """Fetch daily ETF bars, equity benchmark indices, and HS300 valuation."""
    etf_path = RAW_DIR / "etf_daily.parquet"
    if not (etf_path.exists() and not force):
        etf_frames = []
        for code, (sina_sym, alias, asset_class) in ETF_UNIVERSE.items():
            try:
                df = retry_call(
                    lambda s=sina_sym: ak.fund_etf_hist_sina(symbol=s),
                    desc=f"ETF {code} ({alias})",
                )
                df["date"] = pd.to_datetime(df["date"])
                df["code"] = code
                df["symbol"] = sina_sym
                df["alias"] = alias
                df["asset_class"] = asset_class
                keep = ["date", "code", "symbol", "alias", "asset_class", "open", "high", "low", "close", "volume"]
                if "amount" in df.columns:
                    keep.append("amount")
                df = df[[c for c in keep if c in df.columns]].copy()
                for col in ["open", "high", "low", "close", "volume", "amount"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                etf_frames.append(df)
                print(f"  ETF {code} ({alias}): {len(df)} rows, {df['date'].min().date()}..{df['date'].max().date()}", flush=True)
            except Exception as exc:
                print(f"  [WARN] ETF {code} failed: {exc}", flush=True)
            time.sleep(0.15)

        etf_df = (
            pd.concat(etf_frames, ignore_index=True)
            .drop_duplicates(subset=["code", "date"])
            .sort_values(["code", "date"])
            .reset_index(drop=True)
        )
        etf_df.to_parquet(etf_path, index=False)
        print(f"[SAVED] etf_daily.parquet: shape={etf_df.shape}", flush=True)

    idx_path = RAW_DIR / "indices_daily.parquet"
    if not (idx_path.exists() and not force):
        idx_frames = []
        for code, (sina_sym, alias) in INDEX_UNIVERSE.items():
            try:
                df = retry_call(
                    lambda s=sina_sym: ak.index_zh_daily_sina(symbol=s),
                    desc=f"Index {code} ({alias})",
                )
                df["date"] = pd.to_datetime(df["date"])
                df["code"] = code
                df["alias"] = alias
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                idx_frames.append(df[["date", "code", "alias", "open", "high", "low", "close", "volume"]])
                print(f"  Index {code} ({alias}): {len(df)} rows", flush=True)
            except Exception as exc:
                print(f"  [WARN] Index {code} failed: {exc}", flush=True)
            time.sleep(0.15)

        idx_df = (
            pd.concat(idx_frames, ignore_index=True)
            .drop_duplicates(subset=["code", "date"])
            .sort_values(["code", "date"])
            .reset_index(drop=True)
        )
        idx_df.to_parquet(idx_path, index=False)
        print(f"[SAVED] indices_daily.parquet: shape={idx_df.shape}", flush=True)

    val_path = RAW_DIR / "hs300_valuation.parquet"
    if not (val_path.exists() and not force):
        pe_df = retry_call(lambda: ak.stock_index_pe_lg(symbol="沪深300"), desc="HS300 PE")
        pb_df = retry_call(lambda: ak.stock_index_pb_lg(symbol="沪深300"), desc="HS300 PB")
        pe_df["日期"] = pd.to_datetime(pe_df["日期"])
        pb_df["日期"] = pd.to_datetime(pb_df["日期"])
        val_df = pd.merge(
            pe_df.rename(
                columns={
                    "日期": "date",
                    "指数": "hs300_close",
                    "静态市盈率": "pe_static",
                    "等权静态市盈率": "pe_static_ew",
                    "静态市盈率中位数": "pe_static_median",
                    "滚动市盈率": "pe_ttm",
                    "等权滚动市盈率": "pe_ttm_ew",
                    "滚动市盈率中位数": "pe_ttm_median",
                }
            ),
            pb_df.rename(
                columns={
                    "日期": "date",
                    "市净率": "pb",
                    "等权市净率": "pb_ew",
                    "市净率中位数": "pb_median",
                }
            )[["date", "pb", "pb_ew", "pb_median"]],
            on="date",
            how="outer",
        ).sort_values("date").reset_index(drop=True)
        val_df.to_parquet(val_path, index=False)
        print(f"[SAVED] hs300_valuation.parquet: shape={val_df.shape}", flush=True)


def _fetch_one_etf_nav(code: str, alias: str, asset_class: str, start_ymd: str, end_ymd: str) -> Optional[pd.DataFrame]:
    """Fetch official split/dividend-adjusted ETF NAV & daily growth rate."""
    try:
        df = retry_call(
            lambda: ak.fund_exchange_nav_hist_em(fund=code, start_date=start_ymd, end_date=end_ymd),
            retries=3,
            sleep_sec=1.0,
            desc=f"ETF NAV {code}",
        )
        if df is None or len(df) == 0:
            return None
        df = df.rename(
            columns={
                "净值日期": "date",
                "单位净值": "unit_nav",
                "累计净值": "acc_nav",
                "日增长率": "daily_ret_pct",
            }
        )
        df["date"] = pd.to_datetime(df["date"])
        df["code"] = code
        df["alias"] = alias
        df["asset_class"] = asset_class
        for c in ["unit_nav", "acc_nav", "daily_ret_pct"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        keep = ["date", "code", "alias", "asset_class", "unit_nav", "acc_nav", "daily_ret_pct"]
        return df[[c for c in keep if c in df.columns]].drop_duplicates(subset=["date"]).sort_values("date")
    except Exception as exc:
        print(f"  [WARN] ETF NAV {code} failed: {exc}", flush=True)
        return None


def fetch_etf_nav(start_ymd: str = "20150101", end_ymd: str = "20261120", workers: int = 6, force: bool = False) -> None:
    """Fetch official total-return NAV history for all ETFs (handles splits and dividends)."""
    nav_path = RAW_DIR / "etf_nav_daily.parquet"
    if nav_path.exists() and not force:
        df = pd.read_parquet(nav_path)
        print(f"[CACHE] etf_nav_daily.parquet: shape={df.shape}", flush=True)
        return

    frames = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(_fetch_one_etf_nav, code, alias, asset_class, start_ymd, end_ymd): (code, alias)
            for code, (_, alias, asset_class) in ETF_UNIVERSE.items()
        }
        for fut in as_completed(futs):
            code, alias = futs[fut]
            res = fut.result()
            if res is not None and len(res) > 0:
                frames.append(res)
                print(
                    f"  ETF NAV {code} ({alias}): {len(res)} rows, "
                    f"{res['date'].min().date()}..{res['date'].max().date()}",
                    flush=True,
                )

    nav_df = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["code", "date"])
        .sort_values(["code", "date"])
        .reset_index(drop=True)
    )
    nav_df.to_parquet(nav_path, index=False)
    print(f"[SAVED] etf_nav_daily.parquet: shape={nav_df.shape}", flush=True)


def _fetch_one_stock_price(bs_code: str, start_ymd: str, end_ymd: str) -> Optional[pd.DataFrame]:
    """Fetch one stock's daily HFQ OHLCV + amount + outstanding_share + turnover from Sina."""
    exch, num = bs_code.split(".")
    sina_sym = f"{exch}{num}"
    try:
        df = retry_call(
            lambda: ak.stock_zh_a_hist_sina(symbol=sina_sym, start_date=start_ymd, end_date=end_ymd, adjust="hfq"),
            retries=3,
            sleep_sec=0.8,
            desc=f"sina {sina_sym}",
        )
        if df is None or len(df) == 0:
            return None
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["code"] = bs_code
        keep = ["date", "code", "open", "high", "low", "close", "volume", "amount", "outstanding_share", "turnover"]
        df = df[[c for c in keep if c in df.columns]]
        for c in ["open", "high", "low", "close", "volume", "amount", "outstanding_share", "turnover"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
        return df
    except Exception:
        return None


def _fetch_one_stock_valuation(bs_code: str) -> Optional[pd.DataFrame]:
    """Fetch one stock's 10Y historical PE_TTM, PB, PCF, and TotalMV from Baidu Finance."""
    _, num = bs_code.split(".")
    indicators = [
        ("市盈率(TTM)", "pe_ttm", "近十年"),
        ("市净率", "pb", "近十年"),
        ("市现率", "pcf", "近十年"),
        ("总市值", "total_mv_100m", "近十年"),
    ]
    merged: Optional[pd.DataFrame] = None
    for cn_name, col_name, period in indicators:
        try:
            df = retry_call(
                lambda ind=cn_name, p=period: ak.stock_zh_valuation_baidu(symbol=num, indicator=ind, period=p),
                retries=2,
                sleep_sec=0.5,
                desc=f"baidu {num} {cn_name}",
            )
            if df is None or len(df) == 0:
                continue
            df = df[["date", "value"]].copy()
            df["date"] = pd.to_datetime(df["date"])
            df[col_name] = pd.to_numeric(df["value"], errors="coerce").astype("float32")
            df = df[["date", col_name]].drop_duplicates(subset=["date"])
            if merged is None:
                merged = df
            else:
                merged = pd.merge(merged, df, on="date", how="outer")
        except Exception:
            continue
    if merged is None or len(merged) == 0:
        return None
    merged["code"] = bs_code
    for col in ["pe_ttm", "pb", "pcf", "total_mv_100m"]:
        if col not in merged.columns:
            merged[col] = pd.NA
    return merged[["date", "code", "pe_ttm", "pb", "pcf", "total_mv_100m"]].sort_values("date")


def fetch_pit_hs300_and_equities(
    start_date: str = "2016-01-01",
    end_date: str = "2026-11-20",
    workers: int = 8,
    force: bool = False,
) -> None:
    """Fetch point-in-time CSI 300 constituents, industry map, daily HFQ bars, and historical valuation."""
    pit_path = RAW_DIR / "hs300_constituents_pit.parquet"
    ind_path = RAW_DIR / "stock_industry.parquet"

    if not (pit_path.exists() and ind_path.exists()) or force:
        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"Baostock login failed: {lg.error_msg}")

        snapshot_dates = []
        for yr in range(2016, 2027):
            for md in ["01-15", "06-30", "07-15", "12-31"]:
                dt_str = f"{yr}-{md}"
                if dt_str <= end_date:
                    snapshot_dates.append(dt_str)
        if end_date not in snapshot_dates:
            snapshot_dates.append(end_date)

        pit_rows = []
        for dt_str in snapshot_dates:
            rs = bs.query_hs300_stocks(date=dt_str)
            cnt = 0
            while rs.error_code == "0" and rs.next():
                pit_rows.append([dt_str] + rs.get_row_data())
                cnt += 1
        pit_df = pd.DataFrame(pit_rows, columns=["query_date", "update_date", "code", "code_name"])
        pit_df["query_date"] = pd.to_datetime(pit_df["query_date"])
        pit_df["update_date"] = pd.to_datetime(pit_df["update_date"])
        pit_df.to_parquet(pit_path, index=False)
        print(f"[SAVED] hs300_constituents_pit.parquet: {len(pit_df)} rows, {pit_df['code'].nunique()} unique stocks", flush=True)

        rs = bs.query_stock_industry()
        ind_rows = []
        while rs.error_code == "0" and rs.next():
            ind_rows.append(rs.get_row_data())
        ind_df = pd.DataFrame(ind_rows, columns=rs.fields)
        ind_df.to_parquet(ind_path, index=False)
        print(f"[SAVED] stock_industry.parquet: {len(ind_df)} rows", flush=True)
        bs.logout()
    else:
        pit_df = pd.read_parquet(pit_path)
        print(f"[CACHE] hs300_constituents_pit.parquet: {len(pit_df)} rows, {pit_df['code'].nunique()} unique stocks", flush=True)

    unique_codes = sorted(pit_df["code"].unique().tolist())
    start_ymd = start_date.replace("-", "")
    end_ymd = end_date.replace("-", "")

    # 1. Fetch Sina HFQ Daily OHLCV + amount + outstanding_share + turnover
    eq_path = RAW_DIR / "equity_daily_hfq.parquet"
    if not eq_path.exists() or force:
        t0 = time.time()
        print(f"Fetching Sina HFQ daily bars for {len(unique_codes)} historical CSI 300 stocks ({workers} threads)...", flush=True)
        frames = []
        failed = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut_map = {ex.submit(_fetch_one_stock_price, c, start_ymd, end_ymd): c for c in unique_codes}
            done_cnt = 0
            for fut in as_completed(fut_map):
                code = fut_map[fut]
                res = fut.result()
                done_cnt += 1
                if res is not None and len(res) > 0:
                    frames.append(res)
                else:
                    failed.append(code)
                if done_cnt % 100 == 0 or done_cnt == len(unique_codes):
                    print(f"  Price progress: {done_cnt}/{len(unique_codes)} (ok={len(frames)}, fail={len(failed)}, {time.time()-t0:.1f}s)", flush=True)

        eq_df = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset=["code", "date"])
            .sort_values(["date", "code"])
            .reset_index(drop=True)
        )
        eq_df.to_parquet(eq_path, index=False)
        print(
            f"[SAVED] equity_daily_hfq.parquet: shape={eq_df.shape}, stocks={eq_df['code'].nunique()}, "
            f"failed={len(failed)} in {time.time()-t0:.1f}s",
            flush=True,
        )

    # 2. Fetch Baidu 10Y historical valuation (PE_TTM, PB, PCF, TotalMV)
    val_path = RAW_DIR / "equity_valuation_baidu.parquet"
    if not val_path.exists() or force:
        t0 = time.time()
        print(f"Fetching Baidu 10Y valuation (PE/PB/PCF/MV) for {len(unique_codes)} stocks ({workers} threads)...", flush=True)
        v_frames = []
        v_failed = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut_map = {ex.submit(_fetch_one_stock_valuation, c): c for c in unique_codes}
            done_cnt = 0
            for fut in as_completed(fut_map):
                code = fut_map[fut]
                res = fut.result()
                done_cnt += 1
                if res is not None and len(res) > 0:
                    v_frames.append(res)
                else:
                    v_failed.append(code)
                if done_cnt % 100 == 0 or done_cnt == len(unique_codes):
                    print(f"  Valuation progress: {done_cnt}/{len(unique_codes)} (ok={len(v_frames)}, fail={len(v_failed)}, {time.time()-t0:.1f}s)", flush=True)

        v_df = (
            pd.concat(v_frames, ignore_index=True)
            .drop_duplicates(subset=["code", "date"])
            .sort_values(["date", "code"])
            .reset_index(drop=True)
        )
        v_df.to_parquet(val_path, index=False)
        print(
            f"[SAVED] equity_valuation_baidu.parquet: shape={v_df.shape}, stocks={v_df['code'].nunique()}, "
            f"failed={len(v_failed)} in {time.time()-t0:.1f}s",
            flush=True,
        )


def write_manifest() -> None:
    """Write a verified dataset manifest JSON for auditability and offline Codex runs."""
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "raw_dir": str(RAW_DIR),
        "files": {},
    }
    for pq in sorted(RAW_DIR.glob("*.parquet")):
        df = pd.read_parquet(pq)
        entry = {
            "rows": int(len(df)),
            "cols": list(df.columns),
            "size_mb": round(pq.stat().st_size / (1024 * 1024), 3),
        }
        if "date" in df.columns:
            entry["date_min"] = str(pd.to_datetime(df["date"]).min().date())
            entry["date_max"] = str(pd.to_datetime(df["date"]).max().date())
        if "code" in df.columns:
            entry["n_unique_codes"] = int(df["code"].nunique())
        manifest["files"][pq.name] = entry

    manifest_path = RAW_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAVED] {manifest_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Fetch open-source China Multi-Asset dataset.")
    parser.add_argument("--macro-bond", action="store_true", help="Fetch ChinaBond curves, macro, ETFs, indices")
    parser.add_argument("--etf-nav", action="store_true", help="Fetch official split/dividend-adjusted ETF NAVs")
    parser.add_argument("--equity", action="store_true", help="Fetch PIT HS300 constituents, daily bars, and valuation")
    parser.add_argument("--all", action="store_true", help="Fetch everything")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if parquet cache exists")
    parser.add_argument("--workers", type=int, default=8, help="Parallel threads for equity fetch")
    parser.add_argument("--start-date", type=str, default="2016-01-01")
    parser.add_argument("--end-date", type=str, default="2026-11-20")
    args = parser.parse_args()

    if not (args.macro_bond or args.etf_nav or args.equity or args.all):
        args.all = True

    if args.all or args.macro_bond:
        print("=== [1/4] ChinaBond Yield Curves (2015-2026) ===", flush=True)
        fetch_chinabond_yields(2015, 2026, force=args.force)
        print("=== [2/4] China Macro Panel ===", flush=True)
        fetch_macro_data(force=args.force)
        print("=== [3/4] Bond/CB/Equity/Gold ETFs & Indices ===", flush=True)
        fetch_etf_and_indices(force=args.force)

    if args.all or args.macro_bond or args.etf_nav:
        print("=== [3b/4] Official Split/Dividend-Adjusted ETF NAVs ===", flush=True)
        fetch_etf_nav("20150101", args.end_date.replace("-", ""), workers=min(6, args.workers), force=args.force)

    if args.all or args.equity:
        print("=== [4/4] Point-in-Time CSI 300 & Individual Stock Panel ===", flush=True)
        fetch_pit_hs300_and_equities(args.start_date, args.end_date, workers=args.workers, force=args.force)

    write_manifest()


if __name__ == "__main__":
    main()
