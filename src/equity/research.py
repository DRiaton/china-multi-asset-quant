
import numpy as np
import pandas as pd


def nw_t(values: pd.Series, lags: int) -> float:
    values = values.dropna().to_numpy()
    n = len(values)
    if n <= lags:
        return np.nan
    x = values - values.mean()
    variance = x @ x / n
    for lag in range(1, lags + 1):
        variance += 2 * (1 - lag / (lags + 1)) * (x[lag:] @ x[:-lag]) / n
    return values.mean() / np.sqrt(variance / n) if variance > 0 else np.nan


def evaluate_factors(panel, ic, cfg, start):
    names = cfg["factors"]
    oos = panel.loc[panel["label_start"].ge(start)].dropna(subset=["label_end"])
    ic = ic.loc[ic.index.isin(oos["date"])]
    summary, detail = [], []
    groups = cfg["groups"]
    snapshots = list(oos.groupby("date", sort=True))
    for name in names:
        ic_series = ic[name].dropna()
        last = {}
        rows = []
        for date, sub in snapshots:
            bucket = np.minimum((sub[name].rank(method="first", pct=True) * groups).apply(np.ceil), groups)
            row = {"date": date, "factor": name, "rank_ic": ic[name].get(date, np.nan)}
            for group in range(1, groups + 1):
                positions = sub.loc[bucket.eq(group)]
                row[f"q{group}_excess"] = positions["label"].mean()
                weight = pd.Series(1 / len(positions), index=positions["code"]) if len(positions) else pd.Series(dtype=float)
                previous = last.get(group, pd.Series(dtype=float))
                row[f"q{group}_target_turnover"] = (weight.subtract(previous, fill_value=0)).abs().sum() / 2
                last[group] = weight
            row["long_short"] = row[f"q{groups}_excess"] - row["q1_excess"]
            rows.append(row)
        buckets = pd.DataFrame(rows)
        detail.extend(rows)
        standard = ic_series.std()
        mean_q = [buckets[f"q{q}_excess"].mean() for q in range(1, groups + 1)]
        row = {
            "factor": name,
            "months": len(ic_series),
            "mean_rank_ic": ic_series.mean(),
            "std_rank_ic": standard,
            "annualized_icir": np.sqrt(12) * ic_series.mean() / standard if standard > 0 else np.nan,
            "newey_west_t": nw_t(ic_series, cfg["nw_lags"]),
            "ic_win_rate": ic_series.gt(0).mean(),
            "mean_monthly_long_short": buckets["long_short"].mean(),
            "quintile_monotonicity": pd.Series(mean_q).rank().corr(pd.Series(range(groups))),
        }
        for q, mean in enumerate(mean_q, start=1):
            row[f"q{q}_mean_excess"] = mean
            row[f"q{q}_mean_target_turnover"] = buckets[f"q{q}_target_turnover"].mean()
        summary.append(row)
    return pd.DataFrame(summary), pd.DataFrame(detail)
