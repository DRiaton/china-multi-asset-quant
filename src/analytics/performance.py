
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew


def drawdown(returns: pd.Series) -> pd.Series:
    nav = (1 + returns).cumprod()
    return nav / nav.cummax().clip(lower=1) - 1


def recovery_days(returns: pd.Series) -> int:
    nav = (1 + returns).cumprod()
    peak = 1.0
    peak_date = nav.index[0]
    longest = 0
    underwater = False
    for date, wealth in nav.items():
        if underwater or wealth < peak:
            longest = max(longest, (date - peak_date).days)
        underwater = wealth < peak
        if wealth >= peak:
            peak, peak_date = wealth, date
    return longest


def sharpe_ratio(excess: pd.Series, sessions: int) -> float:
    standard = excess.std(ddof=1)
    return float(excess.mean() / standard * np.sqrt(sessions)) if standard > 0 else np.nan


def deflated_sharpe(excess, null_max_sharpe):
    """Probabilistic Sharpe evaluated at the expected winner's daily Sharpe."""
    observations = len(excess)
    sigma = excess.std(ddof=1)
    if observations <= 3 or sigma <= 0:
        return np.nan
    daily_sr = excess.mean() / sigma
    third = skew(excess, bias=False)
    fourth = kurtosis(excess, fisher=False, bias=False)
    variance = 1 - third * daily_sr + (fourth - 1) / 4 * daily_sr ** 2
    return float(norm.cdf((daily_sr - null_max_sharpe) * np.sqrt((observations - 1) / variance))) if variance > 0 else np.nan


def expected_best_daily_sharpe(sharpes) -> float:
    finite = np.asarray(sharpes)
    finite = finite[np.isfinite(finite)]
    n = len(finite)
    if n < 2:
        return 0.0
    euler_gamma = 0.5772156649015329
    expected_max_z = (1 - euler_gamma) * norm.ppf(1 - 1 / n) + euler_gamma * norm.ppf(1 - 1 / (n * np.e))
    return float(expected_max_z * finite.std(ddof=1))


def bootstrap_sharpe(excess, sessions, draws, block, seed):
    """Circular moving-block resampling preserves autocorrelation within each block."""
    values = np.asarray(excess)
    n = len(values)
    rng = np.random.default_rng(seed)
    blocks = np.arange(block)
    ratios = []
    per_draw = (n + block - 1) // block
    for start in range(0, draws, 128):
        size = min(128, draws - start)
        positions = rng.integers(0, n, size=(size, per_draw, 1))
        offsets = (positions + blocks) % n
        samples = values[offsets.reshape(size, -1)[:, :n]]
        std = samples.std(axis=1, ddof=1)
        ratios.extend(samples.mean(axis=1)[std > 0] / std[std > 0] * np.sqrt(sessions))
    return np.quantile(ratios, [0.025, 0.975]) if len(ratios) else (np.nan, np.nan)


def monthly_returns(daily: pd.Series) -> pd.Series:
    return daily.add(1).groupby(daily.index.to_period("M")).prod() - 1


def metrics(daily, cash, config, null_max_sharpe=0.0, benchmark=None):
    r = daily["net_return"]
    if r.isna().any() or not r.index.is_monotonic_increasing:
        raise ValueError("Performance requires ordered and observed portfolio returns")
    sessions = config["dates"]["periods_per_year"]
    cfg = config["analytics"]
    excess = r - cash.reindex(r.index).fillna(0)
    n = len(r)
    cagr = np.expm1(np.log1p(r).sum() * sessions / n)
    max_dd = drawdown(r).min()
    downside = np.sqrt(np.mean(np.minimum(0, excess) ** 2))
    confidence = bootstrap_sharpe(excess, sessions, cfg["bootstrap_draws"], cfg["bootstrap_block"], cfg["seed"])
    result = {
        "sessions": n,
        "cumulative_return": float((1 + r).prod() - 1),
        "cagr": cagr,
        "annualized_volatility": r.std(ddof=1) * np.sqrt(sessions),
        "sharpe": sharpe_ratio(excess, sessions),
        "sortino": excess.mean() * np.sqrt(sessions) / downside if downside > 0 else np.nan,
        "calmar": cagr / -max_dd if max_dd < 0 else np.nan,
        "max_drawdown": max_dd,
        "max_recovery_days": recovery_days(r),
        "monthly_win_rate": monthly_returns(r).gt(0).mean(),
        "annualized_turnover": daily["turnover"].mean() * sessions,
        "annualized_nested_turnover": daily["nested_turnover"].mean() * sessions if "nested_turnover" in daily else 0.0,
        "annualized_cost_bps": daily["cost"].mean() * sessions * 10000,
        "deflated_sharpe_probability": deflated_sharpe(excess, null_max_sharpe),
        "dsr_null_daily_sharpe": null_max_sharpe,
        "sharpe_ci_low": confidence[0],
        "sharpe_ci_high": confidence[1],
    }
    for level in cfg["var_levels"]:
        threshold = r.quantile(1 - level)
        number = round(level * 100)
        result[f"daily_var_{number}"] = -threshold
        result[f"daily_cvar_{number}"] = -r.loc[r.le(threshold)].mean()
    if benchmark is not None:
        bm = benchmark.reindex(r.index)
        active = r - bm
        result.update(
            info_ratio=sharpe_ratio(active, sessions),
            tracking_error=active.std(ddof=1) * np.sqrt(sessions),
            relative_monthly_win_rate=monthly_returns(r).gt(monthly_returns(bm)).mean(),
            benchmark_cagr=np.expm1(np.log1p(bm).sum() * sessions / n),
        )
        result["active_cagr_difference"] = result["cagr"] - result["benchmark_cagr"]
    return result


def stress_table(strategies, windows):
    rows = []
    for event, (start, end) in windows.items():
        for name, daily in strategies.items():
            r = daily.loc[start:end, "net_return"]
            if r.empty:
                continue
            rows.append({
                "event": event, "strategy": name, "start": r.index[0], "end": r.index[-1],
                "sessions": len(r), "cumulative_return": (1 + r).prod() - 1,
                "max_drawdown": drawdown(r).min(),
            })
    return pd.DataFrame(rows)
