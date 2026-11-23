
import pandas as pd

from .curve import curve_pca, nelson_siegel, theoretical_return
from .portfolio import simulate_bond
from .signals import bond_signals


def load_market_assets(processed, codes):
    renames = {value: key for key, value in codes.items()}
    return tuple(
        pd.read_parquet(processed / f"etf_{name}.parquet").rename(columns=renames)
        for name in ("returns", "tradable", "eligible")
    )


def bond_stage(processed, outputs, config):
    cfg = config["bond"]
    series = pd.read_parquet(processed / "csi300_close.parquet")
    macro = pd.read_parquet(processed / "macro_asof.parquet")
    curves = pd.read_parquet(processed / "curves.parquet")
    etfs = load_market_assets(processed, config["data"]["etf_codes"])
    returns, tradable, eligible = (etf[list(cfg["durations"])] for etf in etfs)
    signals, yields = bond_signals(curves, series.index, macro, config)
    signals.to_parquet(processed / "bond_signals.parquet")
    signals.loc[config["dates"]["start"]:].to_parquet(outputs / "bond_signals_oos.parquet")

    dns, fitted = nelson_siegel(yields["CGB"], cfg["maturities"], cfg["ns_lambda_per_year"])
    dns.to_csv(outputs / "bond_dns_factors.csv")
    fitted.to_parquet(processed / "bond_dns_fitted.parquet")
    pc_scores, pca = curve_pca(yields["CGB"], cfg["pca_min_fit_sessions"], config["dates"]["start"])
    pc_scores.to_csv(outputs / "bond_pca_scores.csv")
    pca.to_csv(outputs / "bond_pca_loadings.csv")

    theory, comparisons = {}, []
    for curve, tenor, proxy in (
        ("CGB", 1, "cash"), ("CGB", 5, "cgb5"), ("CGB", 10, "cgb10"), ("MTN_AAA", 3, "credit"),
    ):
        data = yields[curve].dropna(axis=1, how="all")
        dated = data.reindex(series.index.union(data.index)).ffill().reindex(series.index)
        maturities = [cfg["maturities"][cfg["yield_columns"].index(col)] for col in data.columns]
        label = f"{curve}_{tenor}Y"
        daily = theoretical_return(dated, maturities, tenor, config["dates"]["periods_per_year"])
        theory[label] = daily
        aligned = pd.concat([daily.rename("theory"), returns[proxy].rename("actual")], axis=1)
        aligned = aligned.loc[config["dates"]["start"]:].dropna()
        difference = aligned["actual"] - aligned["theory"]
        comparisons.append({
            "theoretical_index": label, "etf": proxy, "sessions": len(aligned),
            "correlation": aligned["theory"].corr(aligned["actual"]),
            "mean_daily_residual_bps": difference.mean() * 10000,
            "annualized_tracking_residual": difference.std() * config["dates"]["periods_per_year"] ** 0.5,
        })
    theoretical = pd.DataFrame(theory).loc[config["dates"]["start"]:]
    theoretical.to_csv(outputs / "bond_theoretical_returns.csv")
    (1 + theoretical).cumprod().to_csv(outputs / "bond_theoretical_nav.csv")
    pd.DataFrame(comparisons).to_csv(outputs / "bond_theoretical_tracking.csv", index=False)

    static = {
        "Bond_Benchmark": cfg["benchmark_weights"],
        "Bond_Rate_Benchmark": {"cgb5": 0.5, "cgb10": 0.5},
        "Bond_5Y": {"cgb5": 1.0},
        "Bond_10Y": {"cgb10": 1.0},
        "Bond_Rate_Credit_50_50": {"cgb5": 0.25, "cgb10": 0.25, "credit": 0.50},
    }
    rows, books, weights = [], [], []

    def run(name, **kwargs):
        stats, _, end, orders = simulate_bond(returns, tradable, eligible, signals, config, **kwargs)
        rows.append(stats.assign(strategy=name))
        weights.append(end.assign(strategy=name))
        books.append(orders.assign(strategy=name))
        return stats["net_return"]

    active = run("Bond_Active_Sleeve")
    duration = run("Bond_Duration_Only", allow_credit=False)
    for name, target in static.items():
        run(name, static=target)
    for multiplier in config["analytics"]["cost_multipliers"]:
        if multiplier != 1:
            run(f"Bond_Active_Cost_{multiplier:g}x", cost_multiplier=multiplier)
            run(f"Bond_Benchmark_Cost_{multiplier:g}x", static=static["Bond_Benchmark"], cost_multiplier=multiplier)
    daily = pd.concat(rows).reset_index()
    daily.to_parquet(outputs / "bond_daily.parquet", index=False)
    pd.concat(weights).reset_index().to_parquet(outputs / "bond_weights.parquet", index=False)
    pd.concat(books).to_csv(outputs / "bond_orders.csv", index=False)

    rate = daily.loc[daily["strategy"].eq("Bond_Rate_Benchmark")].set_index("date")["net_return"]
    pd.DataFrame({
        "duration_excess_daily": duration - rate,
        "credit_excess_daily": active - duration,
        "total_excess_daily": active - rate,
    }).to_csv(outputs / "bond_duration_credit_effects.csv")
    return daily
