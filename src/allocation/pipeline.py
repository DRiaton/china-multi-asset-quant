
import itertools

import pandas as pd

from src.utils.calendar import execution_map
from src.fixed_income.pipeline import load_market_assets
from .portfolio import ABLATION, simulate_allocation
from .regime import macro_features, regime_performance


def load_allocation_market(processed, outputs, config, multiplier=1):
    etfs, traded, eligible = load_market_assets(processed, config["data"]["etf_codes"])
    names = ["equity", "cash", "gold", "convertible"]
    net = etfs[names].rename(columns={"equity": "equity_etf"})
    gross = net.copy()
    costs = pd.DataFrame(0.0, index=net.index, columns=net.columns)
    turnover = costs.copy()
    trades = traded[names].rename(columns={"equity": "equity_etf"}).copy()
    eq = pd.read_parquet(outputs / "equity_daily.parquet")
    bond = pd.read_parquet(outputs / "bond_daily.parquet")
    specs = {
        "equity_active": (eq, "Equity_Alpha_Sleeve" if multiplier == 1 else f"Equity_Cost_{multiplier:g}x"),
        "bond_active": (bond, "Bond_Active_Sleeve" if multiplier == 1 else f"Bond_Active_Cost_{multiplier:g}x"),
        "bond_passive": (bond, "Bond_Benchmark" if multiplier == 1 else f"Bond_Benchmark_Cost_{multiplier:g}x"),
    }
    for label, (frame, sleeve) in specs.items():
        daily = frame.loc[frame["strategy"].eq(sleeve)].set_index("date")
        if daily.empty:
            raise ValueError(f"Missing prerequisite {sleeve}")
        net[label], gross[label], costs[label], turnover[label] = (
            daily[col].reindex(net.index) for col in ("net_return", "gross_return", "cost", "turnover")
        )
        trades[label] = net[label].notna()
    warmup = pd.Timestamp(config["dates"]["warmup_start"])
    # Use contemporaneous passive proxies to initialize covariance before live sleeves exist.
    # On each later decision only realized sleeve data through that decision can enter.
    bond_proxy = etfs[list(config["bond"]["benchmark_weights"])].mean(axis=1)
    cov = net.copy()
    for name in config["allocation"]["satellite_caps"]:
        cov[name] = cov[name].where(trades[name])
    cov["equity_active"] = cov["equity_active"].combine_first(net["equity_etf"])
    for label in ("bond_active", "bond_passive"):
        cov[label] = cov[label].combine_first(bond_proxy)
    return {
        "net": net, "gross": gross, "cost": costs, "turnover": turnover,
        "tradable": trades, "eligible": eligible, "covariance": cov.loc[warmup:],
    }


def allocation_stage(processed, outputs, config):
    macro = pd.read_parquet(processed / "macro_asof.parquet")
    bond = pd.read_parquet(processed / "bond_signals.parquet")
    features, erp = macro_features(macro, bond, config)
    features.to_csv(outputs / "macro_regimes_monthly.csv")
    erp.to_parquet(outputs / "erp_daily.parquet")
    etfs, traded, seasoned = load_market_assets(processed, config["data"]["etf_codes"])
    investable = etfs.where(traded & seasoned)
    regimes = regime_performance(features, investable, execution_map(etfs.index, features.index), config)
    regimes.to_csv(outputs / "macro_regime_asset_stats.csv", index=False)

    market = load_allocation_market(processed, outputs, config)
    frames, starts, ends, orders = {}, {}, {}, []
    active = {"equity_asset": "equity_active", "bond_asset": "bond_active"}
    flagship = {**active, "use_overlay": True}
    strategies = {
        "Static_20_80": {"kind": "static20"},
        "Static_60_40": {"kind": "static60"},
        "Risk_Parity_ERC": {"kind": "erc"},
        "HRP": {"kind": "hrp"},
        "ERP_Tactical_20_80": {"kind": "erp"},
        "Macro_BL_Passive": {"kind": "bl"},
        "Macro_BL_Bond_Active": {"kind": "bl", "bond_asset": "bond_active"},
        "Macro_BL_Equity_Bond_Active": {"kind": "bl", **active},
        "Macro_BL_FixedIncomePlus": {"kind": "bl", **flagship},
        "Macro_BL_Satellites": {"kind": "bl", **flagship, "satellites": True},
        "Macro_BL_Balanced": {"kind": "bl", **active, "balanced": True, "satellites": True, "use_overlay": True},
    }

    def run(name, use_market, **kwargs):
        daily, start, end, book = simulate_allocation(use_market, features, bond, config, **kwargs)
        frames[name], starts[name], ends[name] = daily, start, end
        orders.append(book.assign(strategy=name))
        return daily

    for name, kwargs in strategies.items():
        run(name, market, **kwargs)
    for multiplier in config["analytics"]["cost_multipliers"]:
        if multiplier != 1:
            markets = load_allocation_market(processed, outputs, config, multiplier)
            run(f"Allocation_Cost_{multiplier:g}x", markets, kind="bl", cost_multiplier=multiplier, **flagship)

    params = []
    for delta, tau in itertools.product(config["allocation"]["risk_aversion_grid"], config["allocation"]["tau_grid"]):
        name = f"BL_delta_{delta:g}_tau_{tau:g}"
        run(name, market, kind="bl", delta=delta, tau=tau, **flagship)
        params.append({"strategy": name, "delta": delta, "tau": tau})
    pd.DataFrame(params).to_csv(outputs / "bl_grid_parameters.csv", index=False)

    def export(data, name):
        combined = pd.concat([frame.assign(strategy=key) for key, frame in data.items()])
        combined.reset_index().to_parquet(outputs / name, index=False)

    export(frames, "allocation_daily.parquet")
    export(starts, "allocation_weights_start.parquet")
    export(ends, "allocation_weights_end.parquet")
    pd.concat(orders, ignore_index=True).to_csv(outputs / "allocation_orders.csv", index=False)
    pd.DataFrame({"step": range(len(ABLATION)), "strategy": ABLATION}).to_csv(outputs / "ablation_steps.csv", index=False)
    return frames
