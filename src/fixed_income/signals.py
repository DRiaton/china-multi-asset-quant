
import numpy as np
import pandas as pd

from src.data.macro import align_daily_close
from .curve import carry_roll


def rolling_z(values: pd.Series, window: int, minimum: int | None = None) -> pd.Series:
    trailing = values.rolling(window, min_periods=window if minimum is None else minimum)
    return (values - trailing.mean()) / trailing.std(ddof=0).replace(0, np.nan)


def bond_signals(curves, calendar, macro, config):
    cfg = config["bond"]
    yields = {name: group.set_index("date")[cfg["yield_columns"]] for name, group in curves.groupby("curve_id")}
    aligned = {name: align_daily_close(frame, calendar) for name, frame in yields.items()}
    cgb, mtn, bank = (aligned[key] for key in ("CGB", "MTN_AAA", "BANK_AAA"))
    signals = cgb.add_prefix("cgb_")
    signals["spread_10y_1y"] = cgb["y_10y"] - cgb["y_1y"]
    signals["spread_10y_3y"] = cgb["y_10y"] - cgb["y_3y"]
    signals["butterfly_5y"] = 2 * cgb["y_5y"] - cgb["y_1y"] - cgb["y_10y"]
    signals["credit_spread_3y"] = mtn["y_3y"] - cgb["y_3y"]
    signals["credit_spread_5y"] = mtn["y_5y"] - cgb["y_5y"]
    signals["bank_spread_5y"] = bank["y_5y"] - cgb["y_5y"]
    signals["funding_spread"] = macro["shibor_3m"] - cgb["y_1y"]
    signals["shibor_1w_z"] = rolling_z(macro["shibor_1w"], cfg["shibor_window"])
    signals["funding_spread_z"] = rolling_z(signals["funding_spread"], cfg["shibor_window"])

    quantile_names = [
        "spread_10y_1y", "spread_10y_3y", "butterfly_5y",
        "credit_spread_3y", "credit_spread_5y", "bank_spread_5y",
    ]
    for tenor in (1, 3, 5, 7, 10, 30):
        name = f"carry_roll_{tenor}y"
        signals[name] = carry_roll(cgb[cfg["yield_columns"]], cfg["maturities"], tenor, cfg["roll_months"])
        quantile_names.append(name)
    for window in cfg["rate_momentum_windows"]:
        name = f"rate_mom_{window}d"
        signals[name] = -cgb["y_10y"].diff(window)
        quantile_names.append(name)
    for name in quantile_names:
        signals[f"{name}_pctl"] = signals[name].rolling(cfg["signal_history"], min_periods=cfg["signal_minimum"]).rank(pct=True)

    spread_change = signals["credit_spread_3y"].diff(cfg["credit_shock_window"])
    threshold = spread_change.shift().rolling(
        cfg["signal_history"], min_periods=cfg["signal_minimum"],
    ).quantile(cfg["credit_shock_quantile"]).clip(lower=cfg["credit_shock_min_bps"] / 10000)
    funding = signals["shibor_1w_z"].combine(signals["funding_spread_z"], np.fmax)
    signals["credit_change_20d_bps"] = spread_change * 10000
    signals["credit_gate_threshold_bps"] = threshold * 10000
    signals["credit_gate"] = spread_change.gt(threshold) | funding.gt(cfg["funding_z_gate"])

    momentum = signals[[f"rate_mom_{w}d_pctl" for w in cfg["rate_momentum_windows"]]].mean(axis=1)
    term = signals[["spread_10y_1y_pctl", "spread_10y_3y_pctl", "carry_roll_10y_pctl"]].mean(axis=1)
    macro_defense = (
        macro["pmi_mom_3m"].lt(0).astype(float)
        + macro["cpi_yoy"].diff(63).lt(0).astype(float)
    ) / 2
    momentum_weight, term_weight, macro_weight = cfg["duration_signal_weights"]
    rank = momentum_weight * momentum.fillna(0.5) + term_weight * term.fillna(0.5) + macro_weight * macro_defense
    defensive, offensive = cfg["defensive_duration"], cfg["offensive_duration"]
    benchmark_duration = sum(cfg["durations"][asset] * weight for asset, weight in cfg["benchmark_weights"].items())
    signals["duration_score"] = rank
    signals["benchmark_duration"] = benchmark_duration
    target = pd.Series(
        np.interp(rank, [0, 0.30, 0.50, 0.65, 1], [*defensive, benchmark_duration, *offensive]),
        index=signals.index,
    )
    signals["duration_target"] = target.where(~signals["credit_gate"], target.clip(upper=cfg["gated_duration_ceiling"]))
    signals["credit_score"] = signals[["credit_spread_3y_pctl", "credit_spread_5y_pctl", "bank_spread_5y_pctl"]].mean(axis=1)
    return signals, yields
