from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


INK = "#152B3B"
MUTED = "#475C6A"
GRID = "#DEE6EA"
NAVY = "#204F8C"
TEAL = "#007C76"
ORANGE = "#B54F32"
FLAGSHIP = "Macro_BL_FixedIncomePlus"
MODELS = {
    FLAGSHIP: ("Macro BL Fixed Income Plus", NAVY, "-", 2.4),
    "Static_20_80": ("Static 20/80", "#586E78", "--", 1.7),
    "Static_60_40": ("Static 60/40", ORANGE, "-", 1.5),
    "Risk_Parity_ERC": ("Risk parity (ERC)", "#698541", ":", 2.0),
    "HRP": ("HRP", "#7758A6", "-.", 1.4),
    "ERP_Tactical_20_80": ("ERP tactical", TEAL, "-", 1.6),
}
REGIMES = {
    "Recovery": "#34836C",
    "Overheat": "#C68B3D",
    "Stagflation": "#AD5374",
    "Recession": "#63768B",
}
FIGURE_FILES = {
    "nav": "01_allocation_nav.png",
    "drawdown": "02_allocation_drawdowns.png",
    "regimes": "03_flagship_weights_regimes.png",
    "dns": "04_chinabond_dns.png",
    "bond": "05_bond_duration_credit.png",
    "equity": "06_equity_factor_diagnostics.png",
    "brinson": "07_brinson_waterfall.png",
    "stress": "08_historical_stress.png",
    "bl": "09_bl_parameter_sensitivity.png",
    "ablation": "10_allocation_ablation.png",
}
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelcolor": MUTED,
    "axes.edgecolor": GRID,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.axisbelow": True,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "text.color": INK,
    "legend.frameon": False,
    "legend.fontsize": 9,
}


def _canvas(title, subtitle, source, *, size=(11, 5.6), rect=(0.075, 0.12, 0.97, 0.79)):
    fig = Figure(figsize=size, facecolor="white")
    FigureCanvasAgg(fig)
    fig.text(0.075, 0.943, title, fontsize=21, fontweight="bold")
    fig.text(0.075, 0.882, subtitle, color=MUTED, fontsize=10.2)
    fig.text(0.075, 0.035, "Source  " + source, color=MUTED, fontsize=8)
    return fig, rect


def _axis(fig, rect):
    left, bottom, right, top = rect
    return fig.add_axes([left, bottom, right - left, top - bottom])


def _time(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.margins(x=0)
    ax.grid(axis="x", visible=False)


def _legend(ax, **kwargs):
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.13), borderaxespad=0, **kwargs)


def _allocation_curves(daily, drawdowns=False):
    selected = daily.loc[daily["strategy"].isin(MODELS)]
    nav = selected.pivot(index="date", columns="strategy", values="nav").sort_index()
    if drawdowns:
        title = "V2 allocation drawdowns"
        subtitle = "Reported net drawdown from inception  |  The V2 overlay uses a separate trailing 252-session peak"
        nav = nav.div(nav.cummax().clip(lower=1.0)) - 1
    else:
        title = "V2 net performance of six allocation models"
        subtitle = f"Growth of CNY 1 after trading costs  |  {nav.index.min():%b %Y} to {nav.index.max():%b %Y}"
    fig, rect = _canvas(title, subtitle, "allocation_daily.parquet")
    ax = _axis(fig, rect)
    for strategy, (label, color, style, width) in MODELS.items():
        ax.plot(nav.index, nav[strategy], label=label, color=color, ls=style, lw=width,
                zorder=5 if strategy == FLAGSHIP else 3)
    if drawdowns:
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.axhline(0, color=GRID, lw=0.8)
    else:
        ax.set_ylabel("Net wealth (CNY)")
    _time(ax)
    _legend(ax, ncol=3)
    return fig


def _weights_and_regimes(weights, regime, exposure_summary):
    active = weights.loc[weights["strategy"].eq(FLAGSHIP)].set_index("date").sort_index()
    exposures = pd.DataFrame({
        "Equity": active[["equity_etf", "equity_active"]].sum(axis=1),
        "Bonds": active[["bond_passive", "bond_active"]].sum(axis=1),
        "Cash": active[["cash", "_cash"]].sum(axis=1),
    })
    top = exposure_summary.loc[exposure_summary["layer"].eq("top_level")].set_index("asset")["mean_weight"]
    probability_columns = [f"p_{label}" for label in REGIMES]
    valid = regime.set_index("date").loc[:, probability_columns].sort_index().dropna()
    valid = valid.loc[active.index.min():active.index.max()]
    # The probability known at each decision date is extended forward to the next decision.
    valid.loc[active.index.max()] = valid.iloc[-1]
    fig, rect = _canvas(
        "V2 flagship allocation and the macro backdrop",
        f"Mean top-level weights: equity {top['equity']:.1%}, bonds {top['bond']:.1%}, cash {top['cash']:.1%}  |  End-of-day weights",
        "allocation_weights_end.parquet, flagship_exposure_summary.csv, macro_regimes_monthly.csv",
        size=(11, 6.8),
    )
    left, bottom, right, top = rect
    gs = fig.add_gridspec(2, 1, left=left, bottom=bottom, right=right, top=top, hspace=0.38)
    ax = fig.add_subplot(gs[0])
    prob = fig.add_subplot(gs[1], sharex=ax)
    ax.stackplot(exposures.index, exposures.to_numpy().T, labels=exposures.columns,
                 colors=[NAVY, "#89A8BB", "#DBE7EC"])
    prob.stackplot(valid.index, valid.to_numpy().T, labels=list(REGIMES),
                   colors=list(REGIMES.values()), step="post", alpha=0.94)
    for panel in (ax, prob):
        panel.set_ylim(0, 1)
        panel.set_yticks([0, 0.5, 1])
        panel.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        _time(panel)
        _legend(panel, ncol=4)
    ax.set_ylabel("Portfolio weight")
    prob.set_ylabel("Filtered probability")
    ax.tick_params(labelbottom=False)
    ax.set_xlim(active.index.min(), active.index.max())
    return fig


def _dns(dns, start, end):
    dns = dns.set_index("date").sort_index().loc[start:end]
    fig, rect = _canvas(
        "Three factors summarize the ChinaBond CGB curve",
        f"Daily Nelson–Siegel level, slope and curvature  |  Median fit RMSE: {dns['rmse_bps'].median():.2f} bp",
        "bond_dns_factors.csv",
        size=(11, 6.8),
    )
    left, bottom, right, top = rect
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], left=left, bottom=bottom,
                          right=right, top=top, hspace=0.28)
    ax = fig.add_subplot(gs[0])
    rmse = fig.add_subplot(gs[1], sharex=ax)
    for name, color in zip(("level", "slope", "curvature"), (NAVY, TEAL, ORANGE)):
        ax.plot(dns.index, dns[name], label=name.title(), color=color, lw=1.2)
    ax.axhline(0, color=MUTED, alpha=0.5, lw=0.8)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel("Factor loading")
    _legend(ax, ncol=3)
    rmse.plot(dns.index, dns["rmse_bps"], color="#7758A6", lw=1.0)
    rmse.set_ylabel("Fit RMSE (bp)")
    ax.tick_params(labelbottom=False)
    _time(ax)
    return fig


def _bond(bond):
    strategies = {
        "Bond_Active_Sleeve": ("Active duration + credit", NAVY, "-"),
        "Bond_Duration_Only": ("Duration only", TEAL, "-."),
        "Bond_Benchmark": ("Bond benchmark", MUTED, "--"),
    }
    active = bond.loc[bond["strategy"].eq("Bond_Active_Sleeve")].set_index("date").sort_index()
    fig, rect = _canvas(
        "V2 bond sleeve relative to the passive duration anchor",
        "Net wealth, end-of-day duration and credit sleeve weights  |  Shading marks the active tail-risk gate",
        "bond_daily.parquet",
        size=(11, 8.3),
    )
    left, bottom, right, top = rect
    gs = fig.add_gridspec(3, 1, left=left, bottom=bottom, right=right, top=top, hspace=0.28)
    nav = fig.add_subplot(gs[0])
    duration = fig.add_subplot(gs[1], sharex=nav)
    credit = fig.add_subplot(gs[2], sharex=nav)
    for strategy, (label, color, style) in strategies.items():
        data = bond.loc[bond["strategy"].eq(strategy)].set_index("date").sort_index()
        nav.plot(data.index, data["nav"], label=label, color=color, ls=style, lw=1.6)
        duration.plot(data.index, data["modified_duration"], color=color, ls=style, lw=1.35)
    credit.plot(active.index, active["lgfv_weight"], label="LGFV (511220)", color=TEAL, lw=1.15)
    credit.plot(active.index, active["short_cp_weight"], label="Short CP (511360)", color="#7758A6", lw=1.05)
    _legend(credit, ncol=2)
    gate = active["credit_gate"].fillna(False).astype(bool)
    ends = active.index[gate.ne(gate.shift(fill_value=False))].tolist()
    if len(ends) % 2:
        ends.append(active.index[-1])
    for start, end in zip(ends[::2], ends[1::2]):
        duration.axvspan(start, end, color=ORANGE, alpha=0.18, lw=0)
        credit.axvspan(start, end, color=ORANGE, alpha=0.18, lw=0)
    nav.set_ylabel("Net wealth (CNY)")
    duration.set_ylabel("Duration (years)")
    credit.set_ylabel("ETF weight")
    credit.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    credit.set_ylim(bottom=0)
    _legend(nav, ncol=3)
    for panel in (nav, duration, credit):
        _time(panel)
    nav.tick_params(labelbottom=False)
    duration.tick_params(labelbottom=False)
    return fig


def _diverging(ax, data, tick_labels_x, tick_labels_y, fmt):
    values = np.asarray(data, dtype=float)
    max_abs = max(np.nanmax(np.abs(values)), 1e-9)
    heatmap = ax.imshow(values, cmap="RdBu", norm=TwoSlopeNorm(0, -max_abs, max_abs), aspect="auto")
    ax.set_xticks(np.arange(len(tick_labels_x)), tick_labels_x)
    ax.set_yticks(np.arange(len(tick_labels_y)), tick_labels_y)
    ax.grid(False)
    for (row, column), value in np.ndenumerate(values):
        if np.isfinite(value):
            ax.text(column, row, format(value, fmt), ha="center", va="center",
                    color="white" if abs(value) > 0.63 * max_abs else INK, fontsize=8.8)
    return heatmap


def _equity_factors(factors, ensemble, industry):
    data = factors.set_index("factor").sort_values("mean_rank_ic", ascending=False)
    fig, rect = _canvas(
        "V2 equity factor evidence and causal ensemble",
        f"Industry-and-size neutralized factors  |  Largest constrained rebalance target deviation: {industry['abs_deviation'].max():.2%}",
        "equity_factor_ic_summary.csv, equity_ensemble_weights.csv, equity_industry_deviations.csv",
        size=(11, 10),
        rect=(0.235, 0.105, 0.96, 0.82),
    )
    left, bottom, right, top = rect
    gs = fig.add_gridspec(2, 2, height_ratios=[3.3, 1], width_ratios=[1, 1.4],
                          left=left, bottom=bottom, right=right, top=top, wspace=0.14, hspace=0.5)
    ic = fig.add_subplot(gs[0, 0])
    buckets = fig.add_subplot(gs[0, 1])
    values = data["mean_rank_ic"]
    ic.barh(np.arange(len(values)), values, color=np.where(values.ge(0), NAVY, ORANGE), height=0.66)
    ic.set_yticks(np.arange(len(values)), data.index.str.replace("_", " "))
    ic.set_ylim(len(values) - 0.5, -0.5)
    ic.axvline(0, color=MUTED, lw=0.85)
    ic.grid(axis="y", visible=False)
    ic.grid(axis="x")
    ic.set_title("Mean rank IC", loc="left")
    quintiles = [f"q{i}_mean_excess" for i in range(1, 6)]
    _diverging(buckets, data[quintiles] * 10_000, [f"Q{i}" for i in range(1, 6)], values.index, ".0f")
    buckets.set_yticklabels([])
    buckets.set_title("Monthly excess return (bp)", loc="left")
    ax = fig.add_subplot(gs[1, :])
    model_names = ["E1_ICIR", "E2_Ridge", "E3_LightGBM"]
    weight_cols = [f"weight_{name}" for name in model_names]
    ensemble = ensemble.sort_values("date")
    ax.stackplot(ensemble["date"], ensemble[weight_cols].to_numpy().T,
                 labels=model_names, colors=[NAVY, TEAL, ORANGE], step="post", alpha=0.9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_ylabel("Ensemble weight")
    _time(ax)
    _legend(ax, ncol=3)
    return fig


def _brinson(attribution):
    core = attribution.set_index("effect")["annualized"]
    aliases = {
        "asset_allocation": "Asset allocation",
        "equity_selection": "Equity selection",
        "bond_selection": "Bond selection",
        "fixed_income_selection": "Bond selection",
        "cash_implementation": "Cash implementation",
        "interaction": "Interaction",
        "transaction_cost_drag": "Trading costs",
    }
    names = core.index.to_series().str.lower()
    reported_total = names.str.contains("^active_total$", regex=True)
    residual = names.str.contains("residual|reconciliation", regex=True)
    components = core.loc[~(reported_total | residual)] * 10_000
    actual_total = core.loc[reported_total].iloc[-1] * 10_000 if reported_total.any() else components.sum()
    starts = np.r_[0.0, components.cumsum().to_numpy()]
    x = np.arange(len(components) + 1)
    labels = [aliases.get(name, name.replace("_", " ").title()) for name in components.index]
    fig, rect = _canvas(
        "V2 flagship attribution versus Static 20/80",
        "Linked Brinson effects and incremental trading-cost drag, annualized in basis points",
        "brinson_attribution.csv",
        rect=(0.09, 0.19, 0.97, 0.79),
    )
    ax = _axis(fig, rect)
    for i, value in enumerate(components):
        ax.bar(i, value, bottom=starts[i], width=0.62,
               color=TEAL if value >= 0 else ORANGE, zorder=2)
        ax.hlines(starts[i + 1], i + 0.31, i + 0.69, color=MUTED, linestyles="dotted", lw=0.85)
        ax.annotate(f"{value:+.1f}", (i, starts[i + 1]), xytext=(0, 7 if value >= 0 else -7),
                    textcoords="offset points", ha="center", va="bottom" if value >= 0 else "top",
                    color=INK, fontweight="bold")
    ax.bar(x[-1], actual_total, width=0.62, color=NAVY, zorder=2)
    ax.annotate(f"{actual_total:+.1f}", (x[-1], actual_total),
                xytext=(0, 7 if actual_total >= 0 else -7), textcoords="offset points",
                ha="center", va="bottom" if actual_total >= 0 else "top", fontweight="bold")
    ax.set_xticks(x, labels + ["Total active return"], rotation=14, ha="right")
    ax.set_ylabel("Annualized contribution (bp)")
    ax.axhline(0, color=MUTED, lw=0.85)
    ax.grid(axis="x", visible=False)
    low = min(starts.min(), actual_total)
    high = max(starts.max(), actual_total)
    span = max(high - low, 1.0)
    ax.set_ylim(low - 0.14 * span, high + 0.09 * span)
    return fig


def _stress_windows(stress):
    names = {
        "deleveraging_2018": "2018 deleveraging",
        "covid_q1_2020": "2020 Q1 COVID shock",
        "bond_bear_2020": "2020 bond sell-off",
        "stocks_bonds_2022": "2022 stock/bond sell-off",
        "redemption_2022": "2022 wealth-product redemptions",
        "reversal_2024": "2024 September reversal",
        "low_rate_2025_26": "2025–26 low-rate environment",
    }
    data = stress.loc[stress["strategy"].isin(MODELS)].copy()
    order = data["event"].drop_duplicates().tolist()
    events = [names.get(name.lower(), name.replace("_", " ")) for name in order]
    short = ["Macro BL", "20/80", "60/40", "ERC", "HRP", "ERP"]
    fig, rect = _canvas(
        "V2 allocation models in historical stress windows",
        "Observed cumulative return and maximum drawdown from net daily returns",
        "stress_tests.csv",
        size=(12.8, 5.9),
        rect=(0.24, 0.15, 0.97, 0.77),
    )
    left, bottom, right, top = rect
    gs = fig.add_gridspec(1, 2, left=left, bottom=bottom, right=right, top=top, wspace=0.10)
    for i, (metric, title) in enumerate((("cumulative_return", "Cumulative return"),
                                        ("max_drawdown", "Maximum drawdown"))):
        ax = fig.add_subplot(gs[i])
        cells = data.pivot(index="event", columns="strategy", values=metric).reindex(index=order, columns=list(MODELS))
        _diverging(ax, cells * 100, short, events if i == 0 else [""] * len(events), ".1f")
        ax.set_title(title + " (%)", loc="left")
    return fig


def _parameter_grid(grid):
    fig, rect = _canvas(
        "V2 Black–Litterman parameter sensitivity",
        "Realized outcomes by risk aversion (δ) and prior covariance scale (τ)  |  Fixed core assets and relative views",
        "bl_grid_sensitivity.csv",
        size=(11.5, 5.2),
        rect=(0.075, 0.17, 0.97, 0.76),
    )
    left, bottom, right, top = rect
    gs = fig.add_gridspec(1, 3, left=left, bottom=bottom, right=right, top=top, wspace=0.22)
    metrics = [("cagr", "CAGR (%)", 100), ("sharpe", "Sharpe ratio", 1),
               ("max_drawdown", "Maximum drawdown (%)", 100)]
    for i, (metric, title, scale) in enumerate(metrics):
        ax = fig.add_subplot(gs[i])
        cells = grid.pivot(index="delta", columns="tau", values=metric).sort_index().sort_index(axis=1)
        values = cells.to_numpy() * scale
        low, high = np.nanmin(values), np.nanmax(values)
        midpoint = (low + high) / 2
        # Each panel has its own scale to make the magnitude of parameter effects readable.
        ax.imshow(values, cmap="Blues", vmin=low - 0.25 * max(high - low, 1e-9),
                  vmax=high + 0.08 * max(high - low, 1e-9), aspect="auto")
        ax.set_xticks(np.arange(len(cells.columns)), [f"{v:g}" for v in cells.columns])
        ax.set_yticks(np.arange(len(cells.index)), [f"{v:g}" for v in cells.index] if i == 0 else [])
        ax.grid(False)
        ax.set_title(title, loc="left")
        ax.set_xlabel("τ")
        if i == 0:
            ax.set_ylabel("Risk aversion δ")
        for (row, column), value in np.ndenumerate(values):
            if np.isfinite(value):
                ax.text(column, row, f"{value:.2f}", ha="center", va="center",
                        fontsize=9.6, color="white" if value > midpoint else INK)
    return fig


def _ablation(performance, steps):
    stages = {
        "Static_20_80": "Static 20/80",
        "Macro_BL_Passive": "Macro BL overlay on passive sleeves",
        "Macro_BL_Bond_Active": "Add active bonds",
        "Macro_BL_Equity_Bond_Active": "Add multifactor equities",
        FLAGSHIP: "Add risk-asset-only controls",
        "Macro_BL_Satellites": "Add gold and convertibles",
    }
    data = performance.set_index("strategy").loc[steps.sort_values("step")["strategy"]]
    data = data.rename(index=stages)
    fig, rect = _canvas(
        "What changed as each allocation layer was added?",
        "Sequential ablation of the V2 fixed-income-plus architecture",
        "ablation_steps.csv, performance_summary.csv",
        size=(11.5, 5.2),
        rect=(0.31, 0.14, 0.96, 0.77),
    )
    left, bottom, right, top = rect
    gs = fig.add_gridspec(1, 3, left=left, bottom=bottom, right=right, top=top, wspace=0.25)
    metrics = [("cagr", "CAGR", True), ("annualized_volatility", "Volatility", True),
               ("sharpe", "Sharpe ratio", False)]
    colors = [NAVY if key == stages[FLAGSHIP] else "#8EA9B8" for key in data.index]
    for i, (metric, title, percent) in enumerate(metrics):
        ax = fig.add_subplot(gs[i])
        y = np.arange(len(data))
        ax.barh(y, data[metric], height=0.61, color=colors)
        ax.set_yticks(y, data.index if i == 0 else [])
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_title(title, loc="left")
        if percent:
            ax.yaxis.grid(False)
            ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    return fig


def generate_figures(outputs: Path, figure_dir: Path) -> list[Path]:
    """Render the report figures directly from the saved research results."""
    outputs, figure_dir = Path(outputs), Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    daily = pd.read_parquet(outputs / "allocation_daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    weights = pd.read_parquet(outputs / "allocation_weights_end.parquet")
    weights["date"] = pd.to_datetime(weights["date"])
    bond = pd.read_parquet(outputs / "bond_daily.parquet")
    bond["date"] = pd.to_datetime(bond["date"])
    regime = pd.read_csv(outputs / "macro_regimes_monthly.csv", parse_dates=["date"])
    dns = pd.read_csv(outputs / "bond_dns_factors.csv", parse_dates=["date"])
    factors = pd.read_csv(outputs / "equity_factor_ic_summary.csv")
    ensemble = pd.read_csv(outputs / "equity_ensemble_weights.csv", parse_dates=["date"])
    industry = pd.read_csv(outputs / "equity_industry_deviations.csv")
    exposures = pd.read_csv(outputs / "flagship_exposure_summary.csv")
    attribution = pd.read_csv(outputs / "brinson_attribution.csv")
    stress = pd.read_csv(outputs / "stress_tests.csv")
    grid = pd.read_csv(outputs / "bl_grid_sensitivity.csv")
    performance = pd.read_csv(outputs / "performance_summary.csv")
    steps = pd.read_csv(outputs / "ablation_steps.csv")
    start, end = daily["date"].min(), daily["date"].max()
    ensemble = ensemble.loc[ensemble["date"].between(start, end)]

    with mpl.rc_context(STYLE):
        figures = {
            "nav": _allocation_curves(daily),
            "drawdown": _allocation_curves(daily, drawdowns=True),
            "regimes": _weights_and_regimes(weights, regime, exposures),
            "dns": _dns(dns, start, end),
            "bond": _bond(bond),
            "equity": _equity_factors(factors, ensemble, industry),
            "brinson": _brinson(attribution),
            "stress": _stress_windows(stress),
            "bl": _parameter_grid(grid),
            "ablation": _ablation(performance, steps),
        }
        paths = []
        for name, figure in figures.items():
            path = figure_dir / FIGURE_FILES[name]
            figure.savefig(path, dpi=300, metadata={"Software": "Matplotlib"})
            figure.clear()
            paths.append(path)
    return paths

