"""Render the public research documents from the completed offline run."""

import json
from pathlib import Path

import pandas as pd


def pct(value):
    return "—" if pd.isna(value) else f"{value:.2%}"


def num(value):
    return "—" if pd.isna(value) else f"{value:.2f}"


def table(frame, columns):
    headers = [name for name, _, _ in columns]
    rows = [
        "| " + " | ".join(str(formatter(row[column])).replace("|", "\\|") for _, column, formatter in columns) + " |"
        for row in frame.to_dict("records")
    ]
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "|".join(" --- " for _ in headers) + "|", *rows])


def write(path, content):
    Path(path).write_text(content.strip() + "\n", encoding="utf-8")


def figure_links(root):
    return "\n".join(
        f"- [{path.stem.replace('_', ' ')}](reports/figures/{path.name})"
        for path in sorted((root / "reports" / "figures").glob("*.png"))
    )


def version_comparison(outputs, series, exposures, attribution):
    snapshot_path = outputs / "v1_baseline_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8")) if snapshot_path.exists() else {}
    rows = []

    def add(name, before, after, formatter):
        rows.append({
            "metric": name,
            "v1": formatter(before),
            "v2": formatter(after),
            "delta": formatter(after - before),
        })

    shared = [
        ("CAGR", "cagr", pct), ("年化波动", "annualized_volatility", pct), ("Sharpe", "sharpe", num),
        ("最大回撤", "max_drawdown", pct), ("Calmar", "calmar", num),
    ]
    for layer, frame, strategies in series:
        for strategy in strategies:
            current = frame.loc[strategy]
            prior = snapshot.get(layer, {}).get(strategy, {})
            name = "Ensemble" if strategy == "Equity_Alpha_Sleeve" else strategy
            metrics = shared
            if layer == "equity":
                metrics = shared + [
                    ("IR", "info_ratio", num), ("单边年化换手", "annualized_turnover", pct),
                    ("年化成本（bps）", "annualized_cost_bps", num),
                ]
            for label, field, formatter in metrics:
                add(f"{name} · {label}", prior.get(field, float("nan")), current[field], formatter)
    top = exposures.loc[exposures["layer"].eq("top_level")].set_index("asset")["mean_weight"]
    for asset in ("equity", "bond", "cash"):
        weight = top[asset]
        add(f"旗舰顶层日均权重 · {asset}", snapshot.get("flagship_mean_weights", {}).get(asset, float("nan")), weight, pct)
    metrics_table = table(pd.DataFrame(rows), [
        ("指标", "metric", str), ("V1", "v1", str), ("V2", "v2", str), ("V2 − V1", "delta", str),
    ])

    rows.clear()
    for effect, value in attribution.set_index("effect")["annualized"].items():
        prior = snapshot.get("brinson", {}).get(effect, {}).get("annualized", float("nan"))
        add(effect, prior * 10_000, value * 10_000, num)
    attribution_table = table(pd.DataFrame(rows), [
        ("Brinson 年化效应（bps）", "metric", str), ("V1", "v1", str),
        ("V2", "v2", str), ("V2 − V1", "delta", str),
    ])
    notice = "" if snapshot else "复现目录中没有 V1 快照，表内显示可复现的 V2 指标。\n\n"
    return notice + metrics_table + "\n\n" + attribution_table


def build_reports(root, outputs):
    root, outputs = Path(root), Path(outputs)
    reports = root / "reports"
    reports.mkdir(exist_ok=True)

    def read(name):
        return pd.read_csv(outputs / f"{name}.csv")

    metadata = json.loads((outputs / "run_metadata.json").read_text(encoding="utf-8"))
    dates = metadata["first_session"], metadata["last_session"]
    performance = read("performance_summary")
    display_performance = performance.loc[~performance["strategy"].str.startswith(("BL_delta_", "Allocation_Cost_"))]
    equity = read("equity_performance")
    bond = read("bond_performance")
    stress = read("stress_tests")
    ablation = read("ablation_summary")
    attribution = read("brinson_attribution")
    grid = read("bl_grid_sensitivity")
    costs = read("cost_sensitivity")
    ic = read("equity_factor_ic_summary")
    tracking = read("bond_theoretical_tracking")
    pca = read("bond_pca_loadings")
    dns = read("bond_dns_factors")
    regimes = read("macro_regime_asset_stats")

    flagship = "Macro_BL_FixedIncomePlus"
    benchmark = "Static_20_80"
    index = performance.set_index("strategy")
    active, base = index.loc[flagship], index.loc[benchmark]
    equity_active = equity.set_index("strategy").loc["Equity_Alpha_Sleeve"]
    equity_delay = equity.set_index("strategy").loc["Equity_Execution_Delay"]
    bond_active = bond.set_index("strategy").loc["Bond_Active_Sleeve"]
    bond_base = bond.set_index("strategy").loc["Bond_Benchmark"]
    fit_share = pca["fit_variance_share"].sum()
    oos_share = pca["oos_projected_variance_share"].sum()
    dns_oos = dns.loc[dns["date"].between(*dates), "rmse_bps"]
    proxy_coverage = metadata["industry_proxy_coverage"]
    strict_coverage = metadata["industry_pit_coverage"]
    proxy_in_use = metadata["industry_static_proxy_enabled"]
    target_deviation = metadata["industry_max_target_deviation"]
    executed_deviation = metadata["industry_max_executed_deviation"]
    model_weights = "、".join(
        f"{name.removeprefix('weight_')} {pct(weight)}"
        for name, weight in metadata["ensemble_mean_oos_weights"].items()
    )
    min_grid, max_grid = grid["sharpe"].min(), grid["sharpe"].max()
    best_ic = ic.loc[ic["mean_rank_ic"].idxmax()]
    worst_ic = ic.loc[ic["mean_rank_ic"].idxmin()]
    crisis = stress.loc[stress["event"].eq("redemption_2022")].set_index("strategy")
    crisis_active = crisis.loc[flagship]
    crisis_base = crisis.loc[benchmark]
    reversal = stress.loc[stress["event"].eq("reversal_2024")].set_index("strategy")
    overlay_base = index.loc["Macro_BL_Equity_Bond_Active"]
    satellites = index.loc["Macro_BL_Satellites"]
    allocation_cost = costs.loc[costs["layer"].eq("allocation")].set_index("cost_multiplier")
    bond_bear = stress.loc[stress["event"].eq("bond_bear_2020")].set_index("strategy")
    bond_daily = pd.read_parquet(outputs / "bond_daily.parquet")
    bond_path = bond_daily.loc[bond_daily["strategy"].eq("Bond_Active_Sleeve")].set_index("date")
    gate_transition = bond_path["credit_gate"] & ~bond_path["credit_gate"].shift(fill_value=False)
    event_dates = crisis.loc["Bond_Active_Sleeve", ["start", "end"]]
    gate_transition = gate_transition.loc[event_dates.iloc[0]:event_dates.iloc[1]]
    gate_days = gate_transition[gate_transition].index
    gate_note = "该窗口未观测到信用闸门由关转开的记录。"
    if len(gate_days):
        gate_day = gate_days[0]
        before, after = bond_path.loc[gate_day:].iloc[:2].itertuples()
        gate_note = (
            f"{gate_day:%Y-%m-%d} 收盘信用闸门由关转开，"
            f"估计久期从当日的 {num(before.modified_duration)} 年变为次交易日收盘的 {num(after.modified_duration)} 年，"
            f"信用权重分别为 {pct(before.credit_weight)} 和 {pct(after.credit_weight)}。"
        )
    linked = attribution.set_index("effect")["annualized"]
    equity_models = equity.set_index("strategy")
    exposures = read("flagship_exposure_summary")
    comparison = version_comparison(outputs, [
        ("allocation", index, [benchmark, flagship]),
        ("equity", equity_models, ["E1_ICIR", "Equity_Alpha_Sleeve"]),
        ("bond", bond.set_index("strategy"), ["Bond_Active_Sleeve", "Bond_Benchmark"]),
    ], exposures, attribution)
    equity_gap = equity_active["info_ratio"] - equity_models.at["E1_ICIR", "info_ratio"]
    bond_gap = bond_active["sharpe"] - bond_base["sharpe"]
    results_note = (
        f"旗舰相对静态 20/80 的年化复合收益差为 {pct(active['cagr'] - base['cagr'])}。"
        f"Ensemble 相对 E1 的 IR 差为 {num(equity_gap)}，单边年化换手为 "
        f"{pct(equity_active['annualized_turnover'])}，可对照 220%–350% 的年化换手区间。"
        f"主动债相对均衡债底的 Sharpe 差为 {num(bond_gap)}。"
        f"Brinson 年化残差为 {metadata['brinson_annualized_residual_bps']:.6f} bps。"
    )
    exposure_table = table(exposures, [
        ("层级", "layer", str), ("资产", "asset", str), ("日均权重", "mean_weight", pct),
    ])
    top_cash = metadata["flagship_mean_exposure"]["top_level"]["cash"]
    total_cash = metadata["flagship_mean_exposure"]["lookthrough"]["cash"]
    implementation_note = (
        f"旗舰顶层现金日均 {pct(top_cash)}，穿透现金日均 {pct(total_cash)}。"
        f"优化目标行业偏离绝对值最大为 {pct(target_deviation)}，执行后收盘最大为 {pct(executed_deviation)}。"
        f"集成模型样本外平均权重为 {model_weights}。"
    )
    sample_limit = (
        "V2 根据 V1 在同一历史区间的诊断重构，评估采用按时间推进的回测。"
        "该区间已参与研究决策，独立留出期仍待验证。保留原有 BL 敏感性网格，未据其结果重选参数。"
    )
    sector_setting = "启用" if proxy_in_use else "关闭"
    sector_setting_en = "enables" if proxy_in_use else "disables"
    sector_limit = (
        f"本次配置{sector_setting} Baostock 样本期结束后抓取的证监会一级行业静态代理。"
        f"按月度样本加总，静态代理覆盖率为 {pct(proxy_coverage)}，严格点时覆盖率为 {pct(strict_coverage)}。"
        "代理服务于回顾性行业暴露控制。"
        "资料无法核实历史真实行业或迁移频率。严格模式可关闭代理并保留原始点时审计。"
        "接入经核验的 Wind 历史点时行业表后，可在同一接口重估。"
    )

    core = [
        ("Strategy", "strategy", str), ("CAGR", "cagr", pct), ("Volatility", "annualized_volatility", pct),
        ("Sharpe", "sharpe", num), ("Max drawdown", "max_drawdown", pct),
    ]
    equity_core = core + [("IR", "info_ratio", num), ("TE", "tracking_error", pct)]
    selected_equity = equity.loc[~equity["strategy"].str.startswith(("Equity_Cost", "Equity_monthly", "Equity_bimonthly", "Equity_quarterly"))]
    selected_bond = bond.loc[~bond["strategy"].str.contains("_Cost_")]
    stress_selected = stress.loc[stress["strategy"].isin([benchmark, flagship])]
    main_table = table(display_performance, core)
    figure_index = figure_links(root)
    feature_figures = {
        "01_allocation_nav.png": "Cumulative NAV comparison",
        "03_flagship_weights_regimes.png": "Flagship exposures and macro regime",
        "07_brinson_waterfall.png": "Brinson annualized attribution",
    }
    embeds = "\n\n".join(
        f"![{caption}](reports/figures/{filename})" for filename, caption in feature_figures.items()
        if (reports / "figures" / filename).exists()
    )

    write(root / "README.md", f"""
# China Macro-Regime Fixed-Income-Plus & Multi-Asset Research · V2

An offline research pipeline for China A-share stock selection, CGB yield-curve and credit rotation, and macro-driven stock–bond allocation. The historical evaluation spans **{dates[0]} to {dates[1]}** using the supplied local market data.

## Architecture

```mermaid
flowchart TD
    A["Local Parquet data and source audit"] --> B["Point-in-time joins, release lags and ETF NAV returns"]
    B --> C["A-share factors and walk-forward stock sleeve"]
    B --> D["CGB curve, duration and credit sleeve"]
    B --> E["Macro quadrants, filtered probabilities and equity risk premium"]
    C --> F["Static, ERC, HRP, ERP and Black–Litterman allocation"]
    D --> F
    E --> F
    F --> G["Next-close simulation, costs and overlay"]
    G --> H["Attribution, stress tests, sensitivity and reports"]
```

- **Equity.** Sixteen factors, industry and size residualization, rank-label Ridge and LightGBM, and an ensemble weighted by matured out-of-sample rank IC.
- **Fixed income.** Frozen pre-evaluation PCA, Nelson–Siegel curve estimates and ETF rotation around the 5.27-year benchmark duration, with a tail credit gate.
- **Allocation.** Fully invested Black–Litterman core with 20/80 and 40/60 strategic anchors, bond-relative views and a trailing drawdown overlay on equity and convertibles. The 252-session drawdown thresholds are 3.5% and 6%. The annual volatility targets are 4.5% and 9%. The overlay sends released weight equally to the bond sleeve and cash at next-session execution.
- **Audit.** Exchange-session timing, an expanding forward regime filter, full return and order outputs, and reproducible research exhibits.

## Historical results

{main_table}

The flagship recorded a CAGR of {pct(active["cagr"])}, a Sharpe of {num(active["sharpe"])} and a maximum drawdown of {pct(active["max_drawdown"])}. The corresponding Static 20/80 figures were {pct(base["cagr"])}, {num(base["sharpe"])} and {pct(base["max_drawdown"])}. Sharpe uses the money-market ETF as the risk-free proxy. Results include simulated costs. Detailed metric definitions are in the [research report](reports/Research_Report.md).

## V1 → V2 实证诊断与金融机理迭代复盘

{sample_limit}

核心 BL 满仓和相对债券观点用于修正现金泄漏。债券重设久期锚与尾部信用闸门，股票加入截面秩训练、因果集成、换手预算及行业代理约束。

{comparison}

{results_note}

{implementation_note}

{exposure_table}

百分比指标的差值以百分点表示。敞口分别列示顶层权重与债券袖套穿透后的资产权重。 [旗舰敞口](outputs/flagship_exposure_summary.csv)

{embeds}

{figure_index}

## Reproduce

With the existing project Python environment and the local tables in `data/raw/`, run these commands from the repository root in PowerShell.

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --stage all
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_MultiAsset_Research_Walkthrough.ipynb
```

Individual stages are `data`, `equity`, `bond`, `allocation`, and `report`. They reuse earlier local outputs. The [configuration](configs/strategy_config.yaml), [raw-data audit](outputs/raw_audit.json), [performance CSV](outputs/performance_summary.csv), [research report](reports/Research_Report.md), and [notebook](notebooks/01_MultiAsset_Research_Walkthrough.ipynb) are the main entry points.

## Data limits

Constituent snapshots are discrete and original macro vintages are unavailable. This run {sector_setting_en} the forward-dated Baostock CSRC L1 classification as a retrospective structural exposure proxy. The proxy covers {pct(proxy_coverage)} of eligible stock-month observations. Strict point-in-time coverage is {pct(strict_coverage)}. The source provides no evidence about historical labels or migration frequency. The strict audit and a switch to disable the proxy remain available. Validated point-in-time classifications can replace the proxy. Trades execute at the next session close. An execution-time sector repair uses the decision-date universe and reference with current marks and tradability. Market moves between rebalances can change realized sector exposures. Sleeve accounting omits consolidated security orders, market depth and queue effects. [Raw-data audit](outputs/raw_audit.json)

## 中文概要

项目以 A 股点时选股与债券 ETF 久期、信用轮动为底座，将宏观概率、ERP 和债市信号映射至 Black–Litterman，并用统一的扣费净值、逐层消融、危机窗口和逐日勾稽归因评估策略。旗舰 CAGR 为 **{pct(active["cagr"])}**，静态 20/80 为 **{pct(base["cagr"])}**。历史行业覆盖与数据披露口径见[研究报告](reports/Research_Report.md)。
""")


    risk_columns = [
        ("策略", "strategy", str), ("Sortino", "sortino", num), ("Calmar", "calmar", num),
        ("修复天数", "max_recovery_days", num), ("VaR 95%", "daily_var_95", pct),
        ("CVaR 95%", "daily_cvar_95", pct), ("VaR 99%", "daily_var_99", pct),
        ("CVaR 99%", "daily_cvar_99", pct), ("月胜率", "monthly_win_rate", pct),
        ("年化换手", "annualized_turnover", pct), ("成本 bps/年", "annualized_cost_bps", num),
        ("DSR 概率", "deflated_sharpe_probability", pct), ("夏普区间下沿", "sharpe_ci_low", num),
        ("夏普区间上沿", "sharpe_ci_high", num),
    ]
    stress_table = table(stress_selected, [
        ("事件", "event", str), ("策略", "strategy", str),
        ("累计净收益", "cumulative_return", pct), ("最大回撤", "max_drawdown", pct),
    ])
    tracking_table = table(tracking, [
        ("理论指数", "theoretical_index", str), ("ETF 资产", "etf", str),
        ("对齐交易日", "sessions", str), ("日收益相关", "correlation", num),
        ("日均残差 bps", "mean_daily_residual_bps", num), ("年化跟踪残差", "annualized_tracking_residual", pct),
    ])
    regime_table = table(regimes.loc[regimes["asset"].isin(["equity", "cgb5", "credit", "gold"])], [
        ("模型", "engine", str), ("状态", "regime", str), ("资产", "asset", str),
        ("月份数", "months", str), ("条件年化收益", "annualized_return", pct), ("条件夏普", "sharpe", num),
    ])
    costs_table = table(
        costs, [(col, col, pct if col in {"cagr", "max_drawdown", "annualized_volatility", "annualized_turnover"} else
                 num if pd.api.types.is_numeric_dtype(costs[col]) else str) for col in costs.columns]
    )
    write(reports / "Research_Report.md", f"""
# 中国宏观状态驱动的固收+与多资产组合研究 · V2

## 摘要

时间顺序评估区间为 {dates[0]} 至 {dates[1]}。旗舰策略将股票与债券 ETF 袖套纳入宏观 Black–Litterman 配置，并实施动态风险控制。旗舰样本 CAGR、夏普、最大回撤分别为 **{pct(active["cagr"])}、{num(active["sharpe"])}、{pct(active["max_drawdown"])}**，静态 20/80 对应为 **{pct(base["cagr"])}、{num(base["sharpe"])}、{pct(base["max_drawdown"])}**。

## V1 → V2 实证诊断与金融机理迭代复盘

{sample_limit}

V1 的现金敞口诊断促使 V2 将货币现金移出 BL 核心并将叠加减仓限定在风险资产。主动债底以静态久期锚定常态配置，信用闸门留给尾部冲击。股票端从原收益拟合和等权模型混合改为截面秩标签和因果集成，再用行业代理及换手预算控制暴露与实施成本。四项改动共同进入本轮重跑，表中差值是整体版本变化。

{comparison}

{results_note}

{implementation_note}

{exposure_table}

百分比指标的差值以百分点表示。顶层债券袖套内部仍可持有货币 ETF，穿透现金包括两层持有。 [敞口明细](../outputs/flagship_exposure_summary.csv)

## 1. 信息集、行情与交易时序

- 月末收盘生成股票和顶层基础权重，下一交易日收盘按当时可交易状态执行。成交日收盘前旧仓位承担当日收益，收盘成交后新仓位承担后续涨跌。动态风险控制也遵守次日执行。暂停交易的标的保留仓位，缺失价格沿用最近观察估值，恢复可观察报价后反映变化。缺少盘口、涨跌停队列、退市回收和强制处置价格数据，流动性冲击由固定 bps 近似。
- 44 个成分股查询截面对应 40 个不同的有效快照，决策使用当时最新且已生效的整期成分股。供应商记录可能在生效日之后查询，工程以提供的成分生效日重建历史。快照间缺少临时成分变动的完整记录。该方法降低了以最新成分股倒推所致的幸存者偏差，覆盖程度取决于历史快照和单股数据。
- 个股估值按股票做向后 as-of，并限制年龄。{sector_limit} [原始数据审计](../outputs/raw_audit.json)
- PMI 用次月首个交易日，CPI/PPI、货币和社融、GDP 按配置的次月或季末后日历滞后代理发布时间。日频利率仅用于后续交易决策。原始未修订宏观版本及逐月精确发布时点缺失。2026-04 之后社融数据缺乏更新，超过 120 天时从增长合成剔除。Shibor 最晚观察日期为 2026-11-13，报表保留年龄用于判断陈旧程度。
- ETF 使用官方复权净值收益，现金 ETF 511880 取累计净值变化以避开单位净值年末重置。未复权收盘价用于成交能力核验。动态 ETF 需满 60 个实际交易日，配置协方差还要求至少 120 条完整可用记录。
- 股票按双边佣金 2.5 bps、每边固定滑点 5 bps 计提，卖出另计印花税，2023-08-28 前为 10 bps，此后为 5 bps。ETF 每边 3 bps。单边换手按实际买入与卖出权重绝对变化之和的一半计量，开始日由现金建仓。净额基于自融资成交记账，市值漂移先于调仓。

## 2. 股票研究

价值、质量、动量与反转、低波和交易特征覆盖 16 个指标。例如，PB 除以 PE 为隐含 ROE，EP 为正 PE 的倒数，12–1 动量跳过最近 21 个交易日，残差波动反映对沪深 300 暴露以外的波动。截面先按 5×MAD 缩尾，再做带截距的规模和可得行业中性化，最后标准化。行业未知时用全截面可得中位数作缺失值回退，原始缺失率另行监控。

预测标签取相邻月度成交收盘间的个股收益减沪深 300 指数收益。Ridge 与 LightGBM 以最多 36 个月内各训练截面的相对收益秩（映射至 [−1, 1]）拟合，原收益继续用于评估。模型训练保留标签成熟时点和 purge 隔离。月末集成只使用此前已形成的样本外预测及截至决策时已成熟的标签，按最近 24 个月的非负模型 ICIR 与 60/20/20 先验收缩。回退月数见模型日志。打分轻量 EMA 平滑。目标单边月度换手预算为 25%，初始建仓豁免，约束冲突时记录例外。单股目标上限为 3%，有效持股数至少为 40。行业小组并入 OTHER，使用行业代理和规模做截面 OLS。优化目标对代理行业相对决策日市值基准的偏离上限为 ±4%。执行日仅用预定成分范围、决策日基准和当时可交易状态及估值修复行业敞口，使成交后收盘权重也受 ±4% 约束，保留冻结仓位与原信号。两次调仓间的市场价格仍会引起行业漂移。 [模型权重](../outputs/equity_ensemble_weights.csv) · [行业偏离](../outputs/equity_industry_deviations.csv)

{table(selected_equity, equity_core)}

对沪深 300 **价格指数**计算相对指标，分红口径差异会影响指增 IR 的解读。净 IR 为 {num(equity_active["info_ratio"])}，多延迟一交易日执行的 IR 为 {num(equity_delay["info_ratio"])}。E1 与 LightGBM 的 CAGR 分别为 {pct(equity_models.at["E1_ICIR", "cagr"])} 和 {pct(equity_models.at["E3_LightGBM", "cagr"])}。模型收益同时受到预测、持仓约束和成本路径影响。40/50/60 等权持仓及各调仓频率的敏感性结果见 [股票绩效表](../outputs/equity_performance.csv)。

分月以 Spearman 相关计算 RankIC，年化 ICIR 用月均 IC 除以波动乘 √12。Newey–West 标准误缓解序列相关。最高平均 RankIC 为 {best_ic["factor"]} 的 {num(best_ic["mean_rank_ic"])}，最低为 {worst_ic["factor"]} 的 {num(worst_ic["mean_rank_ic"])}。IC 衡量排序关系，经济价值还受极端组合收益、成交损耗及选股约束影响。

{table(ic, [
        ("因子", "factor", str), ("月份", "months", str), ("平均 RankIC", "mean_rank_ic", num),
        ("年化 ICIR", "annualized_icir", num), ("NW t 值", "newey_west_t", num),
        ("IC 胜率", "ic_win_rate", pct), ("月均五组多空", "mean_monthly_long_short", pct),
        ("分组单调性", "quintile_monotonicity", num),
    ])}

完整五分组收益与目标换手见 [逐因子统计](../outputs/equity_factor_ic_summary.csv) 和 [分组收益](../outputs/equity_factor_groups.csv)。

## 3. 债券研究

对每个交易日的国债曲线按期限一次性最小二乘拟合。设 x = λτ，斜率载荷为 A = (1 − exp(−x))/x，曲率载荷为 A − exp(−x)。拟合收益率等于 β₁ + β₂A + β₃(A − exp(−x))，λ = 0.7308/年。远端的 A 与曲率载荷趋于零，因此 β₁ 表示长期水平。样本外 DNS 拟合的日均和中位 RMSE 分别为 **{num(dns_oos.mean())} bps** 与 **{num(dns_oos.median())} bps**。

前三个 PCA 方向使用 {pca.iloc[0]["fit_start"]} 至 {pca.iloc[0]["fit_end"]} 的 252 条完整样本拟合。训练方差解释率合计 **{pct(fit_share)}**，同一冻结子空间的样本外投影方差占比 **{pct(oos_share)}**。期限端点的载荷随市场主导因子变化，片面将 PC1–PC3 等同于纯平行移动、斜率和曲率会忽略这个变化。

对 T 年平价债，用半年票息现金流的折现权重求修正久期，并对收益率二阶导数求凸性。自然三次样条在昨日曲线上取得同期限和剩余期限收益率。年化一月 Carry+Roll 近似为 y(T) − 12 × Dmod × [y(T−1/12) − y(T)]。理论指数以昨日票息滚存、当日同期限利率变化、昨日曲线滚降之和估算持有收益，价格变化项为 −Dmod × Δy + ½ × 凸性 × Δy²。曲线水平和理论指数服务于信号及校验，真实轮动使用复权 ETF 收益。现金 ETF 与短国债的底层风险不同。

{tracking_table}

主动久期围绕约 5.27 年的等权 5 年国债、10 年国债和城投债基准锚定。得分综合利率动量（40%）、期限斜率和 10 年 Carry+Roll（35%）、宏观防守信号（25%）。组合采用固定 ETF 久期近似。信用闸门结合信用冲击历史 95% 分位、15 bps 下限与资金面 2σ 阈值。触发时目标久期上限为 3.5 年，城投 ETF 上限 5%，货币 ETF 上限由 30% 放宽到 50%。

{table(selected_bond, core)}

债券主动组合与均衡债底 CAGR 差为 **{pct(bond_active["cagr"] - bond_base["cagr"])}**。久期增强定义为久期单策略净收益减利率基准净收益，信用增量为主动组合减久期单策略，两项基于不同的重跑组合，包含替代交易和约束联动。2020 年债熊期间，两者累计净收益分别为 {pct(bond_bear.at["Bond_Active_Sleeve", "cumulative_return"])} 和 {pct(bond_bear.at["Bond_Benchmark", "cumulative_return"])}。2022 年理财赎回子窗口内，主动债与均衡债底分别为 {pct(crisis.at["Bond_Active_Sleeve", "cumulative_return"])} 和 {pct(crisis.at["Bond_Benchmark", "cumulative_return"])}。{gate_note}每日分解见 [久期信用增量](../outputs/bond_duration_credit_effects.csv)。

## 4. 宏观状态与组合构建

增长合成 PMI 水平、PMI 动量和可用社融，通胀合成 CPI/PPI 同比，流动性合成 M1–M2 剪刀差变化、Shibor 变化和期限利差。ERP 等于沪深 300 盈利收益率减 10 年国债收益率。各项只在尾随窗口内标准化，ERP 同时输出 3 年和 5 年分位数及 Z 分数。R1 以增长和通胀 Z 分数的符号判定四象限。R2 以当期特征至四象限锚点的相对得分生成软分类，用截至上一观察时点的扩展状态转移计数形成先验，乘当期观测概率并归一化。参数固定、状态逐期前向更新，保留 filtered probabilities。状态下资产收益按后续持有段测算，样本月数和可投资产的可得区间不同，不能将条件收益差异直接视为普遍的周期规律。

{regime_table}

静态模型按月回到固定股债比例。ERC 用 Ledoit–Wolf 收缩协方差并求解等风险贡献，HRP 用相关距离、单链接和按簇方差的递归二分。在仅含两条腿时，ERC 按逆波动率配置，HRP 按逆方差配置，后者进一步偏向低波动债券。这个资产池对层次分散的检验力有限。ERP 战术仓位由因果滚动分位线性映射。

BL 用战略股债权重（固收+ 20/80、平衡版 40/60）与收缩协方差形成均衡收益 Π = δΣw₀。宏观风险偏好、ERP 的 3 年滚动 Z 分数和流动性形成权益相对债券的观点，卫星版增设可转债相对债券、黄金相对债券的观点。令 C = τΣ，后验均值为 Π + CPᵀ(PCPᵀ + Ω)⁻¹(Q − PΠ)，后验风险为 Σ 加上条件均值估计误差。Ω 的对角项结合相对观点的风险与信号置信度，敏感性网格固定构造 Ω 所用的参考 τ。

凸优化以扣除均值方差风险及 L1 换手惩罚的后验收益为目标，在核心资产内满仓且禁止卖空。固收+权益走廊为 5%–35%，平衡版为 15%–65%。可转债和黄金上限分别为 10% 和 12%。叠加层按尾随 252 个交易日净值高点计算回撤，触及 3.5% 或 6% 时缩减权益和可转债。事前滚动波动率超过年化目标（固收+ 4.5%、平衡版 9%）也会压降这两类敞口。释放权重按债底与现金各 50% 分配，经交易带判断后于下一交易日收盘执行。

## 5. 同口径绩效与消融

{main_table}

{table(display_performance, risk_columns)}

收益按净值复合年化，波动使用日收益标准差按 √252 年化，Sharpe 以货币 ETF 为日无风险代理。最大回撤以初始净值与历次高点为参考。月胜率衡量正净收益月份占比。VaR 和 CVaR 分别取日净收益左尾阈值及左尾平均的相反数，以损失率表示。最长水下时间用自然日计量，期末尚未修复的区间也纳入。夏普区间基于 1,000 次 21 日块重抽样，保留块内序列关系。DSR 考虑试验数量、夏普横截面离散程度和收益高阶矩，相关策略和研究者已尝试的全部试验数量无法完全辨识，解释时需保留这一不确定性。

{table(ablation, [
        ("步骤", "step", str), ("策略", "strategy", str), ("CAGR", "cagr", pct),
        ("Sharpe", "sharpe", num), ("最大回撤", "max_drawdown", pct),
        ("较上一步 CAGR", "delta_cagr", pct), ("较上一步 Sharpe", "delta_sharpe", num),
    ])}

主动底层变更了配置输入的协方差和收益路径，因此消融差分包含底层本身和顶层重新配置的共同效应。叠加前后最大回撤分别为 {pct(overlay_base["max_drawdown"])} 和 {pct(active["max_drawdown"])}，CAGR 分别为 {pct(overlay_base["cagr"])} 和 {pct(active["cagr"])}。加入黄金和转债后 CAGR 为 {pct(satellites["cagr"])}，其中同时包含卫星暴露与原有权重变化。

## 6. Brinson 归因与危机复盘

日度用策略和基准的实际期初权重及组别毛收益拆分配置、选股、债券选择、现金实施及交互项。成本效应等于基准成本减策略成本，保留底层和顶层记账。Carino 用当日对数收益差除以当日简单收益差来链接各日效应，年化列映射到两策略 CAGR 之差。归因的年化加总残差为 {metadata["brinson_annualized_residual_bps"]:.6f} bps。嵌套 sleeve 为报表里的模型敞口，实际下单前需要把底层证券合并并重新评估成交与费用。

{table(attribution, [
        ("效应", "effect", str), ("年化贡献", "annualized", pct), ("累计净值差贡献", "cumulative", pct),
    ])}

{stress_table}

配置、债券选择、股票选择和交易成本的年化贡献分别为 {linked["asset_allocation"] * 10000:+.2f}、{linked["fixed_income_selection"] * 10000:+.2f}、{linked["equity_selection"] * 10000:+.2f} 和 {linked["transaction_cost_drag"] * 10000:+.2f} bps。配置效应汇总权益、债券和现金的配置偏离，解读单一观点需结合持仓路径及消融结果。

赎回潮子窗口内，旗舰净收益 {pct(crisis_active["cumulative_return"])}，静态 20/80 为 {pct(crisis_base["cumulative_return"])}。该窗口同时含利率、信用和权益变动，解释净值差应结合当期信用闸门、持仓久期与权益权重。2024 年快速反转时，旗舰与静态 20/80 的累计收益分别是 {pct(reversal.at[flagship, "cumulative_return"])} 和 {pct(reversal.at[benchmark, "cumulative_return"])}。这段行情可用来检查风险缩放与再平衡对反弹捕捉的影响。完整策略与区间明细见 [历史压力测试](../outputs/stress_tests.csv)。

## 7. 参数敏感性与下一步可验证问题

BL 风险厌恶和先验置信度网格的 Sharpe 范围为 **{num(min_grid)} 至 {num(max_grid)}**。网格用于检查结果是否依赖单个参数。给定相同历史测试期，事后最优点会带入选择偏差。

{table(grid, [("δ", "delta", num), ("τ", "tau", num), ("CAGR", "cagr", pct), ("Sharpe", "sharpe", num)])}

{costs_table}

将全链路费用由基准水平提高至两倍，旗舰两次的 CAGR 分别为 {pct(allocation_cost.at[1.0, "cagr"])} 和 {pct(allocation_cost.at[2.0, "cagr"])}，年化成本分别为 {num(allocation_cost.at[1.0, "annualized_cost_bps"])} 和 {num(allocation_cost.at[2.0, "annualized_cost_bps"])} bps。上下层净收益反馈到再平衡路径，所以成本压力结果基于重跑。

进一步研究的优先验证项是历史行业分类与指数官方权重、宏观原始版本与真实发布时间、分红一致的指增基准，以及用逐券或逐股成交重建评估 sleeve 合并后换手和容量。同一框架还可以冻结预注册参数、延长真正未参与模型迭代的验证期，并对信用模型的闸门触发时点、资金利率数据断点和极端赎回压力进行逐日复盘。

数据表均来自 [pipeline 输出目录](../outputs/)。因子构造、调仓日、成本、风险参数和事件窗口见 [策略配置](../configs/strategy_config.yaml)。
""")




