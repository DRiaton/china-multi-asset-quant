
import logging

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

LOG = logging.getLogger(__name__)
MODELS = ("E1_ICIR", "E2_Ridge", "E3_LightGBM", "Ensemble")


def rank_signal(values) -> np.ndarray:
    ranks = pd.Series(np.asarray(values)).rank(pct=True).to_numpy()
    return 2 * (ranks - ranks.mean())


def realized_labels(panel, marked_close, trade_at_close, executions, benchmark):
    """Attach entry-to-next-entry labels with the exact timestamp of observability."""
    next_execution = executions.shift(-1)
    result = panel[["date", "code"]].copy()
    result["label_start"] = result["date"].map(executions)
    result["label_end"] = result["date"].map(next_execution)
    result["label"] = np.nan
    for _, group in result.dropna(subset=["label_end"]).groupby("date", sort=True):
        entry, exit_ = group[["label_start", "label_end"]].iloc[0]
        members = group["code"]
        opening = marked_close.loc[entry, members].to_numpy()
        closing = marked_close.loc[exit_, members].to_numpy()
        eligible = trade_at_close.loc[entry, members].to_numpy()
        index_return = benchmark.at[exit_] / benchmark.at[entry] - 1
        result.loc[group.index, "label"] = np.where(eligible, closing / opening - 1 - index_return, np.nan)
    return result[["label_start", "label_end", "label"]]


def monthly_rank_ic(panel: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    rows = []
    for day, sub in panel.groupby("date", sort=True):
        if sub["label"].notna().sum() < 2:
            continue
        valid = sub.dropna(subset=["label"])
        correlation = valid[factors].rank().corrwith(valid["label"].rank())
        rows.append({"date": day, "label_end": valid["label_end"].iloc[0], **correlation.to_dict()})
    return pd.DataFrame(rows).set_index("date").sort_index()


def rank_normalize_labels(panel: pd.DataFrame) -> pd.Series:
    """Scale observed labels within their own decision cross-section, with ties averaged."""
    ranks = panel.groupby("date")["label"].rank(method="average", pct=True)
    return (2 * (ranks - ranks.groupby(panel["date"]).transform("mean"))).rename("label_cs_rank")


def ensemble_posterior(realized: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Shrink realized model RankIC evidence toward the strategic model mix."""
    prior = np.asarray(cfg["ensemble_prior_weights"], dtype=float)
    months = realized.count().to_numpy()
    mature = months >= cfg["ensemble_min_months"]
    if not mature.any():
        return prior / prior.sum()
    mean = realized.mean().to_numpy()
    standard = realized.std().to_numpy()
    valid = mature & np.isfinite(standard) & (standard > 0)
    icir = np.divide(mean, standard, out=np.zeros_like(mean), where=valid).clip(0)
    strength = np.zeros_like(prior)
    if icir.sum() > 0:
        strength = icir / icir.sum() * months.max()
    posterior = prior * cfg["ensemble_prior_months"] + strength
    posterior[mature & (mean <= 0)] *= cfg["ensemble_nonpositive_multiplier"]
    return posterior / posterior.sum()


def walk_forward(panel, ic, cfg, seed, *, oos_start=None):
    factors = cfg["factors"]
    if "label_cs_rank" not in panel:
        panel = panel.assign(label_cs_rank=rank_normalize_labels(panel))
    groups = panel.groupby("date", sort=True)
    output, audit = [], []
    submodels = list(MODELS[:-1])
    pending, outcomes = [], []
    previous_signal = pd.Series(dtype=float)
    first_oos = pd.Timestamp.min if oos_start is None else pd.Timestamp(oos_start)
    lightgbm = None
    next_fit = pd.Timestamp.min
    last_light_fit = pd.NaT
    for date, current in groups:
        while pending and pending[0][1] <= date:
            prediction_day, label_end, scored, fitted = pending.pop(0)
            labels = groups.get_group(prediction_day).set_index("code")["label"].dropna()
            model_scores = scored.reindex(labels.index)
            evidence = model_scores.rank().corrwith(labels.rank()) if len(labels) > 1 else pd.Series(np.nan, index=submodels)
            outcomes.append({"date": prediction_day, "label_end": label_end, **evidence.where(fitted).to_dict()})
        cutoff = date - pd.DateOffset(months=cfg["ensemble_window_months"])
        recent = pd.DataFrame(
            [row for row in outcomes if row["label_end"] >= cutoff],
            columns=["date", "label_end", *submodels],
        )
        ensemble_weights = ensemble_posterior(recent[submodels], cfg)

        trailing = date - pd.DateOffset(months=cfg["train_months"])
        history = ic.loc[(ic.index >= trailing) & (ic["label_end"] <= date), factors]
        weights = np.ones(len(factors))
        e1_fallback = len(history) < cfg["min_ic_months"]
        if not e1_fallback:
            icir = history.mean().div(history.std().replace(0, np.nan)).clip(0, 2).fillna(0).to_numpy()
            if icir.sum() > 0:
                weights = icir
        matrix = current[factors]
        e1 = rank_signal(matrix.to_numpy() @ (weights / weights.sum()))
        purge = date - pd.DateOffset(months=cfg["purge_months"])
        train = panel.loc[
            panel["date"].ge(trailing) & panel["label_end"].le(purge) & panel["label"].notna()
        ]
        n_months = train["date"].nunique()
        ridge = e1
        if n_months >= cfg["ridge_min_months"]:
            ridge_model = Ridge(alpha=cfg["ridge_alpha"]).fit(train[factors], train["label_cs_rank"])
            ridge = rank_signal(ridge_model.predict(matrix))
        if n_months >= cfg["lightgbm_min_months"] and date >= next_fit:
            lightgbm = LGBMRegressor(**cfg["lgbm_params"], random_state=seed, n_jobs=1, verbosity=-1)
            lightgbm.fit(train[factors], train["label_cs_rank"])
            last_light_fit = train["label_end"].max()
            next_fit = date + pd.DateOffset(months=cfg["retrain_months"])
            LOG.info("LightGBM trained on %s months through label availability %s", n_months, last_light_fit.date())
        gbm = e1 if lightgbm is None else rank_signal(lightgbm.predict(matrix))
        scores = current[["date", "code", "model_industry", "industry", "total_mv_100m"]].copy()
        scores["E1_ICIR"], scores["E2_Ridge"], scores["E3_LightGBM"] = e1, ridge, gbm
        ensemble = pd.Series(scores[submodels].to_numpy() @ ensemble_weights, index=scores["code"])
        inertia = cfg["ensemble_previous_score_weight"]
        scores["Ensemble"] = ((1 - inertia) * ensemble + inertia * previous_signal.reindex(ensemble.index).fillna(ensemble)).to_numpy()
        previous_signal = scores.set_index("code")["Ensemble"]

        availability = current["label_end"].max()
        entry = current["label_start"].min() if "label_start" in current else date
        if pd.notna(availability) and entry >= first_oos:
            fitted = pd.Series([True, n_months >= cfg["ridge_min_months"], lightgbm is not None], index=submodels)
            pending.append((date, availability, scores.set_index("code")[submodels], fitted))
        output.append(scores)
        audit.append({
            "date": date,
            **{f"weight_{model}": weight for model, weight in zip(submodels, ensemble_weights)},
            **{f"ensemble_ic_months_{model}": int(recent[model].count()) for model in submodels},
            **{f"ensemble_latest_label_end_{model}": recent.loc[recent[model].notna(), "label_end"].max() for model in submodels},
            "ic_observed_months": len(history),
            "ic_latest_label_end": ic.loc[history.index, "label_end"].max(),
            "purge_cutoff": purge,
            "train_months": n_months,
            "ridge_latest_label_end": train["label_end"].max() if n_months >= cfg["ridge_min_months"] else pd.NaT,
            "lgbm_latest_label_end": last_light_fit,
            "icir_equal_weight_cold_start": e1_fallback,
            "ridge_icir_cold_start": n_months < cfg["ridge_min_months"],
            "lgbm_icir_cold_start": lightgbm is None,
        })
    audit = pd.DataFrame(audit).set_index("date")
    for key in ("ridge_icir_cold_start", "lgbm_icir_cold_start"):
        LOG.info("%s: %s decision months", key, audit[key].sum())
    return pd.concat(output, ignore_index=True), audit
