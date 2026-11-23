
import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

from src.utils.optimization import solve


def shrinkage_covariance(returns, decision, cfg, sessions_per_year):
    trailing = returns.loc[:decision].tail(cfg["cov_window"]).dropna()
    if len(trailing) < cfg["cov_min"]:
        raise ValueError(f"Only {len(trailing)} complete covariance rows at {decision:%Y-%m-%d}")
    cov = LedoitWolf().fit(trailing).covariance_ * sessions_per_year
    return cov + np.eye(len(cov)) * 1e-10


def equal_risk_contribution(covariance):
    """Minimize x'Sx/2 minus equal log budgets, then normalize."""
    n = len(covariance)
    if n == 1:
        return np.array([1.0])
    x = cp.Variable(n, pos=True)
    cov = covariance / np.trace(covariance)
    objective = 0.5 * cp.quad_form(x, cp.psd_wrap(cov)) - cp.sum(cp.log(x)) / n
    return solve(cp.Problem(cp.Minimize(objective)), x)


def hrp(covariance):
    n = len(covariance)
    if n == 1:
        return np.array([1.0])
    std = np.sqrt(np.diag(covariance))
    corr = covariance / np.outer(std, std)
    distance = np.sqrt(np.clip((1 - corr) / 2, 0, 1))
    np.fill_diagonal(distance, 0)
    ordered = leaves_list(linkage(squareform(distance, checks=False), method="single"))
    weights = np.ones(n)

    def cluster_variance(indices):
        subcov = covariance[np.ix_(indices, indices)]
        inverse_var = 1 / np.diag(subcov)
        w = inverse_var / inverse_var.sum()
        return w @ subcov @ w

    def bisect(indices):
        if len(indices) < 2:
            return
        left, right = np.split(indices, [len(indices) // 2])
        vleft, vright = cluster_variance(left), cluster_variance(right)
        weights[left] *= vright / (vleft + vright)
        weights[right] *= vleft / (vleft + vright)
        bisect(left)
        bisect(right)

    bisect(ordered)
    return weights


def black_litterman(prior, covariance, views, returns, omega, tau, delta):
    equilibrium = delta * covariance @ prior
    prior_uncertainty = tau * covariance
    cross = prior_uncertainty @ views.T
    view_cov = views @ cross + omega
    blended = equilibrium + cross @ np.linalg.solve(view_cov, returns - views @ equilibrium)
    uncertainty = prior_uncertainty - cross @ np.linalg.solve(view_cov, cross.T)
    posterior = covariance + uncertainty
    return blended, (posterior + posterior.T) / 2


def optimize_bl(assets, covariance, previous, macro, bond, cfg, bounds, *, delta, tau,
                strategic_equity_weight=None):
    equity_asset = next(asset for asset in assets if asset.startswith("equity"))
    bond_asset = next(asset for asset in assets if asset.startswith("bond"))
    allowed = {equity_asset, bond_asset, *cfg["satellite_caps"]}
    if set(assets) - allowed:
        raise ValueError("Black–Litterman accepts only equity, bond and approved satellite assets")
    strategic_weight = cfg["strategic_equity_weight"] if strategic_equity_weight is None else strategic_equity_weight
    strategic = pd.Series({equity_asset: strategic_weight, bond_asset: 1 - strategic_weight})
    prior = strategic.reindex(assets, fill_value=0).to_numpy()
    risk = macro.get("risk_appetite", 0.0)
    erp_z = np.nan_to_num(macro.get("erp_z_3y", 0))
    liquidity = np.nan_to_num(macro.get("liquidity_z", 0))
    funding_gate = bool(bond["credit_gate"])
    amplitudes = cfg["return_view_amplitude"]
    tilts = {
        equity_asset: (amplitudes["equity"], 0.50 * risk + 0.40 * np.tanh(erp_z / 2) + 0.10 * np.tanh(liquidity)),
        "gold": (amplitudes["gold"], 0.55 - risk),
        "convertible": (amplitudes["convertible"], 0.75 * risk + 0.25 * np.tanh(erp_z) - 0.55 * funding_gate),
    }
    views, increments, confidences = [], [], []
    for i, asset in enumerate(assets):
        if asset not in tilts:
            continue
        pick = np.zeros(len(assets))
        pick[i], pick[assets.index(bond_asset)] = 1.0, -1.0
        amp, signal = tilts[asset]
        confidence = np.interp(abs(signal), [0, 1], cfg["confidence_bounds"])
        confidence = np.clip(confidence * cfg["confidence_scale"], 0.01, 0.99)
        views.append(pick)
        increments.append(amp * signal)
        confidences.append(confidence)
    p = np.asarray(views)
    confidence = np.asarray(confidences)
    # At the reference tau this gives the Idzorek diagonal confidence mapping.
    # Holding its scale fixed across the grid independently varies prior uncertainty.
    omega = np.diag(cfg["tau"] * np.diag(p @ covariance @ p.T) * (1 - confidence) / confidence)
    q = p @ (delta * covariance @ prior) + np.asarray(increments)
    mean, posterior = black_litterman(prior, covariance, p, q, omega, tau, delta)
    w = cp.Variable(len(assets))
    eq = assets.index(equity_asset)
    constraints = [w >= 0, cp.sum(w) == 1, w[eq] >= bounds[0], w[eq] <= bounds[1]]
    for satellite, cap in cfg["satellite_caps"].items():
        if satellite in assets:
            constraints.append(w[assets.index(satellite)] <= cap)
    utility = mean @ w - delta / 2 * cp.quad_form(w, cp.psd_wrap(posterior))
    utility -= cfg["turnover_penalty"] * cp.norm1(w - previous.reindex(assets, fill_value=0).to_numpy())
    weights = solve(cp.Problem(cp.Maximize(utility), constraints), w)
    detail = {
        "core_cash_target": 0.0,
        "core_weight_sum": weights.sum(),
        "view_count": len(p),
        "relative_view_max_row_sum": np.abs(p.sum(axis=1)).max(),
        "prior_equity_weight": strategic_weight,
    }
    for i, asset in enumerate(a for a in assets if a in tilts):
        detail[f"view_{asset}_minus_bond"] = q[i]
        detail[f"confidence_{asset}"] = confidence[i]
    detail.update({f"posterior_{asset}": mu for asset, mu in zip(assets, mean)})
    return pd.Series(weights, index=assets), detail
