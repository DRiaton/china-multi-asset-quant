
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from sklearn.decomposition import PCA


def ns_loadings(maturities, decay):
    x = np.asarray(maturities, dtype=float) * decay
    if (x <= 0).any():
        raise ValueError("Nelson–Siegel maturities and decay must be positive")
    slope = -np.expm1(-x) / x
    return np.column_stack([np.ones_like(slope), slope, slope - np.exp(-x)])


def nelson_siegel(yields: pd.DataFrame, maturities, decay) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = yields.dropna()
    design = ns_loadings(maturities, decay)
    coefficients = np.linalg.lstsq(design, clean.to_numpy().T, rcond=None)[0].T
    fitted = pd.DataFrame(coefficients @ design.T, index=clean.index, columns=clean.columns)
    result = pd.DataFrame(coefficients, index=clean.index, columns=["level", "slope", "curvature"])
    result["rmse_bps"] = np.sqrt((clean - fitted).pow(2).mean(axis=1)) * 10000
    return result, fitted


def curve_pca(yields: pd.DataFrame, fit_sessions: int, oos_start) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = yields.dropna()
    train = clean.loc[clean.index < pd.Timestamp(oos_start)].iloc[:fit_sessions]
    if len(train) < fit_sessions:
        raise ValueError("Insufficient pre-OOS data for a frozen PCA fit")
    pca = PCA(n_components=3).fit(train)
    # Orient the levels consistently and expose the actual loadings for interpretation.
    sign = np.sign(pca.components_[:, -1])
    scores = pd.DataFrame(pca.transform(clean) * sign, index=clean.index, columns=["pc1", "pc2", "pc3"])
    scores = scores.loc[scores.index >= train.index[-1]]
    probe = clean.loc[clean.index >= pd.Timestamp(oos_start)]
    centered = probe - probe.mean()
    oos_share = np.sum(pca.transform(probe - probe.mean() + pca.mean_) ** 2, axis=0) / np.sum(centered.to_numpy() ** 2)
    summary = pd.DataFrame(pca.components_ * sign[:, None], index=scores.columns, columns=clean.columns)
    summary["fit_variance_share"] = pca.explained_variance_ratio_
    summary["oos_projected_variance_share"] = oos_share
    summary["fit_start"], summary["fit_end"] = train.index[0], train.index[-1]
    return scores, summary.rename_axis("component")


def par_bond_risk(yield_to_maturity, tenor: float):
    """Semiannual par bond modified duration and convexity, for integer-year tenors."""
    y = np.asarray(yield_to_maturity, dtype=float)[..., None]
    periods = np.arange(1, round(2 * tenor) + 1)
    discounts = (1 + y / 2) ** -periods
    flows = np.broadcast_to(y / 2, discounts.shape).copy()
    flows[..., -1] += 1
    pv = flows * discounts
    prices = pv.sum(axis=-1)
    duration = (pv * periods / 2 / (1 + y / 2)).sum(axis=-1) / prices
    convexity = (pv * periods * (periods + 1) / 4 / (1 + y / 2) ** 2).sum(axis=-1) / prices
    return duration, convexity


def carry_roll(yields: pd.DataFrame, maturities, tenor: float, months: int) -> pd.Series:
    complete = yields.dropna()
    years = months / 12
    spline = CubicSpline(maturities, complete.to_numpy(), axis=1, bc_type="natural")
    par_yield = spline(tenor)
    duration, _ = par_bond_risk(par_yield, tenor)
    result = par_yield - duration * (spline(tenor - years) - par_yield) / years
    return pd.Series(result, index=complete.index).reindex(yields.index)


def theoretical_return(yields: pd.DataFrame, maturities, tenor: float, sessions_per_year: int) -> pd.Series:
    """Constant tenor index, holding yesterday's par bond over one close-to-close session."""
    complete = yields.dropna()
    spline = CubicSpline(maturities, complete.to_numpy(), axis=1, bc_type="natural")
    level = pd.Series(spline(tenor), index=complete.index)
    risk = par_bond_risk(level, tenor)
    duration = pd.Series(risk[0], index=level.index).shift()
    convexity = pd.Series(risk[1], index=level.index).shift()
    roll_change = pd.Series(spline(tenor - 1 / sessions_per_year) - spline(tenor), index=level.index).shift()
    repricing = level.diff() + roll_change
    daily = level.shift() / sessions_per_year - duration * repricing + 0.5 * convexity * repricing.pow(2)
    return daily.reindex(yields.index)
