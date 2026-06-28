"""
Impulse response functions and stochastic simulation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .gensys import GensysResult


# ------------------------------------------------------------------ #
# IRF                                                                  #
# ------------------------------------------------------------------ #

def compute_irf(
    result: GensysResult,
    shock_index: int = 0,
    shock_size: float = 1.0,
    periods: int = 40,
    var_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute impulse responses to a one-unit shock.

    Parameters
    ----------
    result : GensysResult from gensys()
    shock_index : which column of R (the shock) to hit
    shock_size : scale (default 1.0, i.e. 1 std dev)
    periods : number of periods (including impact period 0)
    var_names : labels for columns; defaults to y0, y1, ...

    Returns
    -------
    DataFrame, shape (periods, n_variables)
    """
    if not result.success:
        raise ValueError("Cannot compute IRF: model has no unique solution.")

    n = result.T.shape[0]
    n_e = result.R.shape[1]

    if var_names is None:
        var_names = [f"y{i}" for i in range(n)]

    irf = np.zeros((periods, n))
    # Impact period: shock hits
    eps = np.zeros(n_e)
    eps[shock_index] = shock_size
    irf[0] = result.R @ eps + result.const

    # Subsequent periods: propagate
    for t in range(1, periods):
        irf[t] = result.T @ irf[t - 1] + result.const

    return pd.DataFrame(irf, columns=var_names, index=pd.RangeIndex(periods, name="period"))


def compute_all_irfs(
    result: GensysResult,
    shock_names: Optional[List[str]] = None,
    var_names: Optional[List[str]] = None,
    periods: int = 40,
) -> Dict[str, pd.DataFrame]:
    """
    Compute IRFs for all shocks.  Returns {shock_name: DataFrame}.
    """
    n_e = result.R.shape[1]
    if shock_names is None:
        shock_names = [f"eps{i}" for i in range(n_e)]

    return {
        shock_names[i]: compute_irf(
            result, shock_index=i, periods=periods, var_names=var_names
        )
        for i in range(n_e)
    }


# ------------------------------------------------------------------ #
# Stochastic simulation                                                #
# ------------------------------------------------------------------ #

def simulate(
    result: GensysResult,
    shock_cov: Optional[np.ndarray] = None,
    periods: int = 200,
    burn: int = 50,
    seed: Optional[int] = None,
    var_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Simulate the model forward by drawing shocks.

    Parameters
    ----------
    result : GensysResult
    shock_cov : (n_e, n_e) covariance matrix of eps_t.
                If None uses identity (all shocks iid N(0,1)).
    periods : number of periods to return (after burn-in)
    burn : burn-in periods discarded
    seed : random seed

    Returns
    -------
    DataFrame, shape (periods, n_variables)
    """
    if not result.success:
        raise ValueError("Cannot simulate: model has no unique solution.")

    rng = np.random.default_rng(seed)
    n = result.T.shape[0]
    n_e = result.R.shape[1]

    if shock_cov is None:
        shock_cov = np.eye(n_e)

    L = np.linalg.cholesky(shock_cov)   # lower-triangular factor

    total = burn + periods
    out = np.zeros((total, n))
    y = np.zeros(n)

    for t in range(total):
        eps = L @ rng.standard_normal(n_e)
        y = result.T @ y + result.R @ eps + result.const
        out[t] = y

    if var_names is None:
        var_names = [f"y{i}" for i in range(n)]

    return pd.DataFrame(
        out[burn:], columns=var_names, index=pd.RangeIndex(periods, name="period")
    )


# ------------------------------------------------------------------ #
# Forecast error variance decomposition (FEVD)                         #
# ------------------------------------------------------------------ #

def fevd(
    result: GensysResult,
    shock_cov: Optional[np.ndarray] = None,
    horizon: int = 20,
    var_names: Optional[List[str]] = None,
    shock_names: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Forecast error variance decomposition.

    Returns
    -------
    dict with 'variance' and 'share' DataFrames, both shape (horizon, n_vars).
    Each column = total variance / share for that variable at each horizon.
    Plus per-shock DataFrames under key 'by_shock'.
    """
    if not result.success:
        raise ValueError("Cannot compute FEVD: model has no unique solution.")

    n = result.T.shape[0]
    n_e = result.R.shape[1]

    if shock_cov is None:
        shock_cov = np.eye(n_e)
    if var_names is None:
        var_names = [f"y{i}" for i in range(n)]
    if shock_names is None:
        shock_names = [f"eps{i}" for i in range(n_e)]

    T, R = result.T, result.R
    L = np.linalg.cholesky(shock_cov)
    RL = R @ L   # Cholesky-scaled impact

    # var_by_shock[j][h] = contribution of shock j to variance of each variable at h
    var_total = np.zeros((horizon, n))
    var_by_shock = {s: np.zeros((horizon, n)) for s in shock_names}

    T_power = np.eye(n)  # T^0 at h=0
    for h in range(horizon):
        for j, sname in enumerate(shock_names):
            r = T_power @ RL[:, j]   # response at h to unit Chol shock j
            contrib = r ** 2
            var_by_shock[sname][h] += contrib
            var_total[h] += contrib
        T_power = T @ T_power

    # Shares
    var_share_by_shock = {}
    for sname in shock_names:
        share = np.where(var_total > 0, var_by_shock[sname] / var_total, 0.0)
        var_share_by_shock[sname] = pd.DataFrame(share, columns=var_names)

    return {
        "variance": pd.DataFrame(var_total, columns=var_names),
        "by_shock": {s: pd.DataFrame(var_by_shock[s], columns=var_names) for s in shock_names},
        "share": var_share_by_shock,
    }
