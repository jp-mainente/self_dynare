"""
Steady-state utilities.

For a log-linearised model the log-deviation steady state is always y* = 0.
This module provides tools for the *nonlinear* steady state — useful when
deriving the log-linear approximation or verifying parameters.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import fsolve, root


def solve_steady_state(
    equations: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    method: str = "hybr",
    tol: float = 1e-10,
    verbose: bool = False,
) -> Dict:
    """
    Solve the nonlinear steady-state system f(x) = 0.

    Parameters
    ----------
    equations : callable
        Function f(x) -> residuals.  Must return array of same size as x.
    x0 : initial guess
    method : scipy.optimize.root method ('hybr', 'lm', 'broyden1', ...)
    tol : convergence tolerance
    verbose : print solver output

    Returns
    -------
    dict with keys 'x' (solution), 'success', 'residual', 'message'
    """
    sol = root(equations, x0, method=method, tol=tol,
               options={"disp": verbose})
    return {
        "x": sol.x,
        "success": sol.success,
        "residual": float(np.max(np.abs(sol.fun))),
        "message": sol.message,
    }


def log_linearise(
    equations: Callable[[np.ndarray], np.ndarray],
    ss: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Numerically compute the Jacobian of `equations` at steady state `ss`.

    This is the log-linear coefficient matrix when equations are written in
    terms of log-deviations and ss contains the *level* steady-state values.
    Each column j is (f(ss with x_j perturbed) - f(ss)) / eps,
    scaled by ss[j] to give log-derivative.

    Returns
    -------
    J : (n_eq, n_vars) Jacobian (d f / d log x_j * x_j)
    """
    n = len(ss)
    f0 = equations(ss)
    n_eq = len(f0)
    J = np.zeros((n_eq, n))
    for j in range(n):
        x_plus = ss.copy()
        x_plus[j] = ss[j] * (1 + eps)
        J[:, j] = (equations(x_plus) - f0) / eps
    return J


def check_steady_state(
    equations: Callable[[np.ndarray], np.ndarray],
    ss: np.ndarray,
    tol: float = 1e-8,
) -> bool:
    """Return True if max |f(ss)| < tol."""
    residual = np.max(np.abs(equations(ss)))
    return bool(residual < tol)
