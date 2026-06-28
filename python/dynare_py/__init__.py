"""
dynare_py — Python implementation of Dynare-style DSGE model solving.

Quick start
-----------
>>> from dynare_py import DSGEModel, gensys, compute_irf, simulate

ABC mode (natural DSGE form  A E_t[y+1] + B y + C y-1 + D eps = 0):
>>> model = DSGEModel.from_abc(A, B, C, D, variables=[...], shocks=[...])
>>> result = model.solve()

Sims mode (direct):
>>> from dynare_py import gensys
>>> result = gensys(G0, G1, Psi, Pi)

Simulation:
>>> irf = compute_irf(result, shock_index=0, periods=40, var_names=[...])
>>> sim = simulate(result, periods=200)
>>> decomp = fevd(result, horizon=20)
"""

from .gensys import GensysResult, QZDecomposition, BKAnalysis, gensys
from .model import DSGEModel
from .simulation import compute_irf, compute_all_irfs, simulate, fevd
from .steady_state import solve_steady_state, log_linearise, check_steady_state

__all__ = [
    # Core solver
    "gensys",
    "GensysResult",
    "QZDecomposition",
    "BKAnalysis",
    # Model class
    "DSGEModel",
    # Simulation
    "compute_irf",
    "compute_all_irfs",
    "simulate",
    "fevd",
    # Steady state
    "solve_steady_state",
    "log_linearise",
    "check_steady_state",
]
