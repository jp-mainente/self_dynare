"""
DSGEModel: high-level interface for log-linearised DSGE models.

Two input modes
---------------
1. ABC mode  (natural DSGE notation)
       A E_t[y_{t+1}] + B y_t + C y_{t-1} + D eps_t = 0
   Automatically converts to Sims form.

2. Sims mode (direct)
       G0 y_t = G1 y_{t-1} + Psi eps_t + Pi eta_t

Both produce a GensysResult via .solve().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .gensys import GensysResult, gensys


@dataclass
class DSGEModel:
    """
    Container for a log-linearised DSGE model.

    Parameters (set directly or via constructor)
    -------------------------------------------
    variables : list of str
        Names of all endogenous variables in order.
    shocks : list of str
        Names of exogenous shocks.
    var_labels : dict, optional
        Human-readable labels for variables.

    After specifying matrices call .solve() to obtain a GensysResult.
    """

    variables: List[str] = field(default_factory=list)
    shocks: List[str] = field(default_factory=list)
    var_labels: dict = field(default_factory=dict)

    # ABC-form matrices (set by from_abc or directly)
    _A: Optional[np.ndarray] = field(default=None, repr=False)
    _B: Optional[np.ndarray] = field(default=None, repr=False)
    _C: Optional[np.ndarray] = field(default=None, repr=False)
    _D: Optional[np.ndarray] = field(default=None, repr=False)

    # Sims-form matrices (set by from_sims or derived from ABC)
    _G0: Optional[np.ndarray] = field(default=None, repr=False)
    _G1: Optional[np.ndarray] = field(default=None, repr=False)
    _Psi: Optional[np.ndarray] = field(default=None, repr=False)
    _Pi: Optional[np.ndarray] = field(default=None, repr=False)
    _c: Optional[np.ndarray] = field(default=None, repr=False)

    # Augmented variable names (filled after ABC -> Sims conversion)
    _aug_variables: List[str] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------ #
    # Factory methods                                                       #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_abc(
        cls,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        D: np.ndarray,
        variables: List[str],
        shocks: List[str],
        var_labels: Optional[dict] = None,
    ) -> "DSGEModel":
        """
        Build from the 'natural' DSGE log-linear form:

            A E_t[y_{t+1}] + B y_t + C y_{t-1} + D eps_t = 0

        The ABC->Sims conversion augments the state with expectation variables
        only for variables that appear forward-looking (non-zero columns of A).

        Augmented state: z_t = [y_t ; (E_t[y_{t+1}])_{fwd}]
        Augmented names: variables + [v+'_fwd' for fwd v]
        """
        m = DSGEModel(
            variables=list(variables),
            shocks=list(shocks),
            var_labels=var_labels or {},
        )
        m._A = np.asarray(A, dtype=float)
        m._B = np.asarray(B, dtype=float)
        m._C = np.asarray(C, dtype=float)
        m._D = np.asarray(D, dtype=float)
        m._build_sims_from_abc()
        return m

    @classmethod
    def from_sims(
        cls,
        G0: np.ndarray,
        G1: np.ndarray,
        Psi: np.ndarray,
        Pi: np.ndarray,
        variables: List[str],
        shocks: List[str],
        c: Optional[np.ndarray] = None,
        var_labels: Optional[dict] = None,
    ) -> "DSGEModel":
        """Build directly from Sims-form matrices."""
        m = DSGEModel(
            variables=list(variables),
            shocks=list(shocks),
            var_labels=var_labels or {},
        )
        m._G0 = np.asarray(G0, dtype=float)
        m._G1 = np.asarray(G1, dtype=float)
        m._Psi = np.asarray(Psi, dtype=float)
        m._Pi = np.asarray(Pi, dtype=float)
        m._c = np.zeros(G0.shape[0]) if c is None else np.asarray(c, dtype=float)
        m._aug_variables = list(variables)
        return m

    # ------------------------------------------------------------------ #
    # ABC -> Sims conversion                                               #
    # ------------------------------------------------------------------ #

    def _build_sims_from_abc(self) -> None:
        """
        Internal: convert ABC form to augmented Sims form.

        Augmented state: z = [y ; yf]  where yf = E_t[y_{t+1}] for forward vars.
        Rows in G0 / G1:
          rows 0..n-1    : original equations  (B y_t + A yf_t = -C y_{t-1} - D eps)
          rows n..n+nf-1 : expectation updates (y_{fwd,t} = yf_{fwd,t-1} + eta_t)

        Pi has shape (n+nf, nf).
        """
        A, B, C, D = self._A, self._B, self._C, self._D
        n = A.shape[0]
        ne = D.shape[1]

        # Which original variables are forward-looking?
        fwd_idx = [j for j in range(n) if np.any(A[:, j] != 0)]
        nf = len(fwd_idx)

        n_aug = n + nf
        fwd_names = [self.variables[j] + "_fwd" for j in fwd_idx]
        self._aug_variables = list(self.variables) + fwd_names

        # Build G0 (n_aug x n_aug)
        G0 = np.zeros((n_aug, n_aug))
        G0[:n, :n] = B                       # coeff on y_t
        for row_fwd, orig_j in enumerate(fwd_idx):
            G0[:n, n + row_fwd] = A[:, orig_j]   # coeff on yf_t

        # Update rows: y_{fwd, t} - yf_{fwd, t-1} = eta_t
        for row_fwd, orig_j in enumerate(fwd_idx):
            G0[n + row_fwd, orig_j] = 1.0    # 1 * y_j_t  (left side)

        # Build G1 (n_aug x n_aug)
        G1 = np.zeros((n_aug, n_aug))
        G1[:n, :n] = -C                      # -C y_{t-1}
        for row_fwd in range(nf):
            G1[n + row_fwd, n + row_fwd] = 1.0   # 1 * yf_{fwd, t-1}

        # Psi (n_aug x ne)
        Psi = np.zeros((n_aug, ne))
        Psi[:n, :] = -D

        # Pi (n_aug x nf) -- expectational errors enter update rows
        Pi = np.zeros((n_aug, nf))
        for row_fwd in range(nf):
            Pi[n + row_fwd, row_fwd] = -1.0

        self._G0 = G0
        self._G1 = G1
        self._Psi = Psi
        self._Pi = Pi
        self._c = np.zeros(n_aug)

    # ------------------------------------------------------------------ #
    # Solve                                                                 #
    # ------------------------------------------------------------------ #

    def solve(self, div: float = 1.01) -> GensysResult:
        """
        Run gensys and return a GensysResult.

        The result carries T, R, and all decomposition details.
        Variable ordering follows self.aug_variables.
        """
        if self._G0 is None:
            raise RuntimeError("No matrices set. Use from_abc() or from_sims().")
        return gensys(
            self._G0, self._G1, self._Psi, self._Pi, self._c, div=div
        )

    # ------------------------------------------------------------------ #
    # Properties / utilities                                               #
    # ------------------------------------------------------------------ #

    @property
    def aug_variables(self) -> List[str]:
        """Variable names in the augmented (Sims) state vector."""
        return self._aug_variables if self._aug_variables else list(self.variables)

    @property
    def n(self) -> int:
        return len(self.aug_variables)

    @property
    def n_shocks(self) -> int:
        return len(self.shocks)

    def var_index(self, name: str) -> int:
        return self.aug_variables.index(name)

    def shock_index(self, name: str) -> int:
        return self.shocks.index(name)

    def summary(self) -> str:
        lines = [
            f"DSGEModel",
            f"  variables ({len(self.aug_variables)}): {self.aug_variables}",
            f"  shocks    ({len(self.shocks)}): {self.shocks}",
        ]
        if self._G0 is not None:
            lines.append(f"  state size: {self._G0.shape[0]}")
            lines.append(f"  G0 rank: {int(np.linalg.matrix_rank(self._G0))}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"DSGEModel(variables={self.variables}, shocks={self.shocks})"
        )
