"""
Sims (2002) gensys algorithm for solving linear rational expectations models.

System form:
    G0 @ y_t = G1 @ y_{t-1} + c + Psi @ eps_t + Pi @ eta_t

where:
    y_t    : endogenous variables (n x 1)
    eps_t  : exogenous shocks (n_e x 1)
    eta_t  : expectational errors (n_f x 1),  eta_t = E_{t-1}[y_t] - y_t

Solution (if BK satisfied):
    y_t = T @ y_{t-1} + const + R @ eps_t
"""

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import linalg


@dataclass
class QZDecomposition:
    """
    Raw QZ output.  Q @ G1 @ Z^H = AA,  Q @ G0 @ Z^H = BB.
    Eigenvalues  eig = diag(AA) / diag(BB) are eigenvalues of G0^{-1} G1.
    Rows/columns are ordered: stable first, unstable last.
    """
    AA: np.ndarray        # Schur form corresponding to G1
    BB: np.ndarray        # Schur form corresponding to G0
    Q: np.ndarray         # Left unitary transformation
    Z: np.ndarray         # Right unitary transformation
    eigenvalues: np.ndarray   # complex, |eig| < 1 are stable
    n_stable: int
    n_unstable: int


@dataclass
class BKAnalysis:
    """Blanchard-Kahn conditions check."""
    n_forward: int          # columns of Pi (expected forward-looking equations)
    n_unstable: int         # unstable eigenvalues
    existence: bool         # n_unstable <= n_forward
    uniqueness: bool        # n_unstable == n_forward AND Q2_Pi invertible
    Q2_Pi: np.ndarray       # the BK matrix (n_unstable x n_forward); should be invertible
    Q2_Pi_rank: int


@dataclass
class GensysResult:
    """
    Full solution object.

    T      : policy matrix,  y_t = T @ y_{t-1} + R @ eps_t + const
    R      : shock impact matrix
    const  : constant (zero for log-linear models)
    qz     : QZDecomposition — raw Schur matrices and eigenvalues
    bk     : BKAnalysis — condition numbers, BK matrix
    T_hat  : BB11^{-1} @ AA11  (stable transition in the *transformed* space Z^H y)
    Z1     : first n_stable columns of Z (maps transformed -> original)
    eta_matrix : maps eps_t -> eta_t  (eta_t = eta_matrix @ eps_t)
    """
    T: np.ndarray
    R: np.ndarray
    const: np.ndarray
    qz: QZDecomposition
    bk: BKAnalysis
    T_hat: np.ndarray
    Z1: np.ndarray
    eta_matrix: np.ndarray
    success: bool


def gensys(
    G0: np.ndarray,
    G1: np.ndarray,
    Psi: np.ndarray,
    Pi: np.ndarray,
    c: Optional[np.ndarray] = None,
    div: float = 1.01,
) -> GensysResult:
    """
    Solve the linear rational expectations system using QZ decomposition.

    Parameters
    ----------
    G0, G1 : (n, n) arrays
        System matrices.  G0 y_t = G1 y_{t-1} + ...
    Psi : (n, n_e) array
        Shock loading matrix.
    Pi : (n, n_f) array
        Expectational-error loading matrix.
    c : (n,) array, optional
        Constant vector (default: zeros).
    div : float
        Eigenvalue stability cutoff (default 1.01 to handle unit roots cleanly).

    Returns
    -------
    GensysResult
    """
    n = G0.shape[0]
    if c is None:
        c = np.zeros(n)

    # ------------------------------------------------------------------
    # QZ decomposition
    # ordqz(G1, G0) gives eigenvalues of G0^{-1} G1 as alpha/beta
    # 'iuc' = inside unit circle first  (stable modes first)
    # ------------------------------------------------------------------
    AA, BB, alpha, beta, Q, Z = linalg.ordqz(
        G1, G0, sort="iuc", output="complex"
    )
    # ACTUAL scipy convention: Q^H @ G1 @ Z = AA,  Q^H @ G0 @ Z = BB
    # Transformation: w_t = Z^H y_t,  y_t = Z w_t
    # Stable block:   BB11 w1_t = AA11 w1_{t-1} + (Q^H)[:n1,:] Psi_eff eps_t
    QH = Q.conj().T  # Q^H

    # Generalised eigenvalues
    with np.errstate(invalid="ignore", divide="ignore"):
        eigs = np.where(
            np.abs(np.diag(BB)) < 1e-13,
            np.inf + 0j,
            np.diag(AA) / np.diag(BB),
        )

    unstable_mask = np.abs(eigs) > div
    n_unstable = int(np.sum(unstable_mask))
    n_stable = n - n_unstable

    qz = QZDecomposition(
        AA=AA, BB=BB, Q=Q, Z=Z,
        eigenvalues=eigs,
        n_stable=n_stable,
        n_unstable=n_unstable,
    )

    # ------------------------------------------------------------------
    # Blanchard-Kahn check
    # ------------------------------------------------------------------
    n_forward = Pi.shape[1]
    Q2 = QH[n_stable:, :]         # rows of Q^H for unstable block
    Q2_Pi = Q2 @ Pi               # (n_unstable x n_forward)
    Q2_Pi_rank = int(np.linalg.matrix_rank(Q2_Pi))

    existence  = (n_unstable <= n_forward)
    uniqueness = (n_unstable == n_forward) and (Q2_Pi_rank == n_unstable)

    bk = BKAnalysis(
        n_forward=n_forward,
        n_unstable=n_unstable,
        existence=existence,
        uniqueness=uniqueness,
        Q2_Pi=Q2_Pi,
        Q2_Pi_rank=Q2_Pi_rank,
    )

    if not existence:
        warnings.warn(
            f"BK existence fails: {n_unstable} unstable eigenvalues but only "
            f"{n_forward} forward-looking equations."
        )
    if existence and not uniqueness:
        warnings.warn(
            "BK uniqueness fails: multiple solutions exist (rank(Q2 Pi) < n_unstable)."
        )

    if not (existence and uniqueness):
        dummy = np.full((n, n), np.nan)
        return GensysResult(
            T=dummy, R=dummy, const=np.full(n, np.nan),
            qz=qz, bk=bk,
            T_hat=dummy[:n_stable, :n_stable],
            Z1=np.real(Z[:, :n_stable]),
            eta_matrix=dummy,
            success=False,
        )

    # ------------------------------------------------------------------
    # Compute expectational errors:  eta_t = eta_matrix @ eps_t
    # From boundedness:  Q2 Pi eta_t = -Q2 (Psi eps_t + c)  at impact
    # ------------------------------------------------------------------
    Q2_Psi = Q2 @ Psi             # (n_unstable x n_e)

    if n_unstable > 0:
        # eta_matrix: (n_forward x n_e)
        eta_matrix = -np.linalg.solve(Q2_Pi, Q2_Psi)
    else:
        eta_matrix = np.zeros((n_forward, Psi.shape[1]))

    # Effective shock after absorbing eta:  Psi + Pi @ eta_matrix
    Psi_eff = Psi + Pi @ eta_matrix   # (n x n_e)

    # ------------------------------------------------------------------
    # Stable block recursion
    # w_t = Z^H y_t,  y_t = Z w_t,  w2 = 0 enforced
    # BB11 w1_t = AA11 w1_{t-1} + (Q^H)[:n1,:] (c + Psi_eff eps_t)
    # ------------------------------------------------------------------
    AA11 = AA[:n_stable, :n_stable]
    BB11 = BB[:n_stable, :n_stable]
    Q1   = QH[:n_stable, :]          # first n_stable rows of Q^H
    Z1   = Z[:, :n_stable]           # first n_stable columns of Z; y_t = Z1 w1_t

    T_hat = np.linalg.solve(BB11, AA11)           # (n_stable x n_stable)
    BB11_inv_Q1 = np.linalg.solve(BB11, Q1)       # (n_stable x n)

    # Back to original space:  y_t = Z1 w1_t
    T_full   = np.real(Z1 @ T_hat @ Z1.conj().T)
    R_full   = np.real(Z1 @ BB11_inv_Q1 @ Psi_eff)
    const_out = np.real(Z1 @ BB11_inv_Q1 @ c)

    return GensysResult(
        T=T_full, R=R_full, const=const_out,
        qz=qz, bk=bk,
        T_hat=np.real(T_hat),
        Z1=np.real(Z1),
        eta_matrix=np.real(eta_matrix),
        success=True,
    )
