"""
Sims (2002) gensys algorithm — Julia implementation.

System:  G0 * y_t = G1 * y_{t-1} + c + Psi * eps_t + Pi * eta_t

Solution (if BK holds):  y_t = T * y_{t-1} + const + R * eps_t

Implementation uses the companion matrix T_c = G0 \\ G1 and Julia's built-in
ordschur (stable Schur ordering) rather than a custom generalized QZ reorder.
"""

using LinearAlgebra

# ------------------------------------------------------------------ #
# Result structs                                                        #
# ------------------------------------------------------------------ #

"""
Schur decomposition details stored for transparency.

Fields
------
- `T_schur` : ordered Schur form of T_c = G0⁻¹ G1  (stable block upper-left)
- `Z`       : Schur vectors  (y_t = Z w_t,  w_t = Z' y_t)
- `eigenvalues` : eigenvalues of G0⁻¹ G1 (|eig| < 1 stable, ordered stable first)
- `n_stable`, `n_unstable`
"""
struct SchurDecomposition
    T_schur::Matrix{Float64}
    Z::Matrix{Float64}
    eigenvalues::Vector{ComplexF64}   # complex to handle conjugate pairs
    n_stable::Int
    n_unstable::Int
end

"""
Blanchard-Kahn analysis.
"""
struct BKAnalysis
    n_forward::Int
    n_unstable::Int
    existence::Bool
    uniqueness::Bool
    Q2_Pi::Matrix{Float64}    # BK matrix (n_unstable × n_forward); should be invertible
    Q2_Pi_rank::Int
end

"""
Full solution from gensys.

  T         : policy matrix  y_t = T y_{t-1} + R eps_t + const
  R         : shock impact
  const_vec : constant (zero for log-linear models)
  schur_dec : SchurDecomposition
  bk        : BKAnalysis
  T_hat     : stable block of Schur form  (= T_schur[1:n_stable, 1:n_stable])
  Z1        : first n_stable columns of Z  (stable Schur vectors)
  eta_matrix: (n_forward × n_e)  maps eps_t → expectational errors eta_t
  success
"""
struct GensysResult
    T::Matrix{Float64}
    R::Matrix{Float64}
    const_vec::Vector{Float64}
    schur_dec::SchurDecomposition
    bk::BKAnalysis
    T_hat::Matrix{Float64}
    Z1::Matrix{Float64}
    eta_matrix::Matrix{Float64}
    success::Bool
end

# ------------------------------------------------------------------ #
# Core solver                                                           #
# ------------------------------------------------------------------ #

"""
    gensys(G0, G1, Psi, Pi; c=nothing, div=1.01) -> GensysResult

Solve the linear rational expectations model.

# Arguments
- `G0`, `G1` : (n × n) system matrices
- `Psi`      : (n × n_e) shock loadings
- `Pi`       : (n × n_f) expectational-error loadings
- `c`        : (n,) constant (default zeros)
- `div`      : stability cutoff (default 1.01)
"""
function gensys(
    G0::AbstractMatrix,
    G1::AbstractMatrix,
    Psi::AbstractMatrix,
    Pi::AbstractMatrix;
    c::Union{AbstractVector,Nothing} = nothing,
    div::Float64 = 1.01,
)
    n   = size(G0, 1)
    c_  = c === nothing ? zeros(n) : collect(Float64, c)

    # --- Companion matrix T_c = G0⁻¹ G1  ----------------------------
    # Eigenvalues of T_c = eigenvalues of G0⁻¹ G1
    T_c = G0 \ Matrix{Float64}(G1)       # real matrix: use real Schur

    # --- Schur decomposition + ordering  ----------------------------
    # Use real Schur for real T_c so Schur vectors Z remain real orthogonal.
    # Complex Schur forces complex Z; real(Z) then loses orthonormality.
    F   = schur(T_c)
    sel = abs.(F.values) .< div           # select stable eigenvalues
    G   = ordschur(F, sel)                # move stable to upper-left

    n_stable   = sum(sel)
    n_unstable = n - n_stable

    eigs    = complex.(G.values)          # eigenvalues, stable first
    T_schur = G.T                         # ordered Schur form (quasi-upper-triangular)
    Z       = G.Z                         # real orthogonal Schur vectors

    dec = SchurDecomposition(Matrix{Float64}(T_schur), Matrix{Float64}(Z),
                             eigs, n_stable, n_unstable)

    # --- Blanchard-Kahn check  ----------------------------------------
    # Julia real Schur: T_c = Z T_schur Z'  →  w = Z' y,  y = Z w
    ZT  = Z'                                     # Z' (transpose for real Z)
    Q2  = ZT[n_stable+1:end, :]                  # unstable rows of Z'

    Pi_bar    = G0 \ Matrix{Float64}(Pi)         # G0⁻¹ Pi
    Q2_Pi     = Q2 * Pi_bar                      # (n_unstable × n_forward)
    n_forward = size(Pi, 2)
    Q2_Pi_rank = rank(Q2_Pi)

    existence  = n_unstable <= n_forward
    uniqueness = (n_unstable == n_forward) && (Q2_Pi_rank == n_unstable)

    bk = BKAnalysis(n_forward, n_unstable, existence, uniqueness, Q2_Pi, Q2_Pi_rank)

    if !existence
        @warn "BK existence fails: $n_unstable unstable eigenvalues, $n_forward forward equations."
    elseif !uniqueness
        @warn "BK uniqueness fails: multiple (or no) solutions."
    end

    nan_mat(r, c_) = fill(NaN, r, c_)
    if !(existence && uniqueness)
        return GensysResult(
            nan_mat(n, n), nan_mat(n, size(Psi,2)), fill(NaN, n),
            dec, bk,
            nan_mat(n_stable, n_stable),
            nan_mat(n, n_stable),
            nan_mat(n_forward, size(Psi,2)),
            false
        )
    end

    # --- Expectational errors: eta_t = eta_matrix @ eps_t  -----------
    Psi_bar  = G0 \ Matrix{Float64}(Psi)  # G0⁻¹ Psi
    Q2_Psi   = Q2 * Psi_bar

    eta_matrix = if n_unstable > 0
        -Q2_Pi \ Q2_Psi
    else
        zeros(n_forward, size(Psi, 2))
    end

    # --- Build policy function  --------------------------------------
    Psi_eff = Matrix{Float64}(Psi) + Matrix{Float64}(Pi) * eta_matrix

    # Shock impact: R = G0⁻¹ Psi_eff  (satisfies G0 R = Psi_eff exactly)
    R_full = G0 \ Psi_eff

    # Stable block
    Z1    = Z[:, 1:n_stable]              # (n × n_stable) real Schur vectors
    T_hat = T_schur[1:n_stable, 1:n_stable]  # stable Schur block

    # Policy: T = Z1 T_hat Z1'
    T_full = Z1 * T_hat * Z1'

    # Constant
    const_full = G0 \ c_                 # G0⁻¹ c (usually zero)

    return GensysResult(
        T_full, R_full, const_full,
        dec, bk,
        T_hat, Z1,
        eta_matrix,
        true
    )
end
