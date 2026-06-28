"""
DSGEModel struct and constructors for DynareJL.

Two input modes:
  1. ABC form:  A*Ey[t+1] + B*y[t] + C*y[t-1] + D*eps[t] = 0
  2. Sims form: G0*y[t] = G1*y[t-1] + Psi*eps[t] + Pi*eta[t]
"""

"""
    DSGEModel

Container for a log-linearised DSGE model.

Fields
------
- `variables`     : names of original endogenous variables
- `shocks`        : names of exogenous shocks
- `aug_variables` : names in the augmented Sims state (includes _fwd vars)
- `G0`, `G1`      : Sims-form matrices
- `Psi`, `Pi`     : shock / expectational-error loadings
- `c`             : constant vector
"""
mutable struct DSGEModel
    variables::Vector{String}
    shocks::Vector{String}
    aug_variables::Vector{String}
    G0::Matrix{Float64}
    G1::Matrix{Float64}
    Psi::Matrix{Float64}
    Pi::Matrix{Float64}
    c::Vector{Float64}
end

# ------------------------------------------------------------------ #
# Constructors                                                          #
# ------------------------------------------------------------------ #

"""
    from_abc(A, B, C, D, variables, shocks) -> DSGEModel

Build a model from the natural DSGE form:

    A * E_t[y_{t+1}] + B * y_t + C * y_{t-1} + D * eps_t = 0

The state is augmented with forward-expectation variables for each variable
that appears with non-zero coefficient in A.
"""
function from_abc(
    A::AbstractMatrix,
    B::AbstractMatrix,
    C::AbstractMatrix,
    D::AbstractMatrix,
    variables::AbstractVector{<:AbstractString},
    shocks::AbstractVector{<:AbstractString},
)
    n  = size(A, 1)
    ne = size(D, 2)

    # Forward-looking variables: columns of A with any non-zero entry
    fwd_idx = [j for j in 1:n if any(A[:, j] .!= 0)]
    nf = length(fwd_idx)

    n_aug = n + nf
    fwd_names = [variables[j] * "_fwd" for j in fwd_idx]
    aug_vars  = vcat(collect(variables), fwd_names)

    G0  = zeros(n_aug, n_aug)
    G1  = zeros(n_aug, n_aug)
    Psi = zeros(n_aug, ne)
    Pi  = zeros(n_aug, nf)

    # Original equations  (rows 1:n)
    G0[1:n, 1:n] = B
    for (row_f, orig_j) in enumerate(fwd_idx)
        G0[1:n, n + row_f] = A[:, orig_j]
    end
    G1[1:n, 1:n] = -C
    Psi[1:n, :]  = -D

    # Expectation-update equations  (rows n+1 : n_aug)
    for row_f in 1:nf
        orig_j = fwd_idx[row_f]
        G0[n + row_f, orig_j]  =  1.0      # y_{orig_j, t}
        G1[n + row_f, n + row_f] =  1.0    # yf_{row_f, t-1}
        Pi[n + row_f, row_f]     = -1.0    # -eta_{row_f, t}
    end

    return DSGEModel(
        collect(String, variables),
        collect(String, shocks),
        aug_vars,
        G0, G1, Psi, Pi, zeros(n_aug)
    )
end

"""
    from_sims(G0, G1, Psi, Pi, variables, shocks; c=nothing) -> DSGEModel

Build directly from Sims-form matrices.
"""
function from_sims(
    G0::AbstractMatrix,
    G1::AbstractMatrix,
    Psi::AbstractMatrix,
    Pi::AbstractMatrix,
    variables::AbstractVector{<:AbstractString},
    shocks::AbstractVector{<:AbstractString};
    c::Union{AbstractVector,Nothing} = nothing,
)
    n = size(G0, 1)
    c_ = c === nothing ? zeros(n) : collect(Float64, c)
    return DSGEModel(
        collect(String, variables),
        collect(String, shocks),
        collect(String, variables),
        Matrix{Float64}(G0), Matrix{Float64}(G1),
        Matrix{Float64}(Psi), Matrix{Float64}(Pi), c_
    )
end

# ------------------------------------------------------------------ #
# Solve                                                                 #
# ------------------------------------------------------------------ #

"""
    solve(model; div=1.01) -> GensysResult

Run gensys on the model and return the solution.
"""
function solve(model::DSGEModel; div::Float64 = 1.01)
    return gensys(model.G0, model.G1, model.Psi, model.Pi; c=model.c, div=div)
end

# ------------------------------------------------------------------ #
# Utilities                                                             #
# ------------------------------------------------------------------ #

var_index(m::DSGEModel, name::String) = findfirst(==(name), m.aug_variables)
shock_index(m::DSGEModel, name::String) = findfirst(==(name), m.shocks)

function Base.show(io::IO, m::DSGEModel)
    print(io, "DSGEModel(vars=$(m.variables), shocks=$(m.shocks))")
end
