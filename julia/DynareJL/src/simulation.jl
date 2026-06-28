"""
IRF and stochastic simulation for DynareJL.
"""

using Statistics
using LinearAlgebra

# ------------------------------------------------------------------ #
# IRF                                                                  #
# ------------------------------------------------------------------ #

"""
    compute_irf(result, shock_index; shock_size=1.0, periods=40, var_names=nothing)
              -> Matrix{Float64}  (periods × n_vars)

Compute the impulse response to a single shock.
Returns a matrix; rows = periods (0-indexed), columns = variables.
"""
function compute_irf(
    result::GensysResult,
    shock_index::Int = 1;
    shock_size::Float64 = 1.0,
    periods::Int = 40,
    var_names::Union{Vector{String},Nothing} = nothing,
)
    result.success || error("Cannot compute IRF: model has no unique solution.")

    n   = size(result.T, 1)
    n_e = size(result.R, 2)

    irf = zeros(periods, n)
    eps = zeros(n_e); eps[shock_index] = shock_size

    irf[1, :] = result.R * eps .+ result.const_vec
    for t in 2:periods
        irf[t, :] = result.T * irf[t-1, :] .+ result.const_vec
    end
    return irf
end

"""
    compute_all_irfs(result; shock_names, var_names, periods=40)
                  -> Dict{String, Matrix{Float64}}

Compute IRFs for all shocks.
"""
function compute_all_irfs(
    result::GensysResult;
    shock_names::Union{Vector{String},Nothing} = nothing,
    var_names::Union{Vector{String},Nothing} = nothing,
    periods::Int = 40,
)
    n_e = size(result.R, 2)
    snames = shock_names === nothing ? ["eps$i" for i in 1:n_e] : shock_names
    return Dict(
        snames[i] => compute_irf(result, i; periods=periods, var_names=var_names)
        for i in 1:n_e
    )
end

# ------------------------------------------------------------------ #
# Stochastic simulation                                                #
# ------------------------------------------------------------------ #

"""
    simulate(result; shock_cov=I, periods=200, burn=50, seed=nothing, var_names=nothing)
           -> Matrix{Float64}  (periods × n_vars)

Simulate the model by drawing shocks.

# Arguments
- `shock_cov` : (n_e × n_e) covariance; defaults to identity
- `periods`   : simulation length after burn-in
- `burn`      : discarded burn-in periods
- `seed`      : random seed
"""
function simulate(
    result::GensysResult;
    shock_cov::Union{AbstractMatrix,Nothing} = nothing,
    periods::Int = 200,
    burn::Int = 50,
    seed::Union{Int,Nothing} = nothing,
    var_names::Union{Vector{String},Nothing} = nothing,
)
    result.success || error("Cannot simulate: model has no unique solution.")

    n   = size(result.T, 1)
    n_e = size(result.R, 2)

    Σ = shock_cov === nothing ? Matrix{Float64}(I, n_e, n_e) : Matrix{Float64}(shock_cov)
    L = cholesky(Σ).L

    if seed !== nothing
        Random.seed!(seed)
    end

    total = burn + periods
    out = zeros(total, n)
    y   = zeros(n)

    for t in 1:total
        eps = L * randn(n_e)
        y   = result.T * y .+ result.R * eps .+ result.const_vec
        out[t, :] = y
    end

    return out[burn+1:end, :]
end

# ------------------------------------------------------------------ #
# Forecast error variance decomposition                                #
# ------------------------------------------------------------------ #

"""
    fevd(result; shock_cov=I, horizon=20, shock_names=nothing, var_names=nothing)
       -> NamedTuple(variance, share_by_shock)

Returns:
- `variance`      : (horizon × n_vars) total forecast error variance
- `share_by_shock`: Dict mapping shock name -> (horizon × n_vars) share matrix
"""
function fevd(
    result::GensysResult;
    shock_cov::Union{AbstractMatrix,Nothing} = nothing,
    horizon::Int = 20,
    shock_names::Union{Vector{String},Nothing} = nothing,
    var_names::Union{Vector{String},Nothing} = nothing,
)
    result.success || error("Cannot compute FEVD: model has no unique solution.")

    n   = size(result.T, 1)
    n_e = size(result.R, 2)

    Σ = shock_cov === nothing ? Matrix{Float64}(I, n_e, n_e) : Matrix{Float64}(shock_cov)
    L = cholesky(Σ).L
    RL = result.R * L   # (n × n_e)

    snames = shock_names === nothing ? ["eps$i" for i in 1:n_e] : shock_names

    var_total = zeros(horizon, n)
    var_shock = Dict(s => zeros(horizon, n) for s in snames)

    T_power = Matrix{Float64}(I, n, n)  # T^0
    for h in 1:horizon
        for j in 1:n_e
            r     = T_power * RL[:, j]
            contrib = r .^ 2
            var_shock[snames[j]][h, :] .+= contrib
            var_total[h, :]            .+= contrib
        end
        T_power = result.T * T_power
    end

    share_by_shock = Dict(
        s => [var_total[h,v] > 0 ? var_shock[s][h,v] / var_total[h,v] : 0.0
              for h in 1:horizon, v in 1:n]
        for s in snames
    )

    return (variance=var_total, share_by_shock=share_by_shock)
end
