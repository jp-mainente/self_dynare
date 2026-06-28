"""
DynareJL — Julia implementation of Dynare-style DSGE model solving.

Quick start
-----------
```julia
using DynareJL

# ABC mode: A*Ey[t+1] + B*y[t] + C*y[t-1] + D*eps[t] = 0
model = from_abc(A, B, C, D, ["x","pi","i","rn"], ["eps_rn"])
result = solve(model)

# Sims mode: G0*y[t] = G1*y[t-1] + Psi*eps[t] + Pi*eta[t]
result = gensys(G0, G1, Psi, Pi)

# IRF
irf = compute_irf(result, 1; periods=40)

# Stochastic simulation
sim = simulate(result; periods=200, seed=42)

# FEVD
decomp = fevd(result; horizon=20)
```
"""
module DynareJL

using LinearAlgebra
using Statistics
using Random

include("gensys.jl")
include("model.jl")
include("simulation.jl")
include("steady_state.jl")

export
    # Core solver
    gensys, GensysResult, SchurDecomposition, BKAnalysis,
    # Model
    DSGEModel, from_abc, from_sims, solve,
    var_index, shock_index,
    # Simulation
    compute_irf, compute_all_irfs, simulate, fevd,
    # Steady state
    solve_steady_state, log_linearise, check_steady_state

end
