"""
Test script — equivalent to NK_model_julia.ipynb
Run with:  julia test_julia.jl
"""

push!(LOAD_PATH, joinpath(@__DIR__, "DynareJL", "src"))
include(joinpath(@__DIR__, "DynareJL", "src", "DynareJL.jl"))
using .DynareJL
using LinearAlgebra
using Printf
using Statistics

println("=" ^ 50)
println("DynareJL — NK Model Test")
println("=" ^ 50)

# ── Parameters ──────────────────────────────────────
σ, β, κ, ϕ_π, ϕ_x, ρ = 1.0, 0.99, 0.1, 1.5, 0.5, 0.8

variables = ["x", "pi", "i", "rn"]
shocks    = ["eps_rn"]

# ── ABC matrices ────────────────────────────────────
A = [-1.0  1/σ  0.0  0.0;
      0.0  -β   0.0  0.0;
      0.0   0.0  0.0  0.0;
      0.0   0.0  0.0  0.0]

B = [ 1.0   0.0  1/σ  -1/σ;
     -κ     1.0  0.0   0.0;
     -ϕ_x  -ϕ_π  1.0   0.0;
      0.0   0.0  0.0   1.0]

C = [0.0  0.0  0.0   0.0;
     0.0  0.0  0.0   0.0;
     0.0  0.0  0.0   0.0;
     0.0  0.0  0.0  -ρ  ]

D = [0.0; 0.0; 0.0; -1.0;;]

println("\n[1] Forward-looking variables (non-zero cols of A):")
fwd = [variables[j] for j in 1:4 if any(A[:, j] .!= 0)]
println("    ", fwd)

# ── Build and solve ──────────────────────────────────
model  = from_abc(A, B, C, D, variables, shocks)
result = solve(model)

println("\n[2] Blanchard-Kahn Conditions:")
bk = result.bk
println("    n_forward   = ", bk.n_forward)
println("    n_unstable  = ", bk.n_unstable)
println("    existence   = ", bk.existence)
println("    uniqueness  = ", bk.uniqueness)
println("    Q2Pi rank   = ", bk.Q2_Pi_rank)

# ── Consistency check: G0 R = Psi_eff ───────────────
G0      = model.G0
Psi_eff = model.Psi + model.Pi * result.eta_matrix
residual = maximum(abs.(G0 * result.R - Psi_eff))
println("\n[3] Consistency: max|G0 R - Psi_eff| = ", residual, "  (should be ~1e-14)")

# ── Eigenvalues ──────────────────────────────────────
println("\n[4] Eigenvalues of G0⁻¹G1:")
eigs = result.schur_dec.eigenvalues
for (k, ev) in enumerate(eigs)
    tag = abs(ev) > 1.01 ? "UNSTABLE" : "stable  "
    @printf("    λ_%d = %.5f  [%s]\n", k, abs(ev), tag)
end

# ── IRF ─────────────────────────────────────────────
irf = compute_irf(result, 1; periods=6)
println("\n[5] IRF to eps_rn (6 periods, original 4 variables):")
@printf("    %-6s %-10s %-10s %-10s %-10s\n", "t", "x", "pi", "i", "rn")
for t in 1:6
    @printf("    %-6d %-10.5f %-10.5f %-10.5f %-10.5f\n",
            t-1, irf[t,1], irf[t,2], irf[t,3], irf[t,4])
end
println("    rn should be: 1.0, 0.8, 0.64, 0.512, 0.4096, 0.32768")

# ── rn decay check ───────────────────────────────────
rn_expected = [0.8^k for k in 0:5]
rn_actual   = [irf[t, 4] for t in 1:6]
rn_error    = maximum(abs.(rn_actual .- rn_expected))
println("    max error vs 0.8^t: ", rn_error)

# ── Simulation ───────────────────────────────────────
sim = simulate(result; periods=500, burn=100, seed=42)
println("\n[6] Simulation std devs (500 periods):")
for (j, v) in enumerate(model.variables)
    @printf("    %s: std = %.5f\n", v, std(sim[:, j]))
end

# ── FEVD ─────────────────────────────────────────────
decomp = fevd(result; horizon=20, shock_names=shocks)
println("\n[7] FEVD — share of eps_rn in x at horizon 1, 4, 8, 20:")
for h in [1, 4, 8, 20]
    @printf("    h=%2d: %.6f\n", h, decomp.share_by_shock["eps_rn"][h, 1])
end

# ── Z1 decomposition ─────────────────────────────────
println("\n[8] Z1 (stable Schur vectors, n×n_stable):")
Z1 = result.Z1
for (i, v) in enumerate(model.aug_variables)
    row = join([@sprintf("%8.4f", Z1[i,j]) for j in 1:size(Z1,2)], "  ")
    println("    $v:  $row")
end

println("\n" * "=" ^ 50)
println("All tests complete.")
println("=" ^ 50)
