"""
Steady-state utilities for DynareJL.

For log-linearised models the log-deviation steady state is y* = 0.
This module handles the nonlinear steady state for model derivation/verification.
"""

"""
    solve_steady_state(f!, x0; tol=1e-10, maxiter=500) -> NamedTuple

Find x s.t. f!(x) ≈ 0 using Newton iterations with backtracking.

# Arguments
- `f!` : in-place function f!(residual, x) or out-of-place f(x)
- `x0` : initial guess
"""
function solve_steady_state(f, x0::AbstractVector; tol=1e-10, maxiter=500)
    x = copy(Float64.(x0))
    n = length(x)

    for iter in 1:maxiter
        F = f(x)
        err = maximum(abs, F)
        err < tol && return (x=x, success=true, residual=err, iters=iter)

        # Numerical Jacobian
        J = _numerical_jacobian(f, x)
        dx = -J \ F

        # Backtracking line search
        step = 1.0
        for _ in 1:20
            xnew = x .+ step .* dx
            Fnew = f(xnew)
            norm(Fnew) < norm(F) && (x = xnew; break)
            step *= 0.5
        end
    end

    F = f(x)
    return (x=x, success=false, residual=maximum(abs, F), iters=maxiter)
end

function _numerical_jacobian(f, x; eps=1e-6)
    n  = length(x)
    F0 = f(x)
    J  = zeros(length(F0), n)
    for j in 1:n
        xp    = copy(x); xp[j] += eps
        J[:,j] = (f(xp) .- F0) ./ eps
    end
    return J
end

"""
    log_linearise(f, ss; eps=1e-5) -> Matrix{Float64}

Numerically compute the log-linearised Jacobian at steady state `ss`.
Entry (i,j) = ∂f_i/∂log(x_j) * x_j at x=ss.
"""
function log_linearise(f, ss::AbstractVector; eps=1e-5)
    n   = length(ss)
    F0  = f(ss)
    J   = zeros(length(F0), n)
    for j in 1:n
        xp      = copy(Float64.(ss)); xp[j] *= (1 + eps)
        J[:,j]  = (f(xp) .- F0) ./ eps
    end
    return J
end

"""
    check_steady_state(f, ss; tol=1e-8) -> Bool
"""
check_steady_state(f, ss; tol=1e-8) = maximum(abs, f(ss)) < tol
