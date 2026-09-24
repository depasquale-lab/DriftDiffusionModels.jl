# Exact marginal-likelihood estimation for the multilevel ("extended") DDM.
# Product Gauss–Hermite handles (B, v, a₀). The τ integral uses a
# trial-specific rule because the density vanishes for τ ≥ rt.

"""
    gauss_hermite(q)

Standard-normal Gauss–Hermite nodes and log-weights.
"""
function gauss_hermite(q::Int)
    q >= 2 || throw(ArgumentError("quadrature order must be at least 2"))
    β = [sqrt(i / 2) for i in 1:(q-1)]
    F = eigen(SymTridiagonal(zeros(q), β))
    z = sqrt(2) .* F.values
    w = (F.vectors[1, :]) .^ 2
    p = sortperm(z)
    return z[p], log.(w[p])
end

"""
    gauss_legendre_unit(n)

Gauss–Legendre nodes and log-weights mapped to `(0, 1)`.
"""
function gauss_legendre_unit(n::Int)
    n >= 2 || throw(ArgumentError("quadrature order must be at least 2"))
    β = [i / sqrt(4.0 * i^2 - 1) for i in 1:(n-1)]
    F = eigen(SymTridiagonal(zeros(n), β))
    x = F.values
    w = 2 .* (F.vectors[1, :]) .^ 2
    p = sortperm(x)
    return (x[p] .+ 1) ./ 2, log.(w[p] ./ 2)
end

const LOG2PI = log(2π)

"""
    gauss_laguerre(n)

Gauss–Laguerre nodes and log-weights for the lower τ tail.
"""
function gauss_laguerre(n::Int)
    n >= 1 || throw(ArgumentError("quadrature order must be at least 1"))
    α = [2.0 * i + 1.0 for i in 0:(n-1)]
    β = [float(i) for i in 1:(n-1)]
    F = eigen(SymTridiagonal(α, β))
    t = F.values
    w = (F.vectors[1, :]) .^ 2
    p = sortperm(t)
    return t[p], log.(w[p])
end

"""
    TauRule

Gauss–Legendre panel and Gauss–Laguerre lower-tail rule for `u_τ`.
"""
struct TauRule
    x::Vector{Float64}
    logw::Vector{Float64}
    t::Vector{Float64}
    logwt::Vector{Float64}
    L::Float64
end

const TAU_TAIL_NODES = 4

function TauRule(n::Int, L::Real, n_tail::Int=TAU_TAIL_NODES)
    x, logw = gauss_legendre_unit(n)
    t, logwt = gauss_laguerre(n_tail)
    return TauRule(x, logw, t, logwt, float(L))
end
TauRule(n::Int) = TauRule(n, tau_panel_cut(n))

n_tau_nodes(r::TauRule) = length(r.x) + length(r.t)

"""
    tau_panel_cut(n)

Panel cutoff for an `n`-node τ rule.
"""
tau_panel_cut(n::Int) = clamp(0.5 * n + 10.0, 14.0, 24.0)

"""
    tau_interval(a, r::TauRule)

Return the admissible standard-normal panel for truncation point `a`.

For `a < 0`, the panel follows the shrinking lower-tail mass. The Laguerre
continuation covers everything below it.
"""
@inline function tau_interval(a::Real, r::TauRule)
    # Avoid overflow in `a^2` for small σ_τ.
    d = hypot(min(a, zero(a)), sqrt(2 * r.L))
    return -d, min(a, d)
end

"""
    TauNodes(y_rt, m_τ, σ_τ, r)

Lazy `(z, logweight)` nodes for one trial's τ integral.
"""
struct TauNodes{T<:Real}
    z_lo::T
    h::T
    logh::T
    lw_tail::T
    r::TauRule
end

function TauNodes(rt::Float64, m_τ::T, σ_τ::T, r::TauRule) where {T<:Real}
    a = (log(rt) - m_τ) / σ_τ
    z_lo, z_hi = tau_interval(a, r)
    # Stable form of `z_hi - z_lo` in the deep lower tail.
    h = a < zero(a) ? 2 * r.L / (-z_lo - a) : z_hi - z_lo
    h = max(h, T(1e-300))
    return TauNodes{T}(z_lo, h, log(h),
                       -z_lo * z_lo / 2 - T(LOG2PI) / 2 - log(-z_lo), r)
end

Base.length(tn::TauNodes) = n_tau_nodes(tn.r)

@inline function Base.getindex(tn::TauNodes{T}, j::Int) where {T<:Real}
    n = length(tn.r.x)
    if j <= n
        z = tn.z_lo + tn.h * tn.r.x[j]
        return z, tn.r.logw[j] + tn.logh - z * z / 2 - T(LOG2PI) / 2
    else
        k = j - n
        tk = tn.r.t[k]
        s = tk / (-tn.z_lo)
        return tn.z_lo - s, tn.r.logwt[k] + tn.lw_tail - s * s / 2
    end
end

"""
    QuadGrid

Mapped quadrature nodes shared across trials.
"""
struct QuadGrid{T<:Real}
    B::Vector{T}
    v::Vector{T}
    a₀::Vector{T}
    logw_B::Vector{Float64}
    logw_v::Vector{Float64}
    logw_a₀::Vector{Float64}
    m_τ::T
    σ_τ::T
    τ::TauRule
end

const GHGrid = QuadGrid

"""
    build_grid(m, logσ0, rules)

Map quadrature rules into constrained DDM parameter space.
"""
function build_grid(m::AbstractVector{T}, logσ0::AbstractVector{T}, rules) where {T<:Real}
    σ0 = exp.(logσ0)
    zB, lwB = rules[1]
    rτ = rules[2]::TauRule
    zv, lwv = rules[3]
    za, lwa = rules[4]
    return QuadGrid{T}(exp.(m[1] .+ σ0[1] .* zB),
                       exp.(m[3] .+ σ0[3] .* zv),
                       logistic.(m[4] .+ σ0[4] .* za),
                       lwB, lwv, lwa,
                       m[2], σ0[2], rτ)
end

# Sentinel for invalid inputs or non-finite densities.
const LOG_ZERO_MARGINAL = -1.0e6

"""
    trial_marginal_loglik(y, g)

Marginal log-density of one trial using a streaming log-sum-exp.
"""
function trial_marginal_loglik(y::DDMResult, g::QuadGrid{T}) where {T<:Real}
    y.rt > 0 || return T(LOG_ZERO_MARGINAL)

    tn = TauNodes(y.rt, g.m_τ, g.σ_τ, g.τ)

    nB = length(g.B)
    nv = length(g.v)
    na = length(g.a₀)

    mx = T(-Inf)
    s = zero(T)

    @inbounds for j in 1:length(tn)
        z, lwτ = tn[j]
        τ = exp(g.m_τ + g.σ_τ * z)
        for i4 in 1:na
            lw4 = lwτ + g.logw_a₀[i4]
            a₀ = g.a₀[i4]
            for i3 in 1:nv
                lw3 = lw4 + g.logw_v[i3]
                v = g.v[i3]
                for i1 in 1:nB
                    t = lw3 + g.logw_B[i1] +
                        logdensityof(g.B[i1], v, a₀, τ, y.rt, y.choice, y.s)
                    # Skip non-finite and numerically negligible nodes.
                    if !isfinite(t)
                        continue
                    elseif t > mx
                        s = s * exp(mx - t) + one(T)
                        mx = t
                    elseif t - mx > -700
                        s += exp(t - mx)
                    end
                end
            end
        end
    end

    (isfinite(mx) && s > 0) || return T(LOG_ZERO_MARGINAL)
    return mx + log(s)
end

"""
    marginal_loglik(data, m, logσ0; q=8, qτ=2q)

Total marginal log-likelihood, threaded over trials.

`q` controls `(B, v, a₀)` and `qτ` controls the truncated τ integral.
"""
function marginal_loglik(data::Vector{DDMResult},
                         m::AbstractVector{T},
                         logσ0::AbstractVector{T};
                         q::Int=8, qτ::Int=2q) where {T<:Real}
    rules = quadrature_rules(q, qτ)
    return marginal_loglik(data, m, logσ0, rules)
end

function marginal_loglik(data::Vector{DDMResult},
                         m::AbstractVector{T},
                         logσ0::AbstractVector{T},
                         rules) where {T<:Real}
    g = build_grid(m, logσ0, rules)
    parts = Vector{T}(undef, length(data))
    @threads for i in eachindex(data)
        parts[i] = trial_marginal_loglik(data[i], g)
    end
    return sum(parts)
end

"""
    quadrature_rules(q, qτ=2q)

Build quadrature rules in `(B, τ, v, a₀)` order.
"""
function quadrature_rules(q::Int, qτ::Int=2q)
    r = gauss_hermite(q)
    return (r, TauRule(qτ), r, r)
end

"""
    MLDDMFit

Result of [`fit_mlddm_exact`](@ref).
"""
struct MLDDMFit
    m::Vector{Float64}
    σ0::Vector{Float64}
    loglik::Float64
    bic::Float64
    converged::Bool
    boundary::Vector{Bool}
    n_zero::Int
    quad_err::Float64
    q::Int
    qτ::Int
    n_starts::Int
    n_converged::Int
    n_used::Int
    n_total::Int
    idx::Vector{Int}
    seconds::Float64
end

function Base.show(io::IO, f::MLDDMFit)
    println(io, "MLDDMFit (exact marginal likelihood, q=$(f.q), qτ=$(f.qτ))")
    println(io, "  loglik    = $(round(f.loglik; digits=3))   BIC = $(round(f.bic; digits=3))")
    println(io, "  m  (B τ v a₀) = $(round.(f.m; digits=4))")
    println(io, "  σ0 (B τ v a₀) = $(round.(f.σ0; digits=4))")
    println(io, "  converged = $(f.converged)  ($(f.n_converged)/$(f.n_starts) starts)")
    println(io, "  quadrature error at optimum = $(round(f.quad_err; sigdigits=3)) nats/trial")
    f.n_used < f.n_total &&
        println(io, "  fitted on $(f.n_used) of $(f.n_total) trials (subsampled)")
    any(f.boundary) && println(io, "  !! σ0 at boundary for: ",
                               join(PARAM_ORDER[f.boundary], ", "),
                               " — variance component not supported by the data")
    f.n_zero > 0 && println(io, "  !! $(f.n_zero) trials with zero marginal density")
end

const PARAM_ORDER = ["B", "τ", "v", "a₀"]

const SIGMA0_FLOOR = 1e-3

const QTAU_MAX = 512

"""
    parameter_box(data)

Loose optimizer bounds excluding unresolved or degenerate parameter regions.
"""
function parameter_box(data::Vector{DDMResult})
    rt_max = maximum(d.rt for d in data)
    lower = [log(0.3), log(1e-4), log(1e-3), -10.0,
             log(SIGMA0_FLOOR), log(SIGMA0_FLOOR), log(SIGMA0_FLOOR), log(SIGMA0_FLOOR)]
    upper = [log(1e2), log(rt_max), log(10.0), 10.0,
             log(3.0), log(3.0), log(3.0), log(3.0)]
    return lower, upper
end

"""
    fit_mlddm_exact(data; q=8, qτ=2q, n_starts=4, rng, init=nothing, verbose=true)

Fit multilevel DDM hyperparameters by maximum marginal likelihood.
"""
function fit_mlddm_exact(data::Vector{DDMResult};
                         q::Int=8,
                         qτ::Int=2q,
                         n_starts::Int=4,
                         rng::AbstractRNG=Random.default_rng(),
                         init::Union{Nothing,Tuple{Vector{Float64},Vector{Float64}}}=nothing,
                         iterations::Int=500,
                         max_trials::Int=0,
                         g_tol::Float64=0.0,
                         time_limit::Float64=NaN,
                         escalate::Bool=true,
                         q_max::Int=12,
                         escalate_tol::Float64=1e-3,
                         max_refinements::Int=2,
                         verbose::Bool=true)
    t0 = time()
    rules = quadrature_rules(q, qτ)

    n_total = length(data)
    idx = collect(1:n_total)
    if max_trials > 0 && n_total > max_trials
        idx = sort!(randperm(rng, n_total)[1:max_trials])
        data = data[idx]
        verbose && @info "subsampled $(max_trials) of $(n_total) trials for fitting"
    end

    m0, ls0 = init === nothing ? default_init(data) : (init[1], log.(init[2]))

    # Optim's gradient tolerance applies to the total log-likelihood.
    gtol = g_tol > 0 ? g_tol : 1e-6 * length(data)
    verbose && @info "gradient tolerance $(round(gtol; sigdigits=3)) on $(length(data)) trials"

    lower, upper = parameter_box(data)
    inbox(θ) = clamp.(θ, lower .+ 1e-6, upper .- 1e-6)

    function solve(rs, θ0, label)
        f = θ -> -marginal_loglik(data, collect(θ[1:4]), collect(θ[5:8]), rs)
        cfg = ForwardDiff.GradientConfig(f, θ0, ForwardDiff.Chunk{8}())
        g! = (G, θ) -> ForwardDiff.gradient!(G, f, θ, cfg)
        res = try
            Optim.optimize(f, g!, lower, upper, θ0,
                           Optim.Fminbox(Optim.LBFGS(linesearch=Optim.LineSearches.BackTracking())),
                           Optim.Options(iterations=iterations, g_tol=gtol,
                                         time_limit=isnan(time_limit) ? Inf : time_limit))
        catch err
            verbose && @warn "$label failed" exception = err
            return nothing
        end
        val = Optim.minimum(res)
        isfinite(val) || return nothing
        conv = Optim.converged(res)
        verbose && @info "$label: loglik=$(round(-val; digits=3)) converged=$conv"
        return (Optim.minimizer(res), conv, val)
    end

    function best_over_starts(rs, extra_starts, tag)
        bθ = nothing; bval = Inf; nc = 0
        starts = copy(extra_starts)
        for start in 1:n_starts
            push!(starts, inbox(start == 1 ? vcat(m0, ls0) :
                                vcat(m0 .+ 0.3 .* randn(rng, 4), ls0 .+ 0.5 .* randn(rng, 4))))
        end
        for (i, θ0) in enumerate(starts)
            r = solve(rs, inbox(θ0), "$tag start $i")
            r === nothing && continue
            r[2] && (nc += 1)
            r[3] < bval && (bval = r[3]; bθ = r[1])
        end
        return bθ, bval, nc, length(starts)
    end

    θ, best_val, n_conv, n_tried = best_over_starts(rules, Vector{Float64}[], "q=$q")
    θ === nothing && error("all $(n_starts) starts failed")
    conv = n_conv > 0

    # Validate the objective at a finer quadrature order.
    quad_err = Inf
    if escalate
        for attempt in 0:max_refinements
            rules_ref = quadrature_rules(min(q + 2, q_max), min(4 * qτ, QTAU_MAX))
            ll_ref = marginal_loglik(data, θ[1:4], θ[5:8], rules_ref)
            quad_err = abs(ll_ref + best_val) / length(data)
            verbose && @info "quadrature check at q=$q qτ=$qτ: $(round(quad_err; sigdigits=3)) nats/trial"
            quad_err < escalate_tol && break
            attempt == max_refinements && break
            q, qτ = min(q + 2, q_max), min(4 * qτ, QTAU_MAX)
            rules = rules_ref
            θ_n, val_n, nc_n, nt_n = best_over_starts(rules, [θ], "q=$q qτ=$qτ")
            θ_n === nothing && break
            θ, best_val, n_conv, n_tried = θ_n, val_n, nc_n, nt_n
        end
    end
    conv = n_conv > 0 && quad_err < escalate_tol

    m = θ[1:4]
    σ0 = exp.(θ[5:8])

    g = build_grid(m, θ[5:8], rules)
    n_zero = count(y -> trial_marginal_loglik(y, g) <= LOG_ZERO_MARGINAL / 2, data)

    ll = -best_val
    k = 8
    bic = -2 * ll + k * log(length(data))

    return MLDDMFit(m, σ0, ll, bic, conv, σ0 .<= SIGMA0_FLOOR * 1.01, n_zero,
                    quad_err, q, qτ, n_tried, n_conv, length(data), n_total, idx,
                    time() - t0)
end

"""
    default_init(data)

Data-driven starting values for `(m, logσ0)`.
"""
function default_init(data::Vector{DDMResult})
    rts = [d.rt for d in data]
    choices = [d.choice for d in data]

    τ_init = max(quantile(rts, 0.1), 1e-3)
    B_init = clamp(std(rts) * 2.0, 0.5, 5.0)
    accuracy = mean(choices .== 1)
    v_init = clamp(abs(log((accuracy + 0.01) / (1 - accuracy + 0.01))), 0.3, 3.0)

    m0 = [log(B_init), log(τ_init), log(v_init), 0.0]
    return m0, log.([0.2, 0.15, 0.25, 0.2])
end

"""
    quadrature_check(data, m, σ0; qs=(4, 6, 8, 10), qτ_mult=2)

Evaluate log-likelihood across quadrature orders.
"""
function quadrature_check(data::Vector{DDMResult},
                          m::AbstractVector{<:Real},
                          σ0::AbstractVector{<:Real};
                          qs=(4, 6, 8, 10), qτ_mult::Int=2)
    logσ0 = log.(collect(float.(σ0)))
    mm = collect(float.(m))
    N = length(data)
    out = NamedTuple[]
    prev = nothing
    for q in qs
        qτ = qτ_mult * q
        ll = marginal_loglik(data, mm, logσ0; q=q, qτ=qτ)
        Δ = prev === nothing ? NaN : (ll - prev) / N
        push!(out, (q=q, qτ=qτ, loglik=ll, delta_per_trial=Δ))
        prev = ll
    end
    return out
end

"""
    trial_posteriors(data, m, σ0; q=8, qτ=2q)

Per-trial posterior mean and standard deviation of `u_t`.

Returns two `4 × N` matrices in `(B, τ, v, a₀)` order.
"""
function trial_posteriors(data::Vector{DDMResult},
                          m::AbstractVector{<:Real},
                          σ0::AbstractVector{<:Real};
                          q::Int=8, qτ::Int=2q)
    rules = quadrature_rules(q, qτ)
    mm = collect(float.(m))
    ls = log.(collect(float.(σ0)))
    g = build_grid(mm, ls, rules)

    uB = log.(g.B)
    uv = log.(g.v)
    ua = log.(g.a₀ ./ (1 .- g.a₀))

    N = length(data)
    nτ = n_tau_nodes(g.τ)
    μ = zeros(4, N)
    sd = zeros(4, N)

    @threads for t in 1:N
        y = data[t]
        if !(y.rt > 0)
            μ[:, t] .= NaN
            sd[:, t] .= NaN
            continue
        end

        tn = TauNodes(y.rt, g.m_τ, g.σ_τ, g.τ)
        τs = Vector{Float64}(undef, nτ)
        uτs = Vector{Float64}(undef, nτ)
        lwτ = Vector{Float64}(undef, nτ)
        for j in 1:nτ
            z, lw = tn[j]
            uτs[j] = g.m_τ + g.σ_τ * z
            τs[j] = exp(uτs[j])
            lwτ[j] = lw
        end

        mx = -Inf
        @inbounds for i4 in eachindex(g.a₀), i3 in eachindex(g.v),
                      j in 1:nτ, i1 in eachindex(g.B)
            t_ = g.logw_a₀[i4] + g.logw_v[i3] + lwτ[j] + g.logw_B[i1] +
                 logdensityof(g.B[i1], g.v[i3], g.a₀[i4], τs[j],
                              y.rt, y.choice, y.s)
            t_ > mx && (mx = t_)
        end

        Z = 0.0
        s1 = zeros(4)
        s2 = zeros(4)
        @inbounds for i4 in eachindex(g.a₀), i3 in eachindex(g.v),
                      j in 1:nτ, i1 in eachindex(g.B)
            w = exp(g.logw_a₀[i4] + g.logw_v[i3] + lwτ[j] + g.logw_B[i1] +
                    logdensityof(g.B[i1], g.v[i3], g.a₀[i4], τs[j],
                                 y.rt, y.choice, y.s) - mx)
            w == 0 && continue
            Z += w
            s1[1] += w * uB[i1]; s2[1] += w * uB[i1]^2
            s1[2] += w * uτs[j]; s2[2] += w * uτs[j]^2
            s1[3] += w * uv[i3]; s2[3] += w * uv[i3]^2
            s1[4] += w * ua[i4]; s2[4] += w * ua[i4]^2
        end

        if Z <= 0
            μ[:, t] .= NaN
            sd[:, t] .= NaN
        else
            for d in 1:4
                mean_d = s1[d] / Z
                μ[d, t] = mean_d
                sd[d, t] = sqrt(max(s2[d] / Z - mean_d^2, 0.0))
            end
        end
    end

    return μ, sd
end

"""
    profile_sigma0(data, fit, dim; factors=..., q=fit.q, qτ=fit.qτ)

Conditional profile for one `σ0` component with other parameters fixed.
"""
function profile_sigma0(data::Vector{DDMResult}, fit::MLDDMFit, dim::Int;
                        factors=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0),
                        q::Int=fit.q, qτ::Int=fit.qτ)
    1 <= dim <= 4 || throw(ArgumentError("dim must be 1..4 (B, τ, v, a₀)"))
    length(data) == fit.n_total || throw(ArgumentError(
        "profile expects the same `data` passed to fit_mlddm_exact " *
        "($(fit.n_total) trials, got $(length(data))); the fit's trial indices " *
        "are relative to it"))

    fitted = data[fit.idx]
    rules = quadrature_rules(q, qτ)
    out = NamedTuple[]
    for f in factors
        σ = copy(fit.σ0)
        σ[dim] *= f
        ll = marginal_loglik(fitted, fit.m, log.(σ), rules)
        push!(out, (factor=f, sigma0=σ[dim], loglik=ll, delta=ll - fit.loglik))
    end
    return out
end
