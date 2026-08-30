"""Utilities for validating multilevel-DDM variational inference."""
module VIValidation

using DriftDiffusionModels
using Optim
using Random
using Statistics
using LinearAlgebra
using Printf
using Logging

const DDMs = DriftDiffusionModels

const PARAM_NAMES = ("B", "τ", "v", "a₀")

export PARAM_NAMES,
       to_u, from_u, HyperTruth,
       sample_trial, simulate_iid, simulate_ar1, simulate_switching,
       run_vi, refine_qs, elbo_fixed, hyper_from_qs_fullcov,
       package_init_hyper,
       reference_posterior, run_vi_fullcov,
       acf, acf_null_band, summarize_recovery, write_csv


# Parameter-space helpers

"""
    to_u(B, τ, v, a₀)

Map `(B, τ, v, a₀)` to unconstrained space.
"""
to_u(B, τ, v, a₀) = [log(B), log(τ), log(v), log(a₀ / (1 - a₀))]

"""
    from_u(u)

Map unconstrained parameters to `(B, τ, v, a₀)`.
"""
function from_u(u::AbstractVector{<:Real})
    B, v, a₀, τ = DDMs.transform_params(u)
    return (B, τ, v, a₀)
end

"""
    HyperTruth(m, σ0)

Simulation hyperdistribution in unconstrained `(B, τ, v, a₀)` order.
"""
struct HyperTruth
    m::Vector{Float64}
    σ0::Vector{Float64}
end

"""
    HyperTruth(; B, τ, v, a₀, σ0)

Build simulation truth from constrained parameters.
"""
function HyperTruth(; B::Float64=1.2, τ::Float64=0.15, v::Float64=1.5,
                     a₀::Float64=0.5,
                     σ0::Vector{Float64}=[0.25, 0.15, 0.30, 0.20])
    return HyperTruth(to_u(B, τ, v, a₀), σ0)
end

as_hyper(h::HyperTruth) = DDMs.DDMHyper(copy(h.m), log.(h.σ0))


# Exact trial sampling from the inference likelihood

"""
    sample_trial(rng, B, v, a₀, τ; tmax_mult=12.0, ngrid=2000)

Draw from the first-passage-time likelihood by numerical CDF inversion.
"""
function sample_trial(rng::AbstractRNG, B::Real, v::Real, a₀::Real, τ::Real;
                      tmax_mult::Float64=12.0, ngrid::Int=2000)
    s = rand(rng, (-1, 1))

    # Cover both drift- and diffusion-dominated decision times.
    scale = max(B^2 / max(abs(v), 1e-3), B^2)
    tmax = τ + tmax_mult * scale

    ts = range(τ + 1e-6, tmax; length=ngrid)
    dt = step(ts)

    fl = similar(ts, Float64)
    fu = similar(ts, Float64)
    @inbounds for (i, t) in enumerate(ts)
        fl[i] = exp(DDMs.logdensityof(B, v, a₀, τ, t, -1, s))
        fu[i] = exp(DDMs.logdensityof(B, v, a₀, τ, t, 1, s))
    end

    ml = sum(fl) * dt
    mu = sum(fu) * dt
    total = ml + mu
    total <= 0 && error("degenerate likelihood at B=$B v=$v a₀=$a₀ τ=$τ")

    choice = rand(rng) < (mu / total) ? 1 : -1
    f = choice == 1 ? fu : fl

    cdf = cumsum(f) .* dt
    cdf ./= cdf[end]
    r = rand(rng)
    idx = searchsortedfirst(cdf, r)
    idx = clamp(idx, 1, ngrid)
    rt = ts[idx]

    return DDMResult(rt, choice, s)
end

"""
    simulate_iid(rng, truth, N)

Simulate i.i.d. parameters and trials. Returns `(data, U)`.
"""
function simulate_iid(rng::AbstractRNG, truth::HyperTruth, N::Int)
    U = truth.m .+ truth.σ0 .* randn(rng, 4, N)
    return materialize(rng, U), U
end

"""
    simulate_ar1(rng, truth, N; ρ=0.98)

Simulate AR(1) parameters with the requested stationary marginal.
"""
function simulate_ar1(rng::AbstractRNG, truth::HyperTruth, N::Int; ρ::Float64=0.98)
    U = zeros(4, N)
    z = randn(rng, 4)
    @inbounds for t in 1:N
        z = ρ .* z .+ sqrt(1 - ρ^2) .* randn(rng, 4)
        U[:, t] = truth.m .+ truth.σ0 .* z
    end
    return materialize(rng, U), U
end

"""
    simulate_switching(rng, truth, N; K=3, stay=0.995, spread=1.5)

Simulate parameters from a sticky `K`-state Markov chain.
"""
function simulate_switching(rng::AbstractRNG, truth::HyperTruth, N::Int;
                            K::Int=3, stay::Float64=0.995, spread::Float64=1.5)
    centers = [truth.m .+ spread .* truth.σ0 .* (2 * (k - 1) / max(K - 1, 1) - 1)
               for k in 1:K]
    U = zeros(4, N)
    states = zeros(Int, N)
    k = rand(rng, 1:K)
    @inbounds for t in 1:N
        if rand(rng) > stay
            k = rand(rng, filter(!=(k), 1:K))
        end
        states[t] = k
        U[:, t] = centers[k] .+ 0.25 .* truth.σ0 .* randn(rng, 4)
    end
    return materialize(rng, U), U, states
end

"""
    materialize(rng, U)

Draw one trial per column of true unconstrained parameters.
"""
function materialize(rng::AbstractRNG, U::AbstractMatrix{Float64})
    N = size(U, 2)
    data = Vector{DDMResult}(undef, N)
    for t in 1:N
        B, v, a₀, τ = DDMs.transform_params(@view U[:, t])
        data[t] = sample_trial(rng, B, v, a₀, τ)
    end
    return data
end


# The package's own default initialisation

"""
    package_init_hyper(data)

Reproduce `fit_vi_gaussian(...; init_from_data=true)` initialization.
"""
function package_init_hyper(data::Vector{DDMResult})
    rts = [d.rt for d in data]
    choices = [d.choice for d in data]

    τ_init = quantile(rts, 0.1)
    B_init = clamp(std(rts) * 2.0, 0.5, 5.0)
    accuracy = mean(choices .== 1)
    v_init = clamp(abs(log((accuracy + 0.01) / (1 - accuracy + 0.01))), 0.3, 3.0)

    m0 = [log(B_init), log(τ_init), log(v_init), 0.0]
    return DDMs.DDMHyper(m0, log.([0.7, 0.5, 0.7, 0.5]))
end


# Diagonal-q variational fitting, instrumented

"""
    elbo_fixed(qs, data, hyper, eps_common)

Deterministic ELBO estimate using shared standard-normal draws.
"""
function elbo_fixed(qs::Vector{<:DDMs.TrialVIParams}, data::Vector{DDMResult},
                    hyper::DDMs.DDMHyper, eps_common::Vector{Vector{Float64}})
    total = 0.0
    M = length(eps_common)
    @inbounds for i in eachindex(data)
        q = qs[i]
        σ = exp.(q.logσ)
        ll = 0.0
        for ε in eps_common
            u = q.μ .+ σ .* ε
            B, v, a₀, τ = DDMs.transform_params(u)
            ll += DDMs.logdensityof(B, v, a₀, τ, data[i].rt, data[i].choice, data[i].s)
        end
        total += ll / M - DDMs.kl_gaussian_diag(q, hyper)
    end
    return total
end

"""
    run_vi(data; ...)

Instrumented `fit_vi_gaussian` loop. Returns `(hyper, qs, trace, diag)`.
"""
function run_vi(data::Vector{DDMResult};
                hyper0::DDMs.DDMHyper,
                n_iter::Int=30,
                K::Int=3,
                fresh_eps::Bool=true,
                eval_eps::Union{Nothing,Vector{Vector{Float64}}}=nothing,
                rng::AbstractRNG=Random.default_rng(),
                jitter::Float64=0.1,
                verbose::Bool=false)
    N = length(data)
    hyper = DDMs.DDMHyper(copy(hyper0.m), copy(hyper0.logσ))

    qs = [DDMs.TrialVIParams(hyper0.m .+ jitter .* randn(rng, 4), copy(hyper0.logσ))
          for _ in 1:N]

    fixed = fresh_eps ? nothing : [[randn(rng, 4) for _ in 1:K] for _ in 1:N]
    ev = eval_eps === nothing ? [randn(rng, 4) for _ in 1:64] : eval_eps

    trace = NamedTuple[]
    stuck_total = 0

    for iter in 1:n_iter
        epss = fresh_eps ? [[randn(rng, 4) for _ in 1:K] for _ in 1:N] : fixed

        before = [copy(q.μ) for q in qs]
        # Count failed trials without emitting one warning per trial.
        Logging.with_logger(Logging.NullLogger()) do
            Threads.@threads for i in 1:N
                qs[i] = DDMs.optimize_trial_vi(qs[i], data[i], hyper, epss[i])
            end
        end
        stuck = count(i -> qs[i].μ == before[i], 1:N)
        stuck_total += stuck

        # Score the E- and M-steps separately.
        el_E = elbo_fixed(qs, data, hyper, ev)
        hyper = DDMs.update_hyper_from_qs(qs)
        el_M = elbo_fixed(qs, data, hyper, ev)

        push!(trace, (iter=iter, elbo=el_M, elbo_estep=el_E, elbo_mstep=el_M,
                      stuck=stuck, m=copy(hyper.m), σ0=exp.(hyper.logσ)))
        verbose && @printf("  iter %3d  ELBO afterE %13.2f  afterM %13.2f  stuck %4d
",
                           iter, el_E, el_M, stuck)
    end

    return hyper, qs, trace, (stuck_total=stuck_total, N=N, K=K, n_iter=n_iter)
end

"""
    refine_qs(data, hyper, qs; M=64, rng)

Run an E-step at a fixed hyperdistribution.
"""
function refine_qs(data::Vector{DDMResult}, hyper::DDMs.DDMHyper,
                   qs::Vector{<:DDMs.TrialVIParams};
                   M::Int=64, rng::AbstractRNG=Random.default_rng())
    N = length(data)
    epss = [[randn(rng, 4) for _ in 1:M] for _ in 1:N]
    out = Vector{DDMs.TrialVIParams{Float64}}(undef, N)
    Logging.with_logger(Logging.NullLogger()) do
        Threads.@threads for i in 1:N
            out[i] = DDMs.optimize_trial_vi(qs[i], data[i], hyper, epss[i])
        end
    end
    return out
end


# Reference posterior by dense quadrature

"""
    reference_posterior(y, hyper; ngrid=25, span=5.0)

Dense-grid single-trial posterior. Returns `(μ, Σ, edge_mass)`.
"""
function reference_posterior(y::DDMResult, hyper::DDMs.DDMHyper;
                             ngrid::Int=25, span::Float64=5.0)
    σ0 = exp.(hyper.logσ)
    ax = ntuple(d -> collect(range(hyper.m[d] - span * σ0[d],
                                   hyper.m[d] + span * σ0[d]; length=ngrid)), 4)

    logw = Array{Float64}(undef, ngrid, ngrid, ngrid, ngrid)
    u = zeros(4)
    @inbounds for i4 in 1:ngrid, i3 in 1:ngrid, i2 in 1:ngrid, i1 in 1:ngrid
        u[1] = ax[1][i1]; u[2] = ax[2][i2]; u[3] = ax[3][i3]; u[4] = ax[4][i4]
        B, v, a₀, τ = DDMs.transform_params(u)
        logw[i1, i2, i3, i4] =
            DDMs.logdensityof(B, v, a₀, τ, y.rt, y.choice, y.s) +
            DDMs.prior_logpdf(u, hyper)
    end

    w = exp.(logw .- maximum(logw))
    w ./= sum(w)

    μ = zeros(4)
    M2 = zeros(4, 4)
    edge = 0.0
    @inbounds for i4 in 1:ngrid, i3 in 1:ngrid, i2 in 1:ngrid, i1 in 1:ngrid
        wt = w[i1, i2, i3, i4]
        wt == 0 && continue
        uu = (ax[1][i1], ax[2][i2], ax[3][i3], ax[4][i4])
        if i1 == 1 || i1 == ngrid || i2 == 1 || i2 == ngrid ||
           i3 == 1 || i3 == ngrid || i4 == 1 || i4 == ngrid
            edge += wt
        end
        for d in 1:4
            μ[d] += wt * uu[d]
        end
        for a in 1:4, b in 1:4
            M2[a, b] += wt * uu[a] * uu[b]
        end
    end

    Σ = M2 .- μ * μ'
    Σ = (Σ .+ Σ') ./ 2
    return μ, Σ, edge
end


# Full-covariance variational family

"""
    unpack_full(θ)

Unpack `θ` into `μ` and the Cholesky factor `L`.
"""
function unpack_full(θ::AbstractVector{T}) where {T<:Real}
    μ = θ[1:4]
    L = zeros(T, 4, 4)
    @inbounds for d in 1:4
        L[d, d] = exp(θ[4+d])
    end
    k = 9
    @inbounds for a in 2:4, b in 1:(a-1)
        L[a, b] = θ[k]
        k += 1
    end
    return μ, L
end

pack_full(μ, L) = vcat(μ, log.(diag(L)),
                       [L[a, b] for a in 2:4 for b in 1:(a-1)])

"""Closed-form `KL(N(μ, LL') ‖ N(m, diag σ0²))`."""
function kl_full(μ::AbstractVector{T}, L::AbstractMatrix{T},
                 hyper::DDMs.DDMHyper) where {T<:Real}
    σ0 = exp.(hyper.logσ)
    Σ = L * L'
    tr_term = sum(Σ[d, d] / σ0[d]^2 for d in 1:4)
    quad = sum(((μ[d] - hyper.m[d]) / σ0[d])^2 for d in 1:4)
    logdetΣ = 2 * sum(log(L[d, d]) for d in 1:4)
    logdetΣ0 = 2 * sum(hyper.logσ)
    return 0.5 * (tr_term + quad - 4 + logdetΣ0 - logdetΣ)
end

"""
    optimize_trial_full(θ0, y, hyper, eps)

Full-covariance analogue of `DriftDiffusionModels.optimize_trial_vi`.
"""
function optimize_trial_full(θ0::Vector{Float64}, y::DDMResult,
                             hyper::DDMs.DDMHyper, eps::Vector{Vector{Float64}})
    f = function (θ)
        μ, L = unpack_full(θ)
        ll = zero(eltype(θ))
        for ε in eps
            u = μ .+ L * ε
            B, v, a₀, τ = DDMs.transform_params(u)
            ll += DDMs.logdensityof(B, v, a₀, τ, y.rt, y.choice, y.s)
        end
        val = -(ll / length(eps) - kl_full(μ, L, hyper))
        isfinite(val) ? val : 1e10
    end

    res = Optim.optimize(f, θ0,
                         Optim.BFGS(linesearch=Optim.LineSearches.BackTracking());
                         autodiff=:forward)
    if !Optim.converged(res) || any(isnan, Optim.minimizer(res))
        return θ0
    end
    return Optim.minimizer(res)
end

"""
    hyper_from_qs_fullcov(Θ)

Moment-matching update for full-covariance trial posteriors.
"""
function hyper_from_qs_fullcov(Θ::Vector{Vector{Float64}})
    μs = [unpack_full(θ)[1] for θ in Θ]
    Σd = [diag(unpack_full(θ)[2] * unpack_full(θ)[2]') for θ in Θ]
    m = mean(μs)
    σ0sq = mean([Σd[i] .+ (μs[i] .- m) .^ 2 for i in eachindex(μs)])
    return DDMs.DDMHyper(collect(m), 0.5 .* log.(σ0sq .+ 1e-6))
end

"""
    run_vi_fullcov(data; ...)

Coordinate ascent with full-covariance trial posteriors.
"""
function run_vi_fullcov(data::Vector{DDMResult};
                        hyper0::DDMs.DDMHyper,
                        n_iter::Int=30,
                        K::Int=3,
                        rng::AbstractRNG=Random.default_rng(),
                        jitter::Float64=0.1,
                        verbose::Bool=false)
    N = length(data)
    hyper = DDMs.DDMHyper(copy(hyper0.m), copy(hyper0.logσ))

    Θ = [pack_full(hyper0.m .+ jitter .* randn(rng, 4),
                   Matrix(Diagonal(exp.(hyper0.logσ)))) for _ in 1:N]

    trace = NamedTuple[]
    for iter in 1:n_iter
        epss = [[randn(rng, 4) for _ in 1:K] for _ in 1:N]
        Logging.with_logger(Logging.NullLogger()) do
            Threads.@threads for i in 1:N
                Θ[i] = optimize_trial_full(Θ[i], data[i], hyper, epss[i])
            end
        end
        hyper = hyper_from_qs_fullcov(Θ)
        push!(trace, (iter=iter, m=copy(hyper.m), σ0=exp.(hyper.logσ)))
        verbose && @printf("  [full] iter %3d  m=%s
", iter,
                           string(round.(hyper.m; digits=3)))
    end

    μ_mat = reduce(hcat, [unpack_full(θ)[1] for θ in Θ])
    return hyper, Θ, μ_mat, trace
end


# Diagnostics

"""
    acf(x, maxlag)

Sample autocorrelation of `x` at lags `1:maxlag`.
"""
function acf(x::AbstractVector{<:Real}, maxlag::Int)
    n = length(x)
    xc = x .- mean(x)
    denom = sum(abs2, xc)
    denom == 0 && return zeros(maxlag)
    return [sum(@views xc[1:n-l] .* xc[1+l:n]) / denom for l in 1:maxlag]
end

"""
    acf_null_band(x, maxlag; nperm=200, rng, q=0.975)

Two-sided permutation band for the ACF.
"""
function acf_null_band(x::AbstractVector{<:Real}, maxlag::Int;
                       nperm::Int=200, rng::AbstractRNG=Random.default_rng(),
                       q::Float64=0.975)
    A = zeros(nperm, maxlag)
    for p in 1:nperm
        A[p, :] = acf(shuffle(rng, x), maxlag)
    end
    hi = [quantile(@view(A[:, l]), q) for l in 1:maxlag]
    lo = [quantile(@view(A[:, l]), 1 - q) for l in 1:maxlag]
    return lo, hi
end

"""
    summarize_recovery(hyper, truth)

Per-dimension hyperparameter recovery errors.
"""
function summarize_recovery(hyper::DDMs.DDMHyper, truth::HyperTruth)
    σ0 = exp.(hyper.logσ)
    return (m_est=copy(hyper.m),
            m_true=copy(truth.m),
            m_bias=hyper.m .- truth.m,
            m_bias_in_σ0=(hyper.m .- truth.m) ./ truth.σ0,
            σ0_est=σ0,
            σ0_true=copy(truth.σ0),
            σ0_ratio=σ0 ./ truth.σ0)
end

"""
    write_csv(path, header, rows)

Write rows to CSV without additional dependencies.
"""
function write_csv(path::AbstractString, header::Vector{String}, rows::Vector)
    open(path, "w") do io
        println(io, join(header, ","))
        for r in rows
            println(io, join(string.(r), ","))
        end
    end
    return path
end

end # module
