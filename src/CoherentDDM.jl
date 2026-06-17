"""
    CoherentDDM

A drift diffusion model where the drift rate scales as a power law of stimulus coherence:

    v_trial = s · k · c^α

where `k` is the drift gain, `c ∈ [0,1]` is coherence, `s ∈ {-1,+1}` is stimulus
direction, and `α > 0` is the coherence exponent (α=1 recovers the linear model).
At c=0 the drift is defined as 0 regardless of α.

The `fit_α` flag controls whether the coherence exponent `α` is estimated during
`fit!`. Set `fit_α = false` to hold `α` fixed at its current value. Needed when
sharing a single nonlinearity across HMM states: give every state's emission the
same `α` and `fit_α = false` so each state-specific `fit!` leaves `α` untouched.

The numeric fields are parametric in `T<:Real` so the parameters can carry
`ForwardDiff.Dual` numbers — this lets gradients flow through an HMM's forward
likelihood (see `fit_hmm_gradient!`) all the way into the emission parameters.
"""
mutable struct CoherentDDM{T<:Real}
    B::T          # Boundary separation
    k::T          # Drift gain
    α::T          # Coherence exponent (1.0 = linear)
    a₀::T         # Starting point as fraction of B
    τ::T          # Non-decision time
    fit_α::Bool   # Whether α is estimated (true) or held fixed (false) in fit!
end

# Promoting positional constructor: allows mixing Int/Float/Dual arguments.
function CoherentDDM(B::Real, k::Real, α::Real, a₀::Real, τ::Real, fit_α::Bool=true)
    B, k, α, a₀, τ = promote(B, k, α, a₀, τ)
    return CoherentDDM{typeof(B)}(B, k, α, a₀, τ, fit_α)
end

function CoherentDDM(;
    B   = 5.0,
    k   = 2.0,
    α   = 1.0,
    a₀  = 0.5,
    τ   = 1e-3,
    fit_α::Bool = true,
)
    return CoherentDDM(B, k, α, a₀, τ, fit_α)
end

"""
    CoherentDDMResult

A single trial observation for a CoherentDDM: reaction time, binary choice,
stimulus direction, and coherence level.
"""
struct CoherentDDMResult
    rt::Float64    # Reaction time
    choice::Int    # -1 (lower/Left) or +1 (upper/Right)
    s::Int         # Stimulus direction: -1 or +1
    c::Float64     # Coherence ∈ [0, 1]
end

function CoherentDDMResult(; rt::Float64, choice::Int, s::Int, c::Float64)
    return CoherentDDMResult(rt, choice, s, c)
end

"""
    simulateDDM(model::CoherentDDM, c::Float64, dt::Float64, rng)

Simulate a single CoherentDDM trial at coherence level `c` using Euler-Maruyama.
"""
function simulateDDM(model::CoherentDDM, c::Float64, dt::Float64=1e-5, rng::AbstractRNG=Random.default_rng())
    @unpack B, k, α, a₀, τ = model

    s = rand(rng, [-1, 1])
    v = s * k * (c > 0 ? c^α : 0.0)

    t = 0.0
    a = a₀ * B

    while a < B && a > 0
        if t < τ
            t += dt
        else
            a += v * dt + sqrt(dt) * randn(rng)
            t += dt
        end
    end

    choice = a >= B ? 1 : -1
    return CoherentDDMResult(t, choice, s, c)
end

"""
    simulateDDM(model::CoherentDDM, s::Int, c::Float64, dt::Float64, rng)

Simulate a single CoherentDDM trial with a fixed stimulus direction `s` (±1)
and coherence `c`. Useful for PPCs where the stimulus design is held fixed.
"""
function simulateDDM(model::CoherentDDM, s::Int, c::Float64, dt::Float64=1e-5, rng::AbstractRNG=Random.default_rng())
    @unpack B, k, α, a₀, τ = model

    v = s * k * (c > 0 ? c^α : 0.0)

    t = 0.0
    a = a₀ * B

    while a < B && a > 0
        if t < τ
            t += dt
        else
            a += v * dt + sqrt(dt) * randn(rng)
            t += dt
        end
    end

    choice = a >= B ? 1 : -1
    return CoherentDDMResult(t, choice, s, c)
end

"""
    simulateDDM(model::CoherentDDM, coherences::Vector{Float64}, dt::Float64)

Simulate one trial per entry in `coherences`, multi-threaded.
"""
function simulateDDM(model::CoherentDDM, coherences::Vector{Float64}, dt::Float64=1e-5)
    results = Vector{CoherentDDMResult}(undef, length(coherences))
    @threads for i in eachindex(coherences)
        results[i] = simulateDDM(model, coherences[i], dt)
    end
    return results
end

"""
    Random.rand(rng, model::CoherentDDM)

Sample a single trial at coherence 1.0 — required by HiddenMarkovModels.jl interface.
"""
function Random.rand(rng::AbstractRNG, model::CoherentDDM)
    return simulateDDM(model, 1.0, 1e-6, rng)
end

DensityInterface.DensityKind(::CoherentDDM) = HasDensity()

"""
    DensityInterface.logdensityof(model::CoherentDDM, x::CoherentDDMResult)
"""
function DensityInterface.logdensityof(model::CoherentDDM, x::CoherentDDMResult)
    @unpack B, k, α, a₀, τ = model
    @unpack rt, choice, s, c = x
    return logdensityof(B, k, α, a₀, τ, rt, choice, s, c)
end

"""
    logdensityof(B, k, α, a₀, τ, rt, choice, s, c)

Log-density of a CoherentDDM trial. The signed drift is `v = s·k·c^α`,
with `v = 0` defined at `c = 0` for all α (continuous extension).
Uses the lower-boundary WFPT density with reflection for upper-boundary responses.
"""
function logdensityof(B::TB, k::TK, α::TAlpha, a₀::TA, τ::TT,
                      rt::Float64, choice::Int, s::Int, c::Float64
) where {TB<:Real, TK<:Real, TAlpha<:Real, TA<:Real, TT<:Real}
    if rt <= 0
        return -Inf
    end
    T = promote_type(TB, TK, TAlpha, TA, TT)
    B, k, α, a₀, τ = T(B), T(k), T(α), T(a₀), T(τ)

    @assert (s == 1 || s == -1)           "stimulus s must be ±1"
    @assert (choice == 1 || choice == -1) "choice must be ±1"
    @assert 0 ≤ c ≤ 1                     "coherence c must be in [0, 1]"

    v_signed = s * k * (c > 0 ? c^α : zero(T))

    # wfpt is the lower-boundary density; reflect for upper-boundary responses
    if choice == -1
        v_eff = v_signed
        w_eff = a₀
    else
        v_eff = -v_signed
        w_eff = 1 - a₀
    end

    dens = wfpt(rt, v_eff, B, w_eff, τ)
    dens = max(dens, 1e-16)
    logdens = log(dens)
    return isfinite(logdens) ? logdens : -1e16
end

######################################################################
# Parameter transforms (constrained ↔ unconstrained)
######################################################################

"""
    _constrain_coherent(uB, uk, uα, ua₀, uτ) -> (B, k, α, a₀, τ)

Map an unconstrained parameter tuple to the constrained CoherentDDM space:
`B, k, α, τ > 0` via `exp`, and `a₀ ∈ (0, 1)` via `logistic`. Used by every
gradient-based fit so optimisation can run on all of ℝ without box bounds.
"""
@inline _constrain_coherent(uB, uk, uα, ua₀, uτ) =
    (exp(uB), exp(uk), exp(uα), logistic(ua₀), exp(uτ))

"""
    coherent_to_unconstrained(model::CoherentDDM) -> Vector{Float64}

Pack a CoherentDDM's parameters into an unconstrained vector
`[log B, log k, log α, logit a₀, log τ]`.
"""
coherent_to_unconstrained(m::CoherentDDM) =
    [log(m.B), log(m.k), log(m.α), logit(m.a₀), log(m.τ)]

"""
    coherent_from_unconstrained(u; fit_α=true) -> CoherentDDM

Inverse of [`coherent_to_unconstrained`](@ref): build a CoherentDDM from an
unconstrained vector `u = [uB, uk, uα, ua₀, uτ]`.
"""
function coherent_from_unconstrained(u::AbstractVector; fit_α::Bool=true)
    B, k, α, a₀, τ = _constrain_coherent(u[1], u[2], u[3], u[4], u[5])
    return CoherentDDM(B, k, α, a₀, τ, fit_α)
end

"""
    StatsAPI.fit!(model::CoherentDDM, x::Vector{CoherentDDMResult}, w)

MLE estimation of CoherentDDM parameters via **unconstrained** L-BFGS. Positivity
of `B, k, α, τ` and `a₀ ∈ (0,1)` are enforced through the parameter transform
(see [`_constrain_coherent`](@ref)) rather than box bounds, and the gradient is
obtained by ForwardDiff. Accepts an optional weights vector to support use as an
HMM emission distribution. When `model.fit_α` is `false`, `α` is held fixed at
its current value and dropped from the optimisation.
"""
function StatsAPI.fit!(model::CoherentDDM, x::Vector{CoherentDDMResult},
                       w::AbstractVector{<:Real}=ones(length(x)))
    @unpack B, k, α, a₀, τ, fit_α = model
    αfix = α

    # Optimise in unconstrained space. With fit_α=false, uα is omitted and the
    # fixed α spliced back in, so the likelihood is identical either way.
    function neg_log_likelihood(θ)
        if fit_α
            B_t, k_t, α_t, a₀_t, τ_t = _constrain_coherent(θ[1], θ[2], θ[3], θ[4], θ[5])
        else
            B_t  = exp(θ[1]); k_t = exp(θ[2]); a₀_t = logistic(θ[3]); τ_t = exp(θ[4])
            α_t  = oftype(B_t, αfix)
        end
        ll = zero(eltype(θ))
        @inbounds for i in eachindex(x)
            ll += w[i] * logdensityof(B_t, k_t, α_t, a₀_t, τ_t,
                                       x[i].rt, x[i].choice, x[i].s, x[i].c)
        end
        return -ll
    end

    θ0 = fit_α ? [log(B), log(k), log(α), logit(a₀), log(τ)] :
                 [log(B), log(k), logit(a₀), log(τ)]

    g! = (g, θ) -> ForwardDiff.gradient!(g, neg_log_likelihood, θ)
    result = optimize(neg_log_likelihood, g!, θ0,
                      LBFGS(linesearch=Optim.LineSearches.BackTracking()))

    opt = Optim.minimizer(result)
    if fit_α
        model.B, model.k, model.α, model.a₀, model.τ =
            _constrain_coherent(opt[1], opt[2], opt[3], opt[4], opt[5])
    else
        model.B  = exp(opt[1])
        model.k  = exp(opt[2])
        model.a₀ = logistic(opt[3])
        model.τ  = exp(opt[4])
        # model.α left unchanged
    end

    return model
end

"""
    fit_shared_α!(models::Vector{CoherentDDM}, x, γ)

Jointly estimate per-state CoherentDDM parameters **with a single coherence
exponent `α` shared across all states**. Each state `i` keeps its own
`(B, k, a₀, τ)`, weighted by row `i` of the state-marginal matrix `γ`
(`K × length(x)`), while one common `α` is fit to all states' weighted data at
once. On return every model in `models` carries the same estimated `α`.

The objective is the total weighted log-likelihood

    Σᵢ Σₜ γ[i,t] · logdensityof(Bᵢ, kᵢ, α, a₀ᵢ, τᵢ, xₜ)

maximised over `{(Bᵢ, kᵢ, a₀ᵢ, τᵢ)}ᵢ ∪ {α}` via **unconstrained** L-BFGS with a
ForwardDiff gradient (positivity / (0,1) constraints handled by the parameter
transform). The per-model `fit_α` flag is ignored here — α is always estimated
jointly in this routine.
"""
function fit_shared_α!(models::AbstractVector{<:CoherentDDM},
                       x::AbstractVector{CoherentDDMResult},
                       γ::AbstractMatrix{<:Real})
    K = length(models)
    @assert size(γ, 1) == K            "γ must have one row per state (got $(size(γ,1)) rows, $K models)"
    @assert size(γ, 2) == length(x)    "γ columns must match number of observations"

    nper = 4  # per-state unconstrained params: uB, uk, ua₀, uτ; shared uα trails

    # Unconstrained layout: [uB₁,uk₁,ua₀₁,uτ₁, …, uB_K,uk_K,ua₀_K,uτ_K, uα]
    function neg_log_likelihood(θ)
        α_t = exp(θ[end])
        ll = zero(eltype(θ))
        for i in 1:K
            off  = (i - 1) * nper
            B_t  = exp(θ[off + 1])
            k_t  = exp(θ[off + 2])
            a₀_t = logistic(θ[off + 3])
            τ_t  = exp(θ[off + 4])
            @inbounds for t in eachindex(x)
                wt = γ[i, t]
                wt == 0 && continue
                ll += wt * logdensityof(B_t, k_t, α_t, a₀_t, τ_t,
                                        x[t].rt, x[t].choice, x[t].s, x[t].c)
            end
        end
        return -ll
    end

    θ0 = Float64[]
    for m in models
        append!(θ0, (log(m.B), log(m.k), logit(m.a₀), log(m.τ)))
    end
    push!(θ0, log(models[1].α))   # shared α initialised from the first model

    g! = (g, θ) -> ForwardDiff.gradient!(g, neg_log_likelihood, θ)
    result = optimize(neg_log_likelihood, g!, θ0,
                      LBFGS(linesearch=Optim.LineSearches.BackTracking()))

    opt = Optim.minimizer(result)
    α_shared = exp(opt[end])
    for i in 1:K
        off = (i - 1) * nper
        models[i].B  = exp(opt[off + 1])
        models[i].k  = exp(opt[off + 2])
        models[i].a₀ = logistic(opt[off + 3])
        models[i].τ  = exp(opt[off + 4])
        models[i].α  = α_shared
    end

    return models
end

"""
    StatsAPI.fit!(hmm::PriorHMM{<:Real,<:CoherentDDM}, fb, obs_seq; seq_ends)

Baum–Welch M-step specialised for CoherentDDM emissions. The initial/transition
update is identical to the generic `PriorHMM` M-step. When `hmm.share_α` is
`true`, the emissions are fit *jointly* with a single shared coherence exponent
`α` via [`fit_shared_α!`](@ref); otherwise each state's emission is fit
independently (respecting its own `fit_α` flag).
"""
function StatsAPI.fit!(hmm::PriorHMM{<:Real,<:CoherentDDM},
                       fb::HiddenMarkovModels.ForwardBackwardStorage,
                       obs_seq::AbstractVector; seq_ends)
    K = length(hmm)

    _update_init_trans!(hmm, fb, seq_ends)

    if hmm.share_α
        fit_shared_α!(hmm.dists, obs_seq, fb.γ)
    else
        for i in 1:K
            StatsAPI.fit!(hmm.dists[i], obs_seq, fb.γ[i, :])
        end
    end

    @assert HiddenMarkovModels.valid_hmm(hmm)
    return nothing
end
