"""
    CoherentDDM

A drift diffusion model where the drift rate scales as a power law of stimulus coherence:

    v_trial = s · k · c^α

where `k` is the drift gain, `c ∈ [0,1]` is coherence, `s ∈ {-1,+1}` is stimulus
direction, and `α > 0` is the coherence exponent (α=1 recovers the linear model).
At c=0 the drift is defined as 0 regardless of α.
"""
mutable struct CoherentDDM
    B::Float64    # Boundary separation
    k::Float64    # Drift gain
    α::Float64    # Coherence exponent (1.0 = linear)
    a₀::Float64   # Starting point as fraction of B
    τ::Float64    # Non-decision time
end

function CoherentDDM(;
    B::Float64  = 5.0,
    k::Float64  = 2.0,
    α::Float64  = 1.0,
    a₀::Float64 = 0.5,
    τ::Float64  = 1e-3,
)
    return CoherentDDM(B, k, α, a₀, τ)
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

"""
    StatsAPI.fit!(model::CoherentDDM, x::Vector{CoherentDDMResult}, w)

MLE estimation of CoherentDDM parameters via L-BFGS-B. Accepts an optional
weights vector to support use as an HMM emission distribution.
"""
function StatsAPI.fit!(model::CoherentDDM, x::Vector{CoherentDDMResult},
                       w::AbstractVector{<:Real}=ones(length(x)))
    @unpack B, k, α, a₀, τ = model

    function neg_log_likelihood(params)
        B_t, k_t, α_t, a₀_t, τ_t = params
        if B_t < 0
            return convert(typeof(B_t), Inf)
        end
        ll = zero(eltype(params))
        for i in eachindex(x)
            ll += w[i] * logdensityof(B_t, k_t, α_t, a₀_t, τ_t,
                                       x[i].rt, x[i].choice, x[i].s, x[i].c)
        end
        return -ll
    end

    initial_params = [B,     k,    α,    a₀,         τ   ]
    lower_bounds   = [0.001, 0.0,  0.1,  1e-3,       1e-6]
    upper_bounds   = [50.0,  20.0, 3.0,  1.0-1e-3,   5.0 ]

    g! = (g, params) -> ForwardDiff.gradient!(g, neg_log_likelihood, params)
    result = optimize(neg_log_likelihood, g!, lower_bounds, upper_bounds, initial_params,
                      Fminbox(LBFGS(linesearch=Optim.LineSearches.BackTracking())))

    opt = Optim.minimizer(result)
    model.B  = opt[1]
    model.k  = opt[2]
    model.α  = opt[3]
    model.a₀ = opt[4]
    model.τ  = opt[5]

    return model
end
