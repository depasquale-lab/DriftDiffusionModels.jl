#=
First-Passage-Time DDM with a point-process observation of the latent state.

This is the "proper" neural DDM: the latent accumulator x_t follows a drift
diffusion SDE and runs from stimulus onset until it is absorbed at a boundary.
Neurons emit spikes (a Poisson point process) driven by x_t along the trajectory
up to absorption.

Inference is a sequential Monte Carlo (particle filter) that accounts for
within-bin boundary crossings via the Brownian-bridge correction, so it is a
faithful estimate of the continuous-time first-passage likelihood rather than
a coarse discrete-time approximation.

Conventions match DDM.jl's WFPT implementation so the spike-free marginal
reduces exactly to the Wiener first-passage density: absorbing boundaries at 0
(lower) and B (upper); relative start a₀ ∈ (0,1) with x₀ = a₀·B; unit diffusion
(σ² = 1) by convention; choice -1 = lower, +1 = upper; RT = τ + first-passage time.
=#

#= 
Observation models (Poisson point process on the latent x)

Each model maps the latent x to a per-neuron linear predictor η_n(x); the
Poisson rate is softplus(η_n(x)).
=#

abstract type AbstractObservationModel end

#=
Linear observation: η_n(x) = b_n + w_n·x
- `b` : per-neuron intercept (length N)
- `w` : per-neuron loading on the latent (length N)
=#
mutable struct LinearPoissonObservationModel{T<:Real} <: AbstractObservationModel
    b::Vector{T}
    w::Vector{T}

    # Inner constructor owns the length check. Defining it suppresses Julia's
    # auto-generated `(Vector{T}, Vector{T})` constructor, which would otherwise
    # be dispatched to (and skip validation) on positional calls.
    function LinearPoissonObservationModel{T}(
        b::AbstractVector,
        w::AbstractVector,
    ) where {T<:Real}
        length(b) == length(w) ||
            throw(ArgumentError("b and w must have the same length"))
        return new{T}(collect(T, b), collect(T, w))
    end
end

function LinearPoissonObservationModel(
    b::AbstractVector{<:Real},
    w::AbstractVector{<:Real},
)
    T = promote_type(eltype(b), eltype(w))
    T = T <: AbstractFloat ? T : Float64
    return LinearPoissonObservationModel{T}(b, w)
end

LinearPoissonObservationModel(; b::AbstractVector{<:Real}, w::AbstractVector{<:Real}) =
    LinearPoissonObservationModel(b, w)

function LinearPoissonObservationModel(N::Integer; T::Type{<:Real} = Float64)
    return LinearPoissonObservationModel{T}(zeros(T, N), ones(T, N))
end

#=
Basis-expansion observation: η_n(x) = Σ_k β[k,n]·φ_k(x), where φ_k is a Gaussian
bump centered at centers[k] with width widths[k]. Lets each neuron have a
nonlinear (bump-shaped) tuning to the latent instead of a strict linear ramp.
- `β`       : K × N basis weights
- `centers` : K bump centers (in latent units, typically spanning [0, B])
- `widths`  : K bump widths
- `ρ`       : smoothness / ridge penalty strength (used by fitting, not the likelihood)
=#
mutable struct BasisPoissonObservationModel{T<:Real} <: AbstractObservationModel
    β::Matrix{T}
    centers::Vector{T}
    widths::Vector{T}
    ρ::T
end

function BasisPoissonObservationModel(
    K::Integer,
    N::Integer;
    centers::AbstractVector{<:Real} = K == 1 ? [0.5] :
                                      collect(range(0.0, 1.0; length = K)),
    widths::AbstractVector{<:Real} = fill(1.0 / max(K - 1, 1), K),
    ρ::Real = 1e-3,
    T::Type{<:Real} = Float64,
)
    length(centers) == K || throw(ArgumentError("length(centers) must equal K"))
    length(widths) == K || throw(ArgumentError("length(widths) must equal K"))
    return BasisPoissonObservationModel{T}(
        zeros(T, K, N),
        collect(T, centers),
        collect(T, widths),
        T(ρ),
    )
end

#=
Gaussian-process observation (STUB): a nonparametric tuning curve x ↦ rate via a
sparse/inducing-point GP prior. Not yet implemented — the struct carries the
hyperparameters but a likelihood needs the variational inducing values q(u)
before it can be defined.
=#
mutable struct GPPoissonObservationModel{T<:Real} <: AbstractObservationModel
    inducing_points::Vector{T}
    lengthscale::Vector{T}   # one per neuron, or shared
    amplitude::Vector{T}     # one per neuron, or shared
    mean::Vector{T}          # baseline mean, one per neuron
end

function GPPoissonObservationModel(
    N::Integer;
    inducing_points::AbstractVector{<:Real} = range(0.0, 1.0; length = 10),
    lengthscale::Union{Real,AbstractVector{<:Real}} = 1.0,
    amplitude::Union{Real,AbstractVector{<:Real}} = 1.0,
    mean::Union{Real,AbstractVector{<:Real}} = 0.0,
    T::Type{<:Real} = Float64,
)
    _expand(x) = x isa AbstractVector ? collect(T, x) : fill(T(x), N)
    ls, amp, μ = _expand(lengthscale), _expand(amplitude), _expand(mean)
    length(ls) == N || throw(ArgumentError("lengthscale length must equal N"))
    length(amp) == N || throw(ArgumentError("amplitude length must equal N"))
    length(μ) == N || throw(ArgumentError("mean length must equal N"))
    return GPPoissonObservationModel{T}(collect(T, inducing_points), ls, amp, μ)
end

n_neurons(o::LinearPoissonObservationModel) = length(o.b)
n_neurons(o::BasisPoissonObservationModel) = size(o.β, 2)
n_neurons(o::GPPoissonObservationModel) = length(o.mean)

# --- linear predictor η_n(x) per observation model --------------------------
_drive(o::LinearPoissonObservationModel, x::Real, n::Integer) = o.b[n] + o.w[n] * x

function _basis_features(o::BasisPoissonObservationModel, x::Real)
    K = length(o.centers)
    R = promote_type(eltype(o.centers), eltype(o.widths), typeof(x))
    φ = Vector{R}(undef, K)
    for k = 1:K
        φ[k] = exp(-(x - o.centers[k])^2 / (2 * o.widths[k]^2))
    end
    return φ
end

function _drive(o::BasisPoissonObservationModel, n::Integer, φ::AbstractVector)
    η = zero(promote_type(eltype(o.β), eltype(φ)))
    for k in eachindex(φ)
        η += o.β[k, n] * φ[k]
    end
    return η
end

# Poisson log-density / sampling: rate = softplus(η)·dt
function obs_logpdf(
    o::LinearPoissonObservationModel,
    y::AbstractVector{<:Integer},
    x::Real,
    dt::Real,
)
    N = n_neurons(o)
    length(y) == N || throw(ArgumentError("y length ($(length(y))) must equal n_neurons ($N)"))
    T = promote_type(eltype(o.b), eltype(o.w), typeof(x), typeof(float(dt)))
    ll = zero(T)
    for n = 1:N
        μ = softplus(_drive(o, x, n)) * dt
        ll += (y[n] == 0 ? zero(μ) : y[n] * log(μ)) - μ - loggamma(y[n] + 1)
    end
    return ll
end

function obs_sample(rng::AbstractRNG, o::LinearPoissonObservationModel, x::Real, dt::Real)
    N = n_neurons(o)
    counts = Vector{Int}(undef, N)
    for n = 1:N
        counts[n] = rand(rng, Poisson(softplus(_drive(o, x, n)) * dt))
    end
    return counts
end

function obs_logpdf(
    o::BasisPoissonObservationModel,
    y::AbstractVector{<:Integer},
    x::Real,
    dt::Real,
)
    N = n_neurons(o)
    length(y) == N || throw(ArgumentError("y length ($(length(y))) must equal n_neurons ($N)"))
    φ = _basis_features(o, x)
    T = promote_type(eltype(o.β), eltype(φ), typeof(float(dt)))
    ll = zero(T)
    for n = 1:N
        μ = softplus(_drive(o, n, φ)) * dt
        ll += (y[n] == 0 ? zero(μ) : y[n] * log(μ)) - μ - loggamma(y[n] + 1)
    end
    return ll
end

function obs_sample(rng::AbstractRNG, o::BasisPoissonObservationModel, x::Real, dt::Real)
    N = n_neurons(o)
    φ = _basis_features(o, x)
    counts = Vector{Int}(undef, N)
    for n = 1:N
        counts[n] = rand(rng, Poisson(softplus(_drive(o, n, φ)) * dt))
    end
    return counts
end

function obs_logpdf(::GPPoissonObservationModel, ::AbstractVector{<:Integer}, ::Real, ::Real)
    error(
        "obs_logpdf for GPPoissonObservationModel is not implemented yet — the " *
        "struct needs variational inducing values q(u) before a likelihood is defined.",
    )
end

function obs_sample(::AbstractRNG, ::GPPoissonObservationModel, ::Real, ::Real)
    error(
        "obs_sample for GPPoissonObservationModel is not implemented yet — the " *
        "struct needs inducing values q(u) before sampling can be defined.",
    )
end

#=
FPTDDM model = latent first-passage accumulator + an observation model
=#

"""
    FPTDDM{O,T}

First-passage-time drift-diffusion model with a Poisson point-process observation.

Accumulator fields:
- `v`  : drift scale — the latent drifts at `v · u_t` per unit time
- `B`  : boundary separation; absorbing boundaries at `0` and `B`
- `a₀` : relative start point ∈ (0,1); absolute start `x₀ = a₀·B`
- `λ`  : leak (`0` = classic Wiener DDM; `>0` = leaky / OU accumulator)
- `σ²` : diffusion variance (convention: fix at `1` to anchor the latent scale)
- `τ`  : non-decision time (shifts RT; no accumulation happens during τ)
- `obs`: an `AbstractObservationModel` mapping x to per-neuron Poisson rates
         (`LinearPoissonObservationModel`, `BasisPoissonObservationModel`, …)

Construct with an explicit observation model via `obs=`, or with the convenience
`b=`, `w=` keywords to build a `LinearPoissonObservationModel` directly.
"""
mutable struct FPTDDM{O<:AbstractObservationModel,T<:Real}
    v::T
    B::T
    a₀::T
    λ::T
    σ²::T
    τ::T
    obs::O
end

function FPTDDM(;
    v::Real = 1.0,
    B::Real = 1.0,
    a₀::Real = 0.5,
    λ::Real = 0.0,
    σ²::Real = 1.0,
    τ::Real = 0.0,
    obs::Union{Nothing,AbstractObservationModel} = nothing,
    b::AbstractVector{<:Real} = Float64[],
    w::AbstractVector{<:Real} = Float64[],
)
    if obs === nothing
        obs = LinearPoissonObservationModel(b, w)
    end
    T = promote_type(
        typeof(v),
        typeof(B),
        typeof(a₀),
        typeof(λ),
        typeof(σ²),
        typeof(τ),
    )
    T = T <: AbstractFloat ? T : Float64
    return FPTDDM{typeof(obs),T}(T(v), T(B), T(a₀), T(λ), T(σ²), T(τ), obs)
end

n_neurons(model::FPTDDM) = n_neurons(model.obs)

# Model-level observation calls delegate to the observation model.
obs_logpdf(model::FPTDDM, y::AbstractVector{<:Integer}, x::Real, dt::Real) =
    obs_logpdf(model.obs, y, x, dt)
obs_sample(rng::AbstractRNG, model::FPTDDM, x::Real, dt::Real) =
    obs_sample(rng, model.obs, x, dt)

#=
Brownian-bridge within-bin crossing probabilities
For a Brownian step with variance `varstep = σ²·dt` from `x0` to `x1`, the
probability the continuous path touched the upper barrier B during the bin,
given both endpoints are below B, is exp(-2(B-x0)(B-x1)/varstep). Symmetric
formula for the lower barrier at 0. These are exact for driftless BB and an
excellent approximation for the small-drift-per-bin regime (dt small).
=#
@inline function _p_cross_upper(B::Real, x0::Real, x1::Real, varstep::Real)
    (x0 >= B || x1 >= B) && return one(varstep)
    return exp(-2 * (B - x0) * (B - x1) / varstep)
end

@inline function _p_cross_lower(x0::Real, x1::Real, varstep::Real)
    (x0 <= 0 || x1 <= 0) && return one(varstep)
    return exp(-2 * x0 * x1 / varstep)
end

# Absorbing-boundary particle filter (marginal likelihood)

"""
    fpt_loglik(model, n_bins, choice, dt; u, spikes, N, rng, resample_threshold)

Estimate `log p(choice, absorbed in bin n_bins [, spikes] | model)` for a single
trial whose decision period spans `n_bins` bins of width `dt`. The returned value
is a log-probability; the behavior-only first-passage *density* at
`RT = τ + n_bins·dt` is `exp(fpt_loglik(...)) / dt`.

The filter propagates particles under the free (unbounded) transition and:
  * on every non-final ("survival") bin, reweights each particle by the
    Brownian-bridge probability it touched *neither* barrier, killing any whose
    proposed endpoint already left `(0, B)`;
  * on the final ("absorption") bin, reweights by the probability it was
    absorbed at the *observed* boundary `choice` during that bin.

If `spikes` (an `n_bins × N_neurons` count matrix) is supplied, each bin also
contributes the Poisson observation log-likelihood. `u` is the per-bin stimulus
trace (defaults to all ones).
"""
function fpt_loglik(
    model::FPTDDM,
    n_bins::Integer,
    choice::Integer,
    dt::Real;
    u::Union{Nothing,AbstractVector{<:Real}} = nothing,
    spikes::Union{Nothing,AbstractMatrix{<:Integer}} = nothing,
    N::Integer = 1024,
    rng::AbstractRNG = Random.default_rng(),
    resample_threshold::Real = 0.5,
)
    n_bins >= 1 || throw(ArgumentError("n_bins must be ≥ 1"))
    choice == 1 || choice == -1 || throw(ArgumentError("choice must be ±1"))
    u === nothing || length(u) >= n_bins ||
        throw(ArgumentError("u must have at least n_bins entries"))
    spikes === nothing || size(spikes, 1) >= n_bins ||
        throw(ArgumentError("spikes must have at least n_bins rows"))

    B = model.B
    v = model.v
    λ = model.λ
    varstep = model.σ² * dt
    sσ = sqrt(varstep)
    logN = log(N)

    particles = fill(float(model.a₀ * B), N)
    log_w = fill(-logN, N)
    log_ml = 0.0

    for t = 1:n_bins
        u_t = u === nothing ? 1.0 : u[t]
        absorbing = (t == n_bins)

        for i = 1:N
            x0 = particles[i]
            μ = x0 + (-λ * x0 + v * u_t) * dt
            x1 = μ + sσ * randn(rng)
            particles[i] = x1

            if absorbing
                q =
                    choice == 1 ? _p_cross_upper(B, x0, x1, varstep) :
                    _p_cross_lower(x0, x1, varstep)
                log_w[i] += q > 0 ? log(q) : -Inf
            else
                if x1 <= 0 || x1 >= B
                    log_w[i] = -Inf
                else
                    s =
                        (1 - _p_cross_upper(B, x0, x1, varstep)) *
                        (1 - _p_cross_lower(x0, x1, varstep))
                    log_w[i] += s > 0 ? log(s) : -Inf
                end
            end

            if spikes !== nothing && isfinite(log_w[i])
                log_w[i] += obs_logpdf(model, @view(spikes[t, :]), particles[i], dt)
            end
        end

        lse = logsumexp(log_w)
        log_ml += lse
        isfinite(lse) || return -Inf   # all particles died
        log_w .-= lse

        if !absorbing
            ess = exp(-logsumexp(2 .* log_w))
            if ess < resample_threshold * N
                idx = _systematic_resample(rng, log_w, N)
                particles = particles[idx]
                fill!(log_w, -logN)
            end
        end
    end

    return log_ml
end

# Systematic resampling given normalized log weights; returns N indices.
function _systematic_resample(rng::AbstractRNG, log_w::AbstractVector, N::Integer)
    w = exp.(log_w)
    w ./= sum(w)
    cumw = cumsum(w)
    u0 = rand(rng) / N
    indices = Vector{Int}(undef, N)
    j = 1
    for i = 1:N
        u_i = u0 + (i - 1) / N
        while j < N && cumw[j] < u_i
            j += 1
        end
        indices[i] = j
    end
    return indices
end

# Trial container + likelihood wrapper

"""
    Trial{T}

A single trialized observation for the `FPTDDM`. The decision period is binned
into `n_time` bins of width `dt`; the last bin is the absorption (response) bin,
so `RT ≈ τ + n_time·dt`.

Fields:
- `spikes` : `n_time × n_neurons` integer matrix of per-bin spike counts
- `u`      : `n_time` stimulus trace (e.g. values in {-1, 0, +1})
- `choice` : observed choice, +1 (upper) or -1 (lower)
- `dt`     : bin width
"""
struct Trial{T<:Real}
    spikes::Matrix{Int}
    u::Vector{T}
    choice::Int
    dt::T

    function Trial(
        spikes::AbstractMatrix{<:Integer},
        u::AbstractVector{<:Real},
        choice::Integer,
        dt::Real,
    )
        n_time = size(spikes, 1)
        length(u) == n_time ||
            throw(ArgumentError("u length ($(length(u))) must equal n_time ($n_time)"))
        choice == 1 || choice == -1 || throw(ArgumentError("choice must be ±1"))
        dt > 0 || throw(ArgumentError("dt must be positive"))
        T = typeof(float(dt))
        return new{T}(Matrix{Int}(spikes), collect(T, u), Int(choice), T(dt))
    end
end

n_time(trial::Trial) = size(trial.spikes, 1)
n_neurons(trial::Trial) = size(trial.spikes, 2)

"""
    loglik(model, trial; N, rng, resample_threshold)

Log marginal likelihood `log p(spikes, choice, RT | model)` for a single `Trial`.
"""
function loglik(
    model::FPTDDM,
    trial::Trial;
    N::Integer = 1024,
    rng::AbstractRNG = Random.default_rng(),
    resample_threshold::Real = 0.5,
)
    return fpt_loglik(
        model,
        n_time(trial),
        trial.choice,
        trial.dt;
        u = trial.u,
        spikes = trial.spikes,
        N = N,
        rng = rng,
        resample_threshold = resample_threshold,
    )
end

"""
    loglik(model, trials; N, rng, resample_threshold)

Sum of per-trial log marginal likelihoods over a vector of `trials`.
"""
function loglik(
    model::FPTDDM,
    trials::AbstractVector{<:Trial};
    N::Integer = 1024,
    rng::AbstractRNG = Random.default_rng(),
    resample_threshold::Real = 0.5,
)
    total = 0.0
    for tr in trials
        total += loglik(model, tr; N = N, rng = rng, resample_threshold = resample_threshold)
    end
    return total
end

# =============================================================================
# Forward simulation (exact first passage via fine Euler substeps + spikes)
# =============================================================================

"""
    simulate_trial(rng, model, dt; u, max_time, substeps)

Simulate one `Trial` from `model`: evolve the accumulator from `x₀ = a₀·B` until
it is absorbed at a boundary, emitting Poisson spikes each `dt` bin. Absorption
is detected on a fine grid of `substeps` Euler steps per bin (so the realized
first-passage time is accurate), and the trial is truncated to the absorption
bin. `u` is the stimulus trace (defaults to all ones); the accumulator is
force-stopped at `max_time`.

Returns a `Trial` whose `n_time` bins span the decision period (RT ≈ τ + n_time·dt).
"""
function simulate_trial(
    rng::AbstractRNG,
    model::FPTDDM,
    dt::Real;
    u::Union{Nothing,AbstractVector{<:Real}} = nothing,
    max_time::Real = 100 * dt,
    substeps::Integer = 50,
)
    B = model.B
    v = model.v
    λ = model.λ
    N = n_neurons(model)
    max_bins = max(1, floor(Int, max_time / dt))
    ddt = dt / substeps
    sσ = sqrt(model.σ² * ddt)

    x = float(model.a₀ * B)
    rows = Vector{Vector{Int}}()
    us = Vector{Float64}()
    choice = 0
    absorbed = false

    for t = 1:max_bins
        u_t = u === nothing ? 1.0 : u[min(t, length(u))]
        for _ = 1:substeps
            x += (-λ * x + v * u_t) * ddt + sσ * randn(rng)
            if x >= B
                choice = 1
                absorbed = true
                break
            elseif x <= 0
                choice = -1
                absorbed = true
                break
            end
        end
        push!(rows, N == 0 ? Int[] : obs_sample(rng, model, clamp(x, 0, B), dt))
        push!(us, u_t)
        absorbed && break
    end

    if !absorbed
        choice = x >= B / 2 ? 1 : -1
    end

    spikes = N == 0 ? zeros(Int, length(rows), 0) : reduce(vcat, permutedims.(rows))
    return Trial(spikes, us, choice, dt)
end
