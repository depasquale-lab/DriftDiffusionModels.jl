abstract type AbstractStateModel end
abstract type AbstractObservationModel end

mutable struct LeakyAccumulatorModel{T<:Real} <: AbstractStateModel
    B::T  # soft boundary location
    v::T  # drift rate (scales stimulus input)
    λ::T  # leak
    σ²::T # diffusion coefficient
    μ₀::T # initial mean
    Σ₀::T # initial variance
    τ::T  # non-decision time (kept for RT linking; not used in accumulator dynamics)
    α::T  # boundary sharpness (logistic steepness)
    γ::T  # choice logistic steepness: P(right | x_T) = logistic(γ · x_T)
end

function LeakyAccumulatorModel(;
    B::Real = 1.0,
    v::Real = 1.0,
    λ::Real = 0.0,
    σ²::Real = 1.0,
    μ₀::Real = 0.0,
    Σ₀::Real = 0.0,
    τ::Real = 1e-1,
    α::Real = 5.0,
    γ::Real = 5.0,
)
    T = promote_type(
        typeof(B),
        typeof(v),
        typeof(λ),
        typeof(σ²),
        typeof(μ₀),
        typeof(Σ₀),
        typeof(τ),
        typeof(α),
        typeof(γ),
    )
    return LeakyAccumulatorModel{T}(T(B), T(v), T(λ), T(σ²), T(μ₀), T(Σ₀), T(τ), T(α), T(γ))
end

mutable struct LinearPoissonObservationModel{T<:Real} <: AbstractObservationModel
    b::Vector{T} # intercepts, one per neuron
    w::Vector{T} # weights, one per neuron
end

function LinearPoissonObservationModel(N::Integer; T::Type{<:Real} = Float64)
    return LinearPoissonObservationModel{T}(zeros(T, N), ones(T, N))
end

function LinearPoissonObservationModel(;
    b::AbstractVector{<:Real},
    w::AbstractVector{<:Real},
)
    length(b) == length(w) || throw(ArgumentError("b and w must have the same length"))
    T = promote_type(eltype(b), eltype(w))
    return LinearPoissonObservationModel{T}(collect(T, b), collect(T, w))
end

mutable struct BasisPoissonObservationModel{T<:Real} <: AbstractObservationModel
    β::Matrix{T}        # K × N basis weights
    centers::Vector{T}  # K basis centers
    widths::Vector{T}   # K basis widths
    ρ::T                # smoothness / ridge penalty strength
end

function BasisPoissonObservationModel(
    K::Integer,
    N::Integer;
    centers::AbstractVector{<:Real} = K == 1 ? [0.0] :
                                      collect(range(-1.0, 1.0; length = K)),
    widths::AbstractVector{<:Real} = fill(2.0 / max(K - 1, 1), K),
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

mutable struct GPPoissonObservationModel{T<:Real} <: AbstractObservationModel
    inducing_points::Vector{T}
    lengthscale::Vector{T}      # one per neuron, or shared
    amplitude::Vector{T}        # one per neuron, or shared
    mean::Vector{T}             # baseline mean function, one per neuron
end

function GPPoissonObservationModel(
    N::Integer;
    inducing_points::AbstractVector{<:Real} = range(-1.0, 1.0; length = 10),
    lengthscale::Union{Real,AbstractVector{<:Real}} = 1.0,
    amplitude::Union{Real,AbstractVector{<:Real}} = 1.0,
    mean::Union{Real,AbstractVector{<:Real}} = 0.0,
    T::Type{<:Real} = Float64,
)
    _expand(x) = x isa AbstractVector ? collect(T, x) : fill(T(x), N)
    ls = _expand(lengthscale)
    amp = _expand(amplitude)
    μ = _expand(mean)
    length(ls) == N || throw(ArgumentError("lengthscale length must equal N"))
    length(amp) == N || throw(ArgumentError("amplitude length must equal N"))
    length(μ) == N || throw(ArgumentError("mean length must equal N"))
    return GPPoissonObservationModel{T}(collect(T, inducing_points), ls, amp, μ)
end

mutable struct NeuralDDM{S<:AbstractStateModel,O<:AbstractObservationModel}
    state::S
    obs::O
end

function NeuralDDM(;
    state::AbstractStateModel = LeakyAccumulatorModel(),
    obs::AbstractObservationModel = LinearPoissonObservationModel(1),
)
    return NeuralDDM(state, obs)
end

function init_sample(rng::AbstractRNG, model::NeuralDDM)
    init_sample(rng, model.state)
end

function init_sample(rng::AbstractRNG, model::LeakyAccumulatorModel)
    return randn(rng) * sqrt(model.Σ₀) + model.μ₀
end

function init_logpdf(model::NeuralDDM, x::Real)
    init_logpdf(model.state, x)
end

function init_logpdf(model::LeakyAccumulatorModel, x::Real)
    # Σ₀ = 0 is a degenerate point mass at μ₀ (matches init_sample): log-density
    # is +Inf at x == μ₀ and -Inf elsewhere. Guard so the default model (Σ₀=0)
    # doesn't produce -log(0) = +Inf for every x.
    if iszero(model.Σ₀)
        return x == model.μ₀ ? Inf : -Inf
    end
    return -log(model.Σ₀) / 2 - log(2π) / 2 - (x - model.μ₀)^2 / (2 * model.Σ₀)
end

function transition_sample(
    rng::AbstractRNG,
    model::NeuralDDM,
    x_prev::Real,
    u::Real,
    dt::Real,
)
    return transition_sample(rng, model.state, x_prev, u, dt)
end

function transition_sample(
    rng::AbstractRNG,
    model::LeakyAccumulatorModel,
    x_prev::Real,
    u::Real,
    dt::Real,
)
    μ = x_prev + (-model.λ * x_prev + model.v * u) * dt
    σ = sqrt(model.σ² * dt)
    return μ + σ * randn(rng)
end

function transition_logpdf(model::NeuralDDM, x::Real, x_prev::Real, u::Real, dt::Real)
    return transition_logpdf(model.state, x, x_prev, u, dt)
end

function transition_logpdf(
    model::LeakyAccumulatorModel,
    x::Real,
    x_prev::Real,
    u::Real,
    dt::Real,
)
    μ = x_prev + (-model.λ * x_prev + model.v * u) * dt
    var = model.σ² * dt
    return -log(2π * var) / 2 - (x - μ)^2 / (2 * var)
end

function hazard(model::NeuralDDM, x::Real, α::Real)
    return hazard(model.state, x, α)
end

function hazard(model::LeakyAccumulatorModel, x::Real, α::Real)
    return logistic(α * (abs(x) - model.B))
end

function stop_logpdf(model::NeuralDDM, x::Real, stopped::Bool, α::Real)
    return stop_logpdf(model.state, x, stopped, α)
end

function stop_logpdf(model::LeakyAccumulatorModel, x::Real, stopped::Bool, α::Real)
    z = α * (abs(x) - model.B)
    return stopped ? -softplus(-z) : -softplus(z)
end

function obs_sample(rng::AbstractRNG, model::NeuralDDM, x::Real, dt::Real)
    return obs_sample(rng, model.obs, x, dt)
end

function obs_logpdf(model::NeuralDDM, y::AbstractVector{<:Integer}, x::Real, dt::Real)
    return obs_logpdf(model.obs, y, x, dt)
end

# Poisson rate per neuron: λ_n(x) = softplus(η_n(x)), where η is a linear predictor.
# Linear: η_n(x) = b_n + w_n · x
function _drive(obs::LinearPoissonObservationModel, x::Real, n::Integer)
    return obs.b[n] + obs.w[n] * x
end

function obs_sample(rng::AbstractRNG, obs::LinearPoissonObservationModel, x::Real, dt::Real)
    N = length(obs.b)
    counts = Vector{Int}(undef, N)
    for n = 1:N
        μ = softplus(_drive(obs, x, n)) * dt
        counts[n] = rand(rng, Poisson(μ))
    end
    return counts
end

function obs_logpdf(
    obs::LinearPoissonObservationModel,
    y::AbstractVector{<:Integer},
    x::Real,
    dt::Real,
)
    length(y) == length(obs.b) ||
        throw(ArgumentError("y length must match number of neurons"))
    T = promote_type(eltype(obs.b), eltype(obs.w), typeof(x), typeof(dt))
    ll = zero(T)
    for n in eachindex(obs.b)
        μ = softplus(_drive(obs, x, n)) * dt
        ll += (y[n] == 0 ? zero(μ) : y[n] * log(μ)) - μ - loggamma(y[n] + 1)
    end
    return ll
end

# Basis: η_n(x) = Σ_k β[k,n] · φ_k(x), φ_k Gaussian bump centered at centers[k]
function _basis_features(obs::BasisPoissonObservationModel, x::Real)
    K = length(obs.centers)
    R = promote_type(eltype(obs.centers), eltype(obs.widths), typeof(x))
    φ = Vector{R}(undef, K)
    for k = 1:K
        φ[k] = exp(-(x - obs.centers[k])^2 / (2 * obs.widths[k]^2))
    end
    return φ
end

function _drive(obs::BasisPoissonObservationModel, n::Integer, φ::AbstractVector)
    η = zero(promote_type(eltype(obs.β), eltype(φ)))
    for k in eachindex(φ)
        η += obs.β[k, n] * φ[k]
    end
    return η
end

function obs_sample(rng::AbstractRNG, obs::BasisPoissonObservationModel, x::Real, dt::Real)
    N = size(obs.β, 2)
    φ = _basis_features(obs, x)
    counts = Vector{Int}(undef, N)
    for n = 1:N
        μ = softplus(_drive(obs, n, φ)) * dt
        counts[n] = rand(rng, Poisson(μ))
    end
    return counts
end

function obs_logpdf(
    obs::BasisPoissonObservationModel,
    y::AbstractVector{<:Integer},
    x::Real,
    dt::Real,
)
    N = size(obs.β, 2)
    length(y) == N || throw(ArgumentError("y length must match number of neurons"))
    φ = _basis_features(obs, x)
    T = promote_type(eltype(obs.β), eltype(φ), typeof(dt))
    ll = zero(T)
    for n = 1:N
        μ = softplus(_drive(obs, n, φ)) * dt
        ll += (y[n] == 0 ? zero(μ) : y[n] * log(μ)) - μ - loggamma(y[n] + 1)
    end
    return ll
end

# GP: not implemented, struct needs variational q(u) fields before a likelihood is defined
function obs_sample(::AbstractRNG, ::GPPoissonObservationModel, ::Real, ::Real)
    error(
        "obs_sample for GPPoissonObservationModel is not implemented yet — the struct needs inducing values q(u) before sampling can be defined.",
    )
end

function obs_logpdf(
    ::GPPoissonObservationModel,
    ::AbstractVector{<:Integer},
    ::Real,
    ::Real,
)
    error(
        "obs_logpdf for GPPoissonObservationModel is not implemented yet — the struct needs inducing values q(u) before a likelihood is defined.",
    )
end

# P(right | x_T) = logistic(γ · x_T); choice ∈ {+1 (right), -1 (left)}
function choice_logpdf(model::LeakyAccumulatorModel, x::Real, choice::Integer)
    p_right = logistic(model.γ * x)
    return choice == 1 ? log(p_right) : log1p(-p_right)
end

function choice_logpdf(model::NeuralDDM, x::Real, choice::Integer)
    return choice_logpdf(model.state, x, choice)
end

"""
    Trial{T<:Real}

A single trialized observation for the NeuralDDM.

Fields:
- `spikes`  : `n_time × n_neurons` integer matrix of spike counts per bin
- `u`       : `n_time` stimulus trace, values ∈ {-1, 0, +1}
- `choice`  : observed choice, +1 (rightward) or -1 (leftward)
- `dt`      : bin width in seconds

`n_time` is the number of bins up to and including the stopping bin, so
`RT ≈ n_time * dt`. The last bin is treated as the stopping event.
"""
struct Trial{T<:Real}
    spikes::Matrix{Int}   # n_time × n_neurons
    u::Vector{T}          # n_time stimulus values ∈ {-1, 0, +1}
    choice::Int           # +1 right, -1 left
    dt::T

    function Trial(
        spikes::Matrix{<:Integer},
        u::AbstractVector{<:Real},
        choice::Integer,
        dt::Real,
    )
        n_time = size(spikes, 1)
        length(u) == n_time || throw(
            ArgumentError(
                "u length ($(length(u))) must equal number of time bins ($(n_time))",
            ),
        )
        choice ∈ (1, -1) || throw(ArgumentError("choice must be +1 or -1"))
        dt > 0 || throw(ArgumentError("dt must be positive"))
        T = typeof(float(dt))
        new{T}(Matrix{Int}(spikes), collect(T, u), Int(choice), T(dt))
    end
end

n_time(trial::Trial) = size(trial.spikes, 1)
n_neurons(trial::Trial) = size(trial.spikes, 2)

"""
    particle_filter(rng, model, trial; N=512, resample_threshold=0.5)

Run a bootstrap particle filter for a single `Trial` under `model`.

Returns the log marginal likelihood estimate: log p(spikes, RT, choice | model).

The soft boundary hazard contributes at every timestep; at the final bin
the stopping event and choice are additionally scored. The accumulator
continues running regardless of the hazard (soft boundary — particles never die).

Resampling uses systematic resampling when ESS < `resample_threshold * N`.
"""
function particle_filter(
    rng::AbstractRNG,
    model::NeuralDDM,
    trial::Trial;
    N::Integer = 512,
    resample_threshold::Real = 0.5,
)
    T_bins = n_time(trial)
    log_ml = 0.0                       # accumulated log marginal likelihood
    particles = [init_sample(rng, model) for _ = 1:N]
    # log_w holds *normalized* log weights (∑ exp = 1), initialized uniform.
    logN = log(N)
    log_w = fill(-logN, N)

    for t = 1:T_bins
        u_t = trial.u[t]
        y_t = @view trial.spikes[t, :]
        stopped = (t == T_bins)

        # Propagate each particle through the transition
        for i = 1:N
            particles[i] = transition_sample(rng, model, particles[i], u_t, trial.dt)
        end

        # Weight update: observation + soft-boundary stop/continue
        for i = 1:N
            x_i = particles[i]
            log_w[i] += obs_logpdf(model, y_t, x_i, trial.dt)
            log_w[i] += stop_logpdf(model, x_i, stopped, model.state.α)
            if stopped
                log_w[i] += choice_logpdf(model, x_i, trial.choice)
            end
        end

        # Incremental marginal likelihood: log ∑_i W_{t-1}^i · α_t^i. The carried
        # weights are already normalized, so this is just logsumexp with no
        # -log(N) term; subtracting log(N) here too (the previous behavior) is
        # only correct when resampling every step, and over-counts otherwise.
        lse = logsumexp(log_w)
        log_ml += lse
        log_w .-= lse               # renormalize

        # Systematic resampling when ESS drops below threshold
        ess = exp(-logsumexp(2 .* log_w))
        if ess < resample_threshold * N
            indices = _systematic_resample(rng, log_w, N)
            particles = particles[indices]
            fill!(log_w, -logN)     # reset to uniform *normalized* weights
        end
    end

    return log_ml
end

# Systematic resampling given log normalized weights; returns N indices.
function _systematic_resample(rng::AbstractRNG, log_w::Vector{Float64}, N::Integer)
    w = exp.(log_w)
    w ./= sum(w)                       # ensure exact normalization
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

"""
    log_marginal_likelihood(rng, model, trials; N=512, resample_threshold=0.5)

Sum the per-trial log marginal likelihoods across all `trials`.
Trials are processed in parallel across threads when available.
"""
function log_marginal_likelihood(
    rng::AbstractRNG,
    model::NeuralDDM,
    trials::Vector{<:Trial};
    N::Integer = 512,
    resample_threshold::Real = 0.5,
)
    n_trials = length(trials)
    lmls = Vector{Float64}(undef, n_trials)
    seeds = rand(rng, UInt64, n_trials)
    @batch for k = 1:n_trials
        trial_rng = copy(rng)          # type-preserving; copy(TaskLocalRNG) snapshots to Xoshiro
        Random.seed!(trial_rng, seeds[k])
        lmls[k] = particle_filter(
            trial_rng,
            model,
            trials[k];
            N = N,
            resample_threshold = resample_threshold,
        )
    end
    return sum(lmls)
end

# Nelder-Mead fit on the (stochastic, CRN) particle-filter marginal likelihood.
# Gradient-free; optimizes raw parameters, so it can wander into σ²<0 etc. Kept
# as the `method=:neldermead` option of `fit!`; see NeuralDDMFit.jl for the
# default gradient-based (`:lbfgs`) path.
function _fit_neldermead!(
    model::NeuralDDM,
    trials::Vector{<:Trial};
    N::Integer = 512,
    resample_threshold::Real = 0.5,
    rng::AbstractRNG = Random.default_rng(),
    optim_options::Optim.Options = Optim.Options(show_trace = true, iterations = 500),
)
    # Pack parameters into a vector and back
    θ0, pack! = _make_param_io(model)

    obj =
        θ -> begin
            pack!(model, θ)
            -log_marginal_likelihood(
                rng,
                model,
                trials;
                N = N,
                resample_threshold = resample_threshold,
            )
        end

    result = optimize(obj, θ0, NelderMead(), optim_options)
    pack!(model, Optim.minimizer(result))
    return model, result
end

# Parameter packing for LeakyAccumulatorModel + LinearPoissonObservationModel
function _make_param_io(
    model::NeuralDDM{<:LeakyAccumulatorModel,<:LinearPoissonObservationModel},
)
    s = model.state
    o = model.obs
    θ0 = vcat([s.B, s.v, s.λ, s.σ², s.α, s.γ], o.b, o.w)

    function pack!(m::NeuralDDM, θ::AbstractVector)
        m.state.B = θ[1]
        m.state.v = θ[2]
        m.state.λ = θ[3]
        m.state.σ² = θ[4]
        m.state.α = θ[5]
        m.state.γ = θ[6]
        N = length(m.obs.b)
        m.obs.b .= θ[7:(6+N)]
        m.obs.w .= θ[(7+N):(6+2N)]
    end

    return θ0, pack!
end

# Parameter packing for LeakyAccumulatorModel + BasisPoissonObservationModel
function _make_param_io(
    model::NeuralDDM{<:LeakyAccumulatorModel,<:BasisPoissonObservationModel},
)
    s = model.state
    o = model.obs
    θ0 = vcat([s.B, s.v, s.λ, s.σ², s.α, s.γ], vec(o.β))

    function pack!(m::NeuralDDM, θ::AbstractVector)
        m.state.B = θ[1]
        m.state.v = θ[2]
        m.state.λ = θ[3]
        m.state.σ² = θ[4]
        m.state.α = θ[5]
        m.state.γ = θ[6]
        KN = length(m.obs.β)
        m.obs.β .= reshape(θ[7:(6+KN)], size(m.obs.β))
    end

    return θ0, pack!
end

