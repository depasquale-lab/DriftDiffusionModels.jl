abstract type AbstractStateModel end
abstract type AbstractObservationModel end

mutable struct LeakyAccumulatorModel{T<:Real} <: AbstractStateModel
    B::T # boundary
    v::T # drift rate
    λ::T # leak
    σ²::T # diffusion coefficient
    μ₀::T # initial mean
    Σ₀::T # initial variance
    τ::T # non-decision time
end

function LeakyAccumulatorModel(;
    B::Real=1.0,
    v::Real=1.0,
    λ::Real=0.0,
    σ²::Real=1.0,
    μ₀::Real=0.0,
    Σ₀::Real=0.0,
    τ::Real=1e-1,
)
    T = promote_type(typeof(B), typeof(v), typeof(λ), typeof(σ²),
                     typeof(μ₀), typeof(Σ₀), typeof(τ))
    return LeakyAccumulatorModel{T}(T(B), T(v), T(λ), T(σ²), T(μ₀), T(Σ₀), T(τ))
end

mutable struct LinearPoissonObservationModel{T<:Real} <: AbstractObservationModel
    b::Vector{T} # intercepts, one per neuron
    w::Vector{T} # weights, one per neuron
end

function LinearPoissonObservationModel(N::Integer; T::Type{<:Real}=Float64)
    return LinearPoissonObservationModel{T}(zeros(T, N), ones(T, N))
end

function LinearPoissonObservationModel(; b::AbstractVector{<:Real}, w::AbstractVector{<:Real})
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
    K::Integer, N::Integer;
    centers::AbstractVector{<:Real}=K == 1 ? [0.0] : collect(range(-1.0, 1.0; length=K)),
    widths::AbstractVector{<:Real}=fill(2.0 / max(K - 1, 1), K),
    ρ::Real=1e-3,
    T::Type{<:Real}=Float64,
)
    length(centers) == K || throw(ArgumentError("length(centers) must equal K"))
    length(widths)  == K || throw(ArgumentError("length(widths) must equal K"))
    return BasisPoissonObservationModel{T}(
        zeros(T, K, N), collect(T, centers), collect(T, widths), T(ρ),
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
    inducing_points::AbstractVector{<:Real}=range(-1.0, 1.0; length=10),
    lengthscale::Union{Real,AbstractVector{<:Real}}=1.0,
    amplitude::Union{Real,AbstractVector{<:Real}}=1.0,
    mean::Union{Real,AbstractVector{<:Real}}=0.0,
    T::Type{<:Real}=Float64,
)
    _expand(x) = x isa AbstractVector ? collect(T, x) : fill(T(x), N)
    ls = _expand(lengthscale)
    amp = _expand(amplitude)
    μ = _expand(mean)
    length(ls)  == N || throw(ArgumentError("lengthscale length must equal N"))
    length(amp) == N || throw(ArgumentError("amplitude length must equal N"))
    length(μ)   == N || throw(ArgumentError("mean length must equal N"))
    return GPPoissonObservationModel{T}(collect(T, inducing_points), ls, amp, μ)
end

mutable struct NeuralDDM{
    S<:AbstractStateModel,
    O<:AbstractObservationModel,
}
    state::S
    obs::O
end

function NeuralDDM(;
    state::AbstractStateModel=LeakyAccumulatorModel(),
    obs::AbstractObservationModel=LinearPoissonObservationModel(1),
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
    return -log(sqrt(model.Σ₀)) - log(2π) / 2 - (x - model.μ₀)^2 / (2 * model.Σ₀)
end

function transition_sample(rng::AbstractRNG, model::NeuralDDM, x_prev::Real, u::Real, dt::Real)
    return transition_sample(rng, model.state, x_prev, u, dt)
end

function transition_sample(rng::AbstractRNG, model::LeakyAccumulatorModel, x_prev::Real, u::Real, dt::Real)
    μ = x_prev + (-model.λ * x_prev + model.v * u) * dt
    σ = sqrt(model.σ² * dt)
    return μ + σ * randn(rng)
end

function transition_logpdf(model::NeuralDDM, x::Real, x_prev::Real, u::Real, dt::Real)
    return transition_logpdf(model.state, x, x_prev, u, dt)
end

function transition_logpdf(model::LeakyAccumulatorModel, x::Real, x_prev::Real, u::Real, dt::Real)
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
    for n in 1:N
        μ = softplus(_drive(obs, x, n)) * dt
        counts[n] = rand(rng, Poisson(μ))
    end
    return counts
end

function obs_logpdf(obs::LinearPoissonObservationModel, y::AbstractVector{<:Integer}, x::Real, dt::Real)
    length(y) == length(obs.b) || throw(ArgumentError("y length must match number of neurons"))
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
    for k in 1:K
        φ[k] = exp(-(x - obs.centers[k])^2 / (2 * obs.widths[k]^2))
    end
    return φ
end

function _drive(obs::BasisPoissonObservationModel, n::Integer, φ::AbstractVector)
    η = zero(promote_type(eltype(obs.β), eltype(φ)))
    @inbounds for k in eachindex(φ)
        η += obs.β[k, n] * φ[k]
    end
    return η
end

function obs_sample(rng::AbstractRNG, obs::BasisPoissonObservationModel, x::Real, dt::Real)
    N = size(obs.β, 2)
    φ = _basis_features(obs, x)
    counts = Vector{Int}(undef, N)
    for n in 1:N
        μ = softplus(_drive(obs, n, φ)) * dt
        counts[n] = rand(rng, Poisson(μ))
    end
    return counts
end

function obs_logpdf(obs::BasisPoissonObservationModel, y::AbstractVector{<:Integer}, x::Real, dt::Real)
    N = size(obs.β, 2)
    length(y) == N || throw(ArgumentError("y length must match number of neurons"))
    φ = _basis_features(obs, x)
    T = promote_type(eltype(obs.β), eltype(φ), typeof(dt))
    ll = zero(T)
    for n in 1:N
        μ = softplus(_drive(obs, n, φ)) * dt
        ll += (y[n] == 0 ? zero(μ) : y[n] * log(μ)) - μ - loggamma(y[n] + 1)
    end
    return ll
end

# GP: not implemented — struct needs variational q(u) fields before a likelihood is defined
function obs_sample(::AbstractRNG, ::GPPoissonObservationModel, ::Real, ::Real)
    error("obs_sample for GPPoissonObservationModel is not implemented yet — the struct needs inducing values q(u) before sampling can be defined.")
end

function obs_logpdf(::GPPoissonObservationModel, ::AbstractVector{<:Integer}, ::Real, ::Real)
    error("obs_logpdf for GPPoissonObservationModel is not implemented yet — the struct needs inducing values q(u) before a likelihood is defined.")
end

