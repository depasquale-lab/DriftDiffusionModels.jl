#=
Gradient-based fitting for the FPTDDM via a differentiable particle filter.

The estimator is a GUIDED particle filter run under common random numbers:
  * The transition is proposed from the Gaussian transition TRUNCATED to the open
    interval (0, B) — i.e. particles are never allowed to overshoot the absorbing
    boundary and die. The truncation is sampled by the inverse-CDF map from
    pre-drawn uniforms, which is a smooth (differentiable) function of the model
    parameters. The survival probability mass (the truncation normalizer) and the
    Brownian-bridge within-bin no-crossing factor are folded in as smooth weights.
  * The final ("absorption") bin uses a free Gaussian proposal and weights each
    particle by the probability it crossed the OBSERVED boundary during that bin.
  * There is NO resampling. Resampling (hard/systematic) makes the marginal
    likelihood a discontinuous function of the parameters (index flips) AND biases
    the reparameterization gradient (the stop-gradient through the resampling step
    drops a term); both wreck gradient-based optimization. The guided proposal
    keeps the effective sample size high enough that pure importance sampling is a
    good estimator without resampling — at the cost of needing more particles for
    very long trials.

The result is a marginal-likelihood estimate that is a SMOOTH, deterministic (for
fixed noise) function of the parameters, so ForwardDiff yields an accurate,
low-bias gradient (verified to match finite differences to ~1e-7). `fit!` then
maximizes it with L-BFGS. Parameters `B` (log-space) and `a₀` (logit) are fit in
an unconstrained reparameterization; `σ²` (the latent scale anchor) and `τ` (which
does not enter the particle-filter likelihood) are held fixed.
=#

const _INVSQRT2 = 1 / sqrt(2)
_Φ(z) = erfc(-z * _INVSQRT2) / 2                 # standard normal CDF
_Φinv(p) = -sqrt(2) * erfcinv(2 * p)             # standard normal quantile

# Pre-draw the randomness one filter pass consumes: uniforms for the truncated
# survival-bin proposals and standard normals for the free absorption-bin
# proposal. The start point is a deterministic point mass at a₀·B (no init noise).
function _draw_pf_noise(rng::AbstractRNG, N::Integer, n_bins::Integer)
    εU = rand(rng, n_bins, N)
    εN = randn(rng, N)
    return (εU, εN)
end

"""
    _fpt_loglik_guided(model, n_bins, choice, dt, εU, εN; u, spikes)

Deterministic (given the pre-drawn noise) estimate of
`log p(choice, absorbed in bin n_bins [, spikes] | model)` for a single trial,
using the guided/truncated-proposal particle filter with no resampling. Type-generic
in the model parameters so ForwardDiff can differentiate it.
"""
function _fpt_loglik_guided(
    model::FPTDDM,
    n_bins::Integer,
    choice::Integer,
    dt::Real,
    εU::AbstractMatrix,
    εN::AbstractVector;
    u::Union{Nothing,AbstractVector{<:Real}} = nothing,
    spikes::Union{Nothing,AbstractMatrix{<:Integer}} = nothing,
)
    B = model.B
    v = model.v
    λ = model.λ
    varstep = model.σ² * dt
    sσ = sqrt(varstep)
    Np = size(εU, 2)

    P = promote_type(
        typeof(B),
        typeof(v),
        typeof(λ),
        typeof(model.σ²),
        typeof(model.a₀),
        eltype(εU),
        typeof(float(dt)),
    )

    particles = fill(P(model.a₀ * B), Np)
    logN = log(Np)
    log_w = fill(P(-logN), Np)
    log_ml = zero(P)

    lo = P(1e-12)
    hi = P(1 - 1e-12)

    for t = 1:n_bins
        u_t = u === nothing ? one(P) : u[t]
        absorbing = (t == n_bins)

        for i = 1:Np
            x0 = particles[i]
            μ = x0 + (-λ * x0 + v * u_t) * dt

            if absorbing
                # Free proposal; weight by prob. of crossing the OBSERVED boundary.
                x1 = μ + sσ * εN[i]
                particles[i] = x1
                q =
                    choice == 1 ? (x1 >= B ? one(P) : exp(-2 * (B - x0) * (B - x1) / varstep)) :
                    (x1 <= 0 ? one(P) : exp(-2 * x0 * x1 / varstep))
                log_w[i] += q > 0 ? log(q) : P(-Inf)
            else
                # Proposal = transition truncated to (0, B), sampled by inverse CDF.
                α = (0 - μ) / sσ
                β = (B - μ) / sσ
                Φα = _Φ(α)
                Z = _Φ(β) - Φα
                Uc = clamp(Φα + εU[t, i] * Z, lo, hi)
                x1 = μ + sσ * _Φinv(Uc)
                particles[i] = x1
                # Importance weight = truncation mass Z × Brownian-bridge no-crossing.
                bridge =
                    (1 - exp(-2 * (B - x0) * (B - x1) / varstep)) *
                    (1 - exp(-2 * x0 * x1 / varstep))
                log_w[i] +=
                    (Z > 0 ? log(Z) : P(-Inf)) + (bridge > 0 ? log(bridge) : P(-Inf))
            end

            if spikes !== nothing
                log_w[i] += obs_logpdf(model, @view(spikes[t, :]), particles[i], dt)
            end
        end

        lse = logsumexp(log_w)
        isfinite(lse) || return oftype(log_ml, -Inf)   # all particles died
        log_ml += lse
        log_w = log_w .- lse
    end

    return log_ml
end

# -----------------------------------------------------------------------------
# Unconstrained reparameterization + model reconstruction from a θ vector
# -----------------------------------------------------------------------------
# Layout: θ = [log B, v, logit a₀, λ, <obs params>]. B is fit in log-space and a₀
# through a logit so L-BFGS optimizes unconstrained while the model stays in its
# valid domain (B > 0, a₀ ∈ (0,1)). σ² and τ are held fixed: σ² = 1 anchors the
# latent scale, and τ does not enter the particle-filter likelihood (it only maps
# RT = τ + n_bins·dt), so it is not identifiable here.

_logit(p::Real) = log(p / (1 - p))

_state_θ0(m::FPTDDM) = [log(m.B), m.v, _logit(m.a₀), m.λ]

function _state_from_θ(θ::AbstractVector, template::FPTDDM)
    T = eltype(θ)
    return (B = exp(θ[1]), v = θ[2], a₀ = logistic(θ[3]), λ = θ[4],
        σ² = T(template.σ²), τ = T(template.τ))
end

# --- observation-model packing (dispatched on obs type) ---------------------
_obs_θ0(o::LinearPoissonObservationModel) = vcat(o.b, o.w)
_obs_θ0(o::BasisPoissonObservationModel) = vec(o.β)

function _obs_from_θ(θobs::AbstractVector, template::LinearPoissonObservationModel)
    T = eltype(θobs)
    N = length(template.b)
    return LinearPoissonObservationModel{T}(θobs[1:N], θobs[(N+1):(2N)])
end

function _obs_from_θ(θobs::AbstractVector, template::BasisPoissonObservationModel)
    T = eltype(θobs)
    β = reshape(collect(T, θobs), size(template.β))
    return BasisPoissonObservationModel{T}(
        β,
        collect(T, template.centers),
        collect(T, template.widths),
        T(template.ρ),
    )
end

_unconstrained_θ0(m::FPTDDM) = vcat(_state_θ0(m), _obs_θ0(m.obs))

function _model_from_unconstrained(θ::AbstractVector, template::FPTDDM)
    s = _state_from_θ(θ, template)
    obs = _obs_from_θ(@view(θ[5:end]), template.obs)
    T = eltype(θ)
    return FPTDDM{typeof(obs),T}(s.v, s.B, s.a₀, s.λ, s.σ², s.τ, obs)
end

# Copy fitted (Float64) parameters back into the in-place model.
function _copy_params!(dst::FPTDDM, src::FPTDDM)
    dst.v = src.v
    dst.B = src.B
    dst.a₀ = src.a₀
    dst.λ = src.λ
    dst.σ² = src.σ²
    dst.τ = src.τ
    _copy_obs!(dst.obs, src.obs)
    return dst
end

_copy_obs!(d::LinearPoissonObservationModel, s::LinearPoissonObservationModel) =
    (d.b .= s.b; d.w .= s.w; d)
_copy_obs!(d::BasisPoissonObservationModel, s::BasisPoissonObservationModel) =
    (d.β .= s.β; d)

# -----------------------------------------------------------------------------
# fit!
# -----------------------------------------------------------------------------

"""
    fit!(model, trials; N=2000, rng=Random.default_rng(), optim_options)

Fit the `FPTDDM` parameters to `trials` by maximizing the guided particle-filter
marginal likelihood with L-BFGS (ForwardDiff gradients through the differentiable
filter under common random numbers). Modifies `model` in place and returns
`(model, result)`.

Fit parameters: `B` (log-space), `v`, `a₀` (logit), `λ`, and the observation-model
parameters. `σ²` (scale anchor) and `τ` are held fixed. Increase `N` for datasets
with long trials (importance-sampling weight variance grows with trial length).
"""
function fit!(
    model::FPTDDM,
    trials::AbstractVector{<:Trial};
    N::Integer = 2000,
    rng::AbstractRNG = Random.default_rng(),
    optim_options::Optim.Options = Optim.Options(show_trace = false, iterations = 300),
)
    θ0 = _unconstrained_θ0(model)

    # Common random numbers: draw once, reuse for every objective/gradient eval so
    # the objective is a deterministic, smooth function of θ.
    noise = [_draw_pf_noise(rng, N, n_time(tr)) for tr in trials]

    obj =
        θ -> begin
            m = _model_from_unconstrained(θ, model)
            total = zero(eltype(θ))
            for k in eachindex(trials)
                εU, εN = noise[k]
                total += _fpt_loglik_guided(
                    m,
                    n_time(trials[k]),
                    trials[k].choice,
                    trials[k].dt,
                    εU,
                    εN;
                    u = trials[k].u,
                    spikes = trials[k].spikes,
                )
            end
            return -total
        end

    result = optimize(obj, θ0, LBFGS(), optim_options; autodiff = AutoForwardDiff())
    _copy_params!(model, _model_from_unconstrained(Optim.minimizer(result), model))
    return model, result
end
