# =============================================================================
# Simulation, differentiable particle filter, and gradient-based fitting for the
# NeuralDDM. See NeuralDDM.jl for the model definitions and the (stochastic)
# bootstrap particle filter / Nelder-Mead path.
#
# The fitting approach here is "PS + L-BFGS": we differentiate the particle
# filter's own log marginal likelihood under *fixed common random numbers*
# (the transition noise is pre-drawn and reparameterized, the resampling
# schedule is fixed) so that the realized log-ML estimate is a deterministic,
# (a.e.) differentiable function of the parameters. ForwardDiff then yields a
# gradient that is *exactly* the gradient of the objective L-BFGS evaluates —
# this is the Poyiadjis O(N) score estimator (the derivative of the filter
# log-likelihood is the smoothed score). Resampling indices are held fixed given
# the realized weights, which makes the gradient the standard slightly-biased PF
# score; in practice this is fine for point estimation and behaves well under
# common random numbers.
# =============================================================================

# Number of neurons implied by an observation model.
_obs_n_neurons(o::LinearPoissonObservationModel) = length(o.b)
_obs_n_neurons(o::BasisPoissonObservationModel) = size(o.β, 2)
_obs_n_neurons(o::GPPoissonObservationModel) = length(o.mean)
_obs_n_neurons(m::NeuralDDM) = _obs_n_neurons(m.obs)

"""
    simulate_trial(rng, model, u, dt; max_bins=length(u))

Generate a single `Trial` from `model` under stimulus trace `u` (values in
`{-1, 0, +1}`) with bin width `dt`.

The generative order within each bin matches the likelihood scored by
[`particle_filter`](@ref): the accumulator is propagated, spikes are emitted
from the new state, then a stop is sampled with probability `hazard(x)`. On the
stopping bin the choice is drawn from `logistic(γ · x)`. If no stop occurs by
`max_bins`, the trial is force-stopped at `max_bins`.

Returns a `Trial` truncated to the realized stopping bin (so `RT ≈ n_time·dt`).
"""
function simulate_trial(
    rng::AbstractRNG,
    model::NeuralDDM,
    u::AbstractVector{<:Real},
    dt::Real;
    max_bins::Integer = length(u),
)
    max_bins ≥ 1 || throw(ArgumentError("max_bins must be ≥ 1"))
    length(u) ≥ max_bins || throw(
        ArgumentError(
            "u must have at least max_bins ($(max_bins)) entries, got $(length(u))",
        ),
    )

    α = model.state.α
    γ = model.state.γ
    rows = Vector{Vector{Int}}()
    x = init_sample(rng, model)
    choice = 0
    stop_bin = max_bins

    for t = 1:max_bins
        x = transition_sample(rng, model, x, u[t], dt)
        push!(rows, obs_sample(rng, model, x, dt))
        stopped = (t == max_bins) || (rand(rng) < hazard(model, x, α))
        if stopped
            stop_bin = t
            choice = rand(rng) < logistic(γ * x) ? 1 : -1
            break
        end
    end

    spikes = reduce(vcat, (permutedims(r) for r in rows))   # stop_bin × N_neurons
    return Trial(spikes, collect(u[1:stop_bin]), choice, dt)
end

# -----------------------------------------------------------------------------
# Differentiable particle filter
# -----------------------------------------------------------------------------

# Pre-draw the randomness a single particle-filter pass consumes, so the pass
# becomes a deterministic function of the model parameters (common random
# numbers). `ε_init`/`ε_trans` are standard-normal draws reused via the
# location-scale reparameterization; `u_resample` are uniforms for systematic
# resampling.
function _draw_pf_noise(rng::AbstractRNG, N::Integer, T_bins::Integer)
    ε_init = randn(rng, N)
    ε_trans = randn(rng, T_bins, N)
    u_resample = rand(rng, T_bins)
    return (ε_init, ε_trans, u_resample)
end

# Systematic resampling from pre-drawn uniform `u0base ∈ [0,1)`, accepting
# generic (e.g. Dual) weights. Returns N integer indices computed from the
# primal weights (the index map is piecewise-constant in θ, derivative 0 a.e.).
function _systematic_resample_fixed(log_w::AbstractVector, u0base::Real)
    N = length(log_w)
    w = exp.(log_w)
    w = w ./ sum(w)
    cumw = cumsum(w)
    indices = Vector{Int}(undef, N)
    u0 = u0base / N
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
    _pf_loglik(model, trial, ε_init, ε_trans, u_resample; resample_every=1)

Deterministic (given the pre-drawn noise) estimate of `log p(spikes, RT, choice | model)`
for a single `Trial`. Type-generic in the model parameters so ForwardDiff can
differentiate it. The transition noise is reparameterized
`x_t = μ(x_{t-1}) + √(σ²·dt)·ε`, and resampling happens on a fixed schedule
(every `resample_every` bins; set `0` to disable, i.e. pure importance sampling).
"""
function _pf_loglik(
    model::NeuralDDM,
    trial::Trial,
    ε_init::AbstractVector,
    ε_trans::AbstractMatrix,
    u_resample::AbstractVector;
    resample_every::Integer = 1,
)
    s = model.state
    Tbins = n_time(trial)
    dt = trial.dt
    Np = length(ε_init)

    # Element type that can hold parameter-derivative information.
    P = promote_type(
        typeof(s.B),
        typeof(s.v),
        typeof(s.σ²),
        typeof(s.α),
        eltype(ε_init),
        typeof(float(dt)),
    )

    # Σ₀ is held fixed (not optimized). Guard the degenerate Σ₀=0 point mass:
    # sqrt(0) has an infinite derivative, and Inf·0 (the zero partial) = NaN under
    # ForwardDiff, which would poison the whole gradient even though the value is fine.
    sΣ0 = iszero(s.Σ₀) ? zero(P) : sqrt(s.Σ₀)
    particles = P[s.μ₀ + sΣ0 * ε_init[i] for i = 1:Np]
    log_w = zeros(P, Np)
    log_ml = zero(P)

    for t = 1:Tbins
        u_t = trial.u[t]
        y_t = @view trial.spikes[t, :]
        stopped = (t == Tbins)
        sσ = sqrt(s.σ² * dt)

        for i = 1:Np
            xp = particles[i]
            μ = xp + (-s.λ * xp + s.v * u_t) * dt
            particles[i] = μ + sσ * ε_trans[t, i]
        end

        for i = 1:Np
            x_i = particles[i]
            log_w[i] += obs_logpdf(model, y_t, x_i, dt)
            log_w[i] += stop_logpdf(model, x_i, stopped, s.α)
            if stopped
                log_w[i] += choice_logpdf(model, x_i, trial.choice)
            end
        end

        lse = logsumexp(log_w)
        log_ml += lse - log(Np)
        log_w .-= lse

        # Fixed-schedule resampling (never on the final bin — no propagation follows).
        if resample_every > 0 && t < Tbins && t % resample_every == 0
            idx = _systematic_resample_fixed(log_w, u_resample[t])
            particles = particles[idx]
            fill!(log_w, zero(P))
        end
    end

    return log_ml
end

# -----------------------------------------------------------------------------
# Unconstrained reparameterization + model reconstruction from a θ vector
# -----------------------------------------------------------------------------
# Layout: θ = [log B, v, λ, log σ², log α, γ, <obs params>]. B, σ², α are fit in
# log-space so L-BFGS optimizes unconstrained while the model stays in its valid
# (positive) domain. μ₀, Σ₀, τ are held fixed at their template values.

function _unconstrained_θ0(
    m::NeuralDDM{<:LeakyAccumulatorModel,<:LinearPoissonObservationModel},
)
    s, o = m.state, m.obs
    return vcat([log(s.B), s.v, s.λ, log(s.σ²), log(s.α), s.γ], o.b, o.w)
end

function _model_from_unconstrained(
    θ::AbstractVector,
    template::NeuralDDM{<:LeakyAccumulatorModel,<:LinearPoissonObservationModel},
)
    s = template.state
    T = eltype(θ)
    state = LeakyAccumulatorModel{T}(
        exp(θ[1]),
        θ[2],
        θ[3],
        exp(θ[4]),
        T(s.μ₀),
        T(s.Σ₀),
        T(s.τ),
        exp(θ[5]),
        θ[6],
    )
    N = length(template.obs.b)
    obs = LinearPoissonObservationModel{T}(
        collect(T, θ[7:(6+N)]),
        collect(T, θ[(7+N):(6+2N)]),
    )
    return NeuralDDM(state, obs)
end

function _unconstrained_θ0(
    m::NeuralDDM{<:LeakyAccumulatorModel,<:BasisPoissonObservationModel},
)
    s, o = m.state, m.obs
    return vcat([log(s.B), s.v, s.λ, log(s.σ²), log(s.α), s.γ], vec(o.β))
end

function _model_from_unconstrained(
    θ::AbstractVector,
    template::NeuralDDM{<:LeakyAccumulatorModel,<:BasisPoissonObservationModel},
)
    s, o = template.state, template.obs
    T = eltype(θ)
    state = LeakyAccumulatorModel{T}(
        exp(θ[1]),
        θ[2],
        θ[3],
        exp(θ[4]),
        T(s.μ₀),
        T(s.Σ₀),
        T(s.τ),
        exp(θ[5]),
        θ[6],
    )
    KN = length(o.β)
    β = reshape(collect(T, θ[7:(6+KN)]), size(o.β))
    obs = BasisPoissonObservationModel{T}(
        β,
        collect(T, o.centers),
        collect(T, o.widths),
        T(o.ρ),
    )
    return NeuralDDM(state, obs)
end

# Copy fitted (Float64) parameters back into the in-place model.
function _copy_params!(dst::NeuralDDM, src::NeuralDDM)
    _copy_state!(dst.state, src.state)
    _copy_obs!(dst.obs, src.obs)
    return dst
end

function _copy_state!(d::LeakyAccumulatorModel, s::LeakyAccumulatorModel)
    d.B = s.B;
    d.v = s.v;
    d.λ = s.λ;
    d.σ² = s.σ²
    d.μ₀ = s.μ₀;
    d.Σ₀ = s.Σ₀;
    d.τ = s.τ;
    d.α = s.α;
    d.γ = s.γ
    return d
end

_copy_obs!(d::LinearPoissonObservationModel, s::LinearPoissonObservationModel) =
    (d.b .= s.b; d.w .= s.w; d)
_copy_obs!(d::BasisPoissonObservationModel, s::BasisPoissonObservationModel) =
    (d.β .= s.β; d)

# -----------------------------------------------------------------------------
# fit!
# -----------------------------------------------------------------------------

"""
    fit!(model, trials; method=:lbfgs, N=512, rng=Random.default_rng(), kwargs...)

Fit the `NeuralDDM` parameters to `trials` by maximizing the particle-filter
marginal likelihood. Modifies `model` in place and returns `(model, result)`.

Methods:
- `:lbfgs` (default) — gradient-based L-BFGS on the differentiable particle
  filter under common random numbers (ForwardDiff). Parameters `B, σ², α` are
  optimized in log-space so they stay positive; `(v, λ, γ)` and the observation
  weights are unconstrained. Keyword `resample_every` (default `1`) sets the
  fixed resampling schedule (`0` disables resampling).
- `:neldermead` — gradient-free Nelder-Mead on the stochastic marginal
  likelihood (legacy; no constraint handling). Accepts `resample_threshold`.
"""
function fit!(
    model::NeuralDDM,
    trials::Vector{<:Trial};
    method::Symbol = :lbfgs,
    N::Integer = 512,
    resample_every::Integer = 1,
    resample_threshold::Real = 0.5,
    rng::AbstractRNG = Random.default_rng(),
    optim_options::Optim.Options = Optim.Options(show_trace = false, iterations = 500),
)
    if method === :lbfgs
        return _fit_lbfgs!(
            model,
            trials;
            N = N,
            resample_every = resample_every,
            rng = rng,
            optim_options = optim_options,
        )
    elseif method === :neldermead
        return _fit_neldermead!(
            model,
            trials;
            N = N,
            resample_threshold = resample_threshold,
            rng = rng,
            optim_options = optim_options,
        )
    else
        throw(ArgumentError("unknown fit method :$method (use :lbfgs or :neldermead)"))
    end
end

function _fit_lbfgs!(
    model::NeuralDDM,
    trials::Vector{<:Trial};
    N::Integer,
    resample_every::Integer,
    rng::AbstractRNG,
    optim_options::Optim.Options,
)
    θ0 = _unconstrained_θ0(model)

    # Common random numbers: draw the per-trial noise ONCE, up front, and reuse
    # it for every objective/gradient evaluation. This makes the objective a
    # deterministic, smooth function of θ (so L-BFGS line searches behave) at the
    # cost of a fixed-sample bias controllable via N.
    noise = [_draw_pf_noise(rng, N, n_time(tr)) for tr in trials]

    obj =
        θ -> begin
            m = _model_from_unconstrained(θ, model)
            total = zero(eltype(θ))
            for k in eachindex(trials)
                εi, εt, ur = noise[k]
                total += _pf_loglik(
                    m,
                    trials[k],
                    εi,
                    εt,
                    ur;
                    resample_every = resample_every,
                )
            end
            return -total
        end

    result = optimize(obj, θ0, LBFGS(), optim_options; autodiff = :forward)
    _copy_params!(model, _model_from_unconstrained(Optim.minimizer(result), model))
    return model, result
end
