
"""
    fit_global_ddm(data::Vector{DDMResult}; τ0 = 0.1)

Fit a single DDM to all data using MLE.

# Arguments
- `data::Vector{DDMResult}`: Vector of DDMResult observations.
- `τ0`: Initial guess for nondecision time (passed to constructor).

# Returns
- `model::DriftDiffusionModel`: Fitted DDM model.
"""
function fit_global_ddm(data::Vector{DDMResult}; τ0 = 0.1)
    model = DriftDiffusionModel(; τ = τ0)  # all other values default
    StatsAPI.fit!(model, data)
    return model
end


"""
    init_ddm_emissions(rng::AbstractRNG,
                       global_ddm::DriftDiffusionModel,
                       K::Int)

Initialize a vector of DDM emissions for an HMM by perturbing
parameters of a global DDM, using the provided RNG.

By default, we keep `B` and `τ` close to the global values and mainly
perturb `v` and `a₀`.
"""
function init_ddm_emissions(rng::AbstractRNG,
                            global_ddm::DriftDiffusionModel,
                            K::Int)
    emissions = Vector{DriftDiffusionModel}(undef, K)

    for i in 1:K
        # Keep bound and nondecision time near the global value, but ensure positivity
        B_perturbed = max(global_ddm.B * (1.0 + 0.05 * randn(rng)), 1e-3)
        τ_perturbed = max(global_ddm.τ * (1.0 + 0.05 * randn(rng)), 1e-3)

        # Perturb drift and starting point a bit more
        v_perturbed  = global_ddm.v * (1.0 + 0.1 * randn(rng))
        a₀_perturbed = clamp(global_ddm.a₀ * (1.0 + 0.1 * randn(rng)), 0.1, 0.9)

        emissions[i] = DriftDiffusionModel(B_perturbed, v_perturbed, a₀_perturbed, τ_perturbed)
    end

    return emissions
end

"""
    init_ddm_emissions(global_ddm::DriftDiffusionModel, K::Int)

RNG-defaulting convenience wrapper around `init_ddm_emissions(rng, ...)`.
"""
init_ddm_emissions(global_ddm::DriftDiffusionModel, K::Int) =
    init_ddm_emissions(Random.default_rng(), global_ddm, K)


"""
    init_pi(K::Int)

Initialize deterministic, uniform initial state distribution.
"""
function init_pi(K::Int)
    fill(1.0 / K, K)
end


"""
    init_trans(K::Int; stay_prob = 0.95)

Initialize a sticky transition matrix.
"""
function init_trans(K::Int; stay_prob = 0.95)
    @assert 0.0 < stay_prob ≤ 1.0 "stay_prob must be in (0, 1]."

    if K == 1
        return ones(1, 1)  # degenerate 1-state HMM
    end

    A = fill((1 - stay_prob) / (K - 1), K, K)
    for k in 1:K
        A[k, k] = stay_prob
    end
    return A
end


"""
    init_dirichlet_priors(K::Int;
                          α_sticky   = 10.0,
                          α_offdiag  = 1.0,
                          α_init_val = 2.0)

Initialize Dirichlet hyper-parameters for HMM priors.

Defaults encourage staying in the same state (sticky prior) and
a mildly concentrated, roughly uniform initial distribution.
"""
function init_dirichlet_priors(K::Int;
                               α_sticky   = 10.0,
                               α_offdiag  = 1.0,
                               α_init_val = 2.0)
    α_trans = fill(α_offdiag, K, K)
    for k in 1:K
        α_trans[k, k] = α_sticky
    end
    α_init = fill(α_init_val, K)
    return α_trans, α_init
end


"""
    init_hmm_ddm(rng::AbstractRNG,
                 data::Vector{DDMResult}, K::Int;
                 stay_prob   = 0.95,
                 α_sticky    = 10.0,
                 α_offdiag   = 1.0,
                 α_init_val  = 2.0,
                 τ0          = 0.1)

Initialize a PriorHMM with DDM emissions and Dirichlet priors,
using the provided RNG for stochastic components.
"""
function init_hmm_ddm(rng::AbstractRNG,
                      data::Vector{DDMResult}, K::Int;
                      stay_prob   = 0.95,
                      α_sticky    = 2.0,
                      α_offdiag   = 1.0,
                      α_init_val  = 1.0,
                      τ0          = 0.1)

    # 1. Fit global DDM
    global_ddm = fit_global_ddm(data; τ0 = τ0)

    # 2. Emissions: perturbed copies of the global DDM
    emissions  = init_ddm_emissions(rng, global_ddm, K)

    # 3. Initial state distribution and transitions
    init_dist  = init_pi(K)
    trans_mat  = init_trans(K; stay_prob = stay_prob)

    # 4. Dirichlet priors on init and transitions
    α_trans, α_init = init_dirichlet_priors(K;
                                            α_sticky   = α_sticky,
                                            α_offdiag  = α_offdiag,
                                            α_init_val = α_init_val)

    hmm = PriorHMM(init_dist, trans_mat, emissions;
                   α_trans = α_trans, α_init = α_init)

    # Optional sanity check (comment out if not available)
    # @assert HiddenMarkovModels.valid_hmm(hmm)

    return hmm
end


"""
    init_hmm_ddm(data::Vector{DDMResult}, K::Int; kwargs...)

Convenience wrapper that uses `Random.default_rng()`.

See `init_hmm_ddm(rng, data, K; ...)` for keyword arguments.
"""
function init_hmm_ddm(data::Vector{DDMResult}, K::Int; kwargs...)
    init_hmm_ddm(Random.default_rng(), data, K; kwargs...)
end

logistic(x) = 1.0 / (1.0 + exp(-x))

# Numerically stable softplus: log(1 + exp(x))
softplus(x) = x > 0 ? x + log1p(exp(-x)) : log1p(exp(x))