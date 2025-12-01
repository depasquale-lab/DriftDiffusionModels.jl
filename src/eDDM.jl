function transform_params(u::AbstractVector{<:Real})
    # transform parameters to constrained space
    uB, uτ, uv, ua₀ = u
    B = exp(uB)               # B > 0
    τ = exp(uτ)               # τ > 0
    v = exp(uv)               # v > 0
    a₀ = logistic(ua₀)       # a₀ ∈ (0, 1)
    return B, v, a₀, τ
end

struct DDMHyper{T<:Real}
    m::Vector{T}      # means (length = 4)
    logσ::Vector{T}      # stddevs (length = 4)
end

function prior_logpdf(u::AbstractVector{<:Real}, hyper::DDMHyper)
    m, logσ = hyper.m, hyper.logσ
    σ0 = exp.(logσ)
    z = (u .- m) ./ σ0
    logpdf = -0.5 * sum(z .^ 2) - sum(logσ) - (length(u) / 2) * log(2π)
    return logpdf
end

struct TrialVIParams{T<:Real}
    μ::Vector{T}      # length 4
    logσ::Vector{T}   # length 4, σ = exp.(logσ)
end

function sample_u(q::TrialVIParams, rng::AbstractRNG)
    ε = randn(rng, 4)
    σ = exp.(q.logσ)
    return q.μ .+ σ .* ε
end

function kl_gaussian_diag(q::TrialVIParams, hyper::DDMHyper)
    μ, logσ = q.μ, q.logσ
    m, logσ0 = hyper.m, hyper.logσ
    σ  = exp.(logσ)
    σ0 = exp.(logσ0)

    term = (σ.^2 .+ (μ .- m).^2) ./ (σ0.^2) .- 1 .+ 2 .* logσ0 .- 2 .* logσ
    return 0.5 * sum(term)
end

function elbo_trial(q::TrialVIParams,
                    y::DDMResult,
                    hyper::DDMHyper;
                    K::Int=3,
                    rng::AbstractRNG=Random.default_rng())
    ll_acc = 0.0
    for k in 1:K
        u = sample_u(q, rng)
        B, v, a₀, τ = transform_params(u)

        ll = logdensityof(B, v, a₀, τ, y.rt, y.choice, y.s)
        ll_acc += ll
    end
    E_loglik = ll_acc / K
    kl = kl_gaussian_diag(q, hyper)
    return E_loglik - kl
end

function pack_q(q::TrialVIParams)
    return vcat(q.μ, q.logσ)
end

function unpack_q(θ::AbstractVector{T}) where T<:Real
    @assert length(θ) == 8
    μ    = θ[1:4]
    logσ = θ[5:8]
    return TrialVIParams{T}(collect(μ), collect(logσ))
end

function optimize_trial_vi(q0, y, hyper; K=3, rng=Random.default_rng())
    θ0 = pack_q(q0)
    eps = [randn(rng, 4) for _ in 1:K]  # fixed eps for this trial

    f(θ) = begin
        q = unpack_q(θ)
        # deterministic ELBO estimate given fixed eps
        ll_acc = 0.0
        for ε in eps
            σ = exp.(q.logσ)
            u = q.μ .+ σ .* ε
            B, v, a₀, τ = transform_params(u)
            ll_acc += logdensityof(B, v, a₀, τ, y.rt, y.choice, y.s)
        end
        E_loglik = ll_acc / K
        elbo = E_loglik - kl_gaussian_diag(q, hyper)
        elbo_val = -elbo
        isfinite(elbo_val) ? elbo_val : 1e10
    end

    res = optimize(f, θ0, BFGS(linesearch=Optim.LineSearches.BackTracking()); autodiff = :forward)

    # Check if optimization succeeded
    if !Optim.converged(res) || any(isnan, Optim.minimizer(res))
        # If optimization failed, return the initial parameters
        @warn "Trial optimization failed, keeping initial parameters"
        return q0
    end

    θ̂ = Optim.minimizer(res)
    return unpack_q(θ̂)
end

function update_hyper_from_qs(qs::Vector{<:TrialVIParams})
    d = length(qs[1].μ)

    valid_qs = [q for q in qs if !any(isnan, q.μ) && !any(isnan, q.logσ)]
    N_valid = length(valid_qs)

    if N_valid == 0
        @warn "No valid trials for hyperparameter update, using default values"
        m = [log(2.0), log(0.1), log(1.0), 0.0]
        logσ0 = log.([0.5, 0.2, 0.5, 0.2])
        return DDMHyper(m, logσ0)
    end

    μ_mat  = zeros(d, N_valid)
    σ2_mat = zeros(d, N_valid)

    for (j, q) in enumerate(valid_qs)
        μ_mat[:, j] .= q.μ
        σ = exp.(q.logσ)
        σ2_mat[:, j] .= σ.^2
    end

    m = vec(mean(μ_mat; dims=2))
    σ0_sq = vec(mean(σ2_mat .+ (μ_mat .- m).^2; dims=2))
    logσ0 = 0.5 .* log.(σ0_sq .+ 1e-6)

    if any(isnan, m) || any(isnan, logσ0)
        @warn "NaN detected in hyperparameter update, using default values"
        m = [log(2.0), log(0.1), log(1.0), 0.0]
        logσ0 = log.([0.5, 0.2, 0.5, 0.2])
    end

    return DDMHyper(collect(m), collect(logσ0))
end

function fit_vi_gaussian(data::Vector{DDMResult};
                         n_iter::Int=10,
                         K::Int=3,
                         rng::AbstractRNG=Random.default_rng(),
                         verbose::Bool=true,
                         init_from_data::Bool=true)
    N = length(data)
    d = 4

    # 1. Initialize hyper around something reasonable
    if init_from_data && N > 0
        # Data-driven initialization
        rts = [d.rt for d in data]
        choices = [d.choice for d in data]

        # Estimate τ as a quantile of RT (e.g., 10th percentile)
        τ_init = quantile(rts, 0.1)
        # Estimate B from RT variance
        B_init = std(rts) * 2.0  # rough heuristic
        B_init = clamp(B_init, 0.5, 5.0)  # keep reasonable
        # Estimate v from accuracy
        accuracy = mean(choices .== 1)
        v_init = abs(log((accuracy + 0.01) / (1 - accuracy + 0.01)))  # logit-like transform
        v_init = clamp(v_init, 0.3, 3.0)
        # Estimate a₀ from choice bias
        a₀_init = 0.5  # start unbiased

        m0 = [log(B_init), log(τ_init), log(v_init), 0.0]
        if verbose
            println("Data-driven init: B=$(round(B_init, digits=3)), τ=$(round(τ_init, digits=3)), v=$(round(v_init, digits=3))")
        end
    else
        m0 = [log(2.0), log(0.1), log(1.0), 0.0]
    end

    # Use wider initial variance for more flexibility
    logσ0 = log.([0.7, 0.5, 0.7, 0.5])  # increased from [0.5, 0.5, 0.5, 0.2]
    hyper = DDMHyper(m0, logσ0)

    # 2. Initialize q_i with small random perturbations to break symmetry
    qs = Vector{TrialVIParams{Float64}}(undef, N)
    for i in 1:N
        μ_init = m0 .+ randn(rng, 4) .* 0.1  # small random perturbation
        qs[i] = TrialVIParams{Float64}(μ_init, copy(logσ0))
    end

    # Track ELBO history
    elbo_history = Float64[]

    for iter in 1:n_iter
        if verbose
            println("VI iter $iter")
        end

        # E-step: update each q_i
        failed_count = 0
        for i in 1:N
            q_new = optimize_trial_vi(qs[i], data[i], hyper; K=K, rng=rng)
            # Check if optimization actually improved (if not, q_new == qs[i])
            if q_new === qs[i]
                failed_count += 1
            end
            qs[i] = q_new
        end

        if verbose && failed_count > 0
            println("  $failed_count trials failed to converge")
        end

        # M-step: update hyper from q_i's
        hyper = update_hyper_from_qs(qs)

        # Compute and store total ELBO
        elbo = total_elbo(qs, data, hyper; K=K, rng=rng)
        push!(elbo_history, elbo)

        if verbose
            println("  ELBO: $elbo")
        end
    end

    return hyper, qs, elbo_history
end

function total_elbo(qs::Vector{<:TrialVIParams},
                    data::Vector{DDMResult},
                    hyper::DDMHyper;
                    K::Int=3,
                    rng::AbstractRNG=Random.default_rng())
    total = 0.0
    @inbounds for i in eachindex(data)
        total += elbo_trial(qs[i], data[i], hyper; K=K, rng=rng)
    end
    return total
end