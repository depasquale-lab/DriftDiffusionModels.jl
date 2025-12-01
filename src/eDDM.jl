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

function optimize_trial_vi(q0::TrialVIParams,
                           y::DDMResult,
                           hyper::DDMHyper;
                           K::Int=3,
                           rng::AbstractRNG=Random.default_rng())
    θ0 = pack_q(q0)

    f(θ) = begin
        q = unpack_q(θ)
        -elbo_trial(q, y, hyper; K=K, rng=rng)  # Optim minimizes
    end

    res = optimize(f, θ0, BFGS(); autodiff = :forward)
    θ̂ = Optim.minimizer(res)
    return unpack_q(θ̂)
end

function update_hyper_from_qs(qs::Vector{<:TrialVIParams})
    N = length(qs)
    d = length(qs[1].μ)  # should be 4

    μ_mat    = zeros(d, N)
    σ2_mat   = zeros(d, N)

    for (i, q) in enumerate(qs)
        μ_mat[:, i]  .= q.μ
        σ  = exp.(q.logσ)
        σ2_mat[:, i] .= σ.^2
    end

    m = vec(mean(μ_mat; dims=2))
    σ0_sq = vec(mean(σ2_mat .+ (μ_mat .- m).^2; dims=2))
    logσ0 = 0.5 .* log.(σ0_sq .+ 1e-8)

    return DDMHyper(collect(m), collect(logσ0))
end

function fit_vi_gaussian(data::Vector{DDMResult};
                         n_iter::Int=10,
                         K::Int=3,
                         rng::AbstractRNG=Random.default_rng())
    N = length(data)
    d = 4

    # 1. Initialize hyper around something reasonable
    m0    = [log(5.0), log(0.3), log(1.0), 0.0]       # logB, logτ, logv, logit(a₀)
    logσ0 = log.([0.2, 0.2, 0.2, 0.2])                # prior stds ~ 0.2 in u-space
    hyper = DDMHyper(m0, logσ0)

    # 2. Initialize q_i to the prior
    qs = [TrialVIParams{Float64}(copy(m0), copy(logσ0)) for _ in 1:N]

    for iter in 1:n_iter
        println("VI iter $iter")

        # E-step: update each q_i
        for i in 1:N
            qs[i] = optimize_trial_vi(qs[i], data[i], hyper; K=K, rng=rng)
        end

        # M-step: update hyper from q_i's
        hyper = update_hyper_from_qs(qs)
    end

    return hyper, qs
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