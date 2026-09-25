@testset "Struct construction" begin
    m = CoherentDDM()
    @test m.B > 0
    @test m.k > 0
    @test 0 < m.a₀ < 1
    @test m.τ > 0

    m2 = CoherentDDM(B=3.0, k=1.5, a₀=0.4, τ=0.1)
    @test m2.B  == 3.0
    @test m2.k  == 1.5
    @test m2.a₀ == 0.4
    @test m2.τ  == 0.1
end

@testset "CoherentDDMResult construction" begin
    r = CoherentDDMResult(rt=0.5, choice=1, s=1, c=0.5)
    @test r.rt     == 0.5
    @test r.choice == 1
    @test r.s      == 1
    @test r.c      == 0.5
end

@testset "logdensityof basic properties" begin
    m = CoherentDDM(B=3.0, k=2.0, a₀=0.5, τ=0.1)

    for (s, choice, c) in [(1, 1, 0.5), (1, -1, 0.5), (-1, 1, 0.3), (-1, -1, 1.0)]
        ld = logdensityof(m.B, m.k, m.α, m.a₀, m.τ, 0.5, choice, s, c)
        @test isfinite(ld)
    end

    # Negative RT → -Inf
    @test logdensityof(m.B, m.k, m.α, m.a₀, m.τ, -0.1, 1, 1, 0.5) == -Inf

    # RT before non-decision time → density 0 → log returns finite sentinel
    ld_tau = logdensityof(m.B, m.k, m.α, m.a₀, m.τ, 0.05, 1, 1, 0.5)
    @test ld_tau <= 0.0

    # Zero coherence, zero bias: both choices should have equal log-density
    ld_up   = logdensityof(3.0, 2.0, 1.0, 0.5, 0.1, 0.5,  1, 1, 0.0)
    ld_down = logdensityof(3.0, 2.0, 1.0, 0.5, 0.1, 0.5, -1, 1, 0.0)
    @test isapprox(ld_up, ld_down, atol=1e-8)
end

@testset "Simulation: zero coherence → ~50% accuracy" begin
    m = CoherentDDM(B=3.0, k=2.0, a₀=0.5, τ=0.1)
    rng = MersenneTwister(42)
    results = [simulateDDM(m, 0.0, 1e-4, rng) for _ in 1:2000]
    acc = mean(r.choice == r.s for r in results)
    @test 0.4 < acc < 0.6
end

@testset "Simulation: high coherence → high accuracy" begin
    m = CoherentDDM(B=3.0, k=3.0, a₀=0.5, τ=0.1)
    rng = MersenneTwister(42)
    results = [simulateDDM(m, 1.0, 1e-4, rng) for _ in 1:500]
    acc = mean(r.choice == r.s for r in results)
    @test acc > 0.85
end

@testset "Gradient check (ForwardDiff vs FiniteDiff)" begin
    m = CoherentDDM(B=3.0, k=2.0, a₀=0.5, τ=0.15)
    r = CoherentDDMResult(rt=0.5, choice=1, s=1, c=0.5)

    f(p) = logdensityof(p[1], p[2], p[3], p[4], p[5], r.rt, r.choice, r.s, r.c)
    params = [m.B, m.k, m.α, m.a₀, m.τ]

    grad_ad  = ForwardDiff.gradient(f, params)
    grad_num = FiniteDiff.finite_difference_gradient(f, params)

    @test all(isfinite, grad_ad)
    @test isapprox(grad_ad, grad_num, atol=1e-4)
end

@testset "MLE parameter recovery" begin
    rng = MersenneTwister(2025)
    true_model = CoherentDDM(B=3.0, k=2.0, a₀=0.45, τ=0.12)

    # Mix of coherence levels, 200 trials per level
    coherences = repeat([0.1, 0.3, 0.5, 0.7, 1.0], 200)
    data = [simulateDDM(true_model, c, 1e-4, rng) for c in coherences]

    fit_model = CoherentDDM(B=4.0, k=1.5, a₀=0.5, τ=0.1)
    fit!(fit_model, data)

    @test isfinite(fit_model.B)  && isfinite(fit_model.k)
    @test isfinite(fit_model.a₀) && isfinite(fit_model.τ)
    @test abs(fit_model.B  - true_model.B)  ≤ 1.0
    @test abs(fit_model.k  - true_model.k)  ≤ 1.0
    @test abs(fit_model.a₀ - true_model.a₀) ≤ 0.15
    @test abs(fit_model.τ  - true_model.τ)  ≤ 0.08
end

@testset "Parameter transforms roundtrip" begin
    m = CoherentDDM(B=3.2, k=1.7, α=1.4, a₀=0.42, τ=0.13)
    u = coherent_to_unconstrained(m)
    @test length(u) == 5
    @test all(isfinite, u)
    m2 = coherent_from_unconstrained(u)
    @test isapprox(m2.B,  m.B;  rtol=1e-10)
    @test isapprox(m2.k,  m.k;  rtol=1e-10)
    @test isapprox(m2.α,  m.α;  rtol=1e-10)
    @test isapprox(m2.a₀, m.a₀; rtol=1e-10)
    @test isapprox(m2.τ,  m.τ;  rtol=1e-10)

    # logit/logistic are inverses; softmax rows are valid simplex points
    @test isapprox(logistic(logit(0.3)), 0.3; atol=1e-12)
    p = softmax([0.4, -1.2, 2.0])
    @test isapprox(sum(p), 1.0; atol=1e-12) && all(p .> 0)
    @test isapprox(logsumexp([0.0, 0.0]), log(2); atol=1e-12)
end

@testset "fit! holds α fixed when fit_α=false" begin
    rng = MersenneTwister(99)
    true_model = CoherentDDM(B=3.0, k=2.0, α=1.5, a₀=0.5, τ=0.12)
    cohs = repeat([0.1, 0.4, 0.7, 1.0], 200)
    data = [simulateDDM(true_model, c, 1e-4, rng) for c in cohs]

    fm = CoherentDDM(B=4.0, k=1.5, α=1.5, a₀=0.5, τ=0.1, fit_α=false)
    fit!(fm, data)
    @test fm.α == 1.5                 # untouched
    @test abs(fm.B - true_model.B) ≤ 1.0
    @test abs(fm.k - true_model.k) ≤ 1.0
end

@testset "fit_shared_α! estimates one α across states" begin
    rng = MersenneTwister(7)
    trueα = 1.6
    m1 = CoherentDDM(B=2.5, k=1.5, α=trueα, a₀=0.5, τ=0.10)
    m2 = CoherentDDM(B=4.0, k=3.0, α=trueα, a₀=0.5, τ=0.20)
    cohs = repeat([0.1, 0.3, 0.5, 0.7, 1.0], 300)
    d1 = [simulateDDM(m1, c, 1e-4, rng) for c in cohs]
    d2 = [simulateDDM(m2, c, 1e-4, rng) for c in cohs]
    x  = vcat(d1, d2)

    n1 = length(d1)
    γ = zeros(2, length(x))
    γ[1, 1:n1] .= 1.0
    γ[2, n1+1:end] .= 1.0

    models = [CoherentDDM(B=3.0, k=2.0, α=1.0, a₀=0.5, τ=0.1),
              CoherentDDM(B=3.0, k=2.0, α=1.0, a₀=0.5, τ=0.1)]
    fit_shared_α!(models, x, γ)

    @test models[1].α == models[2].α          # genuinely shared
    @test abs(models[1].α - trueα) ≤ 0.4
    @test abs(models[1].B - m1.B)  ≤ 0.7
    @test abs(models[2].B - m2.B)  ≤ 0.7
end

@testset "Gradient-descent HMM fit (autodiff on forward loglik)" begin
    rng = MersenneTwister(11)
    K = 2
    init  = [0.5, 0.5]
    trans = [0.9 0.1; 0.1 0.9]
    dists = [CoherentDDM(B=2.5, k=1.5, α=1.3, a₀=0.5, τ=0.10),
             CoherentDDM(B=4.0, k=3.0, α=1.3, a₀=0.5, τ=0.15)]
    hmm = PriorHMM(init, trans, dists; α_trans=2.0, share_α=true)

    cohs = repeat([0.2, 0.5, 0.8], 200)
    obs  = [simulateDDM(CoherentDDM(B=3.0, k=2.5, α=1.4, a₀=0.5, τ=0.12), c, 1e-4, rng) for c in cohs]
    seq_ends = [length(obs)]

    θ0 = pack_coherent_hmm(hmm)
    ll0 = coherent_hmm_loglikelihood(θ0, obs; K=K, share_α=true, seq_ends=seq_ends)
    @test isfinite(ll0)

    # Our packed-parameter loglik must equal HiddenMarkovModels.jl's own forward
    # algorithm evaluated on the same HMM.
    ll_hmm = DensityInterface.logdensityof(hmm, obs; seq_ends=seq_ends)
    @test isapprox(ll0, ll_hmm; rtol=1e-8)

    # ForwardDiff differentiates straight through HMM.jl's forward algorithm,
    # including into the emission parameters.
    g = ForwardDiff.gradient(t -> coherent_hmm_loglikelihood(t, obs; K=K, share_α=true, seq_ends=seq_ends), θ0)
    @test length(g) == 2 + K*K + 4*K + 1          # init + trans + emis(4/state) + shared α
    @test all(isfinite, g)
    @test any(abs.(g[(2 + K*K + 1):end]) .> 1e-6)  # emission gradients are nonzero

    fit_hmm_gradient!(hmm, obs; seq_ends=seq_ends, prior=false, iterations=150)
    ll1 = coherent_hmm_loglikelihood(pack_coherent_hmm(hmm), obs; K=K, share_α=true, seq_ends=seq_ends)
    @test ll1 ≥ ll0 - 1e-6                          # likelihood did not decrease
    @test hmm.dists[1].α == hmm.dists[2].α          # shared α preserved
    @test isapprox(sum(hmm.init), 1.0; atol=1e-8)
    @test all(isapprox.(sum(hmm.trans; dims=2), 1.0; atol=1e-8))
end

@testset "MAP gradient HMM fit (Dirichlet prior)" begin
    rng = MersenneTwister(3)
    K = 2
    init  = [0.5, 0.5]
    trans = [0.85 0.15; 0.15 0.85]
    dists = [CoherentDDM(B=2.5, k=1.5, α=1.3, a₀=0.5, τ=0.10),
             CoherentDDM(B=4.0, k=3.0, α=1.3, a₀=0.5, τ=0.15)]
    αT = [20.0 1.0; 1.0 20.0]                       # strong sticky prior
    hmm = PriorHMM(init, trans, dists; α_trans=αT, α_init=2.0, share_α=true)

    cohs = repeat([0.2, 0.5, 0.8], 200)
    obs  = [simulateDDM(CoherentDDM(B=3.0, k=2.5, α=1.4, a₀=0.5, τ=0.12), c, 1e-4, rng) for c in cohs]
    seq_ends = [length(obs)]

    θ0 = pack_coherent_hmm(hmm)

    # log-posterior = forward loglik + Dirichlet log-prior (up to a constant)
    ll  = coherent_hmm_loglikelihood(θ0, obs; K=K, share_α=true, seq_ends=seq_ends)
    lp  = coherent_hmm_logposterior(θ0, hmm, obs; seq_ends=seq_ends)
    prior_term = sum((hmm.α_init .- 1) .* log.(hmm.init)) +
                 sum((αT[1, :] .- 1) .* log.(hmm.trans[1, :])) +
                 sum((αT[2, :] .- 1) .* log.(hmm.trans[2, :]))
    @test isapprox(lp - ll, prior_term; rtol=1e-8)

    g = ForwardDiff.gradient(t -> coherent_hmm_logposterior(t, hmm, obs; seq_ends=seq_ends), θ0)
    @test all(isfinite, g)

    # MAP fit improves the posterior and (sticky prior) yields stickier
    # transitions than the pure-MLE fit on the same data.
    hmap = PriorHMM(copy(init), copy(trans), deepcopy(dists); α_trans=αT, α_init=2.0, share_α=true)
    fit_hmm_gradient!(hmap, obs; seq_ends=seq_ends, prior=true, iterations=200)
    @test coherent_hmm_logposterior(pack_coherent_hmm(hmap), hmap, obs; seq_ends=seq_ends) ≥ lp - 1e-6

    hmle = PriorHMM(copy(init), copy(trans), deepcopy(dists); α_trans=αT, α_init=2.0, share_α=true)
    fit_hmm_gradient!(hmle, obs; seq_ends=seq_ends, prior=false, iterations=200)
    @test (hmap.trans[1,1] + hmap.trans[2,2]) ≥ (hmle.trans[1,1] + hmle.trans[2,2])
    @test all(isapprox.(sum(hmap.trans; dims=2), 1.0; atol=1e-8))
end