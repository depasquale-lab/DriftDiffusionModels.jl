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
        ld = logdensityof(m.B, m.k, m.a₀, m.τ, 0.5, choice, s, c)
        @test isfinite(ld)
    end

    # Negative RT → -Inf
    @test logdensityof(m.B, m.k, m.a₀, m.τ, -0.1, 1, 1, 0.5) == -Inf

    # RT before non-decision time → density 0 → log returns finite sentinel
    ld_tau = logdensityof(m.B, m.k, m.a₀, m.τ, 0.05, 1, 1, 0.5)
    @test ld_tau <= 0.0

    # Zero coherence, zero bias: both choices should have equal log-density
    ld_up   = logdensityof(3.0, 2.0, 0.5, 0.1, 0.5,  1, 1, 0.0)
    ld_down = logdensityof(3.0, 2.0, 0.5, 0.1, 0.5, -1, 1, 0.0)
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

    f(p) = logdensityof(p[1], p[2], p[3], p[4], r.rt, r.choice, r.s, r.c)
    params = [m.B, m.k, m.a₀, m.τ]

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