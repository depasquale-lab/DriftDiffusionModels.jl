using Test
using Random
using Statistics
using Optim
using ForwardDiff
using DriftDiffusionModels
const DDM = DriftDiffusionModels

@testset "FPTDDM - keyword constructor and defaults" begin
    m = FPTDDM()
    @test m.v == 1.0
    @test m.B == 1.0
    @test m.a₀ == 0.5
    @test m.λ == 0.0
    @test m.σ² == 1.0
    @test m.τ == 0.0
    @test m.obs isa LinearPoissonObservationModel
    @test isempty(m.obs.b) && isempty(m.obs.w)
    @test n_neurons(m) == 0
end

@testset "FPTDDM - neurons and type promotion" begin
    m = FPTDDM(; b = [0.1, 0.2], w = [1.0, -0.5])
    @test n_neurons(m) == 2
    @test m.obs.b == [0.1, 0.2]
    @test m.obs.w == [1.0, -0.5]
    # integer scalar inputs promote to a float model
    m2 = FPTDDM(; v = 1, B = 2, a₀ = 0, λ = 0, σ² = 1, τ = 0)
    @test m2 isa FPTDDM
    @test m2.B isa Float64
end

@testset "FPTDDM - explicit observation model via obs=" begin
    obs = LinearPoissonObservationModel(3)
    m = FPTDDM(; B = 1.5, obs = obs)
    @test m.obs === obs
    @test n_neurons(m) == 3
    @test m.obs.b == zeros(3) && m.obs.w == ones(3)
end

@testset "FPTDDM - b/w length mismatch errors" begin
    @test_throws ArgumentError FPTDDM(; b = [0.0, 0.0], w = [1.0])
end

@testset "Observation models - construction" begin
    lin = LinearPoissonObservationModel(; b = [0.1], w = [2.0])
    @test n_neurons(lin) == 1
    basis = BasisPoissonObservationModel(4, 3)
    @test n_neurons(basis) == 3
    @test size(basis.β) == (4, 3)
    @test length(basis.centers) == 4 && length(basis.widths) == 4
    @test_throws ArgumentError BasisPoissonObservationModel(3, 2; centers = [0.0, 1.0])
    gp = GPPoissonObservationModel(2)
    @test n_neurons(gp) == 2
end

@testset "obs_logpdf - linear matches Poisson with softplus rate" begin
    m = FPTDDM(; b = [0.2, -0.1], w = [1.5, 0.5])
    x, dt, y = 0.4, 0.01, [2, 0]
    sp(z) = z > 0 ? z + log1p(exp(-z)) : log1p(exp(z))
    expected = 0.0
    for n = 1:2
        μ = sp(m.obs.b[n] + m.obs.w[n] * x) * dt
        expected += y[n] * log(μ) - μ - sum(log, 1:y[n]; init = 0.0)
    end
    @test DDM.obs_logpdf(m, y, x, dt) ≈ expected atol = 1e-10
end

@testset "obs_logpdf - basis matches manual softplus evaluation" begin
    basis = BasisPoissonObservationModel(3, 2; centers = [0.0, 0.5, 1.0], widths = [0.3, 0.3, 0.3])
    basis.β .= reshape(collect(0.1:0.1:0.6), 3, 2)
    m = FPTDDM(; obs = basis)
    x, dt, y = 0.5, 0.01, [1, 3]
    sp(z) = z > 0 ? z + log1p(exp(-z)) : log1p(exp(z))
    φ = [exp(-(x - c)^2 / (2 * 0.3^2)) for c in basis.centers]
    expected = 0.0
    for n = 1:2
        η = sum(basis.β[:, n] .* φ)
        μ = sp(η) * dt
        expected += y[n] * log(μ) - μ - sum(log, 1:y[n]; init = 0.0)
    end
    @test DDM.obs_logpdf(m, y, x, dt) ≈ expected atol = 1e-10
end

@testset "obs_logpdf - length mismatch errors" begin
    m = FPTDDM(; b = [0.0, 0.0], w = [1.0, 1.0])
    @test_throws ArgumentError DDM.obs_logpdf(m, [1], 0.3, 0.01)
end

@testset "GP observation model - not implemented" begin
    m = FPTDDM(; obs = GPPoissonObservationModel(1))
    @test_throws ErrorException DDM.obs_logpdf(m, [0], 0.3, 0.01)
    @test_throws ErrorException DDM.obs_sample(MersenneTwister(1), m, 0.3, 0.01)
end

@testset "crossing probabilities - basic properties" begin
    varstep = 1.0 * 0.01
    # endpoint outside ⇒ certain crossing
    @test DDM._p_cross_upper(1.0, 0.9, 1.05, varstep) == 1.0
    @test DDM._p_cross_lower(0.5, -0.02, varstep) == 1.0
    # both far inside ⇒ near-zero crossing
    @test DDM._p_cross_upper(1.0, 0.5, 0.5, varstep) < 1e-6
    @test DDM._p_cross_lower(0.5, 0.5, varstep) < 1e-6
    # closer to the boundary ⇒ higher crossing probability
    @test DDM._p_cross_upper(1.0, 0.95, 0.95, varstep) >
          DDM._p_cross_upper(1.0, 0.80, 0.80, varstep)
end

@testset "fpt_loglik - argument validation" begin
    m = FPTDDM()
    @test_throws ArgumentError DDM.fpt_loglik(m, 0, 1, 0.01)
    @test_throws ArgumentError DDM.fpt_loglik(m, 10, 2, 0.01)
end

@testset "fpt_loglik - reproducible with same seed" begin
    m = FPTDDM(; v = 1.0, B = 1.0, a₀ = 0.5)
    a = DDM.fpt_loglik(m, 30, 1, 0.005; N = 500, rng = MersenneTwister(7))
    b = DDM.fpt_loglik(m, 30, 1, 0.005; N = 500, rng = MersenneTwister(7))
    @test a == b
    @test isfinite(a)
end

# === Headline test: spike-free marginal reduces to the WFPT density ===
@testset "fpt_loglik - behavior-only marginal matches WFPT" begin
    B, v, a₀, τ = 1.0, 1.0, 0.5, 0.0
    dt = 0.005
    Npart = 4000
    n_reps = 12
    m = FPTDDM(; v = v, B = B, a₀ = a₀, λ = 0.0, σ² = 1.0, τ = τ)

    ref_density(rt, choice) = exp(DDM.logdensityof(B, v, a₀, τ, rt, choice, +1))
    function pf_density(nb, choice, rng)
        ll = [DDM.fpt_loglik(m, nb, choice, dt; N = Npart, rng = rng) for _ = 1:n_reps]
        return mean(exp.(ll)) / dt
    end

    rng = MersenneTwister(20260703)
    # Check across the bulk of both RT distributions. Errors are Monte Carlo
    # scatter (both signs); require agreement within 15% where the density is
    # non-negligible.
    for choice in (+1, -1)
        for nb in (20, 40, 70, 100, 150)
            rt = (nb - 0.5) * dt
            ref = ref_density(rt, choice)
            pf = pf_density(nb, choice, rng)
            @test isapprox(pf, ref; rtol = 0.15)
        end
    end
end

@testset "simulate_trial - shapes, choice, absorption" begin
    m = FPTDDM(; v = 1.0, B = 1.0, a₀ = 0.5, b = [0.5, 0.0], w = [1.0, -1.0])
    rng = MersenneTwister(1)
    tr = simulate_trial(rng, m, 0.005; max_time = 2.0)
    @test tr isa Trial
    @test tr.choice ∈ (1, -1)
    @test n_neurons(tr) == 2
    @test n_time(tr) == length(tr.u)
    @test size(tr.spikes, 1) == n_time(tr)
    @test all(tr.spikes .>= 0)
end

@testset "simulate_trial - drift sign biases choice" begin
    rng = MersenneTwister(11)
    up = FPTDDM(; v = 3.0, B = 1.0, a₀ = 0.5)
    down = FPTDDM(; v = -3.0, B = 1.0, a₀ = 0.5)
    frac_up = mean(simulate_trial(rng, up, 0.005; max_time = 5.0).choice == 1 for _ = 1:200)
    frac_dn = mean(simulate_trial(rng, down, 0.005; max_time = 5.0).choice == 1 for _ = 1:200)
    @test frac_up > 0.5
    @test frac_dn < 0.5
end

@testset "loglik - Trial wrapper is finite and reproducible" begin
    m = FPTDDM(; v = 1.0, B = 1.0, a₀ = 0.5, b = [0.3], w = [1.2])
    tr = simulate_trial(MersenneTwister(3), m, 0.005; max_time = 2.0)
    a = loglik(m, tr; N = 400, rng = MersenneTwister(5))
    b = loglik(m, tr; N = 400, rng = MersenneTwister(5))
    @test a == b
    @test isfinite(a)
end

# === differentiable fitting filter (FPTDDMFit.jl) ===

@testset "_fpt_loglik_guided - behavior-only marginal matches WFPT" begin
    B, v, a₀ = 1.0, 1.0, 0.5
    dt = 0.005
    m = FPTDDM(; v = v, B = B, a₀ = a₀, σ² = 1.0)   # no neurons
    ref(rt, ch) = exp(DDM.logdensityof(B, v, a₀, 0.0, rt, ch, +1))
    function gdens(nb, ch; N = 3000, reps = 10)
        ll = map(1:reps) do r
            εU, εN = DDM._draw_pf_noise(MersenneTwister(r), N, nb)
            DDM._fpt_loglik_guided(m, nb, ch, dt, εU, εN)
        end
        return mean(exp.(ll)) / dt
    end
    for ch in (+1, -1), nb in (20, 50)
        @test isapprox(gdens(nb, ch), ref((nb - 0.5) * dt, ch); rtol = 0.15)
    end
end

@testset "_fpt_loglik_guided - reproducible and differentiable" begin
    m = FPTDDM(; v = 1.0, B = 1.0, a₀ = 0.5, b = [0.5], w = [2.0])
    tr = simulate_trial(MersenneTwister(3), m, 0.01; max_time = 2.0)
    εU, εN = DDM._draw_pf_noise(MersenneTwister(5), 300, n_time(tr))
    a = DDM._fpt_loglik_guided(m, n_time(tr), tr.choice, tr.dt, εU, εN;
        u = tr.u, spikes = tr.spikes)
    b = DDM._fpt_loglik_guided(m, n_time(tr), tr.choice, tr.dt, εU, εN;
        u = tr.u, spikes = tr.spikes)
    @test a == b && isfinite(a)
    # gradient wrt the unconstrained parameters is finite and non-trivial
    θ0 = DDM._unconstrained_θ0(m)
    g = ForwardDiff.gradient(θ0) do θ
        mm = DDM._model_from_unconstrained(θ, m)
        DDM._fpt_loglik_guided(mm, n_time(tr), tr.choice, tr.dt, εU, εN;
            u = tr.u, spikes = tr.spikes)
    end
    @test all(isfinite, g)
    @test any(!iszero, g)
end

# Fast smoke + improvement test. Rigorous parameter recovery (which needs many
# trials / particles / iterations) is exercised by the validation scripts, not the
# unit suite — finite-sample MLEs and the v↔λ correlation make tight recovery
# assertions flaky and slow.
@testset "fit! - runs, improves likelihood, moves to sane params" begin
    truth = FPTDDM(; v = 1.2, B = 1.0, a₀ = 0.55, λ = 0.3, σ² = 1.0,
                   b = [8.0, 10.0], w = [12.0, -8.0])
    rng = MersenneTwister(2026)
    trials = [simulate_trial(rng, truth, 0.01; max_time = 1.0) for _ = 1:40]

    init = FPTDDM(; v = 0.6, B = 1.4, a₀ = 0.5, λ = 0.0, σ² = 1.0,
                  b = zeros(2), w = fill(1.0, 2))
    before = loglik(init, trials; N = 800, rng = MersenneTwister(1))
    fitted, res = fit!(init, trials; N = 800, rng = MersenneTwister(7),
        optim_options = Optim.Options(iterations = 50))
    after = loglik(fitted, trials; N = 800, rng = MersenneTwister(1))

    @test after > before                                # optimizer improved the fit
    @test fitted.B > 0                                  # boundary stays valid
    @test 0 < fitted.a₀ < 1                             # start point stays valid
    @test all(isfinite, vcat(fitted.v, fitted.B, fitted.a₀, fitted.λ,
        fitted.obs.b, fitted.obs.w))
    @test fitted.obs.w != fill(1.0, 2)                  # parameters actually moved
end
