using DriftDiffusionModels: LeakyAccumulatorModel, LinearPoissonObservationModel,
                            BasisPoissonObservationModel, GPPoissonObservationModel,
                            NeuralDDM, AbstractStateModel, AbstractObservationModel,
                            init_sample, init_logpdf,
                            transition_sample, transition_logpdf,
                            hazard, stop_logpdf, choice_logpdf,
                            obs_sample, obs_logpdf,
                            Trial, n_time, n_neurons,
                            particle_filter, log_marginal_likelihood,
                            simulate_trial, fit!,
                            softplus, logsumexp
import DriftDiffusionModels as DDM
using Random: MersenneTwister
using Statistics: mean, var
using Distributions: Poisson, logpdf
using ForwardDiff: gradient
import Optim

@testset "LeakyAccumulatorModel - default constructor" begin
    m = LeakyAccumulatorModel()
    @test m isa LeakyAccumulatorModel{Float64}
    @test m isa AbstractStateModel
    @test m.B == 1.0
    @test m.v == 1.0
    @test m.λ == 0.0
    @test m.σ² == 1.0
    @test m.μ₀ == 0.0
    @test m.Σ₀ == 0.0
    @test m.τ == 1e-1
    @test m.α == 5.0
    @test m.γ == 5.0
end

@testset "LeakyAccumulatorModel - keyword constructor" begin
    m = LeakyAccumulatorModel(B=2.5, v=0.8, λ=0.3, σ²=1.5, μ₀=0.1, Σ₀=0.05, τ=0.2, α=3.0, γ=4.0)
    @test m.B == 2.5
    @test m.v == 0.8
    @test m.λ == 0.3
    @test m.σ² == 1.5
    @test m.μ₀ == 0.1
    @test m.Σ₀ == 0.05
    @test m.τ == 0.2
    @test m.α == 3.0
    @test m.γ == 4.0
end

@testset "LeakyAccumulatorModel - type promotion" begin
    # Mixed Int and Float should promote to Float64
    m = LeakyAccumulatorModel(B=1, v=1.0, λ=0, σ²=1, μ₀=0, Σ₀=0, τ=1//10, α=5, γ=5)
    @test m isa LeakyAccumulatorModel{Float64}

    # All Float32 stays Float32
    m32 = LeakyAccumulatorModel(B=1.0f0, v=1.0f0, λ=0.0f0, σ²=1.0f0,
                                μ₀=0.0f0, Σ₀=0.0f0, τ=0.1f0, α=5.0f0, γ=5.0f0)
    @test m32 isa LeakyAccumulatorModel{Float32}
end

@testset "LeakyAccumulatorModel - positional constructor still works" begin
    m = LeakyAccumulatorModel{Float64}(1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.1, 5.0, 5.0)
    @test m.B == 1.0 && m.τ == 0.1 && m.α == 5.0 && m.γ == 5.0
end

@testset "LeakyAccumulatorModel - mutability" begin
    m = LeakyAccumulatorModel()
    m.B = 3.0
    m.α = 10.0
    @test m.B == 3.0
    @test m.α == 10.0
end

@testset "LinearPoissonObservationModel - N constructor" begin
    obs = LinearPoissonObservationModel(5)
    @test obs isa LinearPoissonObservationModel{Float64}
    @test obs isa AbstractObservationModel
    @test obs.b == zeros(5)
    @test obs.w == ones(5)
    @test length(obs.b) == 5
end

@testset "LinearPoissonObservationModel - custom T" begin
    obs = LinearPoissonObservationModel(3; T=Float32)
    @test obs isa LinearPoissonObservationModel{Float32}
    @test eltype(obs.b) == Float32
end

@testset "LinearPoissonObservationModel - keyword constructor" begin
    obs = LinearPoissonObservationModel(b=[0.1, 0.2, 0.3], w=[1.0, 2.0, 3.0])
    @test obs.b == [0.1, 0.2, 0.3]
    @test obs.w == [1.0, 2.0, 3.0]
end

@testset "LinearPoissonObservationModel - length mismatch errors" begin
    @test_throws ArgumentError LinearPoissonObservationModel(b=[0.0, 0.0], w=[1.0])
end

@testset "BasisPoissonObservationModel - default constructor" begin
    K, N = 4, 6
    obs = BasisPoissonObservationModel(K, N)
    @test obs isa BasisPoissonObservationModel{Float64}
    @test obs isa AbstractObservationModel
    @test size(obs.β) == (K, N)
    @test all(iszero, obs.β)
    @test length(obs.centers) == K
    @test length(obs.widths) == K
    @test obs.centers[1] == -1.0 && obs.centers[end] == 1.0
    @test obs.ρ == 1e-3
end

@testset "BasisPoissonObservationModel - custom centers/widths" begin
    obs = BasisPoissonObservationModel(3, 2;
        centers=[-2.0, 0.0, 2.0], widths=[0.5, 0.5, 0.5], ρ=0.01)
    @test obs.centers == [-2.0, 0.0, 2.0]
    @test obs.widths == [0.5, 0.5, 0.5]
    @test obs.ρ == 0.01
end

@testset "BasisPoissonObservationModel - K=1 edge case" begin
    obs = BasisPoissonObservationModel(1, 3)
    @test length(obs.centers) == 1
    @test length(obs.widths) == 1
    @test size(obs.β) == (1, 3)
end

@testset "BasisPoissonObservationModel - length mismatch errors" begin
    @test_throws ArgumentError BasisPoissonObservationModel(3, 2; centers=[0.0, 1.0])
    @test_throws ArgumentError BasisPoissonObservationModel(3, 2; widths=[0.5, 0.5])
end

@testset "GPPoissonObservationModel - default constructor" begin
    obs = GPPoissonObservationModel(4)
    @test obs isa GPPoissonObservationModel{Float64}
    @test obs isa AbstractObservationModel
    @test length(obs.inducing_points) == 10
    @test obs.lengthscale == ones(4)
    @test obs.amplitude == ones(4)
    @test obs.mean == zeros(4)
end

@testset "GPPoissonObservationModel - scalar broadcasts to N" begin
    obs = GPPoissonObservationModel(3; lengthscale=2.0, amplitude=0.5, mean=-1.0)
    @test obs.lengthscale == fill(2.0, 3)
    @test obs.amplitude == fill(0.5, 3)
    @test obs.mean == fill(-1.0, 3)
end

@testset "GPPoissonObservationModel - vector of length N" begin
    obs = GPPoissonObservationModel(3;
        lengthscale=[1.0, 2.0, 3.0],
        amplitude=[0.1, 0.2, 0.3],
        mean=[0.0, 0.5, 1.0])
    @test obs.lengthscale == [1.0, 2.0, 3.0]
    @test obs.amplitude == [0.1, 0.2, 0.3]
end

@testset "GPPoissonObservationModel - wrong-length vector errors" begin
    @test_throws ArgumentError GPPoissonObservationModel(3; lengthscale=[1.0, 2.0])
    @test_throws ArgumentError GPPoissonObservationModel(3; amplitude=[1.0, 2.0])
    @test_throws ArgumentError GPPoissonObservationModel(3; mean=[1.0, 2.0])
end

@testset "GPPoissonObservationModel - custom inducing_points and T" begin
    obs = GPPoissonObservationModel(2;
        inducing_points=range(0.0, 1.0; length=5), T=Float32)
    @test obs isa GPPoissonObservationModel{Float32}
    @test length(obs.inducing_points) == 5
    @test eltype(obs.lengthscale) == Float32
end

@testset "NeuralDDM - default constructor" begin
    m = NeuralDDM()
    @test m.state isa LeakyAccumulatorModel
    @test m.obs isa LinearPoissonObservationModel
    @test length(m.obs.b) == 1
end

@testset "NeuralDDM - composed with different obs models" begin
    state = LeakyAccumulatorModel(B=2.0, λ=0.1)
    obs_basis = BasisPoissonObservationModel(5, 10)
    m = NeuralDDM(state=state, obs=obs_basis)
    @test m.state.B == 2.0
    @test m.obs isa BasisPoissonObservationModel
    @test size(m.obs.β) == (5, 10)

    obs_gp = GPPoissonObservationModel(4)
    m2 = NeuralDDM(state=state, obs=obs_gp)
    @test m2.obs isa GPPoissonObservationModel
end

@testset "NeuralDDM - parametric types preserved" begin
    state = LeakyAccumulatorModel()
    obs = BasisPoissonObservationModel(3, 2)
    m = NeuralDDM(state, obs)
    @test m isa NeuralDDM{<:LeakyAccumulatorModel, <:BasisPoissonObservationModel}
end

@testset "init_sample - degenerate Σ₀=0" begin
    rng = MersenneTwister(0)
    m = LeakyAccumulatorModel(μ₀=0.3, Σ₀=0.0)
    @test init_sample(rng, m) ≈ 0.3
end

@testset "init_sample - empirical mean/variance match Σ₀, μ₀" begin
    rng = MersenneTwister(42)
    m = LeakyAccumulatorModel(μ₀=1.0, Σ₀=0.25)
    draws = [init_sample(rng, m) for _ in 1:50_000]
    @test isapprox(mean(draws), 1.0; atol=0.01)
    @test isapprox(var(draws), 0.25; atol=0.01)
end

@testset "init_sample - NeuralDDM dispatches to state" begin
    rng = MersenneTwister(1)
    state = LeakyAccumulatorModel(μ₀=2.5, Σ₀=0.0)
    m = NeuralDDM(state=state)
    @test init_sample(rng, m) ≈ 2.5
end

@testset "init_logpdf - matches Normal log-density" begin
    m = LeakyAccumulatorModel(μ₀=0.5, Σ₀=2.0)
    x = 1.2
    expected = -log(2π * 2.0) / 2 - (x - 0.5)^2 / (2 * 2.0)
    @test init_logpdf(m, x) ≈ expected
    @test init_logpdf(NeuralDDM(state=m), x) ≈ expected
end

@testset "init_logpdf - degenerate Σ₀=0 is a point mass at μ₀" begin
    m = LeakyAccumulatorModel(μ₀=0.3, Σ₀=0.0)
    @test init_logpdf(m, 0.3) == Inf      # at the point mass
    @test init_logpdf(m, 0.0) == -Inf     # everywhere else
    # default model has Σ₀=0; must not return +Inf for arbitrary x
    @test init_logpdf(LeakyAccumulatorModel(), 1.7) == -Inf
end

@testset "transition_sample - mean and variance" begin
    rng = MersenneTwister(7)
    m = LeakyAccumulatorModel(v=2.0, λ=0.5, σ²=0.4)
    x_prev, u, dt = 1.0, 0.8, 0.01
    expected_μ = x_prev + (-m.λ * x_prev + m.v * u) * dt
    expected_var = m.σ² * dt
    draws = [transition_sample(rng, m, x_prev, u, dt) for _ in 1:50_000]
    @test isapprox(mean(draws), expected_μ; atol=5e-3)
    @test isapprox(var(draws), expected_var; rtol=0.05)
end

@testset "transition_sample - NeuralDDM forwards" begin
    rng = MersenneTwister(2)
    state = LeakyAccumulatorModel(σ²=0.0)
    m = NeuralDDM(state=state)
    # σ²=0 => deterministic Euler step
    @test transition_sample(rng, m, 1.0, 0.5, 0.1) ≈
          1.0 + (-state.λ * 1.0 + state.v * 0.5) * 0.1
end

@testset "transition_logpdf - matches Gaussian formula" begin
    m = LeakyAccumulatorModel(v=1.5, λ=0.2, σ²=0.6)
    x_prev, u, dt, x = 0.4, 1.0, 0.05, 0.5
    μ = x_prev + (-m.λ * x_prev + m.v * u) * dt
    var = m.σ² * dt
    expected = -log(2π * var) / 2 - (x - μ)^2 / (2 * var)
    @test transition_logpdf(m, x, x_prev, u, dt) ≈ expected
    @test transition_logpdf(NeuralDDM(state=m), x, x_prev, u, dt) ≈ expected
end

@testset "transition_logpdf - peak at mean" begin
    m = LeakyAccumulatorModel(v=1.0, λ=0.0, σ²=1.0)
    x_prev, u, dt = 0.0, 1.0, 0.1
    μ = x_prev + m.v * u * dt
    @test transition_logpdf(m, μ, x_prev, u, dt) >
          transition_logpdf(m, μ + 0.1, x_prev, u, dt)
end

@testset "hazard - boundary gives 0.5" begin
    m = LeakyAccumulatorModel(B=1.5)
    @test hazard(m, 1.5, 10.0) ≈ 0.5
    @test hazard(m, -1.5, 10.0) ≈ 0.5
end

@testset "hazard - monotone in |x|" begin
    m = LeakyAccumulatorModel(B=1.0)
    @test hazard(m, 0.0, 5.0) < hazard(m, 0.5, 5.0) < hazard(m, 1.0, 5.0) < hazard(m, 2.0, 5.0)
end

@testset "hazard - large α saturates" begin
    m = LeakyAccumulatorModel(B=1.0)
    @test hazard(m, 2.0, 1000.0) ≈ 1.0
    @test hazard(m, 0.0, 1000.0) ≈ 0.0
end

@testset "hazard - symmetric in sign of x" begin
    m = LeakyAccumulatorModel(B=1.0)
    @test hazard(m, 0.7, 3.0) ≈ hazard(m, -0.7, 3.0)
end

@testset "hazard - NeuralDDM forwards" begin
    state = LeakyAccumulatorModel(B=2.0)
    m = NeuralDDM(state=state)
    @test hazard(m, 2.0, 5.0) ≈ 0.5
end

@testset "stop_logpdf - matches log(hazard) and log1p(-hazard)" begin
    m = LeakyAccumulatorModel(B=1.0)
    x, α = 0.6, 4.0
    p = hazard(m, x, α)
    @test stop_logpdf(m, x, true,  α) ≈ log(p)
    @test stop_logpdf(m, x, false, α) ≈ log1p(-p)
end

@testset "stop_logpdf - numerically stable in saturation" begin
    m = LeakyAccumulatorModel(B=1.0)
    # |x| >> B with large α: hazard ≈ 1, so log(1-p) must not blow to -Inf
    lp = stop_logpdf(m, 10.0, false, 50.0)
    @test isfinite(lp)
    @test lp < 0  # log of small probability
    # symmetric saturation for stopped at low |x|
    lp2 = stop_logpdf(m, 0.0, true, 50.0)
    @test isfinite(lp2)
end

@testset "stop_logpdf - probabilities sum to 1" begin
    m = LeakyAccumulatorModel(B=1.2)
    for x in (-2.0, -0.5, 0.0, 0.8, 3.0)
        lp_stop = stop_logpdf(m, x, true,  3.0)
        lp_cont = stop_logpdf(m, x, false, 3.0)
        @test exp(lp_stop) + exp(lp_cont) ≈ 1.0
    end
end

@testset "stop_logpdf - NeuralDDM forwards" begin
    state = LeakyAccumulatorModel(B=1.0)
    m = NeuralDDM(state=state)
    @test stop_logpdf(m, 1.0, true, 5.0) ≈ stop_logpdf(state, 1.0, true, 5.0)
end

# === Observation models ===

@testset "LinearPoisson - obs_sample shape and dtype" begin
    rng = MersenneTwister(0)
    obs = LinearPoissonObservationModel(b=[-1.0, 0.0, 1.0], w=[0.5, 0.5, 0.5])
    y = obs_sample(rng, obs, 0.2, 0.01)
    @test y isa Vector{Int}
    @test length(y) == 3
    @test all(y .>= 0)
end

@testset "LinearPoisson - obs_sample empirical mean matches λ·dt" begin
    rng = MersenneTwister(1)
    # one neuron, single state, single dt → Poisson(softplus(η)·dt)
    # pick large η so softplus(η) ≈ η; η = 20 → rate ≈ 20 /s
    obs = LinearPoissonObservationModel(b=[20.0], w=[0.0])
    dt = 0.1                                                     # bin=100ms
    expected = softplus(20.0) * dt                               # ≈ 2.0
    draws = [obs_sample(rng, obs, 0.0, dt)[1] for _ in 1:20_000]
    @test isapprox(mean(draws), expected; atol=0.05)
    @test isapprox(var(draws),  expected; atol=0.10)  # Poisson: var=mean
end

@testset "LinearPoisson - obs_logpdf matches Distributions.Poisson" begin
    obs = LinearPoissonObservationModel(b=[0.2, -0.3], w=[1.0, 0.5])
    x, dt = 0.4, 0.05
    y = [2, 0]
    expected = sum(logpdf(Poisson(softplus(obs.b[n] + obs.w[n] * x) * dt), y[n]) for n in 1:2)
    @test obs_logpdf(obs, y, x, dt) ≈ expected
end

@testset "LinearPoisson - obs_logpdf y=0 with very small rate is finite, not NaN" begin
    obs = LinearPoissonObservationModel(b=[-50.0], w=[0.0])  # softplus(-50) ≈ 1.9e-22
    lp = obs_logpdf(obs, [0], 0.0, 0.01)
    @test isfinite(lp)
    # P(y=0 | rate≈0) ≈ 1, so logpdf ≈ 0
    @test isapprox(lp, 0.0; atol=1e-10)
end

@testset "LinearPoisson - length mismatch errors" begin
    obs = LinearPoissonObservationModel(3)
    @test_throws ArgumentError obs_logpdf(obs, [1, 2], 0.0, 0.01)
end

@testset "BasisPoisson - obs_sample shape" begin
    rng = MersenneTwister(2)
    obs = BasisPoissonObservationModel(4, 5)
    obs.β .= 0.1
    y = obs_sample(rng, obs, 0.5, 0.01)
    @test y isa Vector{Int}
    @test length(y) == 5
end

@testset "BasisPoisson - feature at center is 1" begin
    obs = BasisPoissonObservationModel(3, 2; centers=[-1.0, 0.0, 1.0], widths=[0.5, 0.5, 0.5])
    # set β so only middle basis matters for neuron 1, evaluated at x=0 → φ=[…,1,…]
    obs.β[:, 1] .= [0.0, 2.0, 0.0]   # η_1(0) = 2.0
    obs.β[:, 2] .= 0.0               # η_2(0) = 0.0
    dt = 0.1
    y = [3, 1]
    expected = logpdf(Poisson(softplus(2.0) * dt), 3) + logpdf(Poisson(softplus(0.0) * dt), 1)
    @test obs_logpdf(obs, y, 0.0, dt) ≈ expected
end

@testset "BasisPoisson - obs_logpdf matches manual evaluation off-center" begin
    obs = BasisPoissonObservationModel(2, 1; centers=[-1.0, 1.0], widths=[0.5, 0.5])
    obs.β[:, 1] .= [0.7, -0.3]
    x, dt = 0.3, 0.02
    φ = [exp(-(x - c)^2 / (2 * 0.5^2)) for c in (-1.0, 1.0)]
    η = sum(obs.β[k, 1] * φ[k] for k in 1:2)
    expected = logpdf(Poisson(softplus(η) * dt), 4)
    @test obs_logpdf(obs, [4], x, dt) ≈ expected
end

@testset "BasisPoisson - length mismatch errors" begin
    obs = BasisPoissonObservationModel(3, 2)
    @test_throws ArgumentError obs_logpdf(obs, [0], 0.0, 0.01)
end

@testset "GPPoisson - sampling/logpdf not implemented" begin
    rng = MersenneTwister(3)
    obs = GPPoissonObservationModel(2)
    @test_throws ErrorException obs_sample(rng, obs, 0.0, 0.01)
    @test_throws ErrorException obs_logpdf(obs, [0, 0], 0.0, 0.01)
end

@testset "obs_sample / obs_logpdf - NeuralDDM forwards" begin
    rng = MersenneTwister(4)
    state = LeakyAccumulatorModel()
    obs = LinearPoissonObservationModel(b=[0.0, 0.5], w=[1.0, 0.0])
    m = NeuralDDM(state=state, obs=obs)
    # sample reproducibility check: same rng seed → same counts
    rng_a = MersenneTwister(99)
    rng_b = MersenneTwister(99)
    @test obs_sample(rng_a, m, 0.3, 0.01) == obs_sample(rng_b, obs, 0.3, 0.01)
    # logpdf forwards
    y = [1, 0]
    @test obs_logpdf(m, y, 0.3, 0.01) ≈ obs_logpdf(obs, y, 0.3, 0.01)
end

# === choice_logpdf ===

@testset "choice_logpdf - at x=0, each choice has probability 0.5" begin
    m = LeakyAccumulatorModel(γ=5.0)
    @test exp(choice_logpdf(m, 0.0, 1))  ≈ 0.5
    @test exp(choice_logpdf(m, 0.0, -1)) ≈ 0.5
end

@testset "choice_logpdf - probabilities sum to 1" begin
    m = LeakyAccumulatorModel(γ=3.0)
    for x in (-2.0, -0.5, 0.0, 0.8, 2.0)
        @test exp(choice_logpdf(m, x, 1)) + exp(choice_logpdf(m, x, -1)) ≈ 1.0
    end
end

@testset "choice_logpdf - positive x favours right" begin
    m = LeakyAccumulatorModel(γ=5.0)
    @test choice_logpdf(m, 1.0, 1) > choice_logpdf(m, 1.0, -1)
end

@testset "choice_logpdf - NeuralDDM forwards to state" begin
    state = LeakyAccumulatorModel(γ=4.0)
    model = NeuralDDM(state=state)
    x = 0.7
    @test choice_logpdf(model, x, 1)  ≈ choice_logpdf(state, x, 1)
    @test choice_logpdf(model, x, -1) ≈ choice_logpdf(state, x, -1)
end

# === Trial ===

@testset "Trial - basic construction" begin
    spikes = [1 0; 0 2; 1 1]   # 3 bins × 2 neurons
    u      = [-1.0, 0.0, 1.0]
    t = Trial(spikes, u, 1, 0.01)
    @test t isa Trial{Float64}
    @test n_time(t)    == 3
    @test n_neurons(t) == 2
    @test t.choice == 1
    @test t.dt ≈ 0.01
end

@testset "Trial - choice -1 is valid" begin
    t = Trial(ones(Int, 5, 3), zeros(5), -1, 0.02)
    @test t.choice == -1
end

@testset "Trial - u length mismatch throws" begin
    @test_throws ArgumentError Trial(ones(Int, 4, 2), zeros(3), 1, 0.01)
end

@testset "Trial - invalid choice throws" begin
    @test_throws ArgumentError Trial(ones(Int, 3, 2), zeros(3), 0, 0.01)
end

@testset "Trial - negative dt throws" begin
    @test_throws ArgumentError Trial(ones(Int, 3, 2), zeros(3), 1, -0.01)
end

# === particle_filter ===

@testset "particle_filter - returns finite scalar" begin
    rng   = MersenneTwister(42)
    state = LeakyAccumulatorModel(B=1.5, v=1.0, λ=0.1, σ²=0.5, α=5.0, γ=5.0)
    obs   = LinearPoissonObservationModel(b=[0.0, 0.0], w=[1.0, 0.5])
    model = NeuralDDM(state=state, obs=obs)

    spikes = rand(MersenneTwister(1), 0:2, 20, 2)
    u      = rand(MersenneTwister(2), [-1.0, 0.0, 1.0], 20)
    trial  = Trial(spikes, u, 1, 0.01)

    lml = particle_filter(rng, model, trial; N=256)
    @test isfinite(lml)
    @test lml < 0.0   # log probability must be negative
end

@testset "particle_filter - reproducible with same seed" begin
    state = LeakyAccumulatorModel(α=5.0, γ=5.0)
    obs   = LinearPoissonObservationModel(1)
    model = NeuralDDM(state=state, obs=obs)
    trial = Trial(zeros(Int, 10, 1), zeros(10), 1, 0.01)

    lml_a = particle_filter(MersenneTwister(7), model, trial; N=128)
    lml_b = particle_filter(MersenneTwister(7), model, trial; N=128)
    @test lml_a ≈ lml_b
end

@testset "log_marginal_likelihood - sums over trials" begin
    rng   = MersenneTwister(0)
    state = LeakyAccumulatorModel(α=5.0, γ=5.0)
    obs   = LinearPoissonObservationModel(1)
    model = NeuralDDM(state=state, obs=obs)

    trials = [Trial(zeros(Int, 5, 1), zeros(5), 1, 0.01) for _ in 1:4]
    lml = log_marginal_likelihood(rng, model, trials; N=128)
    @test isfinite(lml)
    @test lml < 0.0
end

@testset "log_marginal_likelihood - reproducible with same seed (CRN)" begin
    state = LeakyAccumulatorModel(B=1.5, v=1.0, λ=0.1, σ²=0.5, α=5.0, γ=5.0)
    obs   = LinearPoissonObservationModel(b=[0.0, 0.0], w=[1.0, 0.5])
    model = NeuralDDM(state=state, obs=obs)

    trials = [Trial(rand(MersenneTwister(k), 0:2, 12, 2),
                    rand(MersenneTwister(100 + k), [-1.0, 0.0, 1.0], 12), 1, 0.01)
              for k in 1:6]

    lml_a = log_marginal_likelihood(MersenneTwister(7), model, trials; N=128)
    lml_b = log_marginal_likelihood(MersenneTwister(7), model, trials; N=128)
    @test lml_a ≈ lml_b   # deterministic objective regardless of thread scheduling
end

# === simulate_trial ===

@testset "simulate_trial - shapes, choice, truncation" begin
    rng   = MersenneTwister(11)
    state = LeakyAccumulatorModel(B=1.0, v=1.5, λ=0.1, σ²=0.5, α=6.0, γ=5.0)
    obs   = LinearPoissonObservationModel(b=[1.0, 0.5, 0.0], w=[1.0, -0.5, 0.2])
    model = NeuralDDM(state=state, obs=obs)
    u = rand(rng, [-1.0, 0.0, 1.0], 40)

    tr = simulate_trial(rng, model, u, 0.01; max_bins=40)
    @test tr isa Trial
    @test n_neurons(tr) == 3
    @test 1 ≤ n_time(tr) ≤ 40
    @test tr.choice ∈ (1, -1)
    @test size(tr.spikes, 1) == n_time(tr)
    @test length(tr.u) == n_time(tr)         # u truncated to stop bin
    @test all(tr.spikes .>= 0)
end

@testset "simulate_trial - force-stops at max_bins when hazard never fires" begin
    # B huge ⇒ hazard ≈ 0, so it should run to max_bins every time
    rng   = MersenneTwister(3)
    state = LeakyAccumulatorModel(B=1e6, v=0.0, λ=0.0, σ²=0.1, α=5.0, γ=5.0)
    model = NeuralDDM(state=state, obs=LinearPoissonObservationModel(1))
    tr = simulate_trial(rng, model, zeros(15), 0.01; max_bins=15)
    @test n_time(tr) == 15
end

# === differentiable PF + gradient ===

@testset "_pf_loglik - finite, negative, deterministic given noise" begin
    state = LeakyAccumulatorModel(B=1.2, v=1.0, λ=0.1, σ²=0.5, α=5.0, γ=5.0)
    obs   = LinearPoissonObservationModel(b=[0.5, 0.0], w=[1.0, 0.5])
    model = NeuralDDM(state=state, obs=obs)
    trial = simulate_trial(MersenneTwister(1), model,
                           rand(MersenneTwister(2), [-1.0, 0.0, 1.0], 30), 0.01; max_bins=30)

    noise = DDM._draw_pf_noise(MersenneTwister(5), 128, n_time(trial))
    ll1 = DDM._pf_loglik(model, trial, noise...; resample_every=1)
    ll2 = DDM._pf_loglik(model, trial, noise...; resample_every=1)
    @test isfinite(ll1)
    @test ll1 < 0.0
    @test ll1 == ll2                       # deterministic given fixed noise
end

@testset "_pf_loglik - ForwardDiff gradient is finite and non-trivial" begin
    state = LeakyAccumulatorModel(B=1.2, v=1.0, λ=0.1, σ²=0.5, α=5.0, γ=5.0)
    obs   = LinearPoissonObservationModel(b=[0.5, 0.0], w=[1.0, 0.5])
    model = NeuralDDM(state=state, obs=obs)
    trials = [simulate_trial(MersenneTwister(k), model,
                             rand(MersenneTwister(100+k), [-1.0, 0.0, 1.0], 25), 0.01; max_bins=25)
              for k in 1:8]
    noise = [DDM._draw_pf_noise(MersenneTwister(7), 96, n_time(tr)) for tr in trials]

    θ0 = DDM._unconstrained_θ0(model)
    f = θ -> begin
        m = DDM._model_from_unconstrained(θ, model)
        s = zero(eltype(θ))
        for k in eachindex(trials)
            s += DDM._pf_loglik(m, trials[k], noise[k]...; resample_every=1)
        end
        -s
    end
    g = gradient(f, θ0)
    @test all(isfinite, g)
    @test any(!iszero, g)                  # objective actually depends on θ
end

# === fit! :lbfgs — parameter recovery ===

@testset "fit! :lbfgs improves fit and approaches truth" begin
    truth = NeuralDDM(
        state = LeakyAccumulatorModel(B=1.2, v=1.5, λ=0.2, σ²=0.5, α=6.0, γ=5.0),
        obs   = LinearPoissonObservationModel(b=[1.0, 0.5], w=[1.2, -0.8]),
    )
    rng = MersenneTwister(2024)
    trials = [simulate_trial(rng, truth, rand(rng, [-1.0, 0.0, 1.0], 30), 0.01; max_bins=30)
              for _ in 1:60]

    # objective helper at a given model, on a fixed noise realization
    noise = [DDM._draw_pf_noise(MersenneTwister(123), 128, n_time(tr)) for tr in trials]
    negll(m) = -sum(DDM._pf_loglik(m, trials[k], noise[k]...; resample_every=1)
                    for k in eachindex(trials))

    # start from a perturbed initialization
    init = NeuralDDM(
        state = LeakyAccumulatorModel(B=2.0, v=0.5, λ=0.0, σ²=1.0, α=4.0, γ=3.0),
        obs   = LinearPoissonObservationModel(b=[0.0, 0.0], w=[0.5, 0.5]),
    )
    f_init  = negll(init)
    f_truth = negll(truth)

    fitted, result = fit!(init, trials; method=:lbfgs, N=128,
                          rng=MersenneTwister(99),
                          optim_options=Optim.Options(iterations=80))
    f_fit = negll(fitted)

    @test f_fit < f_init                       # optimizer improved the fit
    @test f_fit ≤ f_truth + 0.05 * abs(f_truth)  # fits sample ≈ as well as / better than truth
    @test sign(fitted.state.v) == sign(truth.state.v)  # drift direction recovered
end
