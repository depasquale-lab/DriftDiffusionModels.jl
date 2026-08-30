const DE = DriftDiffusionModels
using StatsFuns: normlogcdf

const REF_M = [log(1.2), log(0.15), log(1.5), 0.0]
const REF_S = [0.25, 0.15, 0.30, 0.20]

"""
    sample_trial_exact(rng, B, v, a₀, τ)

Draw a trial by inverting the first-passage-time CDF.
"""
function sample_trial_exact(rng, B, v, a₀, τ; ngrid=1500)
    s = rand(rng, (-1, 1))
    tmax = τ + 12 * max(B^2 / max(abs(v), 1e-3), B^2)
    ts = range(τ + 1e-6, tmax; length=ngrid)
    dt = step(ts)

    fl = [exp(DE.logdensityof(B, v, a₀, τ, t, -1, s)) for t in ts]
    fu = [exp(DE.logdensityof(B, v, a₀, τ, t, 1, s)) for t in ts]
    ml, mu = sum(fl) * dt, sum(fu) * dt

    choice = rand(rng) < mu / (ml + mu) ? 1 : -1
    f = choice == 1 ? fu : fl
    cdf = cumsum(f) .* dt
    cdf ./= cdf[end]
    idx = clamp(searchsortedfirst(cdf, rand(rng)), 1, ngrid)
    return DDMResult(ts[idx], choice, s)
end

@testset "eDDMExact - Gauss-Hermite rule" begin
    for q in (2, 4, 6, 8, 10)
        z, logw = gauss_hermite(q)
        w = exp.(logw)

        @test length(z) == q
        @test length(w) == q
        @test issorted(z)
        @test all(w .> 0)
        @test sum(w) ≈ 1.0 atol = 1e-12         # normalised for N(0,1)
        @test sum(w .* z) ≈ 0.0 atol = 1e-10    # first moment
        @test sum(w .* z .^ 2) ≈ 1.0 atol = 1e-10
    end

    z, logw = gauss_hermite(4)
    w = exp.(logw)
    @test sum(w .* z .^ 4) ≈ 3.0 atol = 1e-10
    @test sum(w .* z .^ 6) ≈ 15.0 atol = 1e-8

    @test_throws ArgumentError gauss_hermite(1)
end

@testset "eDDMExact - marginal likelihood against brute force" begin
    y = DDMResult(0.43, 1, 1)

    function brute_force(y, m, s0; n=60, span=6.0)
        ax = [range(m[d] - span * s0[d], m[d] + span * s0[d], length=n) for d in 1:4]
        h = prod(step(a) for a in ax)
        hyper = DE.DDMHyper(m, log.(s0))
        tot = 0.0
        u = zeros(4)
        for i4 in 1:n, i3 in 1:n, i2 in 1:n, i1 in 1:n
            u[1] = ax[1][i1]; u[2] = ax[2][i2]; u[3] = ax[3][i3]; u[4] = ax[4][i4]
            tot += exp(DE.prior_logpdf(u, hyper) +
                       DE.logdensityof(exp(u[1]), exp(u[3]), logistic(u[4]), exp(u[2]),
                                       y.rt, y.choice, y.s))
        end
        return log(tot * h)
    end

    ref = brute_force(y, REF_M, REF_S; n=60)

    err6 = abs(marginal_loglik([y], REF_M, log.(REF_S); q=6, qτ=12) - ref)
    err10 = abs(marginal_loglik([y], REF_M, log.(REF_S); q=10, qτ=20) - ref)

    @test err6 < 1e-3
    @test err10 < 1e-3
    @test err10 <= err6                      # refining must not make it worse
end

@testset "eDDMExact - quadrature_check" begin
    rng = MersenneTwister(4242)
    data = [sample_trial_exact(rng, 1.2, 1.5, 0.5, 0.15) for _ in 1:20]

    rows = quadrature_check(data, REF_M, REF_S; qs=(4, 6, 8))
    @test length(rows) == 3
    @test isnan(rows[1].delta_per_trial)     # no previous order to compare against
    @test all(isfinite(r.loglik) for r in rows)

    @test abs(rows[3].delta_per_trial) <= abs(rows[2].delta_per_trial) + 1e-9
end

@testset "eDDMExact - the sentinel is not reachable by quadrature truncation" begin
    rules = DE.quadrature_rules(6, 12)
    g = DE.build_grid(REF_M, log.(REF_S), rules)
    y_fast = DDMResult(1e-6, 1, 1)             # 79 sigma below m_tau

    ll = DE.trial_marginal_loglik(y_fast, g)
    @test isfinite(ll)
    @test ll > DE.LOG_ZERO_MARGINAL / 2

    a = (log(y_fast.rt) - REF_M[2]) / REF_S[2]
    @test ll ≈ normlogcdf(a) + log(1e-16) atol = 1e-6

    @test DE.trial_marginal_loglik(DDMResult(0.43, 1, 1), g) > DE.LOG_ZERO_MARGINAL / 2

    @test DE.trial_marginal_loglik(DDMResult(0.0, 1, 1), g) <= DE.LOG_ZERO_MARGINAL / 2
end

@testset "eDDMExact - marginal likelihood is differentiable" begin
    rng = MersenneTwister(7)
    data = [sample_trial_exact(rng, 1.2, 1.5, 0.5, 0.15) for _ in 1:10]
    rules = DE.quadrature_rules(4, 8)

    f = θ -> -marginal_loglik(data, collect(θ[1:4]), collect(θ[5:8]), rules)
    θ0 = vcat(REF_M, log.(REF_S))

    g_ad = ForwardDiff.gradient(f, θ0)
    g_fd = FiniteDiff.finite_difference_gradient(f, θ0)

    @test all(isfinite, g_ad)
    @test isapprox(g_ad, g_fd; rtol=1e-4, atol=1e-4)
end

@testset "eDDMExact - fit recovers the hyperparameter means" begin
    rng = MersenneTwister(11)

    data = DDMResult[]
    for _ in 1:500
        u = REF_M .+ REF_S .* randn(rng, 4)
        B, v, a₀, τ = DE.transform_params(u)
        push!(data, sample_trial_exact(rng, B, v, a₀, τ))
    end

    f = fit_mlddm_exact(data; q=4, qτ=8, n_starts=2, iterations=200,
                        rng=MersenneTwister(12), verbose=false)

    @test f isa MLDDMFit
    @test length(f.m) == 4
    @test length(f.σ0) == 4
    @test all(isfinite, f.m)
    @test all(isfinite, f.σ0)
    @test all(f.σ0 .> 0)
    @test isfinite(f.loglik)
    @test isfinite(f.bic)
    @test f.n_zero == 0
    @test f.n_converged >= 1

    # Variance components are weakly identified, so only test the means.
    @test all(abs.(f.m .- REF_M) .< REF_S)
end

@testset "eDDMExact - trial_posteriors" begin
    rng = MersenneTwister(13)
    data = [sample_trial_exact(rng, 1.2, 1.5, 0.5, 0.15) for _ in 1:25]

    μ, sd = trial_posteriors(data, REF_M, REF_S; q=4, qτ=8)

    @test size(μ) == (4, 25)
    @test size(sd) == (4, 25)
    @test all(isfinite, μ)
    @test all(isfinite, sd)
    @test all(sd .>= 0)

    for d in 1:4
        @test all(abs.(μ[d, :] .- REF_M[d]) .< 6 * REF_S[d])
        @test all(sd[d, :] .<= 1.5 * REF_S[d])
    end
end

@testset "eDDMExact - profile_sigma0" begin
    rng = MersenneTwister(14)
    data = [sample_trial_exact(rng, 1.2, 1.5, 0.5, 0.15) for _ in 1:60]

    f = fit_mlddm_exact(data; q=4, qτ=8, n_starts=1, iterations=150,
                        rng=MersenneTwister(15), verbose=false)

    rows = profile_sigma0(data, f, 1; factors=(0.5, 1.0, 2.0))
    @test length(rows) == 3
    @test all(isfinite(r.loglik) for r in rows)

    unit = rows[findfirst(r -> r.factor == 1.0, rows)]
    @test unit.delta ≈ 0.0 atol = 1e-6
    @test all(r.delta <= 1e-6 * length(data) for r in rows)

    @test_throws ArgumentError profile_sigma0(data, f, 5)

    @test_throws ArgumentError profile_sigma0(data[1:30], f, 1)
end

@testset "eDDMExact - subsampling is recorded and profiles stay anchored" begin
    rng = MersenneTwister(16)
    data = [sample_trial_exact(rng, 1.2, 1.5, 0.5, 0.15) for _ in 1:300]

    f = fit_mlddm_exact(data; q=4, qτ=8, n_starts=1, iterations=100,
                        max_trials=100, rng=MersenneTwister(17), verbose=false)

    @test f.n_total == 300
    @test f.n_used == 100
    @test length(f.idx) == 100
    @test allunique(f.idx)
    @test all(1 .<= f.idx .<= 300)

    rows = profile_sigma0(data, f, 2; factors=(1.0,))
    @test rows[1].delta ≈ 0.0 atol = 1e-6

    f2 = fit_mlddm_exact(data; q=4, qτ=8, n_starts=1, iterations=50,
                         rng=MersenneTwister(18), verbose=false)
    @test f2.n_used == f2.n_total == 300
    @test f2.idx == collect(1:300)
end

# Independent reference for the truncated τ integral.

"""
    tau_dense_reference(m, sigma0, y; qin, K, ng)

Dense quadrature over `u_τ`; `qin=0` pins the other dimensions.
"""
function tau_dense_reference(m, σ0, y; qin::Int=0, K::Int=600, ng::Int=10)
    a = (log(y.rt) - m[2]) / σ0[2]
    Umax = a + sqrt(a^2 + 400.0)           # far past any node the rule uses
    x, logw = DE.gauss_legendre_unit(ng)
    zi, lwi = qin == 0 ? ([0.0], [0.0]) : gauss_hermite(qin)
    Bs = exp.(m[1] .+ σ0[1] .* zi)
    vs = exp.(m[3] .+ σ0[3] .* zi)
    a0s = logistic.(m[4] .+ σ0[4] .* zi)

    mx = -Inf
    acc = 0.0
    for k in 1:K
        lo = Umax * ((k - 1) / K)^3.5
        hi = Umax * (k / K)^3.5
        h = hi - lo
        h <= 0 && continue
        for j in eachindex(x)
            u = lo + h * x[j]
            z = a - u
            τ = exp(m[2] + σ0[2] * z)
            base = logw[j] + log(h) - z^2 / 2 - log(2π) / 2
            for i4 in eachindex(a0s), i3 in eachindex(vs), i1 in eachindex(Bs)
                t = base + lwi[i4] + lwi[i3] + lwi[i1] +
                    DE.logdensityof(Bs[i1], vs[i3], a0s[i4], τ, y.rt, y.choice, y.s)
                if t > mx
                    acc = acc * exp(mx - t) + 1.0
                    mx = t
                else
                    acc += exp(t - mx)
                end
            end
        end
    end
    return mx + log(acc)
end

tail_trial(a; choice=1, s=1) = DDMResult(exp(REF_M[2] + REF_S[2] * a), choice, s)

@testset "eDDMExact - Gauss-Legendre unit rule" begin
    for n in (2, 4, 8, 16, 24)
        x, logw = DE.gauss_legendre_unit(n)
        w = exp.(logw)
        @test length(x) == n
        @test issorted(x)
        @test all(0 .< x .< 1)
        @test all(w .> 0)
        @test sum(w) ≈ 1.0 atol = 1e-13
        for d in 0:min(2n - 1, 9)
            @test sum(w .* x .^ d) ≈ 1 / (d + 1) atol = 1e-12
        end
    end
    @test_throws ArgumentError DE.gauss_legendre_unit(1)
end

@testset "eDDMExact - tau_interval" begin
    r = DE.TauRule(16)
    @test r.L > 0
    @test length(r.t) == DE.TAU_TAIL_NODES

    for a in (-30.0, -19.0, -10.0, -4.0, -2.0, -0.5, 0.0, 0.5, 3.0, 10.0, 40.0)
        lo, hi = DE.tau_interval(a, r)
        @test lo < hi                       # never degenerate: this is the fix
        @test hi <= a + 1e-12               # never integrates a zero density
        @test isfinite(lo) && isfinite(hi)
    end

    for a in (-6.0, -4.0, -2.0, 0.0, 2.0, 5.0, 5.6, 6.0)
        l1, h1 = DE.tau_interval(a - 1e-6, r)
        l2, h2 = DE.tau_interval(a + 1e-6, r)
        @test abs(l1 - l2) < 1e-4
        @test abs(h1 - h2) < 1e-4
    end

    lo, hi = DE.tau_interval(-19.0, r)
    @test hi == -19.0
    @test 0 < hi - lo < 3.0

    for a in (-19.0, -6.0, -1.0, 0.0, 4.0)
        tn = DE.TauNodes(exp(REF_M[2] + REF_S[2] * a), REF_M[2], REF_S[2], r)
        ntail = DE.TAU_TAIL_NODES
        nmain = length(tn) - ntail
        tail_mass = sum(exp(tn[j][2]) for j in (nmain+1):length(tn))
        @test tail_mass ≈ exp(normlogcdf(tn.z_lo)) rtol = 1e-6
        total = sum(exp(tn[j][2]) for j in 1:length(tn))
        @test total ≈ exp(normlogcdf(min(a, -tn.z_lo))) rtol = 1e-6
        @test all(tn[j][1] < a + 1e-12 for j in 1:length(tn))
    end
end

@testset "eDDMExact - accuracy deep in the truncated tail" begin
    σ1 = [1e-7, REF_S[2], 1e-7, 1e-7]

    for a in (2.0, -2.0, -4.0, -6.0, -10.0)
        for choice in (1, -1)
            y = tail_trial(a; choice=choice)
            ref = tau_dense_reference(REF_M, σ1, y; qin=0, K=1200, ng=12)
            ref2 = tau_dense_reference(REF_M, σ1, y; qin=0, K=2000, ng=14)
            @test isapprox(ref, ref2; atol=1e-10)     # the reference is converged

            e16 = marginal_loglik([y], REF_M, log.(σ1); q=2, qτ=16) - ref2
            e32 = marginal_loglik([y], REF_M, log.(σ1); q=2, qτ=32) - ref2

            @test isfinite(e16) && isfinite(e32)
            @test abs(e16) < 2e-3
            @test abs(e32) < 1e-4
            @test marginal_loglik([y], REF_M, log.(σ1); q=2, qτ=16) >
                  DE.LOG_ZERO_MARGINAL / 2
        end
    end

    for a in (-2.0, -6.0, -10.0)
        y = tail_trial(a)
        ref = tau_dense_reference(REF_M, REF_S, y; qin=6, K=400, ng=10)
        @test abs(marginal_loglik([y], REF_M, log.(REF_S); q=6, qτ=32) - ref) < 5e-3
    end
end

@testset "eDDMExact - AD gradients in the deep tail" begin
    rng = MersenneTwister(2024)
    mid = [sample_trial_exact(rng, 1.2, 1.5, 0.5, 0.15) for _ in 1:6]
    tail = [tail_trial(a) for a in (-2.0, -6.0, -12.0)]
    rules = DE.quadrature_rules(4, 16)

    for data in (mid, tail, vcat(mid, tail))
        for θ0 in (vcat(REF_M, log.(REF_S)),
                   vcat(REF_M .+ [0.2, -0.35, 0.1, -0.2], log.(REF_S) .+ 0.3))
            f = θ -> -marginal_loglik(data, collect(θ[1:4]), collect(θ[5:8]), rules)
            g_ad = ForwardDiff.gradient(f, θ0)
            g_fd = FiniteDiff.finite_difference_gradient(f, θ0)
            @test all(isfinite, g_ad)
            @test isfinite(f(θ0))
            @test isapprox(g_ad, g_fd; rtol=1e-4, atol=1e-4)
        end
    end
end

@testset "eDDMExact - refinement stays monotone with tail trials present" begin
    rng = MersenneTwister(99)
    data = vcat([sample_trial_exact(rng, 1.2, 1.5, 0.5, 0.15) for _ in 1:20],
                [tail_trial(a) for a in (-3.0, -7.0, -11.0)])

    rows = quadrature_check(data, REF_M, REF_S; qs=(4, 6, 8, 10), qτ_mult=3)
    @test all(isfinite(r.loglik) for r in rows)
    @test rows[end].loglik / length(data) > -50.0
    for i in 3:length(rows)
        @test abs(rows[i].delta_per_trial) <= abs(rows[i-1].delta_per_trial) + 1e-9
    end
end
