const OMR = OmissionCoherentDDMResult

@testset "Construction and type stability" begin
    d  = OmissionCoherentDDM(B=3.0, k=2.0, α=1.2, a₀=0.5, τ=0.1)
    om = omission_state()
    @test d isa OmissionCoherentDDM{Float64}
    @test om isa OmissionCoherentDDM{Float64}
    @test typeof(d) == typeof(om)                     # homogeneous emission vector
    @test !is_omission_state(d) && is_omission_state(om)
    @test d.rt_max == 60.0
    @test OmissionCoherentDDM(CoherentDDM(); rt_max=10).rt_max == 10.0

    # Wrapping keeps the same (mutable) CoherentDDM object
    c = CoherentDDM(B=2.0, k=1.0)
    w = OmissionCoherentDDM(c)
    c.B = 7.0
    @test w.ddm.B == 7.0

    # Dual-typed placeholder for the omission state
    omd = omission_state(ForwardDiff.Dual{Nothing,Float64,2})
    @test omd isa OmissionCoherentDDM{ForwardDiff.Dual{Nothing,Float64,2}}

    x  = OMR(rt=0.5, choice=1, s=1, c=0.5)
    xo = OMR(rt=60.0, choice=0, s=-1, c=0.25)
    @test !is_omission(x) && is_omission(xo)
    @test CoherentDDMResult(x) == CoherentDDMResult(0.5, 1, 1, 0.5)
    @test OMR(CoherentDDMResult(0.5, 1, 1, 0.5)) == x
    @test_throws ArgumentError CoherentDDMResult(xo)

    # Log-density return type is the model's numeric type
    @test @inferred(DensityInterface.logdensityof(d, x))  isa Float64
    @test @inferred(DensityInterface.logdensityof(d, xo)) isa Float64
    @test @inferred(DensityInterface.logdensityof(om, x)) isa Float64
end

@testset "Deterministic omission emission" begin
    d  = OmissionCoherentDDM(B=3.0, k=2.0, α=1.0, a₀=0.5, τ=0.1)
    om = omission_state()
    x  = OMR(rt=0.5, choice=1, s=1, c=0.5)
    xo = OMR(rt=60.0, choice=0, s=1, c=0.5)

    # omission state: probability one on omissions, floor elsewhere
    @test DensityInterface.logdensityof(om, xo) == 0.0
    @test DensityInterface.logdensityof(om, x)  == OMISSION_LOGFLOOR
    @test OMISSION_LOGFLOOR == log(1e-16)

    # DDM state: WFPT density on choices, floor on omissions
    ld = DensityInterface.logdensityof(d, x)
    @test ld == logdensityof(d.ddm.B, d.ddm.k, d.ddm.α, d.ddm.a₀, d.ddm.τ, 0.5, 1, 1, 0.5)
    @test DensityInterface.logdensityof(d, xo) == OMISSION_LOGFLOOR
    @test ld > OMISSION_LOGFLOOR

    # Omission state has no parameters: gradient w.r.t. the placeholder is zero
    f(p) = DensityInterface.logdensityof(
        OmissionCoherentDDM(CoherentDDM(p[1], p[2], p[3], p[4], p[5]); rt_max=60.0), x)
    g = ForwardDiff.gradient(f, [3.0, 2.0, 1.0, 0.5, 0.1])
    @test all(isfinite, g) && any(abs.(g) .> 0)
    fo(p) = DensityInterface.logdensityof(
        OmissionCoherentDDM{eltype(p)}(CoherentDDM(p[1], p[2], p[3], p[4], p[5]), true, 60.0), xo)
    @test all(iszero, ForwardDiff.gradient(fo, [3.0, 2.0, 1.0, 0.5, 0.1]))
end

@testset "Simulation" begin
    rng = MersenneTwister(5)
    om = omission_state(rt_max=60.0)
    r = simulateDDM(om, 1, 0.5, 1e-4, rng)
    @test r == OMR(60.0, 0, 1, 0.5)
    @test is_omission(rand(rng, om))

    d = OmissionCoherentDDM(B=3.0, k=3.0, α=1.0, a₀=0.5, τ=0.1)
    res = [simulateDDM(d, 1.0, 1e-4, rng) for _ in 1:300]
    @test all(r -> !is_omission(r) && r.rt > 0.1, res)
    @test mean(r.choice == r.s for r in res) > 0.85

    # Censoring: a tiny window turns every DDM trial into an omission
    dc = OmissionCoherentDDM(CoherentDDM(B=3.0, k=0.0, α=1.0, a₀=0.5, τ=0.1); rt_max=0.12)
    rc = [simulateDDM(dc, 1, 0.0, 1e-3, rng) for _ in 1:20]
    @test all(is_omission, rc) && all(r -> r.rt == 0.12, rc)
end

@testset "Packing roundtrip and parameter count" begin
    d1 = OmissionCoherentDDM(B=2.5, k=1.5, α=1.3, a₀=0.45, τ=0.10)
    d2 = OmissionCoherentDDM(B=4.0, k=3.0, α=1.3, a₀=0.55, τ=0.15)
    init  = [0.45, 0.45, 0.1]
    trans = [0.95 0.03 0.02; 0.03 0.95 0.02; 0.1 0.1 0.8]

    for share in (true, false)
        hmm = PriorHMM(init, trans, [d1, omission_state(), d2]; α_trans=2.0, share_α=share)
        θ = pack_omission_hmm(hmm)
        @test length(θ) == n_omission_hmm_params(hmm) == 3 + 9 + (share ? 4 * 2 + 1 : 5 * 2)
        @test omission_flags(hmm) == [false, true, false]

        i2, t2, dists = unpack_omission_hmm(θ, omission_flags(hmm); share_α=share)
        @test isapprox(i2, init; atol=1e-10)
        @test isapprox(t2, trans; atol=1e-10)
        @test dists[2].omission && !dists[1].omission && !dists[3].omission
        for (a, b) in ((dists[1].ddm, d1.ddm), (dists[3].ddm, d2.ddm))
            @test isapprox(a.B, b.B; rtol=1e-10) && isapprox(a.k, b.k; rtol=1e-10)
            @test isapprox(a.α, b.α; rtol=1e-10) && isapprox(a.a₀, b.a₀; rtol=1e-10)
            @test isapprox(a.τ, b.τ; rtol=1e-10)
        end

        # set_omission_hmm! writes through the wrapped mutable DDMs
        h2 = PriorHMM(init, trans, [OmissionCoherentDDM(), omission_state(), OmissionCoherentDDM()];
                      α_trans=2.0, share_α=share)
        set_omission_hmm!(h2, θ)
        @test isapprox(pack_omission_hmm(h2), θ; atol=1e-10)
        @test h2.dists[2].omission
    end
end

@testset "Gradient-descent fit with a deterministic omission state" begin
    rng = MersenneTwister(21)
    K = 3
    d1 = OmissionCoherentDDM(B=2.5, k=1.5, α=1.3, a₀=0.5, τ=0.10)
    d2 = OmissionCoherentDDM(B=4.0, k=3.0, α=1.3, a₀=0.5, τ=0.15)
    truth = PriorHMM([0.45, 0.45, 0.1], [0.95 0.03 0.02; 0.03 0.95 0.02; 0.1 0.1 0.8],
                     [d1, d2, omission_state()]; α_trans=2.0, share_α=true)
    @test HiddenMarkovModels.valid_hmm(truth)

    sim = rand(rng, truth, 800)
    obs = sim.obs_seq
    @test eltype(obs) == OMR
    @test count(is_omission, obs) == count(==(3), sim.state_seq)     # only state 3 omits
    seq_ends = [400, 800]

    hmm = PriorHMM([0.4, 0.4, 0.2], [0.9 0.05 0.05; 0.05 0.9 0.05; 0.2 0.2 0.6],
                   [OmissionCoherentDDM(B=3.0, k=2.0, α=1.0, a₀=0.5, τ=0.08),
                    OmissionCoherentDDM(B=3.5, k=2.5, α=1.0, a₀=0.5, τ=0.12),
                    omission_state()]; α_trans=2.0, share_α=true)

    θ0  = pack_omission_hmm(hmm)
    ll0 = omission_hmm_loglikelihood(θ0, hmm, obs; seq_ends=seq_ends)
    @test isfinite(ll0)
    @test isapprox(ll0, DensityInterface.logdensityof(hmm, obs; seq_ends=seq_ends); rtol=1e-8)

    # Prior term matches the Dirichlet pseudo-count expression
    lp = omission_hmm_logposterior(θ0, hmm, obs; seq_ends=seq_ends)
    prior_term = sum((hmm.α_init .- 1) .* log.(hmm.init)) +
                 sum(sum((hmm.α_trans[j, :] .- 1) .* log.(hmm.trans[j, :])) for j in 1:K)
    @test isapprox(lp - ll0, prior_term; rtol=1e-8)

    # ForwardDiff reaches every DDM parameter; the omission state has none
    g = ForwardDiff.gradient(t -> omission_hmm_loglikelihood(t, hmm, obs; seq_ends=seq_ends), θ0)
    @test length(g) == n_omission_hmm_params(hmm)
    @test all(isfinite, g)
    @test any(abs.(g[(K + K*K + 1):end]) .> 1e-6)
    g_fd = FiniteDiff.finite_difference_gradient(
        t -> omission_hmm_loglikelihood(t, hmm, obs; seq_ends=seq_ends), θ0)
    @test isapprox(g, g_fd; rtol=1e-4, atol=1e-3)

    hmm, res = fit_hmm_gradient!(hmm, obs; seq_ends=seq_ends, prior=false, iterations=150)
    ll1 = omission_hmm_loglikelihood(pack_omission_hmm(hmm), hmm, obs; seq_ends=seq_ends)
    @test ll1 ≥ ll0 - 1e-6
    @test hmm.dists[3].omission                                    # still the omission state
    @test hmm.dists[1].ddm.α == hmm.dists[2].ddm.α                 # shared α preserved
    @test isapprox(sum(hmm.init), 1.0; atol=1e-8)
    @test all(isapprox.(sum(hmm.trans; dims=2), 1.0; atol=1e-8))
    @test HiddenMarkovModels.valid_hmm(hmm)

    # Rough recovery of the DDM-state bounds and stickiness of the omission state
    Bs = sort([hmm.dists[1].ddm.B, hmm.dists[2].ddm.B])
    @test abs(Bs[1] - 2.5) ≤ 1.0 && abs(Bs[2] - 4.0) ≤ 1.2
    @test hmm.trans[3, 3] > 0.5

    # Decoding: every omission trial is assigned to the omission state
    states = HiddenMarkovModels.viterbi(hmm, obs; seq_ends=seq_ends)[1]
    @test all(states[is_omission.(obs)] .== 3)
    @test !any(states[.!is_omission.(obs)] .== 3)

    # MAP fit runs too
    hmap = deepcopy(hmm)
    _, rmap = fit_hmm_gradient!(hmap, obs; seq_ends=seq_ends, prior=true, iterations=20)
    @test isfinite(DriftDiffusionModels.Optim.minimum(rmap))
end

@testset "Baum–Welch M-step skips the omission state" begin
    rng = MersenneTwister(8)
    truth = PriorHMM([0.5, 0.5], [0.95 0.05; 0.1 0.9],
                     [OmissionCoherentDDM(B=3.0, k=2.0, α=1.0, a₀=0.5, τ=0.1), omission_state()];
                     α_trans=2.0)
    obs = rand(rng, truth, 400).obs_seq
    hmm = PriorHMM([0.5, 0.5], [0.9 0.1; 0.2 0.8],
                   [OmissionCoherentDDM(B=4.0, k=1.5, α=1.0, a₀=0.5, τ=0.08, fit_α=false), omission_state()];
                   α_trans=2.0)
    hfit, logl = HiddenMarkovModels.baum_welch(hmm, obs; max_iterations=5)
    @test length(logl) ≥ 2 && logl[end] ≥ logl[1] - 1e-6
    @test hfit.dists[2].omission
    @test abs(hfit.dists[1].ddm.B - 3.0) < 1.5
end
