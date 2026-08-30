#!/usr/bin/env julia
# Validation suite for the multilevel-DDM variational inference scheme.
# Usage: julia -t auto --project=experiments/vi_validation \
#   experiments/vi_validation/run_experiment.jl [stage] [options]
# Stages: recovery, acf, ksens, refpost, fullcov, exact, all.
# Options:  --preset=quick|full   --seeds=N   --out=DIR

using Printf
using Random
using Statistics
using LinearAlgebra

include(joinpath(@__DIR__, "VIValidation.jl"))
using .VIValidation
using DriftDiffusionModels
const DDMs = DriftDiffusionModels

# Configuration

stage = length(ARGS) >= 1 && !startswith(ARGS[1], "--") ? ARGS[1] : "all"
opt(name, default) = begin
    hit = findfirst(a -> startswith(a, "--$name="), ARGS)
    hit === nothing ? default : split(ARGS[hit], "=", limit=2)[2]
end

preset = opt("preset", "quick")
outdir = opt("out", joinpath(@__DIR__, "results"))
mkpath(outdir)

const QUICK = preset == "quick"

const N_SWEEP     = QUICK ? [250, 1000]        : [250, 1000, 4000]
const N_ITER      = QUICK ? 20                 : 40
const N_SEEDS     = parse(Int, opt("seeds", QUICK ? "3" : "5"))
const N_ACF       = QUICK ? 1500               : 4000
const MAXLAG      = 30
const K_SWEEP     = QUICK ? [3, 10, 50]        : [3, 10, 50, 200]
const N_REFPOST   = QUICK ? 16                 : 48
const REF_NGRID   = QUICK ? 21                 : 27
const N_FULLCOV   = QUICK ? 600                : 1500

const INIT_SCALES = QUICK ? [0.5, 1.0, 2.0]     : [0.25, 0.5, 1.0, 2.0, 4.0]

const TRUTH = HyperTruth()   # rat-like: B=1.2, τ=0.15, v=1.5, a₀=0.5

init_hyper(scale::Real=2.0) = DDMs.DDMHyper(TRUTH.m .+ [0.3, -0.3, 0.3, 0.3],
                                            log.(TRUTH.σ0 .* scale))

const LOG = IOBuffer()
function say(fmt::AbstractString, args...)
    s = isempty(args) ? String(fmt) : Printf.format(Printf.Format(fmt), args...)
    println(s)
    println(LOG, s)
    flush(stdout)
end

say("=" ^ 78)
say("Multilevel-DDM variational inference validation")
say("preset=$preset  stage=$stage  threads=$(Threads.nthreads())  seeds=$N_SEEDS")
say("ground truth (u-space, order B τ v a₀):")
say("  m  = $(round.(TRUTH.m; digits=4))")
say("  σ0 = $(round.(TRUTH.σ0; digits=4))")
say("=" ^ 78)


# Stage 1 — hyperparameter recovery

function stage_recovery()
    say("
### Stage: recovery — can the coordinate-ascent scheme recover (m, σ0)?
")
    rows = Any[]
    trace_rows = Any[]

    # NaN selects the package default initialization.
    for N in N_SWEEP, scale in vcat(INIT_SCALES, NaN), seed in 1:N_SEEDS
        rng = MersenneTwister(1000 + seed)
        data, U = simulate_iid(rng, TRUTH, N)

        h0 = isnan(scale) ? package_init_hyper(data) : init_hyper(scale)
        init_σ0 = exp.(h0.logσ)
        label = isnan(scale) ? "pkg-default" : @sprintf("%.2fx", scale)

        t0 = time()
        hyper, qs, trace, dg = run_vi(data; hyper0=h0, n_iter=N_ITER, K=3, rng=rng)
        el = time() - t0

        r = summarize_recovery(hyper, TRUTH)
        emp_sd = vec(std(U; dims=2))   # realised spread in this finite sample

        for d in 1:4
            push!(rows, (N, label, seed, PARAM_NAMES[d],
                         r.m_true[d], r.m_est[d], r.m_bias[d], r.m_bias_in_σ0[d],
                         r.σ0_true[d], emp_sd[d], init_σ0[d],
                         r.σ0_est[d], r.σ0_ratio[d],
                         r.σ0_est[d] / init_σ0[d]))
        end
        for t in trace
            push!(trace_rows, (N, label, seed, t.iter,
                               t.elbo_estep, t.elbo_mstep, t.stuck,
                               t.m[1], t.m[2], t.m[3], t.m[4],
                               t.σ0[1], t.σ0[2], t.σ0[3], t.σ0[4]))
        end

        to_truth = mean(abs.(log.(r.σ0_est ./ TRUTH.σ0)))
        to_init  = mean(abs.(log.(r.σ0_est ./ init_σ0)))
        diverged = any(!isfinite, r.σ0_est) || maximum(r.σ0_ratio) > 20

        say("  N=%-5d init=%-11s seed=%d  %5.1fs  stuck=%d/%d%s", N, label, seed,
            el, dg.stuck_total, N * N_ITER, diverged ? "   *** DIVERGED ***" : "")
        say("        |m bias|/σ0 = %s", string(round.(abs.(r.m_bias_in_σ0); digits=3)))
        say("        σ0 est/true = %s   (log-dist to truth %.3f vs to init %.3f → %s)",
            string(round.(r.σ0_ratio; digits=3)), to_truth, to_init,
            to_init < to_truth ? "TRACKS INIT" : "tracks truth")

        els_E = [t.elbo_estep for t in trace]
        els_M = [t.elbo_mstep for t in trace]
        mono = all(diff(els_M) .>= -1e-6)
        say("        ELBO %.1f → %.1f  (monotone: %s)", els_M[1], els_M[end],
            mono ? "yes" : "no")
    end

    write_csv(joinpath(outdir, "recovery.csv"),
              ["N", "init", "seed", "param", "m_true", "m_est", "m_bias",
               "m_bias_in_sigma0", "sigma0_true", "sigma0_empirical",
               "sigma0_init", "sigma0_est", "sigma0_ratio_to_truth",
               "sigma0_ratio_to_init"], rows)
    write_csv(joinpath(outdir, "recovery_trace.csv"),
              ["N", "init", "seed", "iter", "elbo_after_estep",
               "elbo_after_mstep", "stuck",
               "m_B", "m_tau", "m_v", "m_a0",
               "s_B", "s_tau", "s_v", "s_a0"], trace_rows)
    say("
  → recovery.csv, recovery_trace.csv")
end


# Stage 2 — serial-correlation control

function stage_acf()
    say("
### Stage: acf — serial-correlation controls
")

    rows = Any[]
    summary = Any[]

    for seed in 1:N_SEEDS
        conditions = Tuple{String,Vector{DDMResult},Matrix{Float64}}[]

        rng = MersenneTwister(2000 + seed)
        d_iid, U_iid = simulate_iid(rng, TRUTH, N_ACF)
        push!(conditions, ("iid", d_iid, U_iid))

        rng = MersenneTwister(3000 + seed)
        d_sw, U_sw, _ = simulate_switching(rng, TRUTH, N_ACF)
        push!(conditions, ("switch", d_sw, U_sw))

        perm = shuffle(MersenneTwister(4000 + seed), 1:N_ACF)
        push!(conditions, ("shuffled", d_sw[perm], U_sw[:, perm]))

        for (cond, data, U) in conditions
            rng = MersenneTwister(5000 + seed)
            hyper, qs, trace, dg = run_vi(data; hyper0=init_hyper(1.0),
                                          n_iter=N_ITER, K=3, rng=rng)
            μ = reduce(hcat, [q.μ for q in qs])   # 4 × N

            for d in 1:4
                a_est = acf(view(μ, d, :), MAXLAG)
                a_true = acf(view(U, d, :), MAXLAG)
                lo, hi = acf_null_band(view(μ, d, :), MAXLAG;
                                       nperm=200, rng=MersenneTwister(7000 + seed))
                for l in 1:MAXLAG
                    push!(rows, (seed, cond, PARAM_NAMES[d], l,
                                 a_est[l], a_true[l], lo[l], hi[l]))
                end
                n_out = count(l -> a_est[l] > hi[l] || a_est[l] < lo[l], 1:MAXLAG)
                push!(summary, (seed, cond, PARAM_NAMES[d],
                                a_est[1], a_true[1], mean(a_est[1:10]), n_out))
            end

            m1 = mean(acf(view(μ, d_, :), MAXLAG)[1] for d_ in 1:4)
            t1 = mean(acf(view(U, d_, :), MAXLAG)[1] for d_ in 1:4)
            say("  seed=%d %-9s mean lag-1 ACF: recovered %+.3f   true %+.3f",
                seed, cond, m1, t1)
        end
    end

    write_csv(joinpath(outdir, "acf.csv"),
              ["seed", "condition", "param", "lag", "acf_recovered", "acf_true",
               "null_lo", "null_hi"], rows)
    write_csv(joinpath(outdir, "acf_summary.csv"),
              ["seed", "condition", "param", "lag1_recovered", "lag1_true",
               "mean_acf_lag1_10", "n_lags_outside_null"], summary)
    say("
  → acf.csv, acf_summary.csv")
end


# Stage 3 — Monte-Carlo sample sensitivity

function stage_ksens()
    say("
### Stage: ksens — Monte-Carlo sample sensitivity
")

    rows = Any[]
    N = QUICK ? 800 : 2000
    ev = [randn(MersenneTwister(999), 4) for _ in 1:256]  # common yardstick

    for seed in 1:N_SEEDS
        rng = MersenneTwister(6000 + seed)
        data, U = simulate_iid(rng, TRUTH, N)

        for K in K_SWEEP, fresh in (true, false)
            rng2 = MersenneTwister(6500 + seed)
            t0 = time()
            hyper, qs, trace, dg = run_vi(data; hyper0=init_hyper(1.0),
                                          n_iter=N_ITER, K=K, fresh_eps=fresh,
                                          eval_eps=ev, rng=rng2)
            el = time() - t0
            r = summarize_recovery(hyper, TRUTH)
            μ = reduce(hcat, [q.μ for q in qs])
            σq = reduce(hcat, [exp.(q.logσ) for q in qs])
            lag1 = mean(acf(view(μ, d, :), 1)[1] for d in 1:4)

            for d in 1:4
                push!(rows, (seed, K, fresh, PARAM_NAMES[d],
                             r.m_bias[d], r.m_bias_in_σ0[d], r.σ0_ratio[d],
                             mean(view(σq, d, :)), trace[end].elbo, lag1, el))
            end
            say("  seed=%d K=%-3d fresh_eps=%-5s %5.1fs  ELBO %12.1f  max|m bias|/σ0 %.3f  lag-1 %+.3f",
                seed, K, string(fresh), el, trace[end].elbo,
                maximum(abs.(r.m_bias_in_σ0)), lag1)
        end
    end

    write_csv(joinpath(outdir, "ksens.csv"),
              ["seed", "K", "fresh_eps", "param", "m_bias", "m_bias_in_sigma0",
               "sigma0_ratio", "mean_q_sd", "final_elbo", "lag1_acf", "seconds"], rows)
    say("
  → ksens.csv")
end


# Stage 4 — mean-field vs. reference posterior

function stage_refpost()
    say("
### Stage: refpost — mean-field q_t vs. dense-quadrature posterior
")
    say("  Grid: $(REF_NGRID)^4 = $(REF_NGRID^4) likelihood evaluations per trial.")

    N = QUICK ? 800 : 2000
    rng = MersenneTwister(8000)
    data, U = simulate_iid(rng, TRUTH, N)

    hyper, qs, trace, dg = run_vi(data; hyper0=init_hyper(1.0), n_iter=N_ITER,
                                  K=3, rng=rng)
    qs = refine_qs(data, hyper, qs; M=64, rng=MersenneTwister(8100))

    rows = Any[]
    corr_rows = Any[]
    idx = round.(Int, range(1, N; length=N_REFPOST))

    for i in idx
        μ_ref, Σ_ref, edge = reference_posterior(data[i], hyper;
                                                 ngrid=REF_NGRID, span=5.0)
        sd_ref = sqrt.(diag(Σ_ref))
        C = Σ_ref ./ (sd_ref * sd_ref')

        q = qs[i]
        σ_q = exp.(q.logσ)

        for d in 1:4
            push!(rows, (i, PARAM_NAMES[d], μ_ref[d], q.μ[d],
                         (q.μ[d] - μ_ref[d]) / sd_ref[d],
                         sd_ref[d], σ_q[d], σ_q[d] / sd_ref[d], edge))
        end
        for a in 1:4, b in (a+1):4
            push!(corr_rows, (i, PARAM_NAMES[a], PARAM_NAMES[b], C[a, b], edge))
        end
    end

    say("
  per-dimension, averaged over $(length(idx)) trials:")
    say("    %-4s  %9s  %9s", "dim", "mean err", "sd ratio")
    for d in 1:4
        errs = [r[5] for r in rows if r[2] == PARAM_NAMES[d]]
        rats = [r[8] for r in rows if r[2] == PARAM_NAMES[d]]
        say("    %-4s  %+9.3f  %9.3f", PARAM_NAMES[d], mean(errs), mean(rats))
    end
    say("
  mean error is in units of the exact posterior sd.
")

    say("  exact posterior correlations the diagonal q cannot represent:")
    for a in 1:4, b in (a+1):4
        cs = [r[4] for r in corr_rows if r[2] == PARAM_NAMES[a] && r[3] == PARAM_NAMES[b]]
        say("    %-3s–%-3s  mean %+0.3f   max|.| %0.3f",
            PARAM_NAMES[a], PARAM_NAMES[b], mean(cs), maximum(abs.(cs)))
    end

    maxedge = maximum(r[9] for r in rows)
    say("
  max grid-boundary mass = %.2e %s", maxedge,
        maxedge > 1e-3 ? "(TRUNCATED — rerun with a wider span)" : "(grid is wide enough)")

    write_csv(joinpath(outdir, "refpost.csv"),
              ["trial", "param", "mu_reference", "mu_q", "err_in_ref_sd",
               "sd_reference", "sd_q", "sd_ratio", "edge_mass"], rows)
    write_csv(joinpath(outdir, "refpost_corr.csv"),
              ["trial", "param_a", "param_b", "correlation", "edge_mass"], corr_rows)
    say("
  → refpost.csv, refpost_corr.csv")
end


# Stage 5 — mean-field vs. full covariance

function stage_fullcov()
    say("
### Stage: fullcov — does relaxing mean-field change any conclusion?
")

    rows = Any[]
    for seed in 1:N_SEEDS
        rng = MersenneTwister(9000 + seed)
        data, U = simulate_switching(rng, TRUTH, N_FULLCOV)

        rngd = MersenneTwister(9500 + seed)
        h_d, qs_d, tr_d, _ = run_vi(data; hyper0=init_hyper(1.0), n_iter=N_ITER,
                                    K=3, rng=rngd)
        μ_d = reduce(hcat, [q.μ for q in qs_d])

        rngf = MersenneTwister(9500 + seed)
        h_f, Θ_f, μ_f, tr_f = run_vi_fullcov(data; hyper0=init_hyper(1.0),
                                             n_iter=N_ITER, K=3, rng=rngf)

        r_d = summarize_recovery(h_d, TRUTH)
        r_f = summarize_recovery(h_f, TRUTH)
        for d in 1:4
            push!(rows, (seed, PARAM_NAMES[d],
                         r_d.m_bias[d], r_f.m_bias[d],
                         r_d.σ0_ratio[d], r_f.σ0_ratio[d],
                         acf(view(μ_d, d, :), 1)[1], acf(view(μ_f, d, :), 1)[1],
                         acf(view(U, d, :), 1)[1]))
        end

        say("  seed=%d  max|m bias|: diag %.3f  full %.3f", seed,
            maximum(abs.(r_d.m_bias)), maximum(abs.(r_f.m_bias)))
        say("           mean lag-1 ACF: diag %+.3f  full %+.3f  true %+.3f",
            mean(acf(view(μ_d, d, :), 1)[1] for d in 1:4),
            mean(acf(view(μ_f, d, :), 1)[1] for d in 1:4),
            mean(acf(view(U, d, :), 1)[1] for d in 1:4))
    end

    write_csv(joinpath(outdir, "fullcov.csv"),
              ["seed", "param", "m_bias_diag", "m_bias_full",
               "sigma0_ratio_diag", "sigma0_ratio_full",
               "lag1_diag", "lag1_full", "lag1_true"], rows)
    say("
  → fullcov.csv")
end



# Stage 6 — variational vs. exact marginal likelihood

function stage_exact()
    say("
### Stage: exact — variational vs. exact marginal likelihood
")
    rows = Any[]
    N = QUICK ? 600 : 2000
    q = QUICK ? 6 : 8

    for seed in 1:N_SEEDS
        rng = MersenneTwister(11000 + seed)
        data, U = simulate_iid(rng, TRUTH, N)

        pkg = package_init_hyper(data)
        inits = [("truth-scale", init_hyper(1.0)), ("pkg-default", pkg)]

        for (lbl, h0) in inits
            hv, qsv, trv, dgv = run_vi(data; hyper0=h0, n_iter=N_ITER, K=3,
                                       rng=MersenneTwister(11500 + seed))
            rv = summarize_recovery(hv, TRUTH)

            fe = fit_mlddm_exact(data; q=q, qτ=2q, n_starts=2,
                                 rng=MersenneTwister(11500 + seed),
                                 init=(h0.m, exp.(h0.logσ)), verbose=false)
            re_ratio = fe.σ0 ./ TRUTH.σ0
            re_bias = (fe.m .- TRUTH.m) ./ TRUTH.σ0

            for d in 1:4
                push!(rows, (seed, lbl, PARAM_NAMES[d],
                             rv.σ0_ratio[d], re_ratio[d],
                             rv.m_bias_in_σ0[d], re_bias[d],
                             trv[end].elbo_mstep, fe.loglik,
                             fe.converged, fe.boundary[d], fe.n_zero))
            end

            say("  seed=%d init=%-11s  σ0 est/true", seed, lbl)
            say("        VI    %s", string(round.(rv.σ0_ratio; digits=3)))
            say("        exact %s   conv=%s%s", string(round.(re_ratio; digits=3)),
                string(fe.converged),
                any(fe.boundary) ? "  boundary: " * join(PARAM_NAMES[fe.boundary], ",") : "")
        end

        # Exclude collapsed variance components from initialization comparisons.
        a = [r for r in rows if r[1] == seed && r[2] == "truth-scale"]
        b = [r for r in rows if r[1] == seed && r[2] == "pkg-default"]
        for (est, col) in (("VI", 4), ("exact", 5))
            live = [d for d in 1:4 if !(a[d][11] || b[d][11])]
            if isempty(live)
                say("        %-5s all σ0 components at boundary", est)
            else
                spread = maximum(abs(log(a[d][col] / b[d][col])) for d in live)
                say("        %-5s max |log σ0 ratio| between the two inits: %.3f%s",
                    est, spread,
                    length(live) < 4 ?
                        "  (over $(length(live))/4 non-boundary components)" : "")
            end
        end
        bnd = [PARAM_NAMES[d] for d in 1:4 if a[d][11]]
        !isempty(bnd) && say("        exact σ0 at boundary: %s — run profile_sigma0",
                             join(bnd, ", "))
    end

    write_csv(joinpath(outdir, "exact_vs_vi.csv"),
              ["seed", "init", "param", "sigma0_ratio_vi", "sigma0_ratio_exact",
               "m_bias_vi", "m_bias_exact", "final_elbo_vi", "loglik_exact",
               "exact_converged", "exact_boundary", "exact_n_zero"], rows)
    say("
  → exact_vs_vi.csv")
end

# Driver

t_start = time()
stage in ("recovery", "all") && stage_recovery()
stage in ("acf", "all")      && stage_acf()
stage in ("ksens", "all")    && stage_ksens()
stage in ("refpost", "all")  && stage_refpost()
stage in ("fullcov", "all")  && stage_fullcov()
stage in ("exact", "all")    && stage_exact()

say("
" * "=" ^ 78)
say("done in %.1f s — results in %s", time() - t_start, outdir)
open(joinpath(outdir, "summary.txt"), "w") do io
    write(io, String(take!(LOG)))
end
