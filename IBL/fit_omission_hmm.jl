#=
fit_omission_hmm.jl — fit omission-aware CoherentDDM HMMs to IBL trial data by
direct gradient descent (ForwardDiff through the HMM forward algorithm).

Model: K DDM states (CoherentDDM emissions, drift = s·k·c^α, one α shared across
states by default) plus ONE deterministic omission state that emits "no choice"
with probability one. Each session is one HMM sequence.

Usage (from the repository root):

    julia --project=IBL -t 8 IBL/fit_omission_hmm.jl [options]

Options
  --subject ID[,ID…]   fit these subject(s) (CSV basename without extension)
  --top N              fit the N subjects with the most trials (default 6)
  --task-id i          fit only the i-th subject of the selected list — pair
                       with an SGE array job; default: all subjects.
                       submit_fit.sh maps one array task to one (subject, K, restart)
  --K 3,4,5            numbers of DDM states to try (default 3,4,5; the HMM has
                       one extra deterministic omission state on top of K)
  --restarts R         random restarts per K (default 10)
  --restart-id r       run ONLY restart r (1-based) — for array jobs where each
                       task is one restart; default 0 = run all R restarts
  --iterations N       L-BFGS iterations per restart (default 300)
  --no-share-alpha     estimate a separate α per DDM state
  --mle                maximise the likelihood instead of the MAP posterior
  --rt-max S           response window in seconds (default 60)
  --min-rt S           drop chosen trials with RT below S (default 0)
  --max-sessions N     use only the first N sessions (debugging)
  --data DIR           trials directory (default IBL/IBL-NM_trials/trials)
  --out DIR            output directory (default IBL/results)
  --seed S             base RNG seed (default 1); each (subject, K, restart)
                       derives its own reproducible seed from it

Every restart writes its outputs the moment it finishes, into
<out>/<subject>/restarts/:
  summary_K<k>_r<r>.csv   one row: loglik, logpost, n_params, AIC, BIC, seed, …
  hmm_K<k>_r<r>.jls       the fitted PriorHMM of that restart

Run IBL/collect_results.jl at any time (also while fits are still running) to
aggregate finished restarts, pick the best per (subject, K) and write the
shareable params / posteriors / model-comparison CSVs.
=#

using Pkg
Pkg.activate(joinpath(@__DIR__), io=devnull)

using CSV
using DataFrames
using DriftDiffusionModels
using HiddenMarkovModels
using Random
using Statistics
using Serialization
using Printf
using Dates
using LinearAlgebra: diagind

const DDM = DriftDiffusionModels

include(joinpath(@__DIR__, "common.jl"))

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

function parse_args(args)
    opts = Dict{String,Any}(
        "subject" => String[], "top" => 6, "task-id" => 0,
        "K" => [3, 4, 5], "restarts" => 10, "restart-id" => 0, "iterations" => 300,
        "share-alpha" => true, "prior" => true,
        "rt-max" => 60.0, "min-rt" => 0.0, "max-sessions" => typemax(Int),
        "data" => joinpath(@__DIR__, "IBL-NM_trials", "trials"),
        "out"  => joinpath(@__DIR__, "results"),
        "seed" => 1,
    )
    i = 1
    while i ≤ length(args)
        a = args[i]
        val() = (i += 1; i ≤ length(args) || error("missing value for $a"); args[i])
        if     a == "--subject";        append!(opts["subject"], split(val(), ","))
        elseif a == "--top";            opts["top"] = parse(Int, val())
        elseif a == "--task-id";        opts["task-id"] = parse(Int, val())
        elseif a == "--K";              opts["K"] = parse.(Int, split(val(), ","))
        elseif a == "--restarts";       opts["restarts"] = parse(Int, val())
        elseif a == "--restart-id";     opts["restart-id"] = parse(Int, val())
        elseif a == "--iterations";     opts["iterations"] = parse(Int, val())
        elseif a == "--no-share-alpha"; opts["share-alpha"] = false
        elseif a == "--mle";            opts["prior"] = false
        elseif a == "--rt-max";         opts["rt-max"] = parse(Float64, val())
        elseif a == "--min-rt";         opts["min-rt"] = parse(Float64, val())
        elseif a == "--max-sessions";   opts["max-sessions"] = parse(Int, val())
        elseif a == "--data";           opts["data"] = val()
        elseif a == "--out";            opts["out"] = val()
        elseif a == "--seed";           opts["seed"] = parse(Int, val())
        else error("unknown option $a")
        end
        i += 1
    end
    return opts
end

# ──────────────────────────────────────────────────────────────────────────────
# Fitting
# ──────────────────────────────────────────────────────────────────────────────

function fit_subject(subject, opts)
    rt_max = opts["rt-max"]
    outdir = joinpath(opts["out"], subject, "restarts")
    mkpath(outdir)
    logmsg(msg) = (println("[$(Dates.format(now(), "HH:MM:SS"))] $subject | $msg"); flush(stdout))

    obs, seq_ends, _ = load_subject(joinpath(opts["data"], subject * ".csv");
                                    rt_max = rt_max, min_rt = opts["min-rt"],
                                    max_sessions = opts["max-sessions"])
    n_om = count(is_omission, obs)
    logmsg("$(length(obs)) trials, $(length(seq_ends)) sessions, $n_om omissions " *
        @sprintf("(%.2f%%)", 100n_om / length(obs)))

    # deterministic (per subject) global starting point, identical across restarts
    g = fit_global_coherent(MersenneTwister(opts["seed"]), obs)
    logmsg(@sprintf("global CoherentDDM: B=%.3f k=%.3f α=%.3f a₀=%.3f τ=%.3f", g.B, g.k, g.α, g.a₀, g.τ))
    p_omit = clamp(n_om / length(obs), 1e-3, 0.5)

    restarts = opts["restart-id"] > 0 ? (opts["restart-id"]:opts["restart-id"]) : (1:opts["restarts"])

    for K in opts["K"], r in restarts
        sfile = joinpath(outdir, "summary_K$(K)_r$(r).csv")
        hfile = joinpath(outdir, "hmm_K$(K)_r$(r).jls")
        # warm-start continuation: a finished restart with fewer iterations than
        # the current target is resumed from its saved model instead of refit
        prev = nothing
        if isfile(sfile) && isfile(hfile)
            prev = CSV.read(sfile, DataFrame)[1, :]
            if prev.iterations ≥ opts["iterations"] || prev.converged
                logmsg("K=$K restart $r already done ($(prev.iterations) iterations, " *
                       "converged=$(prev.converged)), skipping")
                continue
            end
        end
        rseed = restart_seed(subject, K, r, opts["seed"])
        rng = MersenneTwister(rseed)
        if prev === nothing
            hmm = init_omission_hmm(rng, g, K; rt_max = rt_max,
                                    share_α = opts["share-alpha"], p_omit = p_omit)
            done_iters, done_secs = 0, 0.0
        else
            hmm = deserialize(hfile).hmm
            done_iters, done_secs = Int(prev.iterations), Float64(prev.seconds)
            logmsg("K=$K restart $r continuing from $done_iters iterations " *
                   "(+$(opts["iterations"] - done_iters))")
        end
        t0 = time()
        hmm, res = fit_hmm_gradient!(hmm, obs; seq_ends = seq_ends, prior = opts["prior"],
                                     iterations = opts["iterations"] - done_iters)
        secs = time() - t0 + done_secs
        θ  = pack_omission_hmm(hmm)
        ll = omission_hmm_loglikelihood(θ, hmm, obs; seq_ends = seq_ends)
        lp = omission_hmm_logposterior(θ, hmm, obs; seq_ends = seq_ends)
        np = n_omission_hmm_params(hmm) - 1 - (K + 1)    # softmax redundancy: init and each of the K+1 trans rows lose 1 dof
        n  = length(obs)
        tot_iters = done_iters + DDM.Optim.iterations(res)
        logmsg(@sprintf("K=%d restart %d: loglik=%.2f logpost=%.2f iters=%d converged=%s (%.0fs)",
                     K, r, ll, lp, tot_iters, DDM.Optim.converged(res), secs))

        row = DataFrame(subject = subject, K_ddm = K, K_total = K + 1, restart = r,
                        seed = rseed, criterion = opts["prior"] ? "logpost" : "loglik",
                        score = opts["prior"] ? lp : ll,
                        loglik = ll, logpost = lp, n_params = np,
                        n_trials = n, n_sessions = length(seq_ends), n_omissions = n_om,
                        AIC = 2np - 2ll, BIC = np * log(n) - 2ll,
                        converged = DDM.Optim.converged(res),
                        iterations = tot_iters, seconds = secs,
                        share_alpha = opts["share-alpha"], rt_max = rt_max,
                        host = gethostname(), finished = string(now()))
        # write via a temp file so the collector never reads a half-written row
        tmp = sfile * ".tmp"
        CSV.write(tmp, row); mv(tmp, sfile; force = true)
        tmph = hfile * ".tmp"
        serialize(tmph, (; hmm, seq_ends, subject, K, restart = r, seed = rseed, opts = Dict(opts)))
        mv(tmph, hfile; force = true)
        logmsg("K=$K restart $r written → $sfile")
    end
    logmsg("done → $outdir")
end

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

function main(args)
    opts = parse_args(args)
    ranked = rank_subjects(opts["data"])
    subjects = isempty(opts["subject"]) ? first.(ranked[1:min(opts["top"], length(ranked))]) :
                                          String.(opts["subject"])
    if opts["task-id"] > 0
        opts["task-id"] ≤ length(subjects) || error("task-id $(opts["task-id"]) > $(length(subjects)) subjects")
        subjects = subjects[opts["task-id"]:opts["task-id"]]
    end
    println("Julia $(VERSION), $(Threads.nthreads()) threads")
    println("Subjects by trial count: ", join(["$s ($n)" for (s, n) in ranked[1:min(8, length(ranked))]], ", "), " …")
    println("Fitting: ", join(subjects, ", "), " | K = ", opts["K"],
            " | restarts = ", opts["restart-id"] > 0 ? "only #$(opts["restart-id"])" : opts["restarts"],
            " | share_α = ", opts["share-alpha"], " | ", opts["prior"] ? "MAP" : "MLE")
    for s in subjects
        fit_subject(s, opts)
    end
end

main(ARGS)
