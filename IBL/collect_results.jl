#=
collect_results.jl — aggregate finished omission-HMM restarts into shareable
results. Safe (and intended) to run WHILE array-job fits are still running: it
processes whatever restarts have finished so far and is idempotent, so rerun it
whenever you want a refreshed snapshot.

Usage (from the repository root):

    julia --project=IBL IBL/collect_results.jl [options]

Options
  --out DIR        results directory written by fit_omission_hmm.jl (default IBL/results)
  --data DIR       trials directory (default IBL/IBL-NM_trials/trials)
  --no-decode      skip the per-trial Viterbi/posterior CSVs (fast summary-only pass)

Reads  <out>/<subject>/restarts/summary_K<k>_r<r>.csv (+ hmm_K<k>_r<r>.jls) and writes:

  <out>/all_restarts.csv          every finished restart of every subject, one row each
  <out>/model_comparison.csv      best restart per (subject, K): loglik, AIC, BIC, …
  <out>/<subject>/summary_K<k>.csv   all finished restarts of that (subject, K)
  <out>/<subject>/params_K<k>.csv    best restart: per-state B,k,α,a₀,τ + init + transition matrix
  <out>/<subject>/trials_K<k>.csv    best restart: per-trial Viterbi state + posterior state probs
  <out>/<subject>/hmm_K<k>.jls       best restart's serialised PriorHMM (deserialize(...).hmm)

Best = highest `score` (log-posterior for MAP fits, log-likelihood for --mle).
Per-trial decoding is only recomputed when the best restart changed since the
last collection, so incremental reruns are cheap.
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

include(joinpath(@__DIR__, "common.jl"))

function parse_args(args)
    opts = Dict{String,Any}(
        "out"  => joinpath(@__DIR__, "results"),
        "data" => joinpath(@__DIR__, "IBL-NM_trials", "trials"),
        "decode" => true,
    )
    i = 1
    while i ≤ length(args)
        a = args[i]
        val() = (i += 1; i ≤ length(args) || error("missing value for $a"); args[i])
        if     a == "--out";       opts["out"]  = val()
        elseif a == "--data";      opts["data"] = val()
        elseif a == "--no-decode"; opts["decode"] = false
        else error("unknown option $a")
        end
        i += 1
    end
    return opts
end

function main(args)
    opts = parse_args(args)
    out = opts["out"]
    isdir(out) || error("no results directory at $out")

    all_rows = DataFrame[]
    best_rows = DataFrame[]

    subjects = sort(filter(s -> isdir(joinpath(out, s, "restarts")), readdir(out)))
    for subject in subjects
        rdir = joinpath(out, subject, "restarts")
        sfiles = sort(filter(f -> startswith(f, "summary_K") && endswith(f, ".csv"), readdir(rdir)))
        isempty(sfiles) && continue
        rows = reduce(vcat, [CSV.read(joinpath(rdir, f), DataFrame) for f in sfiles]; cols = :union)
        push!(all_rows, rows)

        for K in sort(unique(rows.K_ddm))
            sub = rows[rows.K_ddm .== K, :]
            sort!(sub, :restart)
            CSV.write(joinpath(out, subject, "summary_K$K.csv"), sub)

            ib = argmax(sub.score)
            best = sub[ib, :]
            hsrc = joinpath(rdir, "hmm_K$(K)_r$(best.restart).jls")
            if !isfile(hsrc)
                @warn "$subject K=$K: best restart $(best.restart) has no $hsrc, skipping decode"
                push!(best_rows, DataFrame(best))
                continue
            end
            push!(best_rows, DataFrame(best))

            # skip re-decoding if the best restart hasn't changed since last collection
            marker = joinpath(out, subject, ".best_K$K")
            stamp = "r$(best.restart)"
            unchanged = isfile(marker) && read(marker, String) == stamp &&
                        isfile(joinpath(out, subject, "params_K$K.csv"))

            bundle = deserialize(hsrc)
            hmm = bundle.hmm
            CSV.write(joinpath(out, subject, "params_K$K.csv"), params_frame(hmm))
            cp(hsrc, joinpath(out, subject, "hmm_K$K.jls"); force = true)

            if opts["decode"] && !unchanged
                obs, seq_ends, df = load_subject(joinpath(opts["data"], subject * ".csv");
                                                 rt_max = Float64(best.rt_max))
                tr = trials_frame(hmm, obs, seq_ends, df)
                CSV.write(joinpath(out, subject, "trials_K$K.csv"), tr)
                write(marker, stamp)
            end
            @printf("%-12s K=%d  %2d restarts done  best r%-2d  loglik=%.2f  BIC=%.1f%s\n",
                    subject, K, nrow(sub), best.restart, best.loglik, best.BIC,
                    unchanged ? "  (decode cached)" : "")
        end
    end

    isempty(all_rows) && (println("no finished restarts found under $out"); return)

    allr = reduce(vcat, all_rows; cols = :union)
    sort!(allr, [:subject, :K_ddm, :restart])
    CSV.write(joinpath(out, "all_restarts.csv"), allr)

    cmp = reduce(vcat, best_rows; cols = :union)
    sort!(cmp, [:subject, :K_ddm])
    CSV.write(joinpath(out, "model_comparison.csv"), cmp)

    println("\n$(nrow(allr)) finished restarts across $(length(subjects)) subjects")
    println("→ $(joinpath(out, "all_restarts.csv"))")
    println("→ $(joinpath(out, "model_comparison.csv"))")
end

main(ARGS)
