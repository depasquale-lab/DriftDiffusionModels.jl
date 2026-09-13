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
  --top N              fit the N subjects with the most trials (default 4)
  --task-id i          fit only the i-th subject of the selected list — pair
                       with an SGE array job ($SGE_TASK_ID); default: all.
                       submit_fit.sh maps one array task to one (subject, K) pair
  --K 1,2,3            numbers of DDM states to try (default 1,2,3)
  --restarts R         random restarts per K (default 3)
  --iterations N       L-BFGS iterations per restart (default 300)
  --no-share-alpha     estimate a separate α per DDM state
  --mle                maximise the likelihood instead of the MAP posterior
  --rt-max S           response window in seconds (default 60)
  --min-rt S           drop chosen trials with RT below S (default 0)
  --max-sessions N     use only the first N sessions (debugging)
  --data DIR           trials directory (default IBL/IBL-NM_trials/trials)
  --out DIR            output directory (default IBL/results)
  --seed S             RNG seed (default 1)

Outputs, per subject, in <out>/<subject>/:
  summary_K<k>.csv     one row per restart: loglik, logpost, n_params, AIC, BIC, …
  params_K<k>.csv      fitted emission parameters and transition matrix
  trials_K<k>.csv      per-trial Viterbi state and posterior state probabilities
  hmm_K<k>.jls         serialised PriorHMM (Serialization.deserialize to reload)
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

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

function parse_args(args)
    opts = Dict{String,Any}(
        "subject" => String[], "top" => 4, "task-id" => 0,
        "K" => [1, 2, 3], "restarts" => 3, "iterations" => 300,
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
# Data
# ──────────────────────────────────────────────────────────────────────────────

"""
Rank subjects in `dir` by number of trials (descending). Returns `(subject, n)` pairs.
"""
function rank_subjects(dir::AbstractString)
    files = filter(f -> endswith(f, ".csv"), readdir(dir))
    counts = [(splitext(f)[1], countlines(joinpath(dir, f)) - 1) for f in files]
    sort!(counts; by = last, rev = true)
    return counts
end

_isnum(x) = !ismissing(x) && !(x isa AbstractString) && isfinite(x)

"""
    load_subject(path; rt_max, min_rt, max_sessions)

Read one IBL trials CSV into `(obs, seq_ends, df)`. Encoding:

* stimulus side `s`: contrastLeft present ⇒ -1, contrastRight present ⇒ +1;
* coherence `c`: the presented contrast (0, 0.0625, 0.125, 0.25, 1);
* choice: IBL's `choice` is −1 = CCW / +1 = CW and a correct response to a *left*
  stimulus is +1, so the DDM choice is `-choice_ibl` (then `choice == s` ⇔ correct);
* omission: `choice == 0` (IBL `no_choice`); RT is set to `rt_max`.

Chosen trials with a missing/non-positive RT, or RT below `min_rt`, are dropped.
Each session (`eid`) is one sequence; sessions are ordered by `day_n`, `session_n`.
"""
function load_subject(path::AbstractString; rt_max::Float64 = 60.0, min_rt::Float64 = 0.0,
                      max_sessions::Int = typemax(Int))
    df = CSV.read(path, DataFrame; missingstring = ["", "NA", "nan", "NaN"])
    sort!(df, [:day_n, :session_n, :trial_n])

    obs      = OmissionCoherentDDMResult[]
    seq_ends = Int[]
    keep_idx = Int[]

    eids = unique(df.eid)
    length(eids) > max_sessions && (eids = eids[1:max_sessions])
    for eid in eids
        rows = findall(==(eid), df.eid)
        n_before = length(obs)
        for r in rows
            cl, cr = df.contrastLeft[r], df.contrastRight[r]
            if _isnum(cl)
                s, c = -1, Float64(cl)
            elseif _isnum(cr)
                s, c = +1, Float64(cr)
            else
                continue                                 # no stimulus recorded
            end
            c = clamp(c, 0.0, 1.0)
            ch_ibl = df.choice[r]
            _isnum(ch_ibl) || continue
            ch = Int(round(ch_ibl))
            if ch == 0                                   # omission
                push!(obs, OmissionCoherentDDMResult(rt_max, 0, s, c))
            else
                rt = df.reaction_time[r]
                _isnum(rt) || continue
                rt = Float64(rt)
                (rt > 0 && rt ≥ min_rt) || continue
                if rt ≥ rt_max                           # responded after the window closed
                    push!(obs, OmissionCoherentDDMResult(rt_max, 0, s, c))
                else
                    push!(obs, OmissionCoherentDDMResult(rt, -ch, s, c))
                end
            end
            push!(keep_idx, r)
        end
        length(obs) > n_before && push!(seq_ends, length(obs))
    end
    return obs, seq_ends, df[keep_idx, :]
end

# ──────────────────────────────────────────────────────────────────────────────
# Initialisation
# ──────────────────────────────────────────────────────────────────────────────

"""
Fit one CoherentDDM to (a subsample of) the non-omission trials as a starting point.
"""
function fit_global_coherent(rng, obs; nmax = 6000)
    chosen = [CoherentDDMResult(x) for x in obs if !is_omission(x)]
    length(chosen) > nmax && (chosen = chosen[randperm(rng, length(chosen))[1:nmax]])
    rt_med = median(r.rt for r in chosen)
    rt_min = minimum(r.rt for r in chosen)
    m = CoherentDDM(B = 2.0 * sqrt(rt_med), k = 2.0, α = 1.0, a₀ = 0.5,
                    τ = max(0.5 * rt_min, 1e-3))
    fit!(m, chosen)
    return m
end

"""
Build a `PriorHMM` with `K` perturbed copies of `g` plus one trailing omission state.
"""
function init_omission_hmm(rng, g::CoherentDDM, K::Int; rt_max, share_α, p_omit,
                           stay = 0.95, α_sticky = 10.0, α_offdiag = 1.0, α_init = 2.0)
    Kt = K + 1
    dists = OmissionCoherentDDM{Float64}[]
    for i in 1:K
        # log-normal jitter; spread the bound so states start distinguishable
        B  = g.B  * exp(0.25 * randn(rng)) * (K > 1 ? exp(0.4 * ((i - 1) / (K - 1) - 0.5)) : 1.0)
        k  = g.k  * exp(0.25 * randn(rng))
        τ  = g.τ  * exp(0.10 * randn(rng))
        a₀ = clamp(g.a₀ + 0.05 * randn(rng), 0.2, 0.8)
        push!(dists, OmissionCoherentDDM(CoherentDDM(B, k, g.α, a₀, τ, !share_α); rt_max = rt_max))
    end
    push!(dists, omission_state(; rt_max = rt_max))

    init = fill((1 - p_omit) / K, Kt); init[end] = p_omit
    trans = fill((1 - stay) / max(Kt - 1, 1), Kt, Kt)
    trans[diagind(trans)] .= stay
    trans[end, :] .= (1 - 0.8) / K; trans[end, end] = 0.8      # omission state a bit less sticky
    trans ./= sum(trans; dims = 2)

    αT = fill(α_offdiag, Kt, Kt); αT[diagind(αT)] .= α_sticky
    return PriorHMM(init, trans, dists; α_trans = αT, α_init = fill(α_init, Kt), share_α = share_α)
end

# ──────────────────────────────────────────────────────────────────────────────
# Fitting
# ──────────────────────────────────────────────────────────────────────────────

function fit_subject(subject, opts)
    rt_max = opts["rt-max"]
    outdir = joinpath(opts["out"], subject)
    mkpath(outdir)
    logmsg(msg) = (println("[$(Dates.format(now(), "HH:MM:SS"))] $subject | $msg"); flush(stdout))

    obs, seq_ends, df = load_subject(joinpath(opts["data"], subject * ".csv");
                                     rt_max = rt_max, min_rt = opts["min-rt"],
                                     max_sessions = opts["max-sessions"])
    n_om = count(is_omission, obs)
    logmsg("$(length(obs)) trials, $(length(seq_ends)) sessions, $n_om omissions " *
        @sprintf("(%.2f%%)", 100n_om / length(obs)))

    rng = MersenneTwister(opts["seed"])
    g = fit_global_coherent(rng, obs)
    logmsg(@sprintf("global CoherentDDM: B=%.3f k=%.3f α=%.3f a₀=%.3f τ=%.3f", g.B, g.k, g.α, g.a₀, g.τ))
    p_omit = clamp(n_om / length(obs), 1e-3, 0.5)

    summary = DataFrame(subject = String[], K_ddm = Int[], K_total = Int[], restart = Int[],
                        loglik = Float64[], logpost = Float64[], n_params = Int[],
                        n_trials = Int[], n_sessions = Int[], n_omissions = Int[],
                        AIC = Float64[], BIC = Float64[], converged = Bool[],
                        iterations = Int[], seconds = Float64[])

    for K in opts["K"]
        best = nothing
        for r in 1:opts["restarts"]
            hmm = init_omission_hmm(rng, g, K; rt_max = rt_max, share_α = opts["share-alpha"], p_omit = p_omit)
            t0 = time()
            hmm, res = fit_hmm_gradient!(hmm, obs; seq_ends = seq_ends, prior = opts["prior"],
                                         iterations = opts["iterations"])
            secs = time() - t0
            θ  = pack_omission_hmm(hmm)
            ll = omission_hmm_loglikelihood(θ, hmm, obs; seq_ends = seq_ends)
            lp = omission_hmm_logposterior(θ, hmm, obs; seq_ends = seq_ends)
            np = n_omission_hmm_params(hmm) - 1 - (K + 1)    # softmax redundancy: init and each of the K+1 trans rows lose 1 dof
            n  = length(obs)
            logmsg(@sprintf("K=%d restart %d: loglik=%.2f logpost=%.2f iters=%d converged=%s (%.0fs)",
                         K, r, ll, lp, DDM.Optim.iterations(res), DDM.Optim.converged(res), secs))
            push!(summary, (subject, K, K + 1, r, ll, lp, np, n, length(seq_ends), n_om,
                            2np - 2ll, np * log(n) - 2ll, DDM.Optim.converged(res),
                            DDM.Optim.iterations(res), secs))
            score = opts["prior"] ? lp : ll
            if best === nothing || score > best.score
                best = (; hmm = deepcopy(hmm), score, ll, lp, restart = r)
            end
        end

        hmm = best.hmm
        logmsg(@sprintf("K=%d best restart %d: loglik=%.2f", K, best.restart, best.ll))
        for (i, m) in enumerate(hmm.dists)
            m.omission && (logmsg("  state $i: omission (deterministic)"); continue)
            d = m.ddm
            logmsg(@sprintf("  state %d: B=%.3f k=%.3f α=%.3f a₀=%.3f τ=%.3f", i, d.B, d.k, d.α, d.a₀, d.τ))
        end

        # parameters
        Kt = K + 1
        prm = DataFrame(state = 1:Kt,
                        kind  = [m.omission ? "omission" : "ddm" for m in hmm.dists],
                        B  = [m.omission ? NaN : m.ddm.B  for m in hmm.dists],
                        k  = [m.omission ? NaN : m.ddm.k  for m in hmm.dists],
                        α  = [m.omission ? NaN : m.ddm.α  for m in hmm.dists],
                        a₀ = [m.omission ? NaN : m.ddm.a₀ for m in hmm.dists],
                        τ  = [m.omission ? NaN : m.ddm.τ  for m in hmm.dists],
                        init = hmm.init)
        for j in 1:Kt
            prm[!, "trans_to_$j"] = hmm.trans[:, j]
        end
        CSV.write(joinpath(outdir, "params_K$K.csv"), prm)

        # per-trial decoding
        states = viterbi(hmm, obs; seq_ends = seq_ends)[1]
        γ = forward_backward(hmm, obs; seq_ends = seq_ends)[1]
        tr = DataFrame(eid = df.eid, day_n = df.day_n, session_n = df.session_n, trial_n = df.trial_n,
                       rt = [x.rt for x in obs], choice = [x.choice for x in obs],
                       s = [x.s for x in obs], c = [x.c for x in obs],
                       omission = [is_omission(x) for x in obs], viterbi_state = states)
        for j in 1:Kt
            tr[!, "p_state_$j"] = γ[j, :]
        end
        CSV.write(joinpath(outdir, "trials_K$K.csv"), tr)

        serialize(joinpath(outdir, "hmm_K$K.jls"), (; hmm, seq_ends, subject, K, opts = Dict(opts)))
        # one summary file per K so array-job tasks fitting different K never clobber each other
        CSV.write(joinpath(outdir, "summary_K$K.csv"), summary[summary.K_ddm .== K, :])
    end
    logmsg("done → $outdir")
    return summary
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
    println("Fitting: ", join(subjects, ", "), " | K = ", opts["K"], " | restarts = ", opts["restarts"],
            " | share_α = ", opts["share-alpha"], " | ", opts["prior"] ? "MAP" : "MLE")
    for s in subjects
        fit_subject(s, opts)
    end
end

main(ARGS)
