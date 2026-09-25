#=
common.jl — shared helpers for the IBL omission-HMM pipeline.

Included by fit_omission_hmm.jl (fitting) and collect_results.jl (aggregation),
so data loading, initialisation and output formats stay identical between the
two. Not a module; include() after `using DriftDiffusionModels, ...`.
=#

# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────

_isnum(x) = !ismissing(x) && !(x isa AbstractString) && isfinite(x)

"""
Rank subjects in `dir` by number of trials (descending). Returns `(subject, n)` pairs.
"""
function rank_subjects(dir::AbstractString)
    files = filter(f -> endswith(f, ".csv"), readdir(dir))
    counts = [(splitext(f)[1], countlines(joinpath(dir, f)) - 1) for f in files]
    sort!(counts; by = last, rev = true)
    return counts
end

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
Deterministic per-restart RNG seed: distinct across (subject, K, restart) but
reproducible, so an array task can be rerun and give the same fit.
"""
restart_seed(subject::AbstractString, K::Int, r::Int, seed::Int) =
    Int(hash((subject, K, r, seed)) % Int64(2)^40)

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
# Outputs (formats shared by fitting and collection)
# ──────────────────────────────────────────────────────────────────────────────

"Per-state parameters, initial distribution and transition matrix as a DataFrame."
function params_frame(hmm)
    Kt = length(hmm.dists)
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
    return prm
end

"Per-trial Viterbi state and posterior state probabilities as a DataFrame."
function trials_frame(hmm, obs, seq_ends, df)
    Kt = length(hmm.dists)
    states = viterbi(hmm, obs; seq_ends = seq_ends)[1]
    γ = forward_backward(hmm, obs; seq_ends = seq_ends)[1]
    tr = DataFrame(eid = df.eid, day_n = df.day_n, session_n = df.session_n, trial_n = df.trial_n,
                   rt = [x.rt for x in obs], choice = [x.choice for x in obs],
                   s = [x.s for x in obs], c = [x.c for x in obs],
                   omission = [is_omission(x) for x in obs], viterbi_state = states)
    for j in 1:Kt
        tr[!, "p_state_$j"] = γ[j, :]
    end
    return tr
end
