
# plots.jl
#
# Live classroom plots for the moving-dots data. Two classics, optionally with
# the fitted CoherentDDM overlaid:
#   * psychometric  — P(choose right) vs *signed* coherence (the S-curve)
#   * chronometric  — mean reaction time vs coherence strength (slower when hard)
#
# Needs Plots:  ] add Plots
#
# Usage:
#   include("outreach/load_class_data.jl")
#   include("outreach/plots.jl")
#   class  = load_class_data("class_responses.csv")   # or load_student_csvs("folder")
#   data   = pool_trials(class)
#   model  = CoherentDDM(); fit!(model, data)
#   plot_summary(data; model=model)                    # both panels, with fit
#   # single panels:  plot_psychometric(data; model) , plot_chronometric(data; model)

using DriftDiffusionModels
using Statistics
using Random
using Plots

# Group by coherence robustly across students (all share the same level set).
# `+ 0.0` collapses negative zero (from s=-1 · c=0) onto a single 0.0 level.
_round(c) = round(c; digits=4) + 0.0

"""
    pright_by_signed_coherence(trials) -> (levels, p_right, n)

Proportion of rightward (+1) choices at each signed coherence `s·c`.
"""
function pright_by_signed_coherence(trials)
    keys_ = sort(unique(_round(t.s * t.c) for t in trials))
    pr = Float64[]; n = Int[]
    for lv in keys_
        sel = [t for t in trials if _round(t.s * t.c) == lv]
        push!(pr, mean(t.choice == 1 for t in sel))
        push!(n, length(sel))
    end
    return keys_, pr, n
end

"""
    accuracy_by_coherence(trials) -> (cohs, acc, n)

Proportion correct at each (unsigned) coherence. Note `c=0` is chance (~0.5).
"""
function accuracy_by_coherence(trials)
    cohs = sort(unique(_round(t.c) for t in trials))
    acc = Float64[]; n = Int[]
    for c in cohs
        sel = [t for t in trials if _round(t.c) == c]
        push!(acc, mean(t.choice == t.s for t in sel))
        push!(n, length(sel))
    end
    return cohs, acc, n
end

"""
    meanrt_by_coherence(trials) -> (cohs, mean_rt, sem)

Mean reaction time at each coherence, with standard error.
"""
function meanrt_by_coherence(trials)
    cohs = sort(unique(_round(t.c) for t in trials))
    mrt = Float64[]; sem = Float64[]
    for c in cohs
        rts = [t.rt for t in trials if _round(t.c) == c]
        push!(mrt, mean(rts))
        push!(sem, length(rts) > 1 ? std(rts) / sqrt(length(rts)) : 0.0)
    end
    return cohs, mrt, sem
end

# Monte-Carlo predictions from a fitted model, matched to the data's design.
function _model_curves(model, trials; reps=400, dt=1e-4, rng=Random.default_rng())
    cohs = sort(unique(_round(t.c) for t in trials))
    signed = Float64[]; pr_pred = Float64[]
    rt_pred = Float64[]
    for c in cohs
        rts = Float64[]
        for s in (-1, 1)
            nright = 0
            for _ in 1:reps
                r = simulateDDM(model, s, Float64(c), dt, rng)
                nright += (r.choice == 1)
                push!(rts, r.rt)
            end
            push!(signed, _round(s * c))
            push!(pr_pred, nright / reps)
        end
        push!(rt_pred, mean(rts))
    end
    order = sortperm(signed)
    return cohs, signed[order], pr_pred[order], rt_pred
end

"""
    plot_psychometric(trials; model=nothing, kwargs...)

P(choose right) vs signed coherence. Pass a fitted `model` to overlay its
prediction (dashed). Extra kwargs pass through to `plot`.
"""
function plot_psychometric(trials; model=nothing, ms=6, kwargs...)
    lv, pr, n = pright_by_signed_coherence(trials)
    plt = scatter(lv, pr; ms=ms, label="data (n/level: $(minimum(n)) to $(maximum(n)))",
        xlabel="signed coherence  (s · c)", ylabel="P(choose right)",
        title="Psychometric", legend=:bottomright, ylim=(-0.02, 1.02),
        xlim=(-1.05*maximum(abs, lv), 1.05*maximum(abs, lv)), kwargs...)
    hline!(plt, [0.5]; ls=:dot, c=:gray, label="")
    vline!(plt, [0.0]; ls=:dot, c=:gray, label="")
    if model !== nothing
        _, slv, prp, _ = _model_curves(model, trials)
        plot!(plt, slv, prp; lw=2, ls=:dash, c=:crimson, label="CoherentDDM fit")
    end
    return plt
end

"""
    plot_chronometric(trials; model=nothing, kwargs...)

Mean reaction time vs coherence strength. Pass a fitted `model` to overlay.
"""
function plot_chronometric(trials; model=nothing, ms=6, kwargs...)
    cohs, mrt, sem = meanrt_by_coherence(trials)
    plt = scatter(cohs, mrt; yerror=sem, ms=ms, label="data",
        xlabel="coherence  c", ylabel="mean RT (s)",
        title="Chronometric", legend=:topright, kwargs...)
    if model !== nothing
        mc, _, _, rtp = _model_curves(model, trials)
        plot!(plt, mc, rtp; lw=2, ls=:dash, c=:crimson, label="CoherentDDM fit")
    end
    return plt
end

"""
    plot_summary(trials; model=nothing) -> Plots.Plot

Both panels side by side — the one-liner for the projector.
"""
function plot_summary(trials; model=nothing)
    p1 = plot_psychometric(trials; model=model)
    p2 = plot_chronometric(trials; model=model)
    return plot(p1, p2; layout=(1, 2), size=(980, 420), margin=4Plots.mm)
end

"""
    plot_student_grid(class, panelfn; title="", ncols=4, ms=3, kwargs...)

Small-multiples grid: run `panelfn` (e.g. `plot_psychometric` or
`plot_chronometric`) on every student, one titled panel each. Axis labels are
dropped and panels are sized generously so the grid doesn't look squished.
"""
function plot_student_grid(class, panelfn; title="", ncols=3, ms=5, kwargs...)
    ids   = sort(collect(keys(class)))
    nrows = cld(length(ids), ncols)
    panels = map(ids) do id
        p = panelfn(class[id]; ms=ms)
        plot!(p; title=id, legend=false, titlefontsize=11,
              xlabel="", ylabel="", tickfontsize=8)
        p
    end
    return plot(panels...; layout=(nrows, ncols), size=(440*ncols, 380*nrows),
                plot_title=title, plot_titlefontsize=14,
                left_margin=4Plots.mm, bottom_margin=4Plots.mm, kwargs...)
end

"""
    plot_accuracy(trials; kwargs...)

The simplest "harder = closer to a coin flip" view: percent correct at each
coherence, as bars. The dashed line is 50% (pure guessing).
"""
function plot_accuracy(trials; kwargs...)
    cohs, acc, _ = accuracy_by_coherence(trials)
    plt = bar(string.(cohs), 100 .* acc; legend=false, c=:steelblue,
        xlabel="coherence  c  (0 = no signal, big = easy)", ylabel="percent correct",
        title="How often were they right?", ylim=(0, 100), kwargs...)
    hline!(plt, [50]; ls=:dash, c=:gray, label="")
    return plt
end

"""
    plot_rt_distributions(trials; kwargs...)

Reaction-time histograms for the hardest vs easiest coherence. Weak evidence →
slower and more spread out; strong evidence → fast and tight. That right-skewed
shape is the fingerprint the drift diffusion model exists to explain.
"""
function plot_rt_distributions(trials; kwargs...)
    cohs = sort(unique(_round(t.c) for t in trials))
    easy, hard = maximum(cohs), minimum(cohs)
    rts_h = [t.rt for t in trials if _round(t.c) == hard]
    rts_e = [t.rt for t in trials if _round(t.c) == easy]
    plt = histogram(rts_h; bins=30, alpha=0.55, c=:firebrick, label="hardest (c = $hard)",
        xlabel="reaction time (s)", ylabel="number of trials",
        title="How long did decisions take?", kwargs...)
    histogram!(plt, rts_e; bins=30, alpha=0.55, c=:seagreen, label="easiest (c = $easy)")
    return plt
end
