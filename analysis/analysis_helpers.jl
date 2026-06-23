# analysis_helpers.jl
#
# Extra analysis + teaching helpers used by class_dots.jl, kept here so they can
# be unit-tested headlessly (see the bottom of this file's PR notes) and reused.
#
#   total_loglik / compare_alpha   — model comparison (does free α earn its keep?)
#   fit_all_students               — per-student parameter table (individual diffs)
#   plot_param_scatter             — boundary B vs accuracy, one dot per student
#   ddm_path / plot_race           — the interactive "marker racing to a boundary"
#   simulate_design / plot_ppc     — posterior predictive check (model regenerates us)
#   ddm_guess_from_data            — back-of-the-envelope parameter estimate
#   plot_manual_match / plot_residuals — pre-fit matching + model criticism

using DriftDiffusionModels
using Statistics
using Random
using DataFrames
using Plots

_ah_round(x) = round(x; digits = 4) + 0.0

# ----------------------------------------------------------------------
# Closed-form DDM predictions for classroom "knob playing"
# ----------------------------------------------------------------------

"""
    p_upper(B, v, a0) -> Float64

Probability that evidence with drift `v`, boundaries `0` and `B`, and starting
point `a0*B` hits the upper/right boundary. This is the psychometric curve the
DDM predicts before we simulate or fit anything.
"""
function p_upper(B::Real, v::Real, a0::Real)
    x = a0 * B
    if abs(v) < 1e-8
        return Float64(a0)
    end
    return Float64(-expm1(-2v * x) / -expm1(-2v * B))
end

"""
    mean_decision_time(B, v, a0) -> Float64

Mean boundary-crossing time, not including non-decision delay. At zero drift the
formula collapses to `x*(B-x)`, which gives the handy classroom rule
`RT(c=0) ≈ tau + B^2/4` when `a0 = 0.5`.
"""
function mean_decision_time(B::Real, v::Real, a0::Real)
    x = a0 * B
    if abs(v) < 1e-8
        return Float64(x * (B - x))
    end
    return Float64((B * p_upper(B, v, a0) - x) / v)
end

signed_drift(model::CoherentDDM, signed_c::Real) =
    signed_c == 0 ? 0.0 : sign(signed_c) * model.k * abs(signed_c)^model.α

model_pright(model::CoherentDDM, signed_c::Real) =
    p_upper(model.B, signed_drift(model, signed_c), model.a₀)

model_mean_rt(model::CoherentDDM, s::Int, c::Real) =
    model.τ + mean_decision_time(model.B, s * model.k * (c > 0 ? c^model.α : 0.0), model.a₀)

function empirical_curve_table(trials)
    signed_levels = sort(unique(_ah_round(t.s * t.c) for t in trials))
    rows = NamedTuple[]
    for signed_c in signed_levels
        sel = [t for t in trials if _ah_round(t.s * t.c) == signed_c]
        push!(rows, (view = "choice", x = signed_c,
                     data = mean(t.choice == 1 for t in sel),
                     n = length(sel)))
    end
    for c in sort(unique(_ah_round(t.c) for t in trials))
        sel = [t for t in trials if _ah_round(t.c) == c]
        push!(rows, (view = "rt", x = c,
                     data = mean(t.rt for t in sel),
                     n = length(sel)))
    end
    return DataFrame(rows)
end

"""
    model_curve_table(trials, model) -> DataFrame

Binned data plus closed-form DDM predictions for the same x-values. `view` is
`"choice"` for P(right) by signed coherence and `"rt"` for mean RT by coherence.
"""
function model_curve_table(trials, model::CoherentDDM)
    df = empirical_curve_table(trials)
    pred = Float64[]
    for r in eachrow(df)
        if r.view == "choice"
            push!(pred, model_pright(model, r.x))
        else
            push!(pred, mean(model_mean_rt(model, s, r.x) for s in (-1, 1)))
        end
    end
    df.pred = pred
    df.residual = df.data .- df.pred
    return df
end

"""
    manual_score(trials, model) -> DataFrame

Small scorecard for a hand-tuned model: root-mean-squared error on the
psychometric points and on mean RT points. Lower is better.
"""
function manual_score(trials, model::CoherentDDM)
    df = model_curve_table(trials, model)
    choice = df[df.view .== "choice", :]
    rt = df[df.view .== "rt", :]
    DataFrame(
        target = ["P(choose right)", "mean RT"],
        RMSE = [sqrt(mean((choice.data .- choice.pred).^2)),
                sqrt(mean((rt.data .- rt.pred).^2))],
        unit = ["probability", "seconds"],
    )
end

"""
    plot_manual_match(trials, model)

Overlay data with a hand-picked model, without fitting. This is intentionally
fast and deterministic so sliders feel responsive in Pluto.
"""
function plot_manual_match(trials, model::CoherentDDM)
    df = model_curve_table(trials, model)
    choice = df[df.view .== "choice", :]
    rt = df[df.view .== "rt", :]

    p1 = scatter(choice.x, choice.data; ms = 6, label = "data",
        xlabel = "signed coherence  (s * c)", ylabel = "P(choose right)",
        title = "Can our knobs predict choices?", legend = :bottomright,
        ylim = (-0.02, 1.02))
    hline!(p1, [0.5]; ls = :dot, c = :gray, label = "")
    vline!(p1, [0.0]; ls = :dot, c = :gray, label = "")
    plot!(p1, choice.x, choice.pred; lw = 2, c = :crimson, label = "hand-tuned DDM")

    p2 = scatter(rt.x, rt.data; ms = 6, label = "data",
        xlabel = "coherence  c", ylabel = "mean RT (s)",
        title = "Can our knobs predict speed?", legend = :topright)
    plot!(p2, rt.x, rt.pred; lw = 2, c = :crimson, label = "hand-tuned DDM")

    return plot(p1, p2; layout = (1, 2), size = (980, 420), margin = 4Plots.mm)
end

"""
    ddm_guess_from_data(trials) -> CoherentDDM

A deliberately rough, no-optimizer parameter guess:

* `a0` comes from the zero-coherence side bias.
* `tau` comes from the fastest few responses.
* `B` comes from the zero/hard-coherence RT rule `RT ≈ tau + B^2/4`.
* `k` comes from the psychometric slope near zero, using slope ≈ `B*k/4`.

This is not a replacement for fitting. It is a way to make the final fitted
parameters less mysterious.
"""
function ddm_guess_from_data(trials)
    rts = [t.rt for t in trials]
    tau = clamp(quantile(rts, 0.05) - 0.05, 0.03, 0.8)

    cohs = sort(unique(_ah_round(t.c) for t in trials))
    hard_c = first(cohs)
    hard_trials = [t for t in trials if _ah_round(t.c) == hard_c]
    hard_rt = mean(t.rt for t in hard_trials)
    B = clamp(2 * sqrt(max(hard_rt - tau, 0.05)), 0.4, 5.5)

    zero_trials = [t for t in trials if _ah_round(t.c) == 0.0]
    a0 = isempty(zero_trials) ? 0.5 : mean(t.choice == 1 for t in zero_trials)
    a0 = clamp(a0, 0.05, 0.95)

    curve = empirical_curve_table(trials)
    choice = curve[curve.view .== "choice", :]
    x = choice.x
    y = choice.data
    xbar = mean(x)
    denom = sum((x .- xbar).^2)
    slope = denom == 0 ? 0.0 : sum((x .- xbar) .* (y .- mean(y))) / denom
    k = clamp(4 * abs(slope) / B, 0.05, 12.0)

    return CoherentDDM(B = B, k = k, α = 1.0, a₀ = a0, τ = tau, fit_α = false)
end

function parameter_table(models::Pair{String,<:CoherentDDM}...)
    rows = NamedTuple[]
    for (label, m) in models
        push!(rows, (model = label,
                     B = round(m.B; digits = 2),
                     k = round(m.k; digits = 2),
                     α = round(m.α; digits = 2),
                     a0 = round(m.a₀; digits = 2),
                     tau = round(m.τ; digits = 3)))
    end
    return DataFrame(rows)
end

"""
    plot_residuals(trials, model)

Model criticism plot: data minus prediction. A good model leaves small, patternless
residuals. A structured wave or a single big miss is a clue about what the model
does not understand.
"""
function plot_residuals(trials, model::CoherentDDM)
    df = model_curve_table(trials, model)
    choice = df[df.view .== "choice", :]
    rt = df[df.view .== "rt", :]

    p1 = bar(string.(choice.x), choice.residual; legend = false, c = :steelblue,
        xlabel = "signed coherence", ylabel = "data - model",
        title = "Choice residuals")
    hline!(p1, [0.0]; c = :black, lw = 1, label = "")

    p2 = bar(string.(rt.x), rt.residual; legend = false, c = :darkorange,
        xlabel = "coherence", ylabel = "seconds",
        title = "RT residuals")
    hline!(p2, [0.0]; c = :black, lw = 1, label = "")

    return plot(p1, p2; layout = (1, 2), size = (980, 420), margin = 4Plots.mm)
end

function worst_residuals(trials, model::CoherentDDM; n::Int = 4)
    df = model_curve_table(trials, model)
    df.abs_residual = abs.(df.residual)
    sort!(df, :abs_residual, rev = true)
    return first(df, min(n, nrow(df)))
end


"""
    total_loglik(model, trials) -> Float64

Total log-likelihood of a fitted CoherentDDM over a trial set (higher = the
model thinks the data is more probable).
"""
total_loglik(model::CoherentDDM, trials) =
    sum(logdensityof(model.B, model.k, model.α, model.a₀, model.τ,
                     t.rt, t.choice, t.s, t.c) for t in trials)

"""
    compare_alpha(trials) -> DataFrame

Fit the model two ways — coherence exponent **α fixed at 1** (4 free knobs) vs
**α free** (5 knobs) — and score each with log-likelihood and AIC
(`AIC = 2·#params − 2·loglik`; lower is better). Shows whether the extra
flexibility is worth its complexity. Returns the table and the two fitted models.
"""
function compare_alpha(trials)
    m_fixed = CoherentDDM(fit_α = false); fit!(m_fixed, trials)
    m_free  = CoherentDDM(fit_α = true);  fit!(m_free,  trials)
    ll_fixed = total_loglik(m_fixed, trials)
    ll_free  = total_loglik(m_free,  trials)
    df = DataFrame(
        model    = ["α fixed = 1 (simpler)", "α free (more flexible)"],
        n_params = [4, 5],
        loglik   = round.([ll_fixed, ll_free]; digits = 1),
        AIC      = round.([2*4 - 2*ll_fixed, 2*5 - 2*ll_free]; digits = 1),
        α        = round.([m_fixed.α, m_free.α]; digits = 3),
    )
    return df, m_fixed, m_free
end


"""
    fit_all_students(class; fit_α=false) -> DataFrame

Fit a CoherentDDM to every student separately and collect their knobs plus
accuracy into one table — the raw material for "is everyone the same?".
"""
function fit_all_students(class::AbstractDict; fit_α::Bool = false)
    rows = NamedTuple[]
    for (pid, tr) in sort(collect(class); by = first)
        m = CoherentDDM(fit_α = fit_α); fit!(m, tr)
        acc = 100 * count(t -> t.choice == t.s, tr) / length(tr)
        push!(rows, (participant = pid,
                     B  = round(m.B;  digits = 2),
                     k  = round(m.k;  digits = 2),
                     a0 = round(m.a₀; digits = 2),
                     tau = round(m.τ; digits = 3),
                     accuracy_pct = round(acc; digits = 1),
                     n_trials = length(tr)))
    end
    return DataFrame(rows)
end


"""
    plot_param_scatter(student_df; kwargs...)

Boundary `B` (caution) vs accuracy, one labelled dot per student — connects the
speed–accuracy story to an actual model knob.
"""
function plot_param_scatter(student_df::DataFrame; kwargs...)
    scatter(student_df.B, student_df.accuracy_pct;
        legend = false, ms = 7, c = :darkorange,
        xlabel = "boundary B  (→ more cautious)",
        ylabel = "percent correct",
        title = "Caution vs accuracy (one dot per participant)", kwargs...)
end


"""
    ddm_path(; B, k, c, τ, α=1, s=1, a₀=0.5, dt=1e-3, tmax=5, rng) -> (ts, xs)

One simulated evidence-accumulation trajectory: start at `a₀·B`, drift at
`v = s·k·c^α` with noise, until it hits a boundary (0 or B) or `tmax`.
"""
function ddm_path(; B, k, c, τ, α = 1.0, s = 1, a₀ = 0.5,
                  dt = 1e-3, tmax = 5.0, rng = Random.default_rng())
    v = s * k * (c > 0 ? c^α : 0.0)
    t = 0.0; a = a₀ * B
    ts = [0.0]; xs = [a]
    while a < B && a > 0 && t < tmax
        if t < τ
            t += dt
        else
            a += v * dt + sqrt(dt) * randn(rng)
            t += dt
        end
        push!(ts, t); push!(xs, a)
    end
    return ts, xs
end

"""
    plot_race(; B, k, c, τ, α=1, npaths=6, seed=1, kwargs...)

Draw several evidence-accumulation trajectories racing between the two
boundaries — the picture behind the whole model. Half drift toward each side so
both finish lines are visible. Seeded so it's stable as sliders move.
"""
function plot_race(; B, k, c, τ, α = 1.0, npaths = 6, seed = 1, kwargs...)
    rng = MersenneTwister(seed)
    plt = plot(; xlabel = "time (s)", ylabel = "accumulated evidence",
               title = "Racing to a decision  (c = $(round(c; digits=2)))",
               legend = false, ylim = (-0.05B, 1.05B), kwargs...)
    hline!(plt, [0, B]; c = :black, lw = 2, label = "")               # the two finish lines
    hline!(plt, [0.5B]; c = :gray, ls = :dot, label = "")             # start level
    for i in 1:npaths
        s = isodd(i) ? 1 : -1
        ts, xs = ddm_path(; B, k, c, τ, α, s, rng)
        plot!(plt, ts, xs; lw = 1.5, c = s > 0 ? :seagreen : :firebrick, alpha = 0.8)
    end
    annotate!(plt, [(0.0, 1.02B, text("RIGHT", :left, 9, :seagreen)),
                    (0.0, -0.02B, text("LEFT", :left, 9, :firebrick))])
    return plt
end


"""
    simulate_design(model, trials; reps=1, dt=1e-4, rng) -> Vector{CoherentDDMResult}

Generate fake data from a fitted model using the SAME (s, c) design as `trials`,
`reps` times over. The basis of a posterior predictive check.
"""
function simulate_design(model::CoherentDDM, trials; reps::Int = 1,
                         dt = 1e-4, rng = Random.default_rng())
    out = CoherentDDMResult[]
    for _ in 1:reps, t in trials
        push!(out, simulateDDM(model, t.s, t.c, dt, rng))
    end
    return out
end

"""
    plot_ppc(trials, model; reps=20, seed=1)

Posterior predictive check: overlay the fitted model's *simulated* reaction-time
distribution (red) on the students' real RTs (gray). If the model captures the
class, the shapes line up — "the model can regenerate us".
"""
function plot_ppc(trials, model::CoherentDDM; reps::Int = 20, seed::Int = 1)
    rng = MersenneTwister(seed)
    sim = simulate_design(model, trials; reps = reps, rng = rng)
    real_rt = [t.rt for t in trials]
    sim_rt  = [t.rt for t in sim]
    plt = histogram(real_rt; bins = 30, normalize = :pdf, alpha = 0.5, c = :gray,
        label = "real class RTs", xlabel = "reaction time (s)", ylabel = "density",
        title = "Does the model regenerate us?")
    histogram!(plt, sim_rt; bins = 30, normalize = :pdf, alpha = 0.45, c = :firebrick,
        label = "model's simulated RTs")
    return plt
end
