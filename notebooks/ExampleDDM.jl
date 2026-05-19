### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ a0000001-0000-4000-8000-000000000001
# this cell loads all files necessary for fitting a vanilla DDM (sometimes called the pure DDM)
begin
    using DriftDiffusionModels
    using Plots
    using Random
    using Turing
end

# ╔═╡ a0000002-0000-4000-8000-000000000002
# this cell will simulate a DDM process so we have data we can fit later on, we will later fit to other data that is not simulated
ddm_results = let
    B = 2.0       # bound height
    v = 0.73      # drift rate
    a₀ = 0.62     # initial starting point (as fraction of bound)
    τ = 0.2       # non-decision time (s)

    simulateDDM(DriftDiffusionModel(B, v, a₀, τ), 500, 1e-6) # simulate 500 trials with very small dt
end

# ╔═╡ a0000003-0000-4000-8000-000000000003
# quickly plot the results so confirm it looks like a DDM!
let
    rts = [res.rt for res in ddm_results]
    histogram(rts, bins=50, title="Simulated DDM Response Times", xlabel="Response Time (s)", ylabel="Frequency")
end

# ╔═╡ a0000004-0000-4000-8000-000000000004
# we can fit the model as follows
ddm_fit_info = let
    ddm = DriftDiffusionModel(; τ=0.05)
    fit!(ddm, ddm_results)

    println("Fitted DDM parameters:")
    println("Boundary Height: ", ddm.B)
    println("Drift Rate: ", ddm.v)
    println("Initial Starting Point: ", ddm.a₀)
    println("Non-decision time: ", ddm.τ)
    ddm
end

# ╔═╡ a0000005-0000-4000-8000-000000000005
# now we can look at what if we have data from an experment, we can load it and fit the DDM to it
# assume you have some data like follows
begin
    rts = [0.45, 0.52, 0.39, 0.61, 0.48, 0.55, 0.50, 0.47, 0.60, 0.53] # response times in seconds
    choices = [1, -1, 1, 1, -1, 1, -1, -1, 1, -1] # choices made (1 for right, -1 for left)
    stimuli = [1, 1, -1, 1, -1, 1, -1, -1, 1, -1] # stimuli presented (1 for right, -1 for left)

    # convert to DDMResult format
    exp_results = [DDMResult(rt, choice, stim) for (rt, choice, stim) in zip(rts, choices, stimuli)]

    # fit the DDM to "experimental data"
    new_ddm = DriftDiffusionModel(; τ=0.1)

    fit!(new_ddm, exp_results)
    println("Fitted DDM parameters to experimental data:")
    println("Boundary Height: ", new_ddm.B)
    println("Drift Rate: ", new_ddm.v)
    println("Initial Starting Point: ", new_ddm.a₀)
    println("Non-decision time: ", new_ddm.τ)
    new_ddm
end

# ╔═╡ a0000006-0000-4000-8000-000000000006
@model function lapseBayesDDM(rts::Vector{Float64},
    choice::Vector{Int},
    stimulses::Vector{Int})

    N = length(rts)
    max_rt = maximum(rts)

    # priors
    logB ~ Normal(log(2), 0.5)
    logV ~ Normal(log(1), 0.5)
    a₀ ~ Beta(2, 2)
    τ_rel ~ Beta(2, 5)
    p_lapse ~ Beta(1, 10)    # lapses rare a priori

    τ = τ_rel * max_rt
    B = exp(logB)
    v = exp(logV)

    for i in eachindex(rts)
        rt = rts[i]
        ch = choice[i]
        stim = stimulses[i]

        # DDM log-density for this trial
        ll_ddm = logdensityof(B, v, a₀, τ, rt, ch, stim)

        # Lapse log-density for this trial
        ll_lapse_rt = logpdf(Uniform(0.0, max_rt), rt)

        ch_prop = (ch + 1) / 2
        ll_lapse_choice = logpdf(Bernoulli(0.5), ch_prop)

        ll_lapse = ll_lapse_rt + ll_lapse_choice

        m1 = ll_ddm + log1p(-p_lapse)   # log(1 - p_lapse) + ll_ddm
        m2 = ll_lapse + log(p_lapse)      # log(p_lapse)     + ll_lapse

        mmax = max(m1, m2)
        @addlogprob! (mmax + log(exp(m1 - mmax) + exp(m2 - mmax)))
    end
end

# ╔═╡ a0000007-0000-4000-8000-000000000007
chain = let
    model = lapseBayesDDM(rts, choices, stimuli)
    sample(model, NUTS(), 1000)
end

# ╔═╡ a0000008-0000-4000-8000-000000000008
let
    μ_logB = mean(chain["logB"])
    μ_logV = mean(chain["logV"])
    μ_a₀ = mean(chain["a₀"])
    μ_τ_rel = mean(chain["τ_rel"])
    μ_p_lapse = mean(chain["p_lapse"])

    println("Posterior mean estimates:")
    println("Boundary Height (B): ", exp(μ_logB))
    println("Drift Rate (v): ", exp(μ_logV))
    println("Initial Starting Point (a₀): ", μ_a₀)
    println("Non-decision time (τ): ", μ_τ_rel * maximum(rts))
    println("Lapse Probability (p_lapse): ", μ_p_lapse)
end

# ╔═╡ Cell order:
# ╠═a0000001-0000-4000-8000-000000000001
# ╠═a0000002-0000-4000-8000-000000000002
# ╠═a0000003-0000-4000-8000-000000000003
# ╠═a0000004-0000-4000-8000-000000000004
# ╠═a0000005-0000-4000-8000-000000000005
# ╠═a0000006-0000-4000-8000-000000000006
# ╠═a0000007-0000-4000-8000-000000000007
# ╠═a0000008-0000-4000-8000-000000000008
