### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ b0000001-0000-4000-8000-000000000001
begin
    using DriftDiffusionModels
    using HiddenMarkovModels
    using LinearAlgebra
    using CSV
    using DataFrames
    using Distributions
    using Dates
    using Random
    using Plots
end

# ╔═╡ b0000002-0000-4000-8000-000000000002
begin
    ddir = "../data/mouse_df.csv"
    # load in the data
    df = CSV.File(ddir) |> DataFrame
end

# ╔═╡ b0000003-0000-4000-8000-000000000003
data_prep = let
    # Group by animal name and count trials
    trial_counts = combine(groupby(df, :name), nrow => :trial_count)

    # Sort by count in descending order
    sort!(trial_counts, :trial_count, rev=true)

    # pick a mouse of interest (start with mouse of most trials)
    moi = trial_counts[1, :name]

    # get the data for the mouse of interest
    mouse_df = df[df.name .== moi, :]

    # Filter out trials with "omission" outcome
    valid_trials = findall(outcome -> outcome != "omission", mouse_df.outcome)
    filtered_df = mouse_df[valid_trials, :]

    # Map correct -> 1 and incorrect -> -1 (only for error and correct, omissions are gone)
    numeric_outcomes = [outcome == "correct" ? 1 : -1 for outcome in filtered_df.outcome]

    # Get reaction times, filter out "NAN" values
    valid_rt_indices = findall(rt -> uppercase(string(rt)) != "NAN", filtered_df.rt)
    # valid_trials = findall(rt -> rt > 0.2, filtered_df.rt)

    # Apply both filters to keep data aligned
    final_df = filtered_df[valid_rt_indices, :]
    final_outcomes = numeric_outcomes[valid_rt_indices]

    # Convert RTs to Float64
    # final_rts = [parse(Float64, rt) for rt in final_df.rt]
    final_rts = final_df.rt
    final_stimulus = final_df.correct_side
    final_stimulus = [stim == "right" ? 1 : -1 for stim in final_stimulus]

    # Extract just the date part from the timestamp strings
    dates_vec = [Date(split(dt)[1]) for dt in final_df.trial_datetime]

    # Get unique dates in chronological order
    unique_dates = sort(unique(dates_vec))

    # Create a vector of vectors, where each inner vector contains DDMResults for one day
    results_by_date = Vector{Vector{DDMResult}}()

    for date in unique_dates
        # Get indices for this date
        day_indices = findall(dates_vec .== date)

        # Skip days with no valid data
        if isempty(day_indices)
            continue
        end

        # Extract RTs and outcomes for this date
        day_rts = final_rts[day_indices]
        day_outcomes = final_outcomes[day_indices]
        day_stim_side = final_stimulus[day_indices]

        # Create DDMResult objects for this day
        day_results = [DDMResult(rt, choice, stim) for (rt, choice, stim) in zip(day_rts, day_outcomes, day_stim_side)]

        # Add to our vector of vectors
        push!(results_by_date, day_results)
    end

    # Now calculate the sequence ends (cumulative sum of lengths)
    seq_ends_local = cumsum([length(seq) for seq in results_by_date])

    # Concatenate all results into a single vector
    all_results_local = reduce(vcat, results_by_date)

    (all_results=all_results_local, seq_ends=seq_ends_local)
end

# ╔═╡ b0000004-0000-4000-8000-000000000004
all_results = data_prep.all_results

# ╔═╡ b0000005-0000-4000-8000-000000000005
seq_ends = data_prep.seq_ends

# ╔═╡ b0000006-0000-4000-8000-000000000006
# assume a 3 state model
num_states = 3

# ╔═╡ b0000007-0000-4000-8000-000000000007
hmm_priors = let
    α = 10.0 # concentration paramerter for a Dirichlet prior, where α is the main diagonal concentration
    A = zeros(num_states, num_states)

    for i in 1:num_states
        dir = zeros(num_states)
        for j in 1:num_states
            if i == j
                dir[j] = α
            else
                dir[j] = 1.0
            end
        end
        A[i, :] = rand(Dirichlet(dir))
    end

    π₀ = rand(Dirichlet(fill(1.0, num_states)))

    # Define priors over each paramerter
    logv₀_prior = Normal(log(1.0), 0.5)
    a₀_prior = Beta(10.0, 10.0)
    logB_prior = Normal(log(2.0), 0.5)
    τ_init = 0.1

    ddms = Vector{DriftDiffusionModel}(undef, num_states)
    for i in 1:num_states
        logv₀ = rand(logv₀_prior)
        a₀ = rand(a₀_prior)
        logB = rand(logB_prior)
        ddms[i] = DriftDiffusionModel(exp(logB), exp(logv₀), a₀, τ_init)
    end

    (π₀=π₀, A=A, ddms=ddms)
end

# ╔═╡ b0000008-0000-4000-8000-000000000008
hmm_init = let
    αT = ones(num_states, num_states)
    αT[diagind(αT)] .= 5   # sticky prior
    απ = fill(2.0, num_states)

    PriorHMM(hmm_priors.π₀, hmm_priors.A, hmm_priors.ddms, αT, απ)
end

# ╔═╡ b0000009-0000-4000-8000-000000000009
hmm_est, lls = baum_welch(hmm_init, all_results; seq_ends=seq_ends)

# ╔═╡ Cell order:
# ╠═b0000001-0000-4000-8000-000000000001
# ╠═b0000002-0000-4000-8000-000000000002
# ╠═b0000003-0000-4000-8000-000000000003
# ╠═b0000004-0000-4000-8000-000000000004
# ╠═b0000005-0000-4000-8000-000000000005
# ╠═b0000006-0000-4000-8000-000000000006
# ╠═b0000007-0000-4000-8000-000000000007
# ╠═b0000008-0000-4000-8000-000000000008
# ╠═b0000009-0000-4000-8000-000000000009
