### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ c0000001-0000-4000-8000-000000000001
begin
    using DriftDiffusionModels
    using DataFrames
    using Dates
    using CSV
    using Plots
end

# ╔═╡ c0000002-0000-4000-8000-000000000002
begin
    ddir = "../data/mouse_df.csv"
    # load in the data
    df = CSV.File(ddir) |> DataFrame
end

# ╔═╡ c0000003-0000-4000-8000-000000000003
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

# ╔═╡ c0000004-0000-4000-8000-000000000004
all_results = data_prep.all_results

# ╔═╡ c0000005-0000-4000-8000-000000000005
seq_ends = data_prep.seq_ends

# ╔═╡ c0000006-0000-4000-8000-000000000006
hyper, qs, elbo_history = fit_vi_gaussian(all_results; n_iter=25, K=25, verbose=true)

# ╔═╡ c0000007-0000-4000-8000-000000000007
B = [exp(i.μ[1]) for i in qs]          # boundary separation

# ╔═╡ c0000008-0000-4000-8000-000000000008
τ = [exp(i.μ[2]) for i in qs]          # non-decision time

# ╔═╡ c0000009-0000-4000-8000-000000000009
v = [exp(i.μ[3]) for i in qs]          # drift rate

# ╔═╡ c000000a-0000-4000-8000-00000000000a
a0 = [1 / (1 + exp(-i.μ[4])) for i in qs]  # starting point

# ╔═╡ c000000b-0000-4000-8000-00000000000b
plot(τ)

# ╔═╡ c000000c-0000-4000-8000-00000000000c
DriftDiffusionModels.transform_params(hyper.m)

# ╔═╡ c000000d-0000-4000-8000-00000000000d
using StatsBase

# ╔═╡ c000000e-0000-4000-8000-00000000000e
pacf(a0, 1:20)

# ╔═╡ Cell order:
# ╠═c0000001-0000-4000-8000-000000000001
# ╠═c0000002-0000-4000-8000-000000000002
# ╠═c0000003-0000-4000-8000-000000000003
# ╠═c0000004-0000-4000-8000-000000000004
# ╠═c0000005-0000-4000-8000-000000000005
# ╠═c0000006-0000-4000-8000-000000000006
# ╠═c0000007-0000-4000-8000-000000000007
# ╠═c0000008-0000-4000-8000-000000000008
# ╠═c0000009-0000-4000-8000-000000000009
# ╠═c000000a-0000-4000-8000-00000000000a
# ╠═c000000b-0000-4000-8000-00000000000b
# ╠═c000000c-0000-4000-8000-00000000000c
# ╠═c000000d-0000-4000-8000-00000000000d
# ╠═c000000e-0000-4000-8000-00000000000e
