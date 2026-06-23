# load_class_data.jl
#
# Turns the Google Sheet export from the moving-dots task (rdm_task.html)
# into data the CoherentDDM can fit.
#
# The Sheet has one row per student with columns roughly:
#   Timestamp | participant_id | data
# where `data` is a string of trials encoded as  rt:choice:s:c  joined by ';'
# (see encodeData() in rdm_task.html).
#
# Usage:
#   include("outreach/load_class_data.jl")
#   class = load_class_data("class_responses.csv")     # File > Download > CSV from the Sheet
#   pooled = pool_trials(class)                          # all students together
#   model  = CoherentDDM()
#   fit!(model, pooled)                                  # fit the whole class
#   # or fit one student:  fit!(CoherentDDM(), class["P-..."])

using DriftDiffusionModels
using CSV
using DataFrames

"""
    parse_blob(blob) -> Vector{CoherentDDMResult}

Parse one student's `data` cell ("rt:choice:s:c;rt:choice:s:c;...").
Malformed or out-of-range trials are skipped with a warning rather than
aborting the whole load.
"""
function parse_blob(blob::AbstractString)
    trials = CoherentDDMResult[]
    isempty(strip(blob)) && return trials
    for (i, chunk) in enumerate(split(strip(blob), ';'))
        isempty(chunk) && continue
        parts = split(chunk, ':')
        if length(parts) != 4
            @warn "skipping malformed trial $i" chunk
            continue
        end
        try
            rt     = parse(Float64, parts[1])
            choice = parse(Int, parts[2])
            s      = parse(Int, parts[3])
            c      = parse(Float64, parts[4])
            (rt > 0 && choice in (-1, 1) && s in (-1, 1) && 0 <= c <= 1) || continue
            push!(trials, CoherentDDMResult(rt, choice, s, c))
        catch err
            @warn "skipping unparseable trial $i" chunk err
        end
    end
    return trials
end

"""
    load_class_data(csv_path; pid_col="participant_id", data_col="data")
        -> Dict{String, Vector{CoherentDDMResult}}

Read the Sheet CSV export and return a dict mapping each student's anonymous
id to their trial vector. Column names are matched case-insensitively and by
substring, so the default Google Form headers usually "just work".
"""
function load_class_data(csv_path::AbstractString;
                         pid_col::AbstractString="participant_id",
                         data_col::AbstractString="data")
    df = CSV.read(csv_path, DataFrame; stringtype=String)

    findcol(want) = begin
        idx = findfirst(c -> occursin(lowercase(want), lowercase(String(c))), names(df))
        idx === nothing ? nothing : names(df)[idx]
    end
    pcol = findcol(pid_col)
    dcol = findcol(data_col)
    pcol === nothing && error("no column matching \"$pid_col\" in $(names(df))")
    dcol === nothing && error("no column matching \"$data_col\" in $(names(df))")

    out = Dict{String, Vector{CoherentDDMResult}}()
    for row in eachrow(df)
        ismissing(row[dcol]) && continue
        pid = ismissing(row[pcol]) ? "unknown_$(rand(1000:9999))" : String(row[pcol])
        trials = parse_blob(String(row[dcol]))
        if !isempty(trials)
            # If a pid somehow repeats, append rather than overwrite.
            out[pid] = vcat(get(out, pid, CoherentDDMResult[]), trials)
        end
    end
    return out
end

"""
    load_student_csvs(dir; pattern="rdm_") -> Dict{String, Vector{CoherentDDMResult}}

No-form path: read a folder of the per-student backup files the task downloads
(`rdm_P-XXXX.csv`, columns `participant_id,trial,rt,choice,s,c`). Returns the
same dict shape as [`load_class_data`](@ref), so everything downstream
(`pool_trials`, `summarize`, the plots) is identical.
"""
function load_student_csvs(dir::AbstractString; pattern::AbstractString="rdm_")
    out = Dict{String, Vector{CoherentDDMResult}}()
    files = filter(f -> endswith(lowercase(f), ".csv") && occursin(pattern, f),
                   readdir(dir; join=true))
    isempty(files) && @warn "no matching CSVs found in $dir (pattern \"$pattern\")"
    for f in files
        df = CSV.read(f, DataFrame; stringtype=String)
        cols = lowercase.(String.(names(df)))
        need = ["rt", "choice", "s", "c"]
        all(n -> n in cols, need) || (@warn "skipping $f: missing columns" names(df); continue)
        col(n) = names(df)[findfirst(==(n), cols)]

        # Prefer a participant_id column; fall back to the file name.
        pid_idx = findfirst(in(("participant_id", "pid", "p")), cols)
        pid = if pid_idx !== nothing && !isempty(df) && !ismissing(df[1, pid_idx])
            String(df[1, pid_idx])
        else
            splitext(basename(f))[1]
        end

        trials = CoherentDDMResult[]
        for row in eachrow(df)
            any(ismissing, (row[col("rt")], row[col("choice")], row[col("s")], row[col("c")])) && continue
            rt = Float64(row[col("rt")]); choice = Int(row[col("choice")])
            s  = Int(row[col("s")]);      c = Float64(row[col("c")])
            (rt > 0 && choice in (-1,1) && s in (-1,1) && 0 <= c <= 1) || continue
            push!(trials, CoherentDDMResult(rt, choice, s, c))
        end
        !isempty(trials) && (out[pid] = vcat(get(out, pid, CoherentDDMResult[]), trials))
    end
    return out
end

"""
    pool_trials(class) -> Vector{CoherentDDMResult}

Concatenate every student's trials into one vector (a quick group-level fit).
"""
pool_trials(class::AbstractDict) = reduce(vcat, values(class); init=CoherentDDMResult[])

"""
    summarize(class)

Print a per-student trial count and overall accuracy — a fast sanity check
before fitting.
"""
function summarize(class::AbstractDict)
    println("students: ", length(class))
    total = 0
    for (pid, trials) in sort(collect(class); by=first)
        acc = isempty(trials) ? 0.0 :
              count(t -> t.choice == t.s, trials) / length(trials)
        println("  $pid : $(length(trials)) trials, $(round(100acc; digits=1))% correct")
        total += length(trials)
    end
    println("total trials: ", total)
end
