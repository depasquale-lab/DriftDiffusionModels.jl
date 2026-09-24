# Fitting your own data

This page goes from a table of trials to fitted models. To keep it
self-contained it simulates a dataset shaped like a typical two-alternative
forced-choice experiment; to use your own data, replace the
[simulation step](#Step-0:-a-dataset) with your table and keep the rest.

```@example tut
using DriftDiffusionModels
using HiddenMarkovModels
using Random, Statistics
```

## Step 0: a dataset

We need four columns per trial: session, RT (s), whether the response was
correct, and which side was correct. Here the subject switches between an
*engaged* state (strong drift) and a *disengaged* state (weak drift, slower,
biased to the right) that persists across trials.

```@example tut
rng = MersenneTwister(2024)

engaged    = DriftDiffusionModel(1.5, 2.0, 0.50, 0.20)   # B, v, a₀, τ
disengaged = DriftDiffusionModel(1.2, 0.3, 0.65, 0.30)
states_true = [engaged, disengaged]
P = [0.97 0.03; 0.05 0.95]            # sticky state transitions

n_sessions, n_per = 8, 150
session, rt, correct, correct_side, z_true = Int[], Float64[], Int[], Int[], Int[]
for sess in 1:n_sessions
    z = 1
    for t in 1:n_per
        t > 1 && (z = rand(rng) < P[z, 1] ? 1 : 2)
        y = simulateDDM(states_true[z], 1e-4, rng)
        push!(session, sess); push!(rt, y.rt); push!(z_true, z)
        push!(correct, y.choice == y.s ? 1 : -1)
        push!(correct_side, y.s)
    end
end
length(rt)
```

With a real CSV, the equivalent is:

```julia
using CSV, DataFrames
df = CSV.read("my_trials.csv", DataFrame)
df = filter(r -> !ismissing(r.rt) && r.rt > 0, df)          # drop omissions
session      = df.session
rt           = Float64.(df.rt)                                 # seconds!
correct      = ifelse.(df.outcome .== "correct", 1, -1)
correct_side = ifelse.(df.correct_side .== "right", 1, -1)
```

## Step 1: convert to `DDMResult`

`DDMResult(rt, choice, s)` takes the side the subject **responded** to, not
correctness. Convert, and sort trials so each session is contiguous and in
time order (the HMM uses trial order).

```@example tut
choice = correct .* correct_side            # +1 = responded right, -1 = left

order = sortperm(session; alg = Base.Sort.DEFAULT_STABLE)  # keep within-session order
data  = [DDMResult(rt[i], choice[i], correct_side[i]) for i in order]
first(data, 3)
```

For the HMM, `seq_ends` gives the index of the last trial of each session, so
that no transition is modeled across session boundaries:

```@example tut
seq_ends = cumsum([count(==(s), session) for s in sort(unique(session))])
```

## Step 2: a single DDM

`fit!` does maximum likelihood (L-BFGS within bounds), starting from the
model's current values.

```@example tut
ddm = DriftDiffusionModel(; B = 1.0, v = 1.0, a₀ = 0.5, τ = 0.1)
fit!(ddm, data)
ll_single = sum(logdensityof(ddm, y) for y in data)
ddm
```

A single DDM averages the two regimes: its drift sits between them.

## Step 3: an HMM with DDM emissions

[`init_hmm_ddm`](@ref) fits a global DDM, perturbs it once per state, and puts
sticky Dirichlet priors on the transitions. `baum_welch` (from
HiddenMarkovModels.jl) then runs EM with MAP updates.

```@example tut
K = 2
hmm0 = init_hmm_ddm(MersenneTwister(1), data, K)
hmm, lls = baum_welch(hmm0, data; seq_ends = seq_ends)
hmm.dists
```

```@example tut
round.(hmm.trans; digits = 3)
```

EM can reach local optima; in practice, fit from several seeds and keep the
fit with the highest final `last(lls)`.

State labels are arbitrary. Here we call the state with the larger drift
"engaged":

```@example tut
eng = argmax([d.v for d in hmm.dists])
```

### Decoding states

`viterbi` gives the most likely state sequence; `forward_backward` gives
per-trial posterior state probabilities.

```@example tut
z_hat, _ = viterbi(hmm, data; seq_ends = seq_ends)
γ, _     = forward_backward(hmm, data; seq_ends = seq_ends)
p_engaged = γ[eng, :]

# agreement with the simulated truth (only possible with simulated data)
mean((z_hat .== eng) .== (z_true[order] .== 1))
```

### Comparing to a single DDM

[`calculate_ll_ratio`](@ref) reports the improvement in log-likelihood per
trial, in bits:

```@example tut
calculate_ll_ratio(last(lls), ll_single, length(data))
```

In-sample likelihood always favors more states. To choose `K`, compare
held-out likelihood, e.g. leaving out whole sessions:

```@example tut
test_sess = [7, 8]
is_test   = [session[i] in test_sess for i in order]
train, test = data[.!is_test], data[is_test]
ends(mask) = cumsum([count(==(s), session[order][mask]) for s in unique(session[order][mask])])

for K in 1:3
    h, _ = baum_welch(init_hmm_ddm(MersenneTwister(K), train, K), train;
                      seq_ends = ends(.!is_test))
    println("K = $K   held-out log-lik = ",
            round(logdensityof(h, test; seq_ends = ends(is_test)); digits = 1))
end
```

### Priors

To set the Dirichlet priors yourself, build a [`PriorHMM`](@ref) directly:

```julia
αT = [20.0 1.0; 1.0 20.0]      # strongly sticky
απ = [1.0, 1.0]
hmm0 = PriorHMM([0.5, 0.5], [0.95 0.05; 0.05 0.95],
                [DriftDiffusionModel(1.5, 2.0, 0.5, 0.2),
                 DriftDiffusionModel(1.2, 0.5, 0.5, 0.2)];
                α_trans = αT, α_init = απ)
```

## Step 4: the multilevel DDM

Instead of discrete states, the multilevel DDM lets each trial draw its own
parameters from a Gaussian over unconstrained parameters
``u = (\log B, \log\tau, \log v, \mathrm{logit}\,a_0)``, and fits the mean
`m` and standard deviations `σ0` by exact marginal likelihood.

!!! note "Parameter order"
    `m` and `σ0` are ordered **(B, τ, v, a₀)** and live on the unconstrained
    scale, unlike the `DriftDiffusionModel(B, v, a₀, τ)` constructor.

Cost grows as `n_trials · q³ · qτ`. `max_trials` fits a reproducible subsample:

```@example tut
fit = fit_mlddm_exact(data; q = 6, n_starts = 2, max_trials = 400,
                      rng = MersenneTwister(1), verbose = false)
```

```@example tut
B, τ, v, a₀ = exp(fit.m[1]), exp(fit.m[2]), exp(fit.m[3]), logistic(fit.m[4])
(; B, τ, v, a₀)
```

Before reporting a fit, check that the quadrature has converged (the
log-likelihood should stabilize as `q` grows) and whether any `σ0` sits on its
lower bound (`fit.boundary`), meaning that variability is not supported by the
data. [`profile_sigma0`](@ref) profiles the likelihood for one component, e.g.
`profile_sigma0(data, fit, 3)` for drift variability.

```julia
for r in quadrature_check(data, fit.m, fit.σ0)
    @show r.q, r.loglik, r.delta_per_trial
end
μ, sd = trial_posteriors(data, fit.m, fit.σ0)   # 4 × N per-trial posteriors
```
