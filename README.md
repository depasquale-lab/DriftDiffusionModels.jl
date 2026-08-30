# DriftDiffusionModels.jl

**DriftDiffusionModels.jl** is a Julia package for simulating, fitting, and analyzing Drift Diffusion Models (DDMs), with support for multi-state Hidden Markov Models (HMMs) whose emission distributions are governed by DDMs. The package provides tools for simulation, inference, and model selection via cross-validation.

> **Note**: This package is **not intended as a production-ready DDM toolkit**. For a more complete and sophisticated implementation of sequential sampling models including advanced DDM variants, we recommend using [`SequentialSamplingModels.jl`](https://github.com/itsdfish/SequentialSamplingModels.jl).

---

## Features

* Simulation of DDM trajectories using the Euler–Maruyama method
* Wiener First Passage Time (WFPT) density computation (Navarro & Fuss, 2009)
* Log-likelihood and parameter estimation for DDMs via MLE
* Hidden Markov Models with DDM emissions and Dirichlet priors
* Baum–Welch (EM) training with MAP updates

---

## Installation

Clone the repository and include the module in your Julia environment:

```julia
include("DriftDiffusionModels.jl")
using .DriftDiffusionModels
```

---

## Module Overview

### `DriftDiffusionModel`

The core structure representing a DDM:

```julia
DriftDiffusionModel(B, v, a₀, τ)
```

* `B`: Boundary separation
* `v`: Drift rate
* `a₀`: Initial fraction of the boundary
* `τ`: Non-decision time

### `DDMResult`

Result of a single DDM simulation:

```julia
DDMResult(rt, choice, stimulus)
```

* `rt`: Response time
* `choice`: Decision outcome (1 --> R or -1 --> L)
* `stimulus`: Whether evidence favored left vs. right trials (1 --> R, -1 --> L)

---

## Key Functions

### Simulation

```julia
simulateDDM(model::DriftDiffusionModel, dt::Float64=1e-5)
simulateDDM(model::DriftDiffusionModel, n::Int, dt::Float64=1e-5)
```

Simulates one or multiple trials of the DDM using Euler–Maruyama integration.

### Likelihood

```julia
wfpt(t, v, B, w, τ)
logdensityof(model::DriftDiffusionModel, result::DDMResult)
```

Computes the WFPT density and log-likelihood for observed DDM results.

### Fitting

```julia
StatsAPI.fit!(model::DriftDiffusionModel, data::Vector{DDMResult}, weights=ones(length(data)))
```

Fits a DDM to data using Maximum Likelihood Estimation (MLE), supporting observation weights (useful for HMM training).

---

## Hidden Markov Models with DDM Emissions

### `PriorHMM`

A wrapper for Hidden Markov Models with Dirichlet priors on initial probabilities and transition matrices:

```julia
PriorHMM(init, trans, dists; α_trans, α_init)
```

Supports MAP updates via Baum–Welch.

### Training

```julia
baum_welch(hmm, data; seq_ends)
```

Trains an HMM-DDM model using EM.

---

## Multilevel ("extended") DDM

Each trial has DDM parameters drawn from a Gaussian distribution over
`u = (u_B, u_τ, u_v, u_a₀)`:

```
u_t ~ N(m, diag σ0²),    B = exp(u_B), τ = exp(u_τ), v = exp(u_v), a₀ = σ(u_a₀)
y_t ~ DDM(B_t, v_t, a₀_t, τ_t)
```

### Exact marginal likelihood

```julia
fit = fit_mlddm_exact(data; q = 8, qτ = 16, n_starts = 4)
```

The four-dimensional marginal likelihood is evaluated by quadrature and
maximized over `(m, σ0)`. `q` controls Gauss–Hermite integration over `B`, `v`,
and `a₀`; `qτ` controls a truncation-aware rule for `τ < rt` and defaults to
`2q`. Check convergence before reporting a fit:

```julia
for r in quadrature_check(data, fit.m, fit.σ0)
    @show r.q, r.qτ, r.loglik, r.delta_per_trial
end
```

Useful diagnostics are `fit.converged`, `fit.n_converged`, `fit.quad_err`,
`fit.boundary`, and `fit.n_zero`. Posterior summaries and conditional profiles
are available with:

```julia
μ, sd = trial_posteriors(data, fit.m, fit.σ0)   # exact per-trial posteriors, 4 × N
rows  = profile_sigma0(data, fit, 3)            # profile likelihood for σ0[v]
```

### Cost

Each objective evaluation costs `O(n_trials · q³ · qτ)`. Use `max_trials` to
fit a reproducible subsample when the full dataset is too expensive:

```julia
fit = fit_mlddm_exact(data; max_trials = 5000, rng = MersenneTwister(1))
```

The fit records the selected indices and sample sizes. Between-trial variances
can be weakly identified, so inspect `fit.boundary` and `profile_sigma0` before
interpreting `σ0`.

### Legacy: variational inference

```julia
hyper, qs, elbo_history = fit_vi_gaussian(data; n_iter = 10, K = 3)
```

Retained for reproducing earlier results. See `experiments/vi_validation/` for
validation and known limitations.

## Model Comparison

### Log-Likelihood Ratio

```julia
calculate_ll_ratio(ll, ll₀, n)
```

Computes the per-observation log-likelihood ratio (in bits) between multi-state and single-state models.

---

## References

* Navarro, D. J., & Fuss, I. G. (2009). Fast and accurate calculations for first-passage times in Wiener diffusion models.
* HiddenMarkovModels.jl — backend for HMM routines.

---

## File Structure

* `DriftDiffusionModels.jl` – Main module file
* `DDM.jl` – Drift Diffusion Model definitions and utilities
* `HMMDDM.jl` – HMM wrapper with DDM emissions and training
* `eDDMExact.jl` – Multilevel DDM by exact marginal likelihood (recommended)
* `eDDM.jl` – Multilevel DDM by variational inference (legacy)
* `experiments/vi_validation/` – Validation suite for the above two

---

## Contributions

Feel free to contribute pull requests or file issues to suggest features or report bugs!
