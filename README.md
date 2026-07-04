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

## First-Passage-Time Neural DDM (`FPTDDM`)

`FPTDDM` treats a drift-diffusion process as a **latent** whose trajectory drives
a point-process (Poisson) observation of simultaneously recorded neurons. Unlike
a soft-boundary readout, a choice here *is* the boundary that gets hit and the
response time *is* the first-passage time — so behavior and spikes share a single
generative process:

* latent accumulator with absorbing boundaries at `0` and `B` (start `a₀·B`,
  unit diffusion, optional leak `λ`), matching the `DDM.jl` WFPT convention;
* per-neuron Poisson spikes with rate `softplus(ηₙ(x))`, where the linear
  predictor `ηₙ(x)` comes from a swappable `AbstractObservationModel`:
  `LinearPoissonObservationModel` (`bₙ + wₙ·x`),
  `BasisPoissonObservationModel` (Gaussian-bump basis for nonlinear tuning), and
  a `GPPoissonObservationModel` stub;
* inference by an absorbing-boundary particle filter that accounts for within-bin
  boundary crossings via the **Brownian-bridge correction**, so it estimates the
  true continuous-time first-passage likelihood.

Because the boundary is absorbing (not soft), the **spike-free marginal reduces
to the exact Wiener first-passage density**: dropping the Poisson terms recovers
the `wfpt`/`logdensityof` likelihood from `DDM.jl`. This is verified in
`test/test_FPTDDM.jl` (the PF marginal matches WFPT across both boundaries and
integrates to ~1). Behavior-only is thus a strict special case of the joint
neural model.

```julia
model = FPTDDM(; v = 1.0, B = 1.0, a₀ = 0.5, λ = 0.0, σ² = 1.0,
               b = [0.3, 0.1], w = [1.2, -0.8])   # 2 neurons
trial = simulate_trial(rng, model, 0.005; max_time = 2.0)  # spikes + choice + RT
ll    = loglik(model, trial; N = 1024, rng = rng)          # joint log-likelihood
```

Parameter fitting (differentiable PF under common random numbers) is the next
planned step.

## File Structure

* `DriftDiffusionModels.jl` – Main module file
* `DDM.jl` – Drift Diffusion Model definitions and utilities (WFPT likelihood)
* `HMMDDM.jl` – HMM wrapper with DDM emissions and training
* `eDDM.jl` – Hierarchical (empirical-Bayes VI) per-trial DDM
* `FPTDDM.jl` – First-passage-time neural DDM (latent DDM → point process)

---

## Contributions

Feel free to contribute pull requests or file issues to suggest features or report bugs!
