# DriftDiffusionModels.jl

**DriftDiffusionModels.jl** is a Julia package for simulating, fitting, and analyzing Drift Diffusion Models (DDMs), with support for multi-state Hidden Markov Models (HMMs) whose emission distributions are governed by DDMs, coherence-dependent drift, and variational inference for trial-level parameter uncertainty.

> **Note**: This package is **not intended as a production-ready DDM toolkit**. For a more complete and sophisticated implementation of sequential sampling models including advanced DDM variants, we recommend using [`SequentialSamplingModels.jl`](https://github.com/itsdfish/SequentialSamplingModels.jl).

---

## Features

* Simulation of DDM trajectories using the Euler–Maruyama method
* Wiener First Passage Time (WFPT) density computation (Navarro & Fuss, 2009)
* Log-likelihood and parameter estimation for DDMs via MLE
* Coherence-dependent drift with stimulus bias (`CoherentDDM`)
* Deterministic omission (no-response) HMM state (`OmissionCoherentDDM`)
* Direct gradient-descent fitting of HMM-DDMs via ForwardDiff through the forward algorithm
* Hidden Markov Models with DDM emissions and Dirichlet priors
* Baum–Welch (EM) training with MAP updates
* Empirical/variational inference for per-trial parameter uncertainty (`eDDM`)

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
DDMResult(rt, choice, s)
```

* `rt`: Response time
* `choice`: Decision outcome (`1` → Right, `-1` → Left)
* `s`: Stimulus direction (`1` → Right is correct, `-1` → Left is correct)

---

### `CoherentDDM`

A DDM where drift scales linearly with stimulus coherence and includes a bias term:

```
v_trial = s · k · c + v₀
```

```julia
CoherentDDM(B, k, v₀, a₀, τ)
```

* `B`: Boundary separation
* `k`: Drift gain (slope on coherence)
* `v₀`: Drift bias (stimulus-independent)
* `a₀`: Initial fraction of the boundary
* `τ`: Non-decision time

### `CoherentDDMResult`

Result of a single `CoherentDDM` trial:

```julia
CoherentDDMResult(rt, choice, s, c)
```

* `rt`: Response time
* `choice`: Decision outcome (`1` → Right, `-1` → Left)
* `s`: Stimulus direction (`1` → Right is correct, `-1` → Left is correct)
* `c`: Stimulus coherence ∈ [0, 1]

---

### `OmissionCoherentDDM`

An HMM emission for tasks with a response window, where the animal may not
respond at all (an *omission*). It wraps a `CoherentDDM` and adds a flag that
turns a state into a **deterministic omission state**:

```julia
OmissionCoherentDDM(ddm::CoherentDDM; rt_max = 60.0)   # DDM state
omission_state(; rt_max = 60.0)                        # omission state, no parameters
```

* DDM states score choice/RT trials with the coherence-dependent WFPT density
  and give omission trials the numerical floor `1e-16`.
* The omission state emits an omission with probability one and gives any
  choice/RT trial the floor. It carries no free parameters and is skipped by
  every fit.
* Both kinds share one concrete type, so an HMM's emission vector stays
  homogeneous and type-stable, and the type is parametric in the number type so
  `ForwardDiff` gradients flow through the HMM forward pass.

### `OmissionCoherentDDMResult`

```julia
OmissionCoherentDDMResult(rt, choice, s, c)   # choice == 0 encodes an omission
```

Fit an omission-aware HMM by direct gradient descent (MAP by default):

```julia
hmm = PriorHMM(init, trans, [OmissionCoherentDDM(B=3, k=2), OmissionCoherentDDM(B=5, k=1), omission_state()];
               α_trans = 10.0, share_α = true)
fit_hmm_gradient!(hmm, obs; seq_ends = seq_ends)
```

`pack_omission_hmm`, `unpack_omission_hmm`, `set_omission_hmm!`,
`omission_hmm_loglikelihood` and `omission_hmm_logposterior` expose the packed
unconstrained parameter vector for custom optimisation loops. See
[`IBL/`](IBL/README.md) for a full fitting pipeline on IBL trial data.

---

## Key Functions

### Simulation

```julia
# Standard DDM
simulateDDM(model::DriftDiffusionModel, dt::Float64=1e-5)
simulateDDM(model::DriftDiffusionModel, n::Int, dt::Float64=1e-5)

# Coherence-dependent DDM
simulateDDM(model::CoherentDDM, c::Float64, dt::Float64=1e-5)
simulateDDM(model::CoherentDDM, coherences::Vector{Float64}, dt::Float64=1e-5)
```

Simulates one or multiple trials using Euler–Maruyama integration.

### Likelihood

```julia
wfpt(t, v, B, w, τ)
logdensityof(B, v, a₀, τ, rt, choice, s)
logdensityof(B, k, v₀, a₀, τ, rt, choice, s, c)
```

Computes the WFPT density and log-likelihood for observed results.

### Fitting

```julia
fit!(model::DriftDiffusionModel, data::Vector{DDMResult}, weights=ones(length(data)))
fit!(model::CoherentDDM, data::Vector{CoherentDDMResult}, weights=ones(length(data)))
```

Fits a DDM to data using Maximum Likelihood Estimation (MLE) via L-BFGS-B. Both support observation weights for use in HMM training.

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

## Variational Inference (`eDDM`)

Fits a Gaussian variational approximation over per-trial DDM parameters:

```julia
fit_vi_gaussian(data; n_iter, K, rng, verbose, init_from_data)
```

Returns per-trial posterior means and variances over `[B, τ, v, a₀]`, along with ELBO history across iterations.

---

## References

* Navarro, D. J., & Fuss, I. G. (2009). Fast and accurate calculations for first-passage times in Wiener diffusion models.
* HiddenMarkovModels.jl — backend for HMM routines.

---

## File Structure

* `src/DriftDiffusionModels.jl` – Main module file
* `src/DDM.jl` – Core DDM definitions, WFPT density, simulation, and MLE
* `src/CoherentDDM.jl` – Coherence-dependent DDM with drift gain and bias
* `src/OmissionCoherentDDM.jl` – Omission-aware wrapper with a deterministic omission state
* `src/GradientHMM.jl` – Direct gradient-descent (MAP/MLE) fitting of CoherentDDM / omission HMMs
* `src/HMMDDM.jl` – HMM wrapper with DDM emissions and Dirichlet priors
* `src/Utilities.jl` – Initialization helpers for HMM-DDM models
* `src/eDDM.jl` – Variational inference for per-trial parameter uncertainty

---

## Contributions

Feel free to contribute pull requests or file issues to suggest features or report bugs!
