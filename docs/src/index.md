# DriftDiffusionModels.jl

A Julia package for fitting drift diffusion models (DDMs) to choice and
response-time data, including **hidden Markov models whose states each have
their own DDM** (HMM-DDM) and a **multilevel DDM** with trial-to-trial
parameter variability.

It was written for the analyses in the accompanying paper (see
[Citation](@ref)). For a broader toolkit of sequential sampling models, see
[SequentialSamplingModels.jl](https://github.com/itsdfish/SequentialSamplingModels.jl);
this package focuses on the state-switching (HMM) and multilevel extensions.

## Installation

The package is not yet in the General registry. Install it from GitHub
(Julia ≥ 1.10):

```julia
using Pkg
Pkg.add(url = "https://github.com/depasquale-lab/DriftDiffusionModels.jl")
Pkg.add("HiddenMarkovModels")   # needed for baum_welch / viterbi
```

## The model

Evidence ``a`` starts at ``a_0 B`` and drifts with unit diffusion
(``da = s\,v\,dt + dW``) until it hits ``0`` (left response) or ``B`` (right
response), after a non-decision time ``\tau``.

| Field | Meaning | Constraint |
|:------|:--------|:-----------|
| `B`  | boundary separation | ``B > 0`` |
| `v`  | drift **magnitude**; its sign is set by the stimulus side ``s`` | ``v \ge 0`` |
| `a₀` | starting point as a fraction of `B`; `> 0.5` biases toward *right* | ``0 < a_0 < 1`` |
| `τ`  | non-decision time (same units as RT, normally seconds) | ``0 < \tau < \mathrm{RT}`` |

The diffusion coefficient is fixed to 1, which sets the scale of `B` and `v`.

## Data format

Each trial is a [`DDMResult`](@ref)`(rt, choice, s)`:

- `rt`: response time in **seconds**, `> 0`.
- `choice`: which side the subject **responded**: `+1` = right (upper
  boundary), `-1` = left (lower boundary).
- `s`: which side was **correct**: `+1` = right, `-1` = left.

!!! warning "`choice` is the response side, not accuracy"
    Many datasets record correct/incorrect. Convert it with
    `choice = correct .* s`, where `correct` is `+1`/`-1`. Passing
    correctness directly makes every correct left response look like an
    error.

## Where to go next

- [Fitting your own data](@ref) walks from a table of trials to a fitted
  single DDM, an HMM-DDM, state decoding, choosing the number of states, and
  the multilevel DDM.
- [API reference](@ref) lists every exported function.

## Citation

If you use this package, please cite the paper and the software; see
[`CITATION.cff`](https://github.com/depasquale-lab/DriftDiffusionModels.jl/blob/main/CITATION.cff)
(GitHub's "Cite this repository" button reads it).

The package is released under the MIT license.
