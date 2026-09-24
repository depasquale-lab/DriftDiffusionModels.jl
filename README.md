# DriftDiffusionModels.jl

[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://depasquale-lab.github.io/DriftDiffusionModels.jl/dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Fit drift diffusion models (DDMs) to choice and response-time data in Julia,
including:

* **HMM-DDM**: hidden Markov models in which each latent state has its own DDM
  (e.g. engaged vs. disengaged), fit by EM with Dirichlet priors on the
  transitions, with state decoding through
  [HiddenMarkovModels.jl](https://github.com/gdalle/HiddenMarkovModels.jl).
* **Multilevel DDM**: trial-to-trial variability in all four DDM parameters,
  fit by exact (quadrature) marginal likelihood.
* The single-trial DDM building blocks: WFPT density (Navarro & Fuss, 2009),
  maximum-likelihood fitting, and Euler–Maruyama simulation.

This package was developed for the analyses in the accompanying paper (see
[Citation](#citation)). For a broader library of sequential sampling models,
see [SequentialSamplingModels.jl](https://github.com/itsdfish/SequentialSamplingModels.jl);
this package is not yet integrated with it.

**Documentation, including a step-by-step tutorial for fitting your own data:
<https://depasquale-lab.github.io/DriftDiffusionModels.jl/dev/>**

## Installation

The package is not registered yet. Install from GitHub (Julia ≥ 1.10):

```julia
using Pkg
Pkg.add(url = "https://github.com/depasquale-lab/DriftDiffusionModels.jl")
Pkg.add("HiddenMarkovModels")   # for baum_welch / viterbi / forward_backward
```

## Quick start

Each trial is a `DDMResult(rt, choice, s)`:

| field    | meaning                                                  |
|----------|----------------------------------------------------------|
| `rt`     | response time in **seconds**                             |
| `choice` | side **responded**: `+1` right (upper bound), `-1` left |
| `s`      | side that was **correct**: `+1` right, `-1` left         |

> [!IMPORTANT]
> `choice` is the response side, **not** accuracy. If your data record
> correct/incorrect as `±1`, use `choice = correct .* s`.

```julia
using DriftDiffusionModels, HiddenMarkovModels, Random

# rt, correct (±1), correct_side (±1), session: one entry per trial,
# sorted by session and then by trial order
choice = correct .* correct_side
data   = [DDMResult(rt[i], choice[i], correct_side[i]) for i in eachindex(rt)]
seq_ends = cumsum([count(==(s), session) for s in unique(session)])

# single DDM (maximum likelihood)
ddm = DriftDiffusionModel(; τ = 0.1)          # fields: B, v, a₀, τ
fit!(ddm, data)

# 2-state HMM-DDM
hmm0 = init_hmm_ddm(MersenneTwister(1), data, 2)
hmm, lls = baum_welch(hmm0, data; seq_ends)
hmm.dists, hmm.trans                                  # per-state DDMs, transitions
states, _ = viterbi(hmm, data; seq_ends)              # most likely state per trial
γ, _      = forward_backward(hmm, data; seq_ends)     # posterior state probabilities

# multilevel DDM (m, σ0 are on the unconstrained scale, ordered B, τ, v, a₀)
fit = fit_mlddm_exact(data; q = 8, n_starts = 4)
```

The [tutorial](https://depasquale-lab.github.io/DriftDiffusionModels.jl/dev/tutorial/)
runs this end to end on simulated data, covering data preparation from a CSV,
choosing the number of states by held-out likelihood, setting priors, and
checking multilevel fits.

## Parameters

`DriftDiffusionModel(B, v, a₀, τ)` with unit diffusion noise:

* `B`: boundary separation (`> 0`)
* `v`: drift magnitude (`≥ 0`); its sign is set by the stimulus side `s`
* `a₀`: starting point as a fraction of `B` (`0–1`); `> 0.5` biases toward right
* `τ`: non-decision time (s)

## Reproducing the paper

The figures and analyses in the paper are produced by a separate analysis
repository, which uses this package:
**TODO: link to analysis repository**. It has instructions for obtaining the
data and regenerating each figure.

In this repository:

* `notebooks/`: worked examples (`ExampleDDM`, `ExampleDDMHMM`, `ExampleeDDM`).
  The mouse examples expect the dataset at `data/mouse_df.csv`, which is not
  distributed here. Activate the notebook environment with
  `julia --project=notebooks`.
* `experiments/vi_validation/`: simulation study validating the multilevel DDM
  fits (see its README).
* `test/`: run with `julia --project -e 'using Pkg; Pkg.test()'`.

## Citation

If you use this package, please cite the paper and the software. Citation
metadata is in [`CITATION.cff`](CITATION.cff) (GitHub shows it under "Cite
this repository").

> TODO: paper reference.

## License

MIT. See [LICENSE](LICENSE).

## Contributing

Issues and pull requests are welcome.

## References

* Navarro, D. J., & Fuss, I. G. (2009). Fast and accurate calculations for
  first-passage times in Wiener diffusion models. *Journal of Mathematical
  Psychology*, 53(4), 222–230.
