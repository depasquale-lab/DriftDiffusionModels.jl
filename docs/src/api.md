# API reference

## Single DDM

```@docs
DriftDiffusionModel
DDMResult
simulateDDM
wfpt
DriftDiffusionModels.logdensityof(::DriftDiffusionModel, ::DDMResult)
DriftDiffusionModels.logdensityof(::Real, ::Real, ::Real, ::Real, ::Float64, ::Int, ::Int)
fit!(::DriftDiffusionModel, ::Vector{DDMResult}, ::AbstractVector{<:Real})
```

## HMM with DDM emissions

`baum_welch`, `viterbi`, `forward_backward` and `logdensityof(hmm, obs; seq_ends)`
come from [HiddenMarkovModels.jl](https://github.com/gdalle/HiddenMarkovModels.jl)
and work on [`PriorHMM`](@ref).

```@docs
PriorHMM
init_hmm_ddm
calculate_ll_ratio
```

## Multilevel DDM (exact marginal likelihood)

```@docs
fit_mlddm_exact
MLDDMFit
marginal_loglik
quadrature_check
trial_posteriors
profile_sigma0
gauss_hermite
```

## Multilevel DDM (legacy variational inference)

```@docs
fit_vi_gaussian
```
