# Variational inference validation

This suite tests `fit_vi_gaussian` against simulated data and exact numerical
references. It checks hyperparameter recovery, serial-correlation diagnostics,
Monte Carlo sensitivity, and the mean-field approximation.

The experiments call package internals directly. Trials are sampled by
inverting the first-passage-time CDF, avoiding discretization error from
`simulateDDM`.

## Run

```bash
julia --project=experiments/vi_validation -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia -t auto --project=experiments/vi_validation experiments/vi_validation/run_experiment.jl all --preset=full --seeds=5
```

Options are `--preset=quick|full`, `--seeds=N`, and `--out=DIR`. The default is
`quick`. Results are written as CSV files plus `summary.txt`.

## Stages

| Stage | Test | Output |
| --- | --- | --- |
| `recovery` | Recovery across sample sizes and initial `σ0` values | `recovery.csv`, `recovery_trace.csv` |
| `acf` | Recovered ACF for i.i.d., switching, and shuffled trials | `acf.csv`, `acf_summary.csv` |
| `ksens` | Sensitivity to Monte Carlo sample count and redraw policy | `ksens.csv` |
| `refpost` | Mean-field posterior against dense quadrature | `refpost.csv`, `refpost_corr.csv` |
| `fullcov` | Diagonal versus full-covariance trial posteriors | `fullcov.csv` |
| `exact` | Variational inference versus exact marginal likelihood | `exact_vs_vi.csv` |

`recovery` includes the package's hard-coded initial spread and several
multiples of the true spread. A fitted spread that follows its initial value
indicates weak identification.

`acf` uses three controls: truly i.i.d. parameters, a sticky switching process,
and the switching trials after shuffling. The i.i.d. and shuffled estimates
should be near zero.

`ksens` scores all fits with a shared Monte Carlo sample so ELBO values are
comparable. `refpost` reports posterior mean error, variance ratios,
correlations, and grid-edge mass. `fullcov` isolates the effect of the
mean-field restriction. `exact` compares initialization sensitivity; its
log-likelihood and the variational ELBO are not directly comparable.

## Quick-preset findings

Rerun the full preset before reporting these findings.

- Fitted `σ0` follows its initialization. The package default can diverge.
- The fixed-sample ELBO is not monotone under the current coordinate updates.
- The ACF controls behave correctly: i.i.d. and shuffled ACFs are near zero,
  while switching data remain positive but attenuated.
- `K=3` leaves a substantial ELBO gap relative to larger Monte Carlo samples,
  although the ACF conclusion is stable.
- Dense posterior comparisons show small mean errors, mild variance
  underestimation, and modest correlations. Full covariance does not change the
  qualitative conclusions in the quick run.
- Exact marginal likelihood is stable across the tested initializations, while
  variational estimates are not. Inspect boundary estimates with
  `profile_sigma0`.

## Exact replacement

```julia
fit = fit_mlddm_exact(data; q=8, qτ=16, n_starts=4)
μ, sd = trial_posteriors(data, fit.m, fit.σ0)
```

`fit_vi_gaussian` remains available for reproducing earlier results. Exact
marginal likelihood removes the variational approximation but not weak
identification of between-trial variances.

## Dependencies

The experiment environment pins `Optim = "1"` because the package still uses
the legacy `autodiff=:forward` keyword in `src/DDM.jl` and `src/eDDM.jl`. It also
pins `HiddenMarkovModels = "0.7"`, which provides the API used by
`src/HMMDDM.jl`.

## References

- Galdo, M., Bahg, G., & Turner, B. M. (2020). Variational Bayesian methods for
  cognitive science. *Psychological Methods*, 25, 535–559.
- Turner, B. M., Sederberg, P. B., Brown, S. D., & Steyvers, M. (2013). A method
  for efficiently sampling from distributions with correlated dimensions.
  *Psychological Methods*, 18, 368–384.
- Navarro, D. J., & Fuss, I. G. (2009). Fast and accurate calculations for
  first-passage times in Wiener diffusion models. *JMP*, 53, 222–230.
