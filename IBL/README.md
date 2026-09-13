# IBL omission-aware CoherentDDM HMM fits

Fits hidden Markov models whose states are coherence-dependent drift diffusion
models (`CoherentDDM`, drift `s·k·c^α`) **plus one deterministic omission
state** to IBL trial data, by direct gradient descent on the marginal
log-likelihood (ForwardDiff through the HMM forward algorithm).

## Model

* `K` DDM states: `B, k, a₀, τ` each, one coherence exponent `α` shared across
  states (drop with `--no-share-alpha`).
* One omission state: emits "no choice" with probability one, no parameters.
  DDM states assign omissions the numerical floor `1e-16`; the omission state
  assigns that floor to any choice/RT observation.
* Each session (`eid`) is one HMM sequence. Sticky Dirichlet prior on the
  transition rows (MAP by default; `--mle` for pure maximum likelihood).

Trial encoding (`load_subject` in `fit_omission_hmm.jl`):

| field | source |
|---|---|
| `s` | `-1` if `contrastLeft` is present, `+1` if `contrastRight` is present |
| `c` | the presented contrast (0, 0.0625, 0.125, 0.25, 1) |
| `choice` | `-choice_ibl`, so `choice == s` is a correct trial; `0` = omission |
| `rt` | `reaction_time`; omissions get `rt_max` (60 s, the response window) |

## Run locally

```bash
julia --project=IBL -e 'using Pkg; Pkg.instantiate()'     # once
julia --project=IBL -t 8 IBL/fit_omission_hmm.jl --subject ZFM-04019 --K 1,2 --restarts 1 --iterations 50
```

`--top 4` (default) selects the four subjects with the most trials; `--task-id i`
picks the i-th of those for array jobs. See the header of the script for all options.

## Run on the SCC

```bash
module load julia
julia --project=IBL -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
# edit -P (project) in IBL/submit_fit.sh, then from the repository root:
qsub IBL/submit_fit.sh
```

The array job runs one (subject, K) pair per task: `-t 1-12` with K ∈ {1,2,3}
covers the top four subjects by trial count. Each task uses `-pe omp 8`
threads; HiddenMarkovModels.jl parallelises the forward pass over sessions.
Logs land in `IBL/logs/`.

Timing measured locally on the largest subject (44k trials, 51 sessions, 8
threads): about 35 s per L-BFGS iteration at K=1 and 70–80 s at K=2–3, so the
default 2 restarts × 200 iterations is roughly 4 h (K=1) to 9 h (K=3) per task.
Lower `--iterations`/`--restarts` in `submit_fit.sh` for a quick first pass.

## Outputs

`IBL/results/<subject>/`

* `summary_K<k>.csv` — one row per restart: log-likelihood, log-posterior, free
  parameter count, AIC/BIC, convergence flag, wall time (concatenate across K
  to compare models).
* `params_K<k>.csv` — per-state `B, k, α, a₀, τ`, initial probabilities and
  transition matrix (`trans_to_j` columns) of the best restart.
* `trials_K<k>.csv` — per-trial Viterbi state and posterior state probabilities,
  keyed by `eid`, `day_n`, `session_n`, `trial_n`.
* `hmm_K<k>.jls` — the fitted `PriorHMM`; reload with
  `using Serialization, DriftDiffusionModels; deserialize("hmm_K2.jls").hmm`.
