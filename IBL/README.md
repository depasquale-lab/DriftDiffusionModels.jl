# IBL omission-aware CoherentDDM HMM fits

Fits hidden Markov models whose states are coherence-dependent drift diffusion
models (`CoherentDDM`, drift `s·k·c^α`) **plus one deterministic omission
state** to IBL trial data, by direct gradient descent on the marginal
log-likelihood (ForwardDiff through the HMM forward algorithm).

## Model

* `K` DDM states: `B, k, a₀, τ` each, one coherence exponent `α` shared across
  states (drop with `--no-share-alpha`). Default sweep: K = 3, 4, 5.
* One omission state: emits "no choice" with probability one, no parameters.
  DDM states assign omissions the numerical floor `1e-16`; the omission state
  assigns that floor to any choice/RT observation.
* Each session (`eid`) is one HMM sequence. Sticky Dirichlet prior on the
  transition rows (MAP by default; `--mle` for pure maximum likelihood).
* 10 random restarts per (subject, K) by default; each restart's seed is derived
  deterministically from (subject, K, restart, --seed), so any restart can be
  rerun and reproduced in isolation (`--restart-id r`).

Trial encoding (`load_subject` in `common.jl`):

| field | source |
|---|---|
| `s` | `-1` if `contrastLeft` is present, `+1` if `contrastRight` is present |
| `c` | the presented contrast (0, 0.0625, 0.125, 0.25, 1) |
| `choice` | `-choice_ibl`, so `choice == s` is a correct trial; `0` = omission |
| `rt` | `reaction_time`; omissions get `rt_max` (60 s, the response window) |

## Files

* `common.jl` — data loading, HMM initialisation, output frames (shared).
* `fit_omission_hmm.jl` — fit restarts; writes each restart's summary + model
  the moment it finishes (`<out>/<subject>/restarts/`).
* `collect_results.jl` — aggregate finished restarts, pick the best per
  (subject, K), write the shareable CSVs. Idempotent; run it any time, also
  while fits are still running.
* `submit_fit.sh` — SGE array job, one task per (subject, K, restart).
* `collect.sh` — SGE job wrapping the collector (submit held on the array).

## Run locally

```bash
julia --project=IBL -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'   # once
julia --project=IBL -t 8 IBL/fit_omission_hmm.jl --subject ZFM-04019 --K 3 --restarts 2 --iterations 50
julia --project=IBL IBL/collect_results.jl
```

`--top 6` (default) selects the six subjects with the most trials; `--task-id i`
picks the i-th of those, `--restart-id r` runs a single restart (both for array
jobs). See the header of the script for all options.

## Run on the SCC

```bash
julia --project=IBL -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate(); Pkg.precompile()'
# from the repository root:
qsub IBL/submit_fit.sh
qsub -hold_jid ibl_omission_ddm IBL/collect.sh      # auto-collect when all tasks finish
```

The array runs one task per (subject, K, restart): with the defaults (top 6
subjects by trial count, K ∈ {3,4,5}, 10 restarts) that is `-t 1-180`. Tasks
are ordered subject-major, so the animals with the most trials are scheduled
first. Keep the `-t` range in sync with TOP/KS/RESTARTS in `submit_fit.sh`.
Logs land in `IBL/logs/`.

While the array is running, get a snapshot of everything finished so far with

```bash
julia --project=IBL IBL/collect_results.jl
```

Timing (44k-trial subject, 8 threads): ~1–4 s per L-BFGS iteration at K=3–5,
so a 1000-iteration restart-task takes tens of minutes to a few hours (h_rt is
36 h). Resubmitting with a higher ITERATIONS warm-starts every finished restart
from its saved model and runs only the remaining iterations; restarts that
already converged or reached the target are skipped, so resubmission is cheap
and idempotent.

## Outputs

Written incrementally by the fit tasks, `IBL/results/<subject>/restarts/`:

* `summary_K<k>_r<r>.csv` — one row: log-likelihood, log-posterior, `score`
  (the model-selection criterion actually optimised), free parameter count,
  AIC/BIC, seed, convergence flag, wall time, host, timestamp.
* `hmm_K<k>_r<r>.jls` — that restart's fitted `PriorHMM`
  (`Serialization.deserialize(...).hmm` after `using DriftDiffusionModels`).

Written by `collect_results.jl` (rerun it for a fresh snapshot):

* `IBL/results/all_restarts.csv` — every finished restart, one row each.
* `IBL/results/model_comparison.csv` — best restart per (subject, K) with
  loglik / AIC / BIC, for the K = 3 vs 4 vs 5 comparison.
* `IBL/results/<subject>/summary_K<k>.csv` — all finished restarts of that K.
* `IBL/results/<subject>/params_K<k>.csv` — best restart: per-state
  `B, k, α, a₀, τ`, initial probabilities, transition matrix (`trans_to_j`).
* `IBL/results/<subject>/trials_K<k>.csv` — best restart: per-trial Viterbi
  state and posterior state probabilities (`p_state_j`), keyed by
  `eid, day_n, session_n, trial_n`.
* `IBL/results/<subject>/hmm_K<k>.jls` — the best restart's model.

CSVs are the shareable record (open anywhere: Python/R/Excel); the `.jls`
files are for exact reloading in Julia. We deliberately don't write BSON:
BSON.jl embeds Julia type definitions, so files break whenever a struct
changes — everything a collaborator needs is in the CSVs.
