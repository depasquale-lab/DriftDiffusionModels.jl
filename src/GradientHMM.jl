"""
    GradientHMM

Fit a [`PriorHMM`](@ref) with [`CoherentDDM`](@ref) emissions **directly by
gradient descent on the marginal log-likelihood**, instead of EM / Baum–Welch.

The full parameter set is mapped to an unconstrained vector `θ` (initial
distribution and transition rows via `softmax`, emission parameters via the
CoherentDDM transforms). We rebuild an HMM from `θ` and evaluate its likelihood
with HiddenMarkovModels.jl's own forward algorithm (`logdensityof(hmm, obs;
seq_ends)`); ForwardDiff differentiates straight through it. Because
`CoherentDDM` is parametric, the rebuilt emissions carry `Dual` numbers, so the
gradient reaches every parameter — transitions and emissions alike.

Shared-`α` is honoured: when `hmm.share_α` is `true`, a single coherence exponent
is estimated for all states (one `uα` entry in `θ`); otherwise each state carries
its own.

By default `fit_hmm_gradient!` returns the **MAP** estimate: it adds the
`PriorHMM`'s Dirichlet log-prior on the initial distribution and transition rows
to the forward log-likelihood (emissions keep a flat prior, i.e. MLE). This is
the same objective the Baum–Welch M-step maximises — initial/transition counts
plus `α − 1` Dirichlet pseudo-counts — so the gradient fit and EM target the same
posterior mode. Pass `prior=false` for a pure maximum-likelihood fit.

Optimising the softmax-reparameterised objective recovers the same mode as
optimising over the simplex directly: the location of an argmax is invariant
under reparameterisation, so no Jacobian/volume correction is needed (that term
would only matter for sampling or for the mode of a transformed *density*).
"""

"""
    unpack_coherent_hmm(θ, K; share_α=false) -> (init, trans, dists)

Decode an unconstrained vector `θ` into a probability vector `init` (length `K`),
a row-stochastic transition matrix `trans` (`K×K`), and `dists`, a length-`K`
vector of `CoherentDDM` emissions.

Layout of `θ`:
1. `K` initial-distribution logits → `softmax`,
2. `K` blocks of `K` transition-row logits → row-wise `softmax`,
3. emissions: `5K` entries (`uB,uk,uα,ua₀,uτ` per state), or — when
   `share_α=true` — `4K` entries (`uB,uk,ua₀,uτ` per state) plus one trailing
   shared `uα`.
"""
function unpack_coherent_hmm(θ::AbstractVector, K::Int; share_α::Bool=false)
    T = eltype(θ)

    init = softmax(θ[1:K])

    off = K
    trans = Matrix{T}(undef, K, K)
    for j in 1:K
        trans[j, :] = softmax(θ[off + (j - 1) * K + 1 : off + j * K])
    end
    off += K * K

    dists = Vector{CoherentDDM{T}}(undef, K)
    if share_α
        α = exp(θ[end])
        for i in 1:K
            o = off + (i - 1) * 4
            dists[i] = CoherentDDM(exp(θ[o + 1]), exp(θ[o + 2]), α, logistic(θ[o + 3]), exp(θ[o + 4]), false)
        end
    else
        for i in 1:K
            o = off + (i - 1) * 5
            dists[i] = CoherentDDM(exp(θ[o + 1]), exp(θ[o + 2]), exp(θ[o + 3]), logistic(θ[o + 4]), exp(θ[o + 5]), true)
        end
    end

    return init, trans, dists
end

"""
    coherent_hmm_loglikelihood(θ, obs; K, share_α=false, seq_ends=[length(obs)]) -> Real

Marginal log-likelihood `log p(obs | θ)` of a CoherentDDM-emission HMM whose
parameters are encoded in the unconstrained vector `θ` (see
[`unpack_coherent_hmm`](@ref)). The HMM is rebuilt from `θ` and its likelihood is
computed by HiddenMarkovModels.jl's forward algorithm. Differentiate this with
ForwardDiff to drive a custom gradient-descent loop:

```julia
θ = pack_coherent_hmm(hmm)
f(θ) = -coherent_hmm_loglikelihood(θ, obs; K=length(hmm), share_α=hmm.share_α, seq_ends=seq_ends)
g = ForwardDiff.gradient(f, θ)
```
"""
function coherent_hmm_loglikelihood(θ::AbstractVector, obs::AbstractVector{CoherentDDMResult};
                                    K::Int, share_α::Bool=false,
                                    seq_ends=[length(obs)])
    init, trans, dists = unpack_coherent_hmm(θ, K; share_α=share_α)
    hmm = HiddenMarkovModels.HMM(init, trans, dists)
    return DensityInterface.logdensityof(hmm, obs; seq_ends=seq_ends)
end

"""
    _dirichlet_logprior(init, trans, α_init, α_trans) -> Real

Dirichlet log-prior on the initial distribution and each transition row, up to a
constant (the `α`-dependent normaliser is dropped — it does not depend on the
parameters and so is irrelevant to the MAP argmax). Equals the `α − 1`
pseudo-count term used by the Baum–Welch MAP M-step.
"""
function _dirichlet_logprior(init, trans, α_init, α_trans)
    K = length(init)
    lp = sum((α_init .- 1) .* log.(init))
    for j in 1:K
        lp += sum((α_trans[j, :] .- 1) .* log.(@view trans[j, :]))
    end
    return lp
end

"""
    coherent_hmm_logposterior(θ, hmm, obs; seq_ends=[length(obs)]) -> Real

Unnormalised log-posterior of `hmm`'s parameters (encoded in `θ`): the
forward-algorithm log-likelihood plus the `PriorHMM` Dirichlet log-prior on the
initial distribution and transition rows (emissions have a flat prior). Drop the
prior with [`coherent_hmm_loglikelihood`](@ref). Differentiable in `θ`.
"""
function coherent_hmm_logposterior(θ::AbstractVector,
                                   hmm::PriorHMM{<:Real,<:CoherentDDM},
                                   obs::AbstractVector{CoherentDDMResult};
                                   seq_ends=[length(obs)])
    K = length(hmm)
    init, trans, dists = unpack_coherent_hmm(θ, K; share_α=hmm.share_α)
    h = HiddenMarkovModels.HMM(init, trans, dists)
    ll = DensityInterface.logdensityof(h, obs; seq_ends=seq_ends)
    return ll + _dirichlet_logprior(init, trans, hmm.α_init, hmm.α_trans)
end

"""
    pack_coherent_hmm(hmm::PriorHMM{<:Real,<:CoherentDDM}) -> Vector{Float64}

Encode the current parameters of `hmm` into the unconstrained vector `θ`
consumed by [`unpack_coherent_hmm`](@ref) / [`coherent_hmm_loglikelihood`](@ref).
The `share_α` layout is selected from `hmm.share_α`.
"""
function pack_coherent_hmm(hmm::PriorHMM{<:Real,<:CoherentDDM})
    K = length(hmm)
    θ = Float64[]
    append!(θ, log.(hmm.init))                 # softmax-invariant logits
    for j in 1:K
        append!(θ, log.(hmm.trans[j, :]))
    end
    if hmm.share_α
        for m in hmm.dists
            append!(θ, (log(m.B), log(m.k), logit(m.a₀), log(m.τ)))
        end
        push!(θ, log(hmm.dists[1].α))
    else
        for m in hmm.dists
            append!(θ, (log(m.B), log(m.k), log(m.α), logit(m.a₀), log(m.τ)))
        end
    end
    return θ
end

"""
    set_coherent_hmm!(hmm::PriorHMM{<:Real,<:CoherentDDM}, θ) -> hmm

Write the parameters encoded in `θ` back into `hmm` in place (initial
distribution, transition matrix, and each emission's `B, k, α, a₀, τ`).
"""
function set_coherent_hmm!(hmm::PriorHMM{<:Real,<:CoherentDDM}, θ::AbstractVector)
    K = length(hmm)
    init, trans, dists = unpack_coherent_hmm(θ, K; share_α=hmm.share_α)
    hmm.init  .= init
    hmm.trans .= trans
    for i in 1:K
        m, d = hmm.dists[i], dists[i]
        m.B, m.k, m.α, m.a₀, m.τ = d.B, d.k, d.α, d.a₀, d.τ
    end
    return hmm
end

"""
    fit_hmm_gradient!(hmm::PriorHMM{<:Real,<:CoherentDDM}, obs;
                      seq_ends=[length(obs)], prior=true,
                      optimizer=LBFGS(linesearch=Optim.LineSearches.BackTracking()),
                      iterations=200, show_trace=false) -> (hmm, result)

Fit `hmm` to `obs` by maximising — directly, with a ForwardDiff gradient through
HiddenMarkovModels.jl's forward pass — the log-posterior (`prior=true`, the
**MAP** default: forward log-likelihood + `PriorHMM` Dirichlet log-prior) or the
plain log-likelihood (`prior=false`, MLE). The optimisation runs unconstrained on
the packed parameter vector; the optimum is written back into `hmm` in place.
Returns the (mutated) `hmm` and the Optim result.

`seq_ends` follows the HiddenMarkovModels convention (cumulative end indices of
each sequence). Shared-`α` estimation is controlled by `hmm.share_α`.
"""
function fit_hmm_gradient!(hmm::PriorHMM{<:Real,<:CoherentDDM},
                           obs::AbstractVector{CoherentDDMResult};
                           seq_ends=[length(obs)],
                           prior::Bool=true,
                           optimizer=LBFGS(linesearch=Optim.LineSearches.BackTracking()),
                           iterations::Int=200,
                           show_trace::Bool=false)
    θ0 = pack_coherent_hmm(hmm)
    negobj(θ) = prior ?
        -coherent_hmm_logposterior(θ, hmm, obs; seq_ends=seq_ends) :
        -coherent_hmm_loglikelihood(θ, obs; K=length(hmm), share_α=hmm.share_α, seq_ends=seq_ends)
    g! = (g, θ) -> ForwardDiff.gradient!(g, negobj, θ)

    result = optimize(negobj, g!, θ0, optimizer,
                      Optim.Options(iterations=iterations, show_trace=show_trace))

    set_coherent_hmm!(hmm, Optim.minimizer(result))
    return hmm, result
end
