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


#=
 Shared helpers: initial distribution + transition matrix (un)packing.
=#

"""
    _unpack_init_trans(θ, K) -> (init, trans, off)

Decode the leading `K + K²` entries of `θ` into `init = softmax(θ[1:K])` and a
row-stochastic `trans` (row `j` is the softmax of its `K` logits). Returns the
offset `off = K + K²` at which the emission block starts.
"""
function _unpack_init_trans(θ::AbstractVector, K::Int)
    T = eltype(θ)
    init = softmax(θ[1:K])
    off = K
    trans = Matrix{T}(undef, K, K)
    for j in 1:K
        trans[j, :] = softmax(θ[off + (j - 1) * K + 1 : off + j * K])
    end
    return init, trans, off + K * K
end

"""
    _pack_init_trans(hmm) -> Vector{Float64}

Encode `hmm.init` and the rows of `hmm.trans` as softmax-invariant logits.
"""
function _pack_init_trans(hmm::PriorHMM)
    K = length(hmm)
    θ = Float64[]
    append!(θ, log.(hmm.init))
    for j in 1:K
        append!(θ, log.(hmm.trans[j, :]))
    end
    return θ
end

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
    init, trans, off = _unpack_init_trans(θ, K)

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
    θ = _pack_init_trans(hmm)                  # softmax-invariant logits
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
    cfg = ForwardDiff.GradientConfig(negobj, θ0)
    g! = (g, θ) -> ForwardDiff.gradient!(g, negobj, θ, cfg)

    result = optimize(negobj, g!, θ0, optimizer,
                      Optim.Options(iterations=iterations, show_trace=show_trace))

    set_coherent_hmm!(hmm, Optim.minimizer(result))
    return hmm, result
end

#=
 Omission-aware HMM: PriorHMM{<:Real,<:OmissionCoherentDDM}
=#

"""
    omission_flags(hmm::PriorHMM{<:Real,<:OmissionCoherentDDM}) -> Vector{Bool}

`true` for each state that is the deterministic omission state.
"""
omission_flags(hmm::PriorHMM{<:Real,<:OmissionCoherentDDM}) = Bool[m.omission for m in hmm.dists]

"""
    n_omission_hmm_params(hmm::PriorHMM{<:Real,<:OmissionCoherentDDM}) -> Int

Length of the packed parameter vector: `K + K²` init/transition logits plus
`5` per DDM state (`4` per DDM state plus one shared `α` when `hmm.share_α`).
Omission states contribute nothing.
"""
function n_omission_hmm_params(hmm::PriorHMM{<:Real,<:OmissionCoherentDDM})
    K = length(hmm)
    nd = count(!, omission_flags(hmm))
    return K + K * K + (hmm.share_α ? 4 * nd + 1 : 5 * nd)
end

"""
    unpack_omission_hmm(θ, omission::AbstractVector{Bool}; share_α=false, rt_max=60.0)
        -> (init, trans, dists)

Decode an unconstrained vector into `init`, `trans` and a homogeneous
`Vector{OmissionCoherentDDM{eltype(θ)}}`. `omission[i]` marks state `i` as the
deterministic omission state, which consumes **no** entries of `θ`; DDM states
consume `uB,uk,uα,ua₀,uτ` each (or `uB,uk,ua₀,uτ` plus a single trailing shared
`uα` when `share_α`). `rt_max` may be a scalar or a length-`K` vector.

Layout of `θ`: `K` init logits, `K` blocks of `K` transition logits, then the
DDM-state emission blocks in state order, then (optionally) the shared `uα`.
"""
function unpack_omission_hmm(θ::AbstractVector, omission::AbstractVector{Bool};
                             share_α::Bool=false, rt_max=60.0)
    T = eltype(θ)
    K = length(omission)
    init, trans, off = _unpack_init_trans(θ, K)

    dists = Vector{OmissionCoherentDDM{T}}(undef, K)
    α_shared = share_α ? exp(θ[end]) : zero(T)
    o = off
    @inbounds for i in 1:K
        rtm = rt_max isa Number ? Float64(rt_max) : Float64(rt_max[i])
        if omission[i]
            dists[i] = OmissionCoherentDDM{T}(_placeholder_ddm(T), true, rtm)
        elseif share_α
            ddm = CoherentDDM{T}(exp(θ[o + 1]), exp(θ[o + 2]), α_shared,
                                 logistic(θ[o + 3]), exp(θ[o + 4]), false)
            dists[i] = OmissionCoherentDDM{T}(ddm, false, rtm)
            o += 4
        else
            ddm = CoherentDDM{T}(exp(θ[o + 1]), exp(θ[o + 2]), exp(θ[o + 3]),
                                 logistic(θ[o + 4]), exp(θ[o + 5]), true)
            dists[i] = OmissionCoherentDDM{T}(ddm, false, rtm)
            o += 5
        end
    end
    return init, trans, dists
end

"""
    pack_omission_hmm(hmm::PriorHMM{<:Real,<:OmissionCoherentDDM}) -> Vector{Float64}

Encode `hmm` into the unconstrained vector consumed by
[`unpack_omission_hmm`](@ref). Omission states contribute no entries.
"""
function pack_omission_hmm(hmm::PriorHMM{<:Real,<:OmissionCoherentDDM})
    θ = _pack_init_trans(hmm)
    ddms = [m.ddm for m in hmm.dists if !m.omission]
    if hmm.share_α
        isempty(ddms) && throw(ArgumentError("share_α requires at least one DDM state"))
        for d in ddms
            append!(θ, (log(d.B), log(d.k), logit(d.a₀), log(d.τ)))
        end
        push!(θ, log(ddms[1].α))
    else
        for d in ddms
            append!(θ, (log(d.B), log(d.k), log(d.α), logit(d.a₀), log(d.τ)))
        end
    end
    return θ
end

"""
    set_omission_hmm!(hmm::PriorHMM{<:Real,<:OmissionCoherentDDM}, θ) -> hmm

Write the parameters encoded in `θ` back into `hmm` in place. Emission structs
are immutable but wrap mutable `CoherentDDM`s, so the DDM parameters are updated
through the wrapped objects; omission states are untouched.
"""
function set_omission_hmm!(hmm::PriorHMM{<:Real,<:OmissionCoherentDDM}, θ::AbstractVector)
    init, trans, dists = unpack_omission_hmm(θ, omission_flags(hmm); share_α=hmm.share_α)
    hmm.init  .= init
    hmm.trans .= trans
    for i in eachindex(hmm.dists)
        hmm.dists[i].omission && continue
        m, d = hmm.dists[i].ddm, dists[i].ddm
        m.B, m.k, m.α, m.a₀, m.τ = d.B, d.k, d.α, d.a₀, d.τ
    end
    return hmm
end

"""
    omission_hmm_loglikelihood(θ, hmm, obs; seq_ends=[length(obs)]) -> Real

Marginal log-likelihood of `obs` under the omission-aware HMM whose structure
(number of states, omission flags, `share_α`) is taken from `hmm` and whose
parameters are the unconstrained vector `θ`. Computed with
HiddenMarkovModels.jl's forward algorithm; differentiable in `θ`.
"""
function omission_hmm_loglikelihood(θ::AbstractVector,
                                    hmm::PriorHMM{<:Real,<:OmissionCoherentDDM},
                                    obs::AbstractVector{OmissionCoherentDDMResult};
                                    seq_ends=[length(obs)])
    init, trans, dists = unpack_omission_hmm(θ, omission_flags(hmm); share_α=hmm.share_α,
                                             rt_max=[m.rt_max for m in hmm.dists])
    h = HiddenMarkovModels.HMM(init, trans, dists)
    return DensityInterface.logdensityof(h, obs; seq_ends=seq_ends)
end

"""
    omission_hmm_logposterior(θ, hmm, obs; seq_ends=[length(obs)]) -> Real

Forward log-likelihood plus the `PriorHMM` Dirichlet log-prior on the initial
distribution and transition rows (flat prior on emissions). Differentiable in `θ`.
"""
function omission_hmm_logposterior(θ::AbstractVector,
                                   hmm::PriorHMM{<:Real,<:OmissionCoherentDDM},
                                   obs::AbstractVector{OmissionCoherentDDMResult};
                                   seq_ends=[length(obs)])
    init, trans, dists = unpack_omission_hmm(θ, omission_flags(hmm); share_α=hmm.share_α,
                                             rt_max=[m.rt_max for m in hmm.dists])
    h = HiddenMarkovModels.HMM(init, trans, dists)
    ll = DensityInterface.logdensityof(h, obs; seq_ends=seq_ends)
    return ll + _dirichlet_logprior(init, trans, hmm.α_init, hmm.α_trans)
end

"""
    fit_hmm_gradient!(hmm::PriorHMM{<:Real,<:OmissionCoherentDDM}, obs;
                      seq_ends=[length(obs)], prior=true, optimizer=LBFGS(...),
                      iterations=200, show_trace=false) -> (hmm, result)

Direct gradient-descent fit (ForwardDiff through the forward algorithm) of an
omission-aware HMM: MAP by default (`prior=true`), MLE with `prior=false`. The
deterministic omission state has no parameters and is skipped by the
optimiser; DDM states are estimated exactly as in the `CoherentDDM` method,
including shared `α` when `hmm.share_α`. The optimum is written back into `hmm`.
"""
function fit_hmm_gradient!(hmm::PriorHMM{<:Real,<:OmissionCoherentDDM},
                           obs::AbstractVector{OmissionCoherentDDMResult};
                           seq_ends=[length(obs)],
                           prior::Bool=true,
                           optimizer=LBFGS(linesearch=Optim.LineSearches.BackTracking()),
                           iterations::Int=200,
                           show_trace::Bool=false)
    θ0 = pack_omission_hmm(hmm)
    negobj(θ) = prior ?
        -omission_hmm_logposterior(θ, hmm, obs; seq_ends=seq_ends) :
        -omission_hmm_loglikelihood(θ, hmm, obs; seq_ends=seq_ends)
    cfg = ForwardDiff.GradientConfig(negobj, θ0)
    g! = (g, θ) -> ForwardDiff.gradient!(g, negobj, θ, cfg)

    result = optimize(negobj, g!, θ0, optimizer,
                      Optim.Options(iterations=iterations, show_trace=show_trace))

    set_omission_hmm!(hmm, Optim.minimizer(result))
    return hmm, result
end
