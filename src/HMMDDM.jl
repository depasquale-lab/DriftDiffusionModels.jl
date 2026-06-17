"""
    calculate_ll_ratio(ℓℓ::Float64, ℓℓ₀::Float64, n::Int) -> Float64

Calculate the log-likelihood ratio between a multi-state DDM and a single-state DDM. Importantly, we express this ratio in bits.
"""
function calculate_ll_ratio(ℓℓ::Float64, ℓℓ₀::Float64, n::Int)
    ll_ratio = ℓℓ - ℓℓ₀
    return ll_ratio / (n * log(2))
end


"""
    PriorHMM(init, trans, dists; α_trans = 1, α_init = 1)
    PriorHMM(init, trans, dists, α_trans, α_init)

Construct a hidden‑Markov model with Dirichlet priors.

# Arguments
- `init::Vector{<:Real}`: Initial state probabilities `π` (must sum to 1).
- `trans::Matrix{<:Real}`: Transition matrix `A` where each row sums to 1.
- `dists::Vector{D}`: Emission distributions for each state.
- `α_trans`: Hyper‑parameters for the transition rows.  Accepts a scalar
  (symmetric prior) **or** a `K×K` matrix for asymmetric priors.
- `α_init`: Hyper‑parameters for the initial distribution.  Accepts a
  scalar **or** a length‑`K` vector.

All hyper‑parameters must be strictly positive.  In the EM M‑step they
enter as *α − 1* pseudo‑counts.

# Examples
```julia
K = 3
αT = ones(K, K);      αT[diagind(αT)] .= 15          # sticky prior
απ = fill(2.0, K)

hmm = PriorHMM(init, trans, dists; α_trans = αT, α_init = απ)
baum_welch(hmm, sequences; loglikelihood_increasing = false)
```
"""
struct PriorHMM{T<:Real,D} <: HiddenMarkovModels.AbstractHMM
    "Initial state probabilities (π)."
    init     :: Vector{T}
    "Transition matrix (A)."
    trans    :: Matrix{T}
    "Emission distributions—one per hidden state."
    dists    :: Vector{D}
    "Dirichlet hyper‑parameters for *rows* of `A`."
    α_trans  :: Matrix{T}
    "Dirichlet hyper‑parameters for `π`."
    α_init   :: Vector{T}
    "Estimate a single emission nonlinearity (`α`) shared across all states."
    share_α  :: Bool

    function PriorHMM(init::Vector{T}, trans::Matrix{T}, dists::Vector{D};
                      α_trans = one(T), α_init = one(T), share_α::Bool = false) where {T<:Real,D}
        K = length(init)
        @assert size(trans) == (K, K)       "`trans` must be K×K"
        @assert abs(sum(init) - one(T)) < 1e-8       "`init` must sum to 1"
        @assert all(abs.(sum(trans; dims = 2) .- 1) .< 1e-8) "rows of `trans` must sum to 1"

        αT = isa(α_trans, Number) ? fill(T(α_trans), K, K) : Matrix{T}(α_trans)
        αI = isa(α_init,  Number) ? fill(T(α_init),  K)   : Vector{T}(α_init)

        @assert all(αT .> zero(T)) "α_trans must be positive"
        @assert all(αI .> zero(T)) "α_init  must be positive"

        new{T,D}(copy(init), copy(trans), deepcopy(dists), αT, αI, share_α)
    end
end

# Positional‑argument constructor
PriorHMM(init, trans, dists, αT, αI; share_α::Bool = false) =
    PriorHMM(init, trans, dists; α_trans = αT, α_init = αI, share_α = share_α)

Base.length(hmm::PriorHMM) = length(hmm.init)

HiddenMarkovModels.initialization(hmm::PriorHMM)    = hmm.init
HiddenMarkovModels.transition_matrix(hmm::PriorHMM) = hmm.trans
HiddenMarkovModels.obs_distributions(hmm::PriorHMM) = hmm.dists

######################################################################
# Prior log‑density
######################################################################

"""
    logdensityof(hmm::PriorHMM) -> Real

Return the natural‑log density of the Dirichlet priors
`p(π, A | α_init, α_trans)` evaluated at the current parameters.

Used for posterior computation.
"""
function DensityInterface.logdensityof(hmm::PriorHMM)
    K  = length(hmm)
    lp = 0.0
    # transition rows
    for i in 1:K
        α = hmm.α_trans[i, :]
        lp += lgamma(sum(α)) - sum(lgamma.(α)) + sum((α .- 1) .* log.(hmm.trans[i, :]))
    end
    # initial distribution
    απ = hmm.α_init
    lp += lgamma(sum(απ)) - sum(lgamma.(απ)) + sum((απ .- 1) .* log.(hmm.init))
    return lp
end

######################################################################
# Baum–Welch M‑step (MAP)
######################################################################

"""
    StatsAPI.fit!(hmm::PriorHMM,
                  fb::HiddenMarkovModels.ForwardBackwardStorage,
                  obs_seq::AbstractVector; seq_ends)

Perform the M‑step of the Baum–Welch algorithm **with Dirichlet
pseudo‑counts**.

The forward–backward storage `fb` must already contain γ and ξ arrays.
Dirichlet parameters contribute `α − 1` additional counts before
normalising.

This method is called internally by `HiddenMarkovModels.baum_welch!` and
is not intended for direct use.
"""
function StatsAPI.fit!(hmm::PriorHMM,
                       fb::HiddenMarkovModels.ForwardBackwardStorage,
                       obs_seq::AbstractVector; seq_ends)
    K = length(hmm)

    _update_init_trans!(hmm, fb, seq_ends)

    # update each emission model using state marginals γ
    for i in 1:K
        StatsAPI.fit!(hmm.dists[i], obs_seq, fb.γ[i, :])
    end

    @assert HiddenMarkovModels.valid_hmm(hmm)
    return nothing
end

"""
    _update_init_trans!(hmm::PriorHMM, fb, seq_ends)

Apply the Baum–Welch M-step update for the initial distribution `π` and the
transition matrix `A`, including the Dirichlet `α − 1` pseudo-counts. Factored
out so emission-specific `fit!` methods (e.g. shared-`α` CoherentDDM) can reuse
the identical transition update.
"""
function _update_init_trans!(hmm::PriorHMM,
                             fb::HiddenMarkovModels.ForwardBackwardStorage,
                             seq_ends)
    init_counts  = hmm.α_init  .- 1           # prior for π
    trans_counts = hmm.α_trans .- 1           # prior for A

    for k in eachindex(seq_ends)
        t1, t2 = HiddenMarkovModels.seq_limits(seq_ends, k)
        init_counts  .+= fb.γ[:, t1]
        trans_counts .+= sum(fb.ξ[t1:t2])
    end

    hmm.init  .= init_counts ./ sum(init_counts)
    hmm.trans .= trans_counts ./ sum(trans_counts; dims = 2)
    return nothing
end
