"""
    Transforms

Bijective maps between constrained model parameters and an unconstrained
real-valued space, plus the simplex/log-sum-exp helpers needed to optimise
HMM probabilities without box constraints. All functions are written to be
differentiable (ForwardDiff-friendly) so they can be composed into objectives
that are minimised with plain gradient-based optimisers.
"""

"""
    logit(p)

Inverse of [`logistic`](@ref): maps `p ∈ (0, 1)` to the whole real line.
"""
logit(p) = log(p) - log1p(-p)

"""
    logsumexp(v)

Numerically stable `log(sum(exp, v))` for a real vector `v`. Returns `-Inf`
when every entry is `-Inf`.
"""
function logsumexp(v::AbstractVector{<:Real})
    m = maximum(v)
    (isinf(m) && m < 0) && return m          # all -Inf ⇒ log(0)
    return m + log(sum(x -> exp(x - m), v))
end

"""
    logsoftmax(v)

Stable elementwise `v .- logsumexp(v)`. The result `r` satisfies
`sum(exp.(r)) == 1`, i.e. `exp.(logsoftmax(v))` is a probability vector.
"""
logsoftmax(v::AbstractVector{<:Real}) = v .- logsumexp(v)

"""
    softmax(v)

Map an unconstrained real vector to a point on the probability simplex.
"""
softmax(v::AbstractVector{<:Real}) = exp.(logsoftmax(v))
