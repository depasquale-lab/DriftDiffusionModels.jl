"""
    OmissionCoherentDDM

An HMM emission model for tasks where the animal may fail to respond within the
response window (an *omission*). Two kinds of hidden state share this one
concrete type so an HMM's emission vector stays homogeneous (no dynamic
dispatch in the forward pass):

* **DDM state** (`omission = false`): wraps a [`CoherentDDM`](@ref). Choice/RT
  observations are scored with the coherence-dependent WFPT density; omission
  observations receive the numerical floor `log(1e-16)` (the same floor the
  WFPT likelihood uses for impossible RTs).
* **Omission state** (`omission = true`): a *deterministic* emission. It emits
  an omission with probability one (log-density `0`) and any choice/RT
  observation with the floor. It carries **no free parameters** — the wrapped
  `ddm` is a placeholder that is never evaluated.

`rt_max` is the response window (IBL: 60 s). It is the RT recorded for
simulated omissions and censors simulated DDM trials: a decision that has not
terminated by `rt_max` is an omission.

The numeric type `T` is parametric so the wrapped `CoherentDDM{T}` can carry
`ForwardDiff.Dual` numbers, letting gradients flow through the HMM forward
algorithm (see [`fit_hmm_gradient!`](@ref)). Every log-density returned is of
type `T`, so the emission is type-stable under autodiff.

See also [`omission_state`](@ref), [`OmissionCoherentDDMResult`](@ref).
"""
struct OmissionCoherentDDM{T<:Real}
    ddm::CoherentDDM{T}   # wrapped DDM (placeholder for the omission state)
    omission::Bool        # true ⇒ deterministic omission state
    rt_max::Float64       # response window / RT assigned to omissions
end

"""
    OmissionCoherentDDM(ddm::CoherentDDM; rt_max=60.0)

Wrap `ddm` as a (non-omission) DDM emission state with response window `rt_max`.
"""
OmissionCoherentDDM(ddm::CoherentDDM{T}; rt_max::Real=60.0) where {T<:Real} =
    OmissionCoherentDDM{T}(ddm, false, Float64(rt_max))

"""
    OmissionCoherentDDM(; B, k, α, a₀, τ, fit_α=true, rt_max=60.0)

Build a DDM emission state from CoherentDDM keyword parameters.
"""
OmissionCoherentDDM(; rt_max::Real=60.0, kwargs...) =
    OmissionCoherentDDM(CoherentDDM(; kwargs...); rt_max=rt_max)

"""
    omission_state([T=Float64]; rt_max=60.0) -> OmissionCoherentDDM{T}

Construct the deterministic omission state. The wrapped `CoherentDDM{T}` is a
placeholder with unit parameters; it is never evaluated and holds no free
parameters in any fit.
"""
omission_state(::Type{T}=Float64; rt_max::Real=60.0) where {T<:Real} =
    OmissionCoherentDDM{T}(_placeholder_ddm(T), true, Float64(rt_max))

@inline _placeholder_ddm(::Type{T}) where {T<:Real} =
    CoherentDDM{T}(one(T), one(T), one(T), T(0.5), one(T), false)

"""
    is_omission_state(m::OmissionCoherentDDM) -> Bool
"""
is_omission_state(m::OmissionCoherentDDM) = m.omission

"""
    OmissionCoherentDDMResult

A single trial observation for an [`OmissionCoherentDDM`](@ref): reaction time,
choice, stimulus direction and coherence. `choice == 0` encodes an omission
(no response inside the window); the recorded `rt` of an omission is by
convention the response window `rt_max` and is not used by the likelihood.
"""
struct OmissionCoherentDDMResult
    rt::Float64    # Reaction time (omissions: the response window)
    choice::Int    # -1 (lower/Left), +1 (upper/Right), or 0 = omission
    s::Int         # Stimulus direction: -1 or +1
    c::Float64     # Coherence ∈ [0, 1]
end

function OmissionCoherentDDMResult(; rt::Real, choice::Int, s::Int, c::Real)
    return OmissionCoherentDDMResult(Float64(rt), choice, s, Float64(c))
end

"""
    OmissionCoherentDDMResult(x::CoherentDDMResult)

Lift an ordinary CoherentDDM observation (never an omission).
"""
OmissionCoherentDDMResult(x::CoherentDDMResult) =
    OmissionCoherentDDMResult(x.rt, x.choice, x.s, x.c)

"""
    CoherentDDMResult(x::OmissionCoherentDDMResult)

Drop the omission encoding. Throws for an omission trial.
"""
function CoherentDDMResult(x::OmissionCoherentDDMResult)
    x.choice == 0 && throw(ArgumentError("cannot convert an omission trial to CoherentDDMResult"))
    return CoherentDDMResult(x.rt, x.choice, x.s, x.c)
end

"""
    is_omission(x::OmissionCoherentDDMResult) -> Bool
"""
is_omission(x::OmissionCoherentDDMResult) = x.choice == 0

#=
 Density
=#

"Log of the numerical density floor shared with the WFPT likelihood (`1e-16`)."
const OMISSION_LOGFLOOR = log(1e-16)

DensityInterface.DensityKind(::OmissionCoherentDDM) = HasDensity()

"""
    DensityInterface.logdensityof(m::OmissionCoherentDDM{T}, x::OmissionCoherentDDMResult) -> T

* omission state: `0` for an omission trial, `log(1e-16)` otherwise;
* DDM state: `log(1e-16)` for an omission trial, otherwise the CoherentDDM
  WFPT log-density of `(rt, choice)` given `(s, c)`.

The omission state never touches its placeholder `ddm`, so it is free of
parameters and costs no WFPT evaluation.
"""
@inline function DensityInterface.logdensityof(m::OmissionCoherentDDM{T},
                                               x::OmissionCoherentDDMResult) where {T<:Real}
    if m.omission
        return x.choice == 0 ? zero(T) : T(OMISSION_LOGFLOOR)
    else
        x.choice == 0 && return T(OMISSION_LOGFLOOR)
        d = m.ddm
        return T(logdensityof(d.B, d.k, d.α, d.a₀, d.τ, x.rt, x.choice, x.s, x.c))
    end
end

#=
 Simulation
=#

"""
    simulateDDM(m::OmissionCoherentDDM, s::Int, c::Float64, dt=1e-5, rng)

Simulate one trial at stimulus direction `s` and coherence `c`. The omission
state returns `(rt_max, 0, s, c)`. A DDM state runs Euler–Maruyama and is
censored at `rt_max`: if neither bound is hit inside the window the trial is an
omission.
"""
function simulateDDM(m::OmissionCoherentDDM, s::Int, c::Float64, dt::Float64=1e-5,
                     rng::AbstractRNG=Random.default_rng())
    m.omission && return OmissionCoherentDDMResult(m.rt_max, 0, s, c)

    d = m.ddm
    B  = Float64(d.B); k = Float64(d.k); α = Float64(d.α)
    a₀ = Float64(d.a₀); τ = Float64(d.τ)
    v = s * k * (c > 0 ? c^α : 0.0)

    t = 0.0
    a = a₀ * B
    sq = sqrt(dt)
    while a < B && a > 0
        if t < τ
            t += dt
        else
            a += v * dt + sq * randn(rng)
            t += dt
        end
        t ≥ m.rt_max && return OmissionCoherentDDMResult(m.rt_max, 0, s, c)
    end
    choice = a >= B ? 1 : -1
    return OmissionCoherentDDMResult(t, choice, s, c)
end

"""
    simulateDDM(m::OmissionCoherentDDM, c::Float64, dt=1e-5, rng)

Simulate one trial at coherence `c` with a random stimulus direction.
"""
simulateDDM(m::OmissionCoherentDDM, c::Float64, dt::Float64=1e-5,
            rng::AbstractRNG=Random.default_rng()) =
    simulateDDM(m, rand(rng, (-1, 1)), c, dt, rng)

"""
    Random.rand(rng, m::OmissionCoherentDDM)

Sample one trial at coherence 1.0 — required by the HiddenMarkovModels.jl
interface (`rand(hmm, T)`).
"""
Random.rand(rng::AbstractRNG, m::OmissionCoherentDDM) = simulateDDM(m, 1.0, 1e-6, rng)

#=
 EM (Baum–Welch) support — kept so `baum_welch` also works with this emission.
=#

"""
    StatsAPI.fit!(m::OmissionCoherentDDM, x::Vector{OmissionCoherentDDMResult}, w)

Weighted MLE of the wrapped `CoherentDDM` on the non-omission trials. The
omission state has nothing to fit and is returned unchanged.
"""
function StatsAPI.fit!(m::OmissionCoherentDDM, x::AbstractVector{OmissionCoherentDDMResult},
                       w::AbstractVector{<:Real}=ones(length(x)))
    m.omission && return m
    idx  = findall(r -> r.choice != 0, x)
    data = [CoherentDDMResult(x[i]) for i in idx]
    StatsAPI.fit!(m.ddm, data, w[idx])
    return m
end

"""
    StatsAPI.fit!(hmm::PriorHMM{<:Real,<:OmissionCoherentDDM}, fb, obs_seq; seq_ends)

Baum–Welch M-step for omission-aware emissions: initial/transition MAP update,
then the DDM states are refit on the non-omission trials (jointly with one
shared `α` when `hmm.share_α`). Omission states are left untouched.
"""
function StatsAPI.fit!(hmm::PriorHMM{<:Real,<:OmissionCoherentDDM},
                       fb::HiddenMarkovModels.ForwardBackwardStorage,
                       obs_seq::AbstractVector; seq_ends)
    _update_init_trans!(hmm, fb, seq_ends)

    ddm_idx = findall(m -> !m.omission, hmm.dists)
    if hmm.share_α && !isempty(ddm_idx)
        keep = findall(r -> r.choice != 0, obs_seq)
        data = [CoherentDDMResult(obs_seq[t]) for t in keep]
        fit_shared_α!([hmm.dists[i].ddm for i in ddm_idx], data, fb.γ[ddm_idx, keep])
    else
        for i in ddm_idx
            StatsAPI.fit!(hmm.dists[i], obs_seq, fb.γ[i, :])
        end
    end

    @assert HiddenMarkovModels.valid_hmm(hmm)
    return nothing
end
