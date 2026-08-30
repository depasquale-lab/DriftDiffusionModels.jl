"""
    DriftDiffusionModel

Implement a simple drift diffusion model: da/dt = v + σdW.
"""
mutable struct DriftDiffusionModel
    B::Float64 # Boundary Separation
    v::Float64 # Drift Rate
    a₀::Float64 # Initial Accumulation: parameterized as a fraction of B
    τ::Float64 # Non-decision time
end

function DriftDiffusionModel(;
    B::Float64=5.0, #Bound Height
    v::Float64=1.0, # Drift Rate
    a₀::Float64=0.5, # Initial Accumulation
    τ::Float64=1e-3, # Non-decision time
) 
    return DriftDiffusionModel(B, v, a₀, τ)
end

"""
    DDMResult

A tuple of RT and choice. The first element is the RT, the second is the choice.
"""
struct DDMResult
    rt::Float64 # RT
    choice::Int # Choice (-1 or 1)
    s::Int # Stimulus (-1 or 1)
end

function DDMResult(;rt::Float64, choice::Int, s::Int)
    return DDMResult(rt, choice, s)
end

"""
    wfpt(t, v, B, z, τ, err=1e-8)

Calculate the Wiener First Passage Time (WFPT) density at time t for a drift diffusion model
with drift rate v, boundary separation B, starting point α₀, non-decision time τ, and error tolerance err.

This implementation follows the algorithm described in Navarro & Fuss (2009).
"""
function wfpt(t::Real, v::Real, B::Real, w::Real, τ::Real, err::Real=1e-8)
    if !(isfinite(t) && isfinite(v) && isfinite(B) && isfinite(w) && isfinite(τ)) || B <= 0
        return oftype(float(t * v * B * w * τ), NaN)
    end

    # Check for valid inputs
    if t <= τ
        return 0.0
    end
    
    # Use normalized time and relative start point
    tt = (t - τ) / (B^2)
    
    # Calculate number of terms needed for large t version
    if π * tt * err < 1  # if error threshold is set low enough
        kl = sqrt(-2 * log(π * tt * err) / (π^2 * tt))  # bound
        kl = max(kl, 1 / (π * sqrt(tt)))  # ensure boundary conditions met
    else  # if error threshold set too high
        kl = 1 / (π * sqrt(tt))  # set to boundary condition
    end
    
    # Calculate number of terms needed for small t version
    if 2 * sqrt(2 * π * tt) * err < 1  # if error threshold is set low enough
        ks = 2 + sqrt(-2 * tt * log(2 * sqrt(2 * π * tt) * err))  # bound
        ks = max(ks, sqrt(tt) + 1)  # ensure boundary conditions are met
    else  # if error threshold was set too high
        ks = 2  # minimal kappa for that case
    end
    
    # Compute f(tt|0,1,w)
    # Bound work for extreme inputs.
    Kcap = 1.0e4
    p = 0.0  # initialize density
    if ks < kl  # if small t is better...
        isfinite(ks) || return oftype(float(t * v * B * w * τ), NaN)
        ks = min(ks, Kcap)  # cap kappa to avoid long loops
        K = ceil(Int, ks)  # round to smallest integer meeting error
        for k in -floor(Int, (K-1)/2):ceil(Int, (K-1)/2)  # loop over k
            p += (w + 2 * k) * exp(-((w + 2 * k)^2) / 2 / tt)  # increment sum
        end
        p /= sqrt(2 * π * tt^3)  # add constant term
    else  # if large t is better...
        isfinite(kl) || return oftype(float(t * v * B * w * τ), NaN)
        kl = min(kl, Kcap)  # cap kappa to avoid long loops
        K = ceil(Int, kl)  # round to smallest integer meeting error
        for k in 1:K
            p += k * exp(-(k^2) * (π^2) * tt / 2) * sin(k * π * w)  # increment sum
        end
        p *= π  # add constant term
    end
    
    # Convert to f(t|v,B,w)
    density = p * exp(-v * B * w - (v^2) * (t - τ) / 2) / (B^2)
    return max(density, 0.0)  # ensure non-negative density (occasionaly generates neg values e.g., -1e-21) (maybe return +ϵ instead?)
end

"""
    simulateDDM(model::DriftDiffusionModel, dt::FLoat64=1e-5)

Generate a single trial of a drift diffusion model using the Euler-Maruyama method.
"""
function simulateDDM(model::DriftDiffusionModel, dt::Float64=1e-5, rng::AbstractRNG=Random.default_rng())
    @unpack B, v, a₀, τ = model

    # generate a stimulus side randomly
    s = rand(rng, [-1, 1])  # -1 = Left is correct, +1 = Right is correct
    v = s * v  # make drift rate signed according to stimulus

    # initialize variables
    t = 0.0
    a = a₀ * B  # initial accumulation

    while a < B && a > 0
        # run the model forward in time
        if t < τ
            t += dt
        else
            a += v * dt + (sqrt(dt) * randn(rng))
            t += dt
        end
    end

    if a >= B
        choice = 1  # upper boundary hit
    else
        choice = -1  # lower boundary hit
    end

    return DDMResult(t, choice, s)
end

"""
    simulateDDM(model::DriftDiffusionModel, n::Int, dt::Float64=1e-5)
"""
function simulateDDM(model::DriftDiffusionModel, n::Int, dt::Float64=1e-5)
    results = Vector{DDMResult}(undef, n)
    @threads for i in 1:n
        results[i] = simulateDDM(model, dt)
    end
    return results
end

"""
    Random.rand(rng::AbstractRNG, model::DriftDiffusionModel)

Generate a single trial of the drift diffusion model using the Euler-Maruyama method--needed for HiddenMarkovModels.jl
"""
function Random.rand(rng::AbstractRNG, model::DriftDiffusionModel)
    # Generate a single trial of the drift diffusion model
    return simulateDDM(model, 1e-6, rng)
end

# Tell Density Interface that DDMResult is a distribution
DensityInterface.DensityKind(::DriftDiffusionModel) = HasDensity()

"""
    DensityInterface.logdensityof(model::DriftDiffusionModel, x::DDMResult)

Calculate the loglikelihood of a drift diffusion model given a DDMResult.
"""
function DensityInterface.logdensityof(model::DriftDiffusionModel, x::DDMResult)
    @unpack B, v, a₀, τ = model
    @unpack rt, choice, s = x
    
    return logdensityof(B, v, a₀, τ, rt, choice, s)
end

"""
    logdensityof(B, v, a₀, τ, rt, choice; s=+1)

Trial log-density given stimulus side s ∈ {-1,+1}.
- choice: -1 = lower (Left), +1 = upper (Right)
- s: +1 means "Right is correct", -1 means "Left is correct"
"""
function logdensityof(B::TB, v::TV, a₀::TA, τ::TT,
                      rt::Float64, choice::Int, s::Int
) where {TB<:Real, TV<:Real, TA<:Real, TT<:Real}
    if rt <= 0
        return -Inf
    end
    T = promote_type(TB, TV, TA, TT)
    B, v, a₀, τ = T(B), T(v), T(a₀), T(τ)

    @assert (s == 1 || s == -1) "stimulus s must be ±1"
    @assert (choice == 1 || choice == -1) "choice must be ±1"

    # v is a MAGNITUDE, use:
    v_trial = s * abs(v)

    # wfpt is LOWER-boundary density. For upper responses, reflect:
    if choice == -1             # lower/Left
        v_eff =  v_trial
        w_eff =  a₀
    else                        # upper/Right
        v_eff = -v_trial
        w_eff = 1 - a₀
    end

    dens = wfpt(rt, v_eff, B, w_eff, τ)
    dens = max(dens, 1e-16)  # avoid log(0)

    logdens = log(dens)
    return isfinite(logdens) ? logdens : -1e16
end


"""
    StatsAPI.fit!(model::DriftDiffusionModel, x::Vector{DDMResult}, w::Vector{Float64}=ones(length(x)))

Perform parameter estimation of a drift diffusion model using MLE given a vector of DDM observtions. Takes an optional weights vector to support for use in an HMM.
"""
function StatsAPI.fit!(model::DriftDiffusionModel, x::Vector{DDMResult}, w::AbstractVector{<:Real}=ones(length(x)))
    @unpack B, v, a₀, τ = model
    
    # Define negative log-likelihood function for optimization
    function neg_log_likelihood(params)
        # We optimize B, drift rate, and a₀ as a fraction of B
        B_temp, v_temp, a₀_temp, τ_temp = params
        
        # Early return for invalid boundary (must be positive)
        if B_temp < 0
            return convert(typeof(B_temp), Inf)
        end
        
        # Calculate log-likelihood using the raw parameters version
        ll = 0.0
        for i in 1:length(x)
            ll += w[i] * logdensityof(B_temp, v_temp, a₀_temp, τ_temp, x[i].rt, x[i].choice, x[i].s)
        end
        
        # Return negative since optimizers minimize
        return -ll
    end
    
    # Set up optimization
    initial_params = [B, v, a₀, τ]
    
    # Add bounds - a₀_frac must be between -1 and 1
    lower_bounds = [0.001, 0, 0, 1e-3]
    upper_bounds = [50.0, 10.0, 1.0, 5.0]

    # Optimize using L-BFGS-B to respect the bounds
    result = optimize(neg_log_likelihood, lower_bounds, upper_bounds, initial_params, Fminbox(LBFGS(linesearch=Optim.LineSearches.BackTracking())), autodiff=:forward)

    # Extract the optimized parameters
    optimal_params = Optim.minimizer(result)
    
    # Update the model with new parameter estimates
    model.B = optimal_params[1]
    model.v = optimal_params[2]
    model.a₀ = optimal_params[3]
    model.τ = optimal_params[4]

    return model
end

    