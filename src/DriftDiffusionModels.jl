module DriftDiffusionModels

using LinearAlgebra
using Random
using Statistics
using StatsAPI
using Distributions
using Optim
using ADTypes: AutoForwardDiff
using ForwardDiff
using UnPack
using DensityInterface
using HiddenMarkovModels
using SpecialFunctions
using Base.Threads: @threads


# Import the fit! function specifically--its being weird about fit!
import StatsAPI: fit!

include("DDM.jl")
include("HMMDDM.jl")
include("Utilities.jl")
include("eDDM.jl")
include("FPTDDM.jl")
include("FPTDDMFit.jl")

export DriftDiffusionModel,
    DDMResult,
    rand,
    logdensityof,
    fit!,
    crossvalidate,
    PriorHMM,
    simulateDDM,
    wfpt,
    randomDDM,
    logistic,
    softplus,
    logsumexp,
    fit_vi_gaussian

# FPTDDM (first-passage-time neural DDM) exports
export FPTDDM
export AbstractObservationModel
export LinearPoissonObservationModel,
    BasisPoissonObservationModel, GPPoissonObservationModel
export obs_logpdf, obs_sample
export fpt_loglik, loglik
export Trial, n_time, n_neurons
export simulate_trial

end
