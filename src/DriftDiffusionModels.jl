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
using Polyester: @batch


# Import the fit! function specifically--its being weird about fit!
import StatsAPI: fit!

include("DDM.jl")
include("HMMDDM.jl")
include("Utilities.jl")
include("eDDM.jl")
include("NeuralDDM.jl")

export DriftDiffusionModel, DDMResult, rand, logdensityof, fit!, crossvalidate, PriorHMM, simulateDDM, wfpt, randomDDM, logistic, softplus, logsumexp, fit_vi_gaussian

# NeuralDDM exports
export AbstractStateModel, AbstractObservationModel
export LeakyAccumulatorModel
export LinearPoissonObservationModel, BasisPoissonObservationModel, GPPoissonObservationModel
export NeuralDDM
export init_sample, init_logpdf, transition_sample, transition_logpdf
export hazard, stop_logpdf, obs_sample, obs_logpdf, choice_logpdf
export Trial, n_time, n_neurons
export particle_filter, log_marginal_likelihood

end