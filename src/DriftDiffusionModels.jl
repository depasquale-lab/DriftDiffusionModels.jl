module DriftDiffusionModels

using LinearAlgebra
using Random
using Statistics
using StatsAPI
using Distributions
using Optim
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
include("Transforms.jl")
include("CoherentDDM.jl")
include("GradientHMM.jl")

export DriftDiffusionModel, DDMResult, rand, logdensityof, fit!, crossvalidate, PriorHMM, simulateDDM, wfpt, randomDDM, logistic, fit_vi_gaussian,
       CoherentDDM, CoherentDDMResult, fit_shared_α!,
       logit, softmax, logsoftmax, logsumexp,
       coherent_to_unconstrained, coherent_from_unconstrained,
       pack_coherent_hmm, unpack_coherent_hmm, set_coherent_hmm!,
       coherent_hmm_loglikelihood, coherent_hmm_logposterior, fit_hmm_gradient!

end