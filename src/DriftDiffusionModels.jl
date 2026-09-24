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
include("eDDMExact.jl")

export DriftDiffusionModel, DDMResult, rand, logdensityof, fit!, PriorHMM, init_hmm_ddm, calculate_ll_ratio, simulateDDM, wfpt, logistic, fit_vi_gaussian,
       fit_mlddm_exact, MLDDMFit, marginal_loglik, quadrature_check,
       trial_posteriors, profile_sigma0, gauss_hermite

end
