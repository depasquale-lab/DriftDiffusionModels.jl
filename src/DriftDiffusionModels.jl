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
include("OmissionCoherentDDM.jl")
include("GradientHMM.jl")
include("eDDMExact.jl")

export DriftDiffusionModel, DDMResult, rand, logdensityof, fit!, PriorHMM, init_hmm_ddm, calculate_ll_ratio, simulateDDM, wfpt, logistic, fit_vi_gaussian,
       fit_mlddm_exact, MLDDMFit, marginal_loglik, quadrature_check,
       trial_posteriors, profile_sigma0, gauss_hermite,
       CoherentDDM, CoherentDDMResult, fit_shared_α!,
       logit, softmax, logsoftmax, logsumexp,
       coherent_to_unconstrained, coherent_from_unconstrained,
       pack_coherent_hmm, unpack_coherent_hmm, set_coherent_hmm!,
       coherent_hmm_loglikelihood, coherent_hmm_logposterior, fit_hmm_gradient!,
       OmissionCoherentDDM, OmissionCoherentDDMResult, omission_state, is_omission, is_omission_state,
       OMISSION_LOGFLOOR, omission_flags, n_omission_hmm_params,
       pack_omission_hmm, unpack_omission_hmm, set_omission_hmm!,
       omission_hmm_loglikelihood, omission_hmm_logposterior

end
