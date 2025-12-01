module DriftDiffusionModels

using LinearAlgebra
using Random
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

export DriftDiffusionModel, DDMResult, rand, logdensityof, fit!, crossvalidate, PriorHMM, simulateDDM, wfpt, randomDDM, logistic

end