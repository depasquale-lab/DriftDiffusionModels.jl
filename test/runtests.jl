using Test
using Random
using Statistics
using FiniteDiff
using ForwardDiff
import HiddenMarkovModels          # registers logdensityof(::AbstractHMM, …); imported (not used) to keep bare `logdensityof` unambiguous
import DensityInterface
using DriftDiffusionModels

@testset verbose=true "DriftDiffusionModels.jl Tests" begin
    @testset verbose=true "WFPT Tests" begin
        include("test_wfpt.jl")
    end
    @testset verbose=true "Stimulus Coding Tests" begin
        include("test_stimulus_coding.jl")
    end
    @testset verbose=true "Fit Tests" begin
        include("test_fit.jl")
    end
    @testset verbose=true "eDDM Tests" begin
        include("test_eDDM.jl")
    end
    @testset verbose=true "CoherentDDM Tests" begin
        include("test_coherent_ddm.jl")
    end
end

