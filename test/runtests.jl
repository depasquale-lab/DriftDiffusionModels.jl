using Test
using Random
using FiniteDiff
using ForwardDiff
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
    @testset verbose=true "FPTDDM Tests" begin
        include("test_FPTDDM.jl")
    end
end

