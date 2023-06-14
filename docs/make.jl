using Documenter, HomotopyContinuation #, GMMParameterEstimation
include("../src/GMMParameterEstimation.jl")
using .GMMParameterEstimation

push!(LOAD_PATH,"../src/")

makedocs(sitename="GMMParameterEstimation.jl",
         modules = [GMMParameterEstimation],
         pages = ["Home" => "index.md"]
         )

deploydocs(
    repo = "github.com/HaleyColgateKottler/GMMParameterEstimation.jl",
)
