using Documenter, GMMParameterEstimation, HomotopyContinuation

makedocs(sitename="GMMParameterEstimation.jl")

deploydocs(
    repo = "github.com/HaleyColgateKottler/GMMParameterEstimation.jl.git",
    versions = nothing,
    branch = "gh-pages"
)
