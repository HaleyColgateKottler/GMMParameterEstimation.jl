using Documenter, GMMParameterEstimation, HomotopyContinuation
push!(LOAD_PATH,"../src/")

makedocs(sitename="GMMParameterEstimation.jl")

deploydocs(
    repo = "github.com/HaleyColgateKottler/GMMParameterEstimation.jl.git",
    versions = nothing,
    branch = "gh-pages"
)
