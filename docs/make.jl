using Documenter, HomotopyContinuation, GMMParameterEstimation

push!(LOAD_PATH,"../src/")

makedocs(sitename="GMMParameterEstimation.jl Documentation",
         pages = ["Documentation" => "index.md"]
         )

deploydocs(
    repo = "github.com/HaleyColgateKottler/GMMParameterEstimation.jl.git",
)
