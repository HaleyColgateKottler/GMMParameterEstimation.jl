using Documenter, GMMParameterEstimation, HomotopyContinuation
push!(LOAD_PATH,"../src/")

makedocs(sitename="GMMParameterEstimation.jl",
         pages = ["Introduction" => "index.md"]
         )

deploydocs(
    repo = "github.com/HaleyColgateKottler/GMMParameterEstimation.jl.git",
)
