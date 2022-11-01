push!(LOAD_PATH,"../src/")
using Documenter, GMMParameterEstimation

makedocs(sitename="GMMParameterEstimation.jl")

# deploydocs(
#     repo = "github.com/HaleyColgateKottler/GMMParameterEstimation.jl.git",
#     versions = nothing,
#     deploy_config = Documenter.GitHubActions(GITHUB_EVENT_NAME="workflow_dispatch", GITHUB_REF = "main", GITHUB_ACTOR = ""),
# )
