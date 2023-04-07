# GMMParameterEstimation

[![Build Status](https://github.com/HaleyColgateKottler/GMMParameterEstimation.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/HaleyColgateKottler/GMMParameterEstimation.jl/actions/workflows/CI.yml?query=branch%3Amain)

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://HaleyColgateKottler.github.io/GMMParameterEstimation.jl/)

## Example of basic use

```julia
using GMMParameterEstimation
d = 3
k = 2
first_moments = [1.0, -0.67, 2.44, -4.34, 17.4, -46.16, 201.67]
diagonal_moments = [-0.28 2.11 -2.46 15.29 -31.77; 0.4 4.25 3.88 54.75 59.10]
off_diag_system = Dict{Vector{Int64}, Float64}([2, 1, 0] => 1.8506, [1, 0, 1] => -0.329, [2, 0, 1] => 0.0291, [0, 2, 1] => 1.5869, [1, 1, 0] => -1.374, [0, 1, 1] => -0.333)
is_solution_found, (mixing_coefficients, means, covariances) = estimate_parameters(d, k, first_moments, diagonal_moments, off_diag_system)
```
