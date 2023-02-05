# GMMParameterEstimation

[![Build Status](https://github.com/HaleyColgateKottler/GMMParameterEstimation.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/HaleyColgateKottler/GMMParameterEstimation.jl/actions/workflows/CI.yml?query=branch%3Amain)

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://HaleyColgateKottler.github.io/GMMParameterEstimation.jl/)

## Example of basic use

```julia
using GMMParameterEstimation
d = 3
k = 2
first_moments = [1.0, 0.980, 1.938, 3.478, 8.909, 20.526, 64.303]
diagonal_moments = [-0.580 5.682 -11.430 97.890 -341.161; -0.480 1.696 -2.650 11.872 -33.239]
off_diag_system = Dict{Vector{Int64}, Float64}([0, 1, 2] => -1.075, [1, 0, 1] => -0.252, [1, 2, 0] => 6.021, [1, 0, 2] => 1.117, [1, 1, 0] => -0.830, [0, 1, 1] => 0.884)
pass, (mixing_coefficients, means, covariances) = estimate_parameters(d, k, first_moments, diagonal_moments, off_diag_system)
```
