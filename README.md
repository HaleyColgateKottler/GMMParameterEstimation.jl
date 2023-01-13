# GMMParameterEstimation

[![Build Status](https://github.com/HaleyColgateKottler/GMMParameterEstimation.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/HaleyColgateKottler/GMMParameterEstimation.jl/actions/workflows/CI.yml?query=branch%3Amain)

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://HaleyColgateKottler.github.io/GMMParameterEstimation.jl/)

## Example of basic use

```julia
using GMMParameterEstimation
d = 3
k = 2
diagonal = true
num_samples = 10^4
w, true_means, true_covariances = generateGaussians(d, k, diagonal)
sample = getSample(num_samples, w, true_means, true_covariances)
first_moms, diagonal_moms, off_diagonals = sampleMoments(sample, k)
pass, (mixing_coefficients, means, covariances) = estimate_parameters(d, k, first_moms, diagonal_moms, off_diagonals, diagonal)
```
