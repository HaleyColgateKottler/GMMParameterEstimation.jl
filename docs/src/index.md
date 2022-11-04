# GMMParameterEstimation.jl Documentation

## Introduction
GMMParameterEstimation.jl is a package for estimating the parameters of Gaussian `k` mixture models using the method of moments for `k`=2,3,4.

```@contents
```

## Example of basic use

```julia
using GMMParameterEstimation
d = 3
k = 2
diagonal = true
w, true_means, true_covariances = generateGaussians(d, k, diagonal)
sample = getSample(num_samples, w, true_means, true_covariances)
pass, (mixing_coefficients, means, covariances) = estimate_parameters(d, k, sample, diagonal)
```

## Parameter estimation 

The main functionality of the package is the [`estimate_parameters`](@ref) function which estimates parameters for a Gaussian Mixture Model from a sample.

```@docs
estimate_parameters(d::Integer, k::Integer, sample::Array{Float64}, diagonal::Bool, w::Array{Float64})
```
```@docs
unknown_coefficients(d::Integer, k::Integer, w::Array{Float64}, true_means::Array{Float64,2}, true_covariances::Array{Float64,3}, diagonal::Bool)
known_coefficients(d::Integer, k::Integer, w::Array{Float64}, true_means::Array{Float64,2}, true_covariances::Array{Float64,3}, diagonal::Bool)
```

## Generate random Gaussian `k` mixtures and samples

Generating random Gaussian `k` mixtures and sampling from them can be useful for simulation.
```@docs
makeCovarianceMatrix(d::Integer, diagonal::Bool)
generateGaussians(d::Integer, k::Integer, diagonal::Bool)
getSample(numb::Integer, w::Vector{Float64}, means::Matrix{Float64}, covariances::Array{Float64, 3})
```

## Build the polynomial systems

```@docs
get1Dmoments(sample::Matrix{Float64}, dimension::Integer, m::Integer)
```
```@docs
build1DSystem(k::Integer, m::Integer, a::Union{Vector{Float64}, Vector{Variable}})
```
```@docs
tensorPower(tensor, power::Integer)
convert_indexing(moment_i, d)
mixedMomentSystem(d, k, mixing, ms, vs)
```

## Index

```@index
```
