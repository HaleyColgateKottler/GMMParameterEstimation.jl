# GMMParameterEstimation.jl Documentation

```@contents
```

## Functions

```@docs
makeCovarianceMatrix(d::Integer; diagonal::Bool = false)
generateGaussians(d::Integer, k::Integer; diagonal::Bool = false)
get1Dmoments(sample::Matrix{Float64}, dimension::Integer, m::Integer)
getSample(numb::Integer, w::Vector{Float64}, means::Matrix{Float64}, covariances::Array{Float64, 3})
build1DSystem(k::Integer, m::Integer[, a::Union{Vector{Float64}, Vector{Variable}}])
tensorPower(tensor, power::Integer)
convert_indexing(moment_i, d)
mixedMomentSystem(d, k, mixing, ms, vs)
unknown_coefficients(d::Integer, k::Integer, w::Array{Float64}, true_means::Array{Float64,2}, true_covariances::Array{Float64,3}; diagonal::Bool = false)
known_coefficients(d::Integer, k::Integer, w::Array{Float64}, true_means::Array{Float64,2}, true_covariances::Array{Float64,3}; diagonal::Bool = false)
estimate_parameters(d::Integer, k::Integer, sample::Array{Float64}, diagonal::Bool[, w::Array{Float64}])
```

## Index

```@index
```
