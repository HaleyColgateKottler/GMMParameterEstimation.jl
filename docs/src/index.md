# GMMParameterEstimation.jl Documentation

```@contents
```

## Parameter Estimation

The main functionality of this package stems from 

```@docs
estimate_parameters
```

For example, the following code snippet will generate a 3D 2-mixture, take a sample, compute the necessary moments, and then return an estimate of the parameters using the method of moments.

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

## Generating and Sampling from Gaussian Mixture Models

```@docs
makeCovarianceMatrix
generateGaussians
getSample
sampleMoments
perfectMoments
```

## Useful Functions

```@docs
build1DSystem
selectSol
tensorPower
convert_indexing
mixedMomentSystem
```

## Index

```@index
```
