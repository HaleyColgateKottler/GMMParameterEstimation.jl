# GMMParameterEstimation.jl Documentation

GMMParameterEstimation.jl is a package for estimating the parameters of Gaussian k mixture models using the method of moments. It works for general k with known mixing coefficients, and for k=2,3,4 for unknown mixing coefficients.

```@contents
```

## Parameter estimation

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

## Generate and sample from Gaussian Mixture Models

```@docs
makeCovarianceMatrix
generateGaussians
getSample
sampleMoments
perfectMoments
```

## Build the polynomial systems

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
