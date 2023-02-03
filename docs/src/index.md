# GMMParameterEstimation.jl Documentation

GMMParameterEstimation.jl is a package for estimating the parameters of Gaussian k-mixture models using the method of moments. It works for general k with known mixing coefficients, and for k=2,3,4 for unknown mixing coefficients.

```@contents
```

## Example
The following code snippet will use the given moments to return an estimate of the parameters using the method of moments.

```julia
using GMMParameterEstimation
d = 3
k = 2
first_moments = [1.0, 0.980, 1.938, 3.478, 8.909, 20.526, 64.303]
diagonal_moments = [-0.580 5.682 -11.430 97.890 -341.161; -0.480 1.696 -2.650 11.872 -33.239]
off_diag_system = Dict{Vector{Int64}, Float64}([0, 1, 2] => -1.075, [1, 0, 1] => -0.252, [1, 2, 0] => 6.021, [1, 0, 2] => 1.117, [1, 1, 0] => -0.830, [0, 1, 1] => 0.884)
pass, (mixing_coefficients, means, covariances) = estimate_parameters(d, k, first_moments, diagonal_moments, off_diag_system)
```
``\\~\\``
 
 
## Parameter estimation

The main functionality of this package stems from 
```@docs
estimate_parameters
```
which computes the parameter recovery using Algorithm 1 from [Estimating Gaussian Mixtures Using Sparse Polynomial Moment Systems](https://arxiv.org/abs/2106.15675).

In one dimension, for a random variable ``X`` with density ``f`` we define the ``i``th moment as 
``m_i=E[X^i]=\int xf(x)dx``.  
For a Gaussian mixture model, this results in a polynomial in the parameters.  For a sample ``\{y_1,y_2,\dots,y_N\}``, we define the ``i``th sample moment as 
``\overline{m_i}=\frac{1}{N}\sum_{j=1}^N y_j^i``.  
The sample moments approach the true moments as ``N\rightarrow\infty``, so by setting the polynomials equal to the empirical moments, we can then solve the polynomial system to recover the parameters.

For a multivariate random variable ``X`` with density ``f_X`` we define the moments as 
``m_{i_1,\dots,i_n} = E[X_1^{i_1}\cdots X_n^{i_n}] = \int\cdots\int x_1^{i_1}\cdots x_n^{i_n}f_X(x_1,\dots,x_n)dx_1\cdots dx_n`` 
and the empirical moments as 
``\overline{m}_{i_1,\dots,i_n} = \frac{1}{N}\sum_{j=1}^Ny_{j_1}^{i_1}\cdots y_{j_n}^{i_n}``.  
And again, by setting the polynomials equal to the empirical moments, we can then solve the system of polynomials to recover the parameters.  However, choosing which moments becomes more complicated.

 ``\\~\\``

## Generate and sample from Gaussian Mixture Models

```@docs
makeCovarianceMatrix
```
Note that the entries of the resulting covariance matrices are generated from a normal distribution centered at 0 with variance 1.

 ``\\~\\``

```@docs
generateGaussians
```
The parameters are returned as a tuple, with weights in a 1D vector, means as a k x d array, and variances as a k x d x d array.  Note that each entry of each parameter is generated from a normal distribution centered at 0 with variance 1.

 ``\\~\\``

```@docs
getSample
```
This relies on the [Distributions](https://juliastats.org/Distributions.jl/stable/) package.

 ``\\~\\``
 
```@docs
sampleMoments
diagonalPerfectMoments
densePerfectMoments
```
Both expect parameters to be given with weights in a 1D vector, means as a k x d array, and variances as a k x d x d array.

 ``\\~\\``

## Build the polynomial systems

```@docs
build1DSystem
```

```@docs
selectSol
```

```@docs
tensorPower
```

```@docs
convert_indexing
```

```@docs
mixedMomentSystem
```

## Index

```@index
```
