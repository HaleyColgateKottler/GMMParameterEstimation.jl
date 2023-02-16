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
which computes the parameter recovery using Algorithm 1 from [Estimating Gaussian Mixtures Using Sparse Polynomial Moment Systems](https://arxiv.org/abs/2106.15675).  Note that the unknown mixing coefficient cases load a set of generic moments and the corresponding solutions to the first 1-D polynomial system from `sys1_k2.jld2`, `sys1_k3.jld2`, or `sys1_k4.jld2` depending on `k`.

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

### Generation

```@docs
makeCovarianceMatrix
```
Note that the entries of the resulting covariance matrices are generated from a normal distribution centered at 0 with variance 1.

 ``\\~\\``

```@docs
generateGaussians
```
The parameters are returned as a tuple, with weights in a 1D vector, means as a k x d array, and variances as a k x d x d array if `diagonal` is false or as a list of `Diagonal{Float64, Vector{Float64}}` if `diagonal` is true to save memory.  Note that each entry of each parameter is generated from a normal distribution centered at 0 with variance 1.

 ``\\~\\``

```@docs
getSample
```
This relies on the [Distributions](https://juliastats.org/Distributions.jl/stable/) package.

 ``\\~\\``
 
### Computing moments

```@docs
sampleMoments
diagonalPerfectMoments
densePerfectMoments
```
These expect parameters to be given with weights in a 1D vector, means as a k x d array, and covariances as a k x d x d array for dense covariance matrices or as a list of diagonal matrices for diagonal covariance matrices.

 ``\\~\\``

## Build the polynomial systems

```@docs
build1DSystem
```
This uses the usual recursive formula for moments of a univariate Gaussian in terms of the mean and variance, and then takes a convex combination with either variable mixing coefficients or the provided mixing coefficients.

 ``\\~\\``

```@docs
selectSol
```
Statistically significant means has positive variances here.  This is used to select which solution from the parameter homotopy will be used.

 ``\\~\\``

```@docs
tensorPower
```

```@docs
convert_indexing
```
As far as I am aware, the only closed form formula for the mixed dimensional moments of a multivariate Gaussian is that provided by Jo``\~{a}``o M. Pereira, Joe Kileel, and Tamara G. Kolda in [Tensor Moments of Gaussian Mixture Models: Theory and Applications](https://arxiv.org/abs/2202.06930).  However, the tensor moments are indexed in a different way than the multivariate moment notation we used.  Let  ``m_{a_1\cdots a_n}`` be a d-th order multivariate moment and let ``M_{i_1\cdots i_d}^{(d)}`` be an entry of the d-th order tensor moment.  Then ``m_{a_1\cdots a_n}=M_{i_1\cdots i_d}^{(d)}`` where 
``a_j=|\{i_k=j\}|``.  Note that due to symmetry, the indexing of the tensor moment is non-unique.  For example, ``m_{102} = M_{133}^{(3)}=M_{331}^{(3)}=M_{313}^{(3)}=m_{102}``.

 ``\\~\\``

```@docs
mixedMomentSystem
```

The final step in our method of moments parameter recovery for non-diagonal covariance matrices is building and solving a system of ``N:=\frac{k}{2}(d^2-d)`` linear equations in the same number of unknowns to fill in the off diagonal.  The polynomial for ``m_{a_1\cdots a_n}`` is linear if all but two ``a_i=0`` and at least one ``a_1=1``.  There are ``n^2-n`` of these for each order ``\geq2``, so we need these equations for up to ``\lceil \frac{k}{2}\rceil``-th order. 

Note: the polynomial is still linear when 3 ``a_i=1`` and the rest of the ``a_i`` are 0 but this complicates generating the system so we did not include those.
 
Again referring back to [Pereira et al.](https://arxiv.org/abs/2202.06930) for a closed form method of generating the necessary moment polynomials, we generate the linear system using the already computed mixing coefficients, means, and diagonals of the covariances, and return it as a dictionary of index=>polynomial pairs that can then be matched with the corresponding moments.

## Index

```@index
```
