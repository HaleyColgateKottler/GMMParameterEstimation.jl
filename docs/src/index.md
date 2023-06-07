# GMMParameterEstimation.jl Documentation

GMMParameterEstimation.jl is a package for estimating the parameters of Gaussian k-mixture models using the method of moments. It can potentially find the parameters for arbitrary `k` with known or unknown mixing coefficients.  However, since the number of possible solutions to the polynomial system that determines the first dimension parameters and mixing coefficients for ``k>4`` is unknown, for the unknown mixing coefficient case with ``k>4`` failure of the package to find the parameters might occur if an insufficient number of solutions to the system were found

```@contents
```

## Examples
The following code snippet will use the given moments to return an estimate of the parameters using the method of moments with unknown mixing coefficients and dense covariance matrices.

```julia
using GMMParameterEstimation
d = 3
k = 2
first_moments = [1.0, -0.67, 2.44, -4.34, 17.4, -46.16, 201.67]
diagonal_moments = [-0.28 2.11 -2.46 15.29 -31.77; 0.4 4.25 3.88 54.75 59.10]
off_diag_system = Dict{Vector{Int64}, Float64}([2, 1, 0] => 1.8506, [1, 0, 1] => -0.329, [2, 0, 1] => 0.0291, [0, 2, 1] => 1.5869, [1, 1, 0] => -1.374, [0, 1, 1] => -0.333)
is_solution_found, (mixing_coefficients, means, covariances) = estimate_parameters(d, k, first_moments, diagonal_moments, off_diag_system)
```

### Inputs:

1. The number of dimensions `d`

2. The number of mixture components `k`

3. Optional: A vector of mixing coefficients `w` with length `k`

4. A list of the first ``3k+1`` moments (including moment 0) of the first dimension as `first_moments`

5. A matrix where row `i` contains the first ``2k+1`` moments (not including moment 0) of the `i`th dimension as `diagonal_moments`

6. Optional: A dictionary mapping the index of a mixed dimensional moment as a list of integers to the corresponding moment `off_diag_system` (See [mixedMomentSystem](https://haleycolgatekottler.github.io/GMMParameterEstimation.jl/#GMMParameterEstimation.mixedMomentSystem) for clarrification on which moments to include.)


### Outputs:

1. An indicator of success in finding the parameters `is_solution_found`

2. A tuple of the parameters `(mixing_coefficients, means, covariances)` 
``\\~\\``
 

The following code snippet will generate the denoised moments necessary for parameter recovery from the given parameters.

```julia
using GMMParameterEstimation

d=3
k=2
diagonal = false

means = [0.83 0.24 -1.53; 0.22 0.04 -0.71]
covariances = [0.8828527552401668 0.27735188899130847 1.6710529671002674; 2.257873093006253 -1.644707016523332 -0.533030022431624;;; 0.27735188899130847 1.2623673813995742 3.5270452552353238; -1.644707016523332 2.577324062116896 -0.5049891831614162;;; 1.6710529671002674 3.5270452552353238 16.696895556824817; -0.533030022431624 -0.5049891831614162 1.7733508773418585]
mixing_coefficients = [.3, .7]

if diagonal
    true_first, true_diag = diagonalPerfectMoments(d, k, w, true_means, true_covariances)
else
    true_first, true_diag, true_others = densePerfectMoments(d, k, w, true_means, true_covariances)
end
```

``\\~\\``
 
## Parameter estimation

The main functionality of this package stems from 
```@docs
estimate_parameters
```
which computes the parameter recovery using Algorithm 1 from [Estimating Gaussian Mixtures Using Sparse Polynomial Moment Systems](https://arxiv.org/abs/2106.15675).  Note that the unknown mixing coefficient cases with ``k\in\{2,3,4\}`` load a set of generic moments and the corresponding solutions to the first 1-D polynomial system from `sys1_k2.jld2`, `sys1_k3.jld2`, or `sys1_k4.jld2` for a slight speedup.  If `k` is not specified, k=1 will be assumed, and the resulting polynomial system will be solved explicitly and directly.   

In one dimension, for a random variable ``X`` with density ``f`` define the ``i``th moment as 
``m_i=E[X^i]=\int x^if(x)dx``.  
For a Gaussian mixture model, this results in a polynomial in the parameters.  For a sample ``\{y_1,y_2,\dots,y_N\}``, define the ``i``th sample moment as 
``\overline{m_i}=\frac{1}{N}\sum_{j=1}^N y_j^i``.  
The sample moments approach the true moments as ``N\rightarrow\infty``, so by setting the polynomials equal to the empirical moments, we can then solve the polynomial system to recover the parameters.

For a multivariate random variable ``X`` with density ``f_X`` define the moments as 
``m_{i_1,\dots,i_n} = E[X_1^{i_1}\cdots X_n^{i_n}] = \int\cdots\int x_1^{i_1}\cdots x_n^{i_n}f_X(x_1,\dots,x_n)dx_1\cdots dx_n`` 
and the empirical moments as 
``\overline{m}_{i_1,\dots,i_n} = \frac{1}{N}\sum_{j=1}^Ny_{j_1}^{i_1}\cdots y_{j_n}^{i_n}``.  
And again, by setting the polynomials equal to the empirical moments, we can then solve the system of polynomials to recover the parameters.  However, choosing which moments becomes more complicated.  If we know the mixing coefficients, we can use the first ``2k+1`` moments of each dimension to find the means and the diagonal entries of the covariance matrices.  If we do not know the mixing coefficients, we need the first ``3k`` moments of the first dimension to also find the mixing coefficients.  See [mixedMomentSystem](https://haleycolgatekottler.github.io/GMMParameterEstimation.jl/#GMMParameterEstimation.mixedMomentSystem) for which moments to include to fill in the off-diagonals of the covariance matrices if needed.

On a standard laptop we have successfully recovered parameters with unknown mixing coefficients for ``k\leq 4`` and known mixing coefficients for ``k\leq 5``, with ``d\leq 10^5`` for the diagonal covariance case and ``d\leq 50`` for the dense covariance case.  Higher `k` values or higher `d` values have led to issues with running out of RAM.

 ``\\~\\``

One potential difficulty in estimating the mixing coefficients is the resulting dependence on higher moments in the first dimension.  In sample data, if another dimension leads to more accurate moments, using that dimension to recover mixing coefficients and then proceeding can address this difficulty.  The following function was designed for this purpose.  Note that it can either be provided with an d x n sample, or a d x 3k+1 array of moments 0 through 3k for each dimension, augmented by a dictionary of the off-diagonal moments if seeking non-diagonal covariance matrices.

```@docs
dimension_cycle
```

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
To our knowledge, the only closed form formula for the mixed dimensional moments of a multivariate Gaussian is that provided by Jo``\~{a}``o M. Pereira, Joe Kileel, and Tamara G. Kolda in [Tensor Moments of Gaussian Mixture Models: Theory and Applications](https://arxiv.org/abs/2202.06930).  However, the tensor moments are indexed in a different way than the multivariate moment notation we used.  Let  ``m_{a_1\cdots a_n}`` be a d-th order multivariate moment and let ``M_{i_1\cdots i_d}^{(d)}`` be an entry of the d-th order tensor moment.  Then ``m_{a_1\cdots a_n}=M_{i_1\cdots i_d}^{(d)}`` where 
``a_j=|\{i_k=j\}|``.  Note that due to symmetry, the indexing of the tensor moment is non-unique.  For example, ``m_{102} = M_{133}^{(3)}=M_{331}^{(3)}=M_{313}^{(3)}=m_{102}``.

 ``\\~\\``

```@docs
mixedMomentSystem
```

The final step in our method of moments parameter recovery for non-diagonal covariance matrices is building and solving a system of ``N:=\frac{k}{2}(d^2-d)`` linear equations in the same number of unknowns to fill in the off diagonal.  The polynomial for ``m_{a_1\cdots a_n}`` is linear if all but two ``a_i=0`` and at least one ``a_1=1``.  There are ``n^2-n`` of these for each order ``\geq2``, so we need these equations for up to ``\lceil \frac{k}{2}\rceil``-th order.  These moments should be provided to the solver using minimal possible degree, and if only half the possible moments for a degree are necessary (k/2 is even) provide the moments with higher power in earlier dimension, e.g. use [2,1,0] instead of [1,2,0].

Note: the polynomial is still linear when 3 ``a_i=1`` and the rest of the ``a_i`` are 0 but this complicates generating the system so we did not include those.
 
Referring back to [Pereira et al.](https://arxiv.org/abs/2202.06930) for a closed form method of generating the necessary moment polynomials, we generate the linear system using the already computed mixing coefficients, means, and diagonals of the covariances, and return it as a dictionary of index=>polynomial pairs that can then be matched with the corresponding moments.

## Index

```@index
```
