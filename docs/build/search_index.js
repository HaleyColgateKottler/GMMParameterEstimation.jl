var documenterSearchIndex = {"docs":
[{"location":"#GMMParameterEstimation.jl-Documentation","page":"Home","title":"GMMParameterEstimation.jl Documentation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"GMMParameterEstimation.jl is a package for estimating the parameters of Gaussian k-mixture models using the method of moments. It can potentially find the parameters for arbitrary k with known or unknown mixing coefficients.  However, since the number of possible solutions to the polynomial system that determines the first dimension parameters and mixing coefficients for k4 is unknown, for the unknown mixing coefficient case with k4 failure of the package to find the parameters might occur if an insufficient number of solutions to the system were found","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"#Examples","page":"Home","title":"Examples","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The following code snippet will use the given moments to return an estimate of the parameters using the method of moments with unknown mixing coefficients and dense covariance matrices.","category":"page"},{"location":"","page":"Home","title":"Home","text":"using GMMParameterEstimation\nd = 3\nk = 2\nfirst_moments = [1.0, -0.67, 2.44, -4.34, 17.4, -46.16, 201.67]\ndiagonal_moments = [-0.28 2.11 -2.46 15.29 -31.77; 0.4 4.25 3.88 54.75 59.10]\noff_diag_system = Dict{Vector{Int64}, Float64}([2, 1, 0] => 1.8506, [1, 0, 1] => -0.329, [2, 0, 1] => 0.0291, [0, 2, 1] => 1.5869, [1, 1, 0] => -1.374, [0, 1, 1] => -0.333)\nis_solution_found, (mixing_coefficients, means, covariances) = estimate_parameters(d, k, first_moments, diagonal_moments, off_diag_system)","category":"page"},{"location":"#Inputs:","page":"Home","title":"Inputs:","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The number of dimensions d\nThe number of mixture components k\nOptional: A vector of mixing coefficients w with length k\nA list of the first 3k+1 moments (including moment 0) of the first dimension as first_moments\nA matrix where row i contains the first 2k+1 moments (not including moment 0) of the ith dimension as diagonal_moments\nOptional: A dictionary mapping the index of a mixed dimensional moment as a list of integers to the corresponding moment off_diag_system (See mixedMomentSystem for clarrification on which moments to include.)","category":"page"},{"location":"#Outputs:","page":"Home","title":"Outputs:","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"An indicator of success in finding the parameters is_solution_found\nA tuple of the parameters (mixing_coefficients, means, covariances) ","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"The following code snippet will generate the denoised moments necessary for parameter recovery from the given parameters.","category":"page"},{"location":"","page":"Home","title":"Home","text":"using GMMParameterEstimation\n\nd=3\nk=2\ndiagonal = false\n\nmeans = [0.83 0.24 -1.53; 0.22 0.04 -0.71]\ncovariances = [0.8828527552401668 0.27735188899130847 1.6710529671002674; 2.257873093006253 -1.644707016523332 -0.533030022431624;;; 0.27735188899130847 1.2623673813995742 3.5270452552353238; -1.644707016523332 2.577324062116896 -0.5049891831614162;;; 1.6710529671002674 3.5270452552353238 16.696895556824817; -0.533030022431624 -0.5049891831614162 1.7733508773418585]\nmixing_coefficients = [.3, .7]\n\nif diagonal\n    true_first, true_diag = diagonalPerfectMoments(d, k, w, true_means, true_covariances)\nelse\n    true_first, true_diag, true_others = densePerfectMoments(d, k, w, true_means, true_covariances)\nend","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"#Parameter-estimation","page":"Home","title":"Parameter estimation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The main functionality of this package stems from ","category":"page"},{"location":"","page":"Home","title":"Home","text":"estimate_parameters","category":"page"},{"location":"#GMMParameterEstimation.estimate_parameters","page":"Home","title":"GMMParameterEstimation.estimate_parameters","text":"estimate_parameters(d::Integer, k::Integer, first::Vector{Float64}, second::Matrix{Float64}, last::Dict{Vector{Int64}, Float64})\n\nCompute an estimate for the parameters of a d-dimensional Gaussian k-mixture model from the moments.\n\nFor the unknown mixing coefficient dense covariance matrix case, first should be a list of moments 0 through 3k for the first dimension, second should be a matrix of moments 1 through 2k+1 for the remaining dimensions, and last should be a dictionary of the indices as lists of integers and the corresponding moments.\n\n\n\n\n\nestimate_parameters(d::Integer, k::Integer, w::Array{Float64}, first::Vector{Float64}, second::Matrix{Float64}, last::Dict{Vector{Int64}, Float64})\n\nCompute an estimate for the parameters of a d-dimensional Gaussian k-mixture model from the moments.\n\nFor the known mixing coefficient dense covariance matrix case, w should be a vector of the mixing coefficients first should be a list of moments 0 through 3k for the first dimension, second should be a matrix of moments 1 through 2k+1 for the remaining dimensions, and last should be a dictionary of the indices as lists of integers and the corresponding moments.\n\n\n\n\n\nestimate_parameters(d::Integer, k::Integer, first::Vector{Float64}, second::Matrix{Float64})\n\nCompute an estimate for the parameters of a d-dimensional Gaussian k-mixture model from the moments.\n\nFor the unknown mixing coefficient diagonal covariance matrix case, first should be a list of moments 0 through 3k for the first dimension, and second should be a matrix of moments 1 through 2k+1 for the remaining dimensions.\n\n\n\n\n\nestimate_parameters(d::Integer, k::Integer, w::Array{Float64}, first::Vector{Float64}, second::Matrix{Float64})\n\nCompute an estimate for the parameters of a d-dimensional Gaussian k-mixture model from the moments.\n\nFor the known mixing coefficient diagonal covariance matrix case, w should be a vector of the mixing coefficients first should be a list of moments 0 through 3k for the first dimension, and second should be a matrix of moments 1 through 2k+1 for the remaining dimensions..\n\n\n\n\n\nestimate_parameters(d::Integer, first::Vector{Float64}, second::Matrix{Float64})\n\nCompute an estimate for the parameters of a d-dimensional Gaussian model from the moments.\n\nFor the unknown mixing coefficient diagonal covariance matrix case, first should be a list of moments 0 through 3 for the first dimension, and second should be a matrix of moments 1 through 3 for the remaining dimensions.\n\n\n\n\n\nestimate_parameters(d::Integer, first::Vector{Float64}, second::Matrix{Float64}, last::Dict{Vector{Int64}, Float64})\n\nCompute an estimate for the parameters of a d-dimensional Gaussian k-mixture model from the moments.\n\nFor the unknown mixing coefficient dense covariance matrix case, first should be a list of moments 0 through 3 for the first dimension, second should be a matrix of moments 1 through 3 for the remaining dimensions, and last should be a dictionary of the indices as lists of integers and the corresponding moments.\n\n\n\n\n\nestimate_parameters(k::Integer, shared_cov::Matrix{Float64}, first_moms::Vector{Float64}, second_moms::Matrix{Float64})\n\nCompute an estimate for the means of a Gaussian k-mixture model with equal mixing coefficients and known shared covariances from the moments.\n\nThe shared covariance matrix shared_cov will determine the dimension. Then first_moms should be a list of moments 0 through k for the first dimension, second_moms should be a matrix of moments m{je1+e_i} for j in 0 to k-1 and i in 2 to d as a matrix where d is the dimension, i varies across rows, and j varies down columns.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"which computes the parameter recovery using Algorithm 1 from Estimating Gaussian Mixtures Using Sparse Polynomial Moment Systems.  Note that the unknown mixing coefficient cases with kin234 load a set of generic moments and the corresponding solutions to the first 1-D polynomial system from sys1_k2.jld2, sys1_k3.jld2, or sys1_k4.jld2 for a slight speedup.  If k is not specified, k=1 will be assumed, and the resulting polynomial system will be solved explicitly and directly.   ","category":"page"},{"location":"","page":"Home","title":"Home","text":"In one dimension, for a random variable X with density f define the ith moment as  m_i=EX^i=int x^if(x)dx.   For a Gaussian mixture model, this results in a polynomial in the parameters.  For a sample y_1y_2dotsy_N, define the ith sample moment as  overlinem_i=frac1Nsum_j=1^N y_j^i.   The sample moments approach the true moments as Nrightarrowinfty, so by setting the polynomials equal to the empirical moments, we can then solve the polynomial system to recover the parameters.","category":"page"},{"location":"","page":"Home","title":"Home","text":"For a multivariate random variable X with density f_X define the moments as  m_i_1dotsi_n = EX_1^i_1cdots X_n^i_n = intcdotsint x_1^i_1cdots x_n^i_nf_X(x_1dotsx_n)dx_1cdots dx_n  and the empirical moments as  overlinem_i_1dotsi_n = frac1Nsum_j=1^Ny_j_1^i_1cdots y_j_n^i_n.   And again, by setting the polynomials equal to the empirical moments, we can then solve the system of polynomials to recover the parameters.  However, choosing which moments becomes more complicated.  If we know the mixing coefficients, we can use the first 2k+1 moments of each dimension to find the means and the diagonal entries of the covariance matrices.  If we do not know the mixing coefficients, we need the first 3k moments of the first dimension to also find the mixing coefficients.  See mixedMomentSystem for which moments to include to fill in the off-diagonals of the covariance matrices if needed.","category":"page"},{"location":"","page":"Home","title":"Home","text":"On a standard laptop we have successfully recovered parameters with unknown mixing coefficients for kleq 4 and known mixing coefficients for kleq 5, with dleq 10^5 for the diagonal covariance case and dleq 50 for the dense covariance case.  Higher k values or higher d values have led to issues with running out of RAM.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"One potential difficulty in estimating the mixing coefficients is the resulting dependence on higher moments in the first dimension.  In sample data, if another dimension leads to more accurate moments, using that dimension to recover mixing coefficients and then proceeding can address this difficulty.  The following function was designed for this purpose.  Note that it can either be provided with an d x n sample, or a d x 3k+1 array of moments 0 through 3k for each dimension, augmented by a dictionary of the off-diagonal moments if seeking non-diagonal covariance matrices.","category":"page"},{"location":"","page":"Home","title":"Home","text":"dimension_cycle","category":"page"},{"location":"#GMMParameterEstimation.dimension_cycle","page":"Home","title":"GMMParameterEstimation.dimension_cycle","text":"dimension_cycle(k::Integer, sample::Matrix{Float64}, diagonal)\n\nCycle over the dimensions of the sample to find candidate mixing coefficients, then solve for parameters based on those.\n\nThis will take longer than estimate_parameters since it does multiple tries.  Will try each dimension to attempt to find mixing coefficients, and if found will try to solve for parameters.  Returns pass = false if no dimension results in mixing coefficients that allow for a solution.\n\n\n\n\n\ndimension_cycle(d::Integer, k::Integer, cycle_moments::Array{Float64}, indexes::Dict{Vector{Int64}, Float64})\n\nCycle over the dimensions of cycle_moments to find candidate mixing coefficients, then solve for parameters based on those.\n\nThis will take longer than estimate_parameters since it does multiple tries.  Will try each dimension to attempt to find mixing coefficients, and if found will try to solve for parameters.  Returns pass = false if no dimension results in mixing coefficients that allow for a solution.  cycle_moments should be an array of the 0 through 3k moments for each dimension. If no indexes is given, assumes diagonal covariance matrices.\n\n\n\n\n\n","category":"function"},{"location":"#Generate-and-sample-from-Gaussian-Mixture-Models","page":"Home","title":"Generate and sample from Gaussian Mixture Models","text":"","category":"section"},{"location":"#Generation","page":"Home","title":"Generation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"makeCovarianceMatrix","category":"page"},{"location":"#GMMParameterEstimation.makeCovarianceMatrix","page":"Home","title":"GMMParameterEstimation.makeCovarianceMatrix","text":"makeCovarianceMatrix(d::Integer, diagonal::Bool)\n\nGenerate random dxd covariance matrix.\n\nIf diagonal==true, returns a diagonal covariance matrix.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"Note that the entries of the resulting covariance matrices are generated from a normal distribution centered at 0 with variance 1.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"generateGaussians","category":"page"},{"location":"#GMMParameterEstimation.generateGaussians","page":"Home","title":"GMMParameterEstimation.generateGaussians","text":"generateGaussians(d::Integer, k::Integer, diagonal::Bool)\n\nGenerate means and covariances for k Gaussians with dimension d.\n\ndiagonal should be true for spherical case, and false for dense covariance matrices.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"The parameters are returned as a tuple, with weights in a 1D vector, means as a k x d array, and variances as a k x d x d array if diagonal is false or as a list of Diagonal{Float64, Vector{Float64}} if diagonal is true to save memory.  Note that each entry of each parameter is generated from a normal distribution centered at 0 with variance 1.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"getSample","category":"page"},{"location":"#GMMParameterEstimation.getSample","page":"Home","title":"GMMParameterEstimation.getSample","text":"getSample(numb::Integer, w::Vector{Float64}, means::Matrix{Float64}, covariances::Vector)\n\nGenerate a Gaussian mixture model sample with numb entries, mixing coefficients w, means means, and covariances covariances.\n\n\n\n\n\ngetSample(numb::Integer, w::Vector{Float64}, means::Matrix{Float64}, covariances::Array{Float64, 3})\n\nGenerate a Gaussian mixture model sample with numb entries, mixing coefficients w, means means, and covariances covariances.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"This relies on the Distributions package.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"#Computing-moments","page":"Home","title":"Computing moments","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"sampleMoments\ndiagonalPerfectMoments\ndensePerfectMoments\nmoments_for_cycle\nequalMixCovarianceKnown_moments","category":"page"},{"location":"#GMMParameterEstimation.sampleMoments","page":"Home","title":"GMMParameterEstimation.sampleMoments","text":"sampleMoments(sample::Matrix{Float64}, k; diagonal = false)\n\nUse the sample to compute the moments necessary for parameter estimation using method of moments with general covariance matrices and mixing coefficients.\n\nReturns moments 0 to 3k for the first dimension, moments 1 through 2k+1 for the other dimensions as a matrix, and a dictionary with indices and moments for the off-diagonal system if diagonal is false.\n\n\n\n\n\n","category":"function"},{"location":"#GMMParameterEstimation.diagonalPerfectMoments","page":"Home","title":"GMMParameterEstimation.diagonalPerfectMoments","text":"diagonalPerfectMoments(d, k, w, true_means, true_covariances)\n\nUse the given parameters to compute the exact moments necessary for parameter estimation with diagonal covariance matrices.\n\nReturns moments 0 to 3k for the first dimension, and moments 1 through 2k+1 for the other dimensions as a matrix.\n\n\n\n\n\n","category":"function"},{"location":"#GMMParameterEstimation.densePerfectMoments","page":"Home","title":"GMMParameterEstimation.densePerfectMoments","text":"densePerfectMoments(d, k, w, true_means, true_covariances)\n\nUse the given parameters to compute the exact moments necessary for parameter estimation with dense covariance matrices.\n\nReturns moments 0 to 3k for the first dimension, moments 1 through 2k+1 for the other dimensions as a matrix, and a dictionary with indices and moments for the off-diagonal system.\n\n\n\n\n\n","category":"function"},{"location":"#GMMParameterEstimation.moments_for_cycle","page":"Home","title":"GMMParameterEstimation.moments_for_cycle","text":"moments_for_cycle(d, k, w, means, covars, diagonal)\n\nCalculate 0 through 3k+1 denoised moments for every dimension.\n\nUsed as input for cycling over the dimensions to find candidate mixing coefficients.\n\n\n\n\n\n","category":"function"},{"location":"#GMMParameterEstimation.equalMixCovarianceKnown_moments","page":"Home","title":"GMMParameterEstimation.equalMixCovarianceKnown_moments","text":"equalMixCovarianceKnown_moments(k, mean, shared_cov)\n\nUse the given parameters to compute the exact moments necessary for parameter estimation with equal mixing coefficients and shared known covariances.\n\nReturns moments 0 to k for the first dimension, and moments m{je1+e_i} for j in 0 to k-1 and i in 2 to d as a matrix where d is the dimension, i varies across rows, and j varies down columns.\n\n\n\n\n\nequalMixCovarianceKnown_moments(k, sample)\n\nUse the given parameters to compute the sample moments necessary for parameter estimation with equal mixing coefficients and shared known covariances.\n\nReturns moments 0 to k for the first dimension, and moments m{je1+e_i} for j in 0 to k-1 and i in 2 to d as a matrix where d is the dimension, i varies across rows, and j varies down columns.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"These expect parameters to be given with weights in a 1D vector, means as a k x d array, and covariances as a k x d x d array for dense covariance matrices or as a list of diagonal matrices for diagonal covariance matrices.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"#Build-the-polynomial-systems","page":"Home","title":"Build the polynomial systems","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"build1DSystem","category":"page"},{"location":"#GMMParameterEstimation.build1DSystem","page":"Home","title":"GMMParameterEstimation.build1DSystem","text":"build1DSystem(k::Integer, m::Integer)\n\nBuild the polynomial system for a mixture of 1D Gaussians where 'm'-1 is the highest desired moment and the mixing coefficients are unknown.\n\n\n\n\n\nbuild1DSystem(k::Integer, m::Integer, a::Union{Vector{Float64}, Vector{Variable}})\n\nBuild the polynomial system for a mixture of 1D Gaussians where 'm'-1 is the highest desired moment, and a is a vector of the mixing coefficients.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"This uses the usual recursive formula for moments of a univariate Gaussian in terms of the mean and variance, and then takes a convex combination with either variable mixing coefficients or the provided mixing coefficients.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"selectSol","category":"page"},{"location":"#GMMParameterEstimation.selectSol","page":"Home","title":"GMMParameterEstimation.selectSol","text":"selectSol(k::Integer, solution::Result, polynomial::Expression, moment::Number)\n\nSelect a k mixture solution from solution accounting for polynomial and moment.\n\nSort out a k mixture statistically significant solutions from solution, and return the one closest to moment when polynomial is evaluated at those values.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"Statistically significant means has positive variances here.  This is used to select which solution from the parameter homotopy will be used.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"tensorPower","category":"page"},{"location":"#GMMParameterEstimation.tensorPower","page":"Home","title":"GMMParameterEstimation.tensorPower","text":"tensorPower(tensor, power::Integer)\n\nCompute the power tensor power of tensor.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"convert_indexing","category":"page"},{"location":"#GMMParameterEstimation.convert_indexing","page":"Home","title":"GMMParameterEstimation.convert_indexing","text":"convert_indexing(moment_i, d)\n\nConvert the d dimensional multivariate moment_i index to the corresponding tensor moment index.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"To our knowledge, the only closed form formula for the mixed dimensional moments of a multivariate Gaussian is that provided by Joao M. Pereira, Joe Kileel, and Tamara G. Kolda in Tensor Moments of Gaussian Mixture Models: Theory and Applications.  However, the tensor moments are indexed in a different way than the multivariate moment notation we used.  Let  m_a_1cdots a_n be a d-th order multivariate moment and let M_i_1cdots i_d^(d) be an entry of the d-th order tensor moment.  Then m_a_1cdots a_n=M_i_1cdots i_d^(d) where  a_j=i_k=j.  Note that due to symmetry, the indexing of the tensor moment is non-unique.  For example, m_102 = M_133^(3)=M_331^(3)=M_313^(3)=m_102.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"mixedMomentSystem","category":"page"},{"location":"#GMMParameterEstimation.mixedMomentSystem","page":"Home","title":"GMMParameterEstimation.mixedMomentSystem","text":"mixedMomentSystem(d, k, mixing, ms, vs)\n\nBuild a linear system for finding the off-diagonal covariances entries.\n\nFor a d dimensional Gaussian k-mixture model with mixing coefficients mixing, means ms, and covariances vs where the diagonal entries have been filled in and the off diagonals are variables.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"The final step in our method of moments parameter recovery for non-diagonal covariance matrices is building and solving a system of N=frack2(d^2-d) linear equations in the same number of unknowns to fill in the off diagonal.  The polynomial for m_a_1cdots a_n is linear if all but two a_i=0 and at least one a_1=1.  There are n^2-n of these for each order geq2, so we need these equations for up to lceil frack2rceil-th order.  These moments should be provided to the solver using minimal possible degree, and if only half the possible moments for a degree are necessary (k/2 is even) provide the moments with higher power in earlier dimension, e.g. use [2,1,0] instead of [1,2,0].","category":"page"},{"location":"","page":"Home","title":"Home","text":"Note: the polynomial is still linear when 3 a_i=1 and the rest of the a_i are 0 but this complicates generating the system so we did not include those.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Referring back to Pereira et al. for a closed form method of generating the necessary moment polynomials, we generate the linear system using the already computed mixing coefficients, means, and diagonals of the covariances, and return it as a dictionary of index=>polynomial pairs that can then be matched with the corresponding moments.","category":"page"},{"location":"#Index","page":"Home","title":"Index","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"}]
}
