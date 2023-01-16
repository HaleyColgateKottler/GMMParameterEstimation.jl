var documenterSearchIndex = {"docs":
[{"location":"#GMMParameterEstimation.jl-Documentation","page":"Introduction","title":"GMMParameterEstimation.jl Documentation","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"GMMParameterEstimation.jl is a package for estimating the parameters of Gaussian k mixture models using the method of moments. It works for general k with known mixing coefficients, and for k=2,3,4 for unknown mixing coefficients.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"","category":"page"},{"location":"#Parameter-estimation","page":"Introduction","title":"Parameter estimation","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"The main functionality of this package stems from ","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"estimate_parameters","category":"page"},{"location":"#GMMParameterEstimation.estimate_parameters","page":"Introduction","title":"GMMParameterEstimation.estimate_parameters","text":"estimate_parameters(d::Integer, k::Integer, sample::Array{Float64}, diagonal::Bool)\n\nCompute an estimate for the parameters of a d-dimensional Gaussian k-mixture model from the moments.\n\nIf diagonal is true, the covariance matrices are assumed to be diagonal. If w is provided it is taken as the mixing coefficients, otherwise those are computed as well. first should be a list of moments 0 through 3k for the first dimension, second should be a matrix of moments 1 through 2k+1 for the remaining dimensions, and last should be a dictionary of the indices as lists of integers and the corresponding moments or nothing if diagonal is true.\n\n\n\n\n\nestimate_parameters(d::Integer, k::Integer, sample::Array{Float64}, diagonal::Bool)\n\nCompute an estimate for the parameters of a d-dimensional Gaussian k-mixture model from the moments.\n\nIf diagonal is true, the covariance matrices are assumed to be diagonal. If w is provided it is taken as the mixing coefficients, otherwise those are computed as well. first should be a list of moments 0 through 3k for the first dimension, second should be a matrix of moments 1 through 2k+1 for the remaining dimensions, and last should be a dictionary of the indices as lists of integers and the corresponding moments or nothing if diagonal is true.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Introduction","title":"Introduction","text":"For example, the following code snippet will generate a 3D 2-mixture, take a sample, compute the necessary moments, and then return an estimate of the parameters using the method of moments.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"using GMMParameterEstimation\nd = 3\nk = 2\ndiagonal = true\nnum_samples = 10^4\nw, true_means, true_covariances = generateGaussians(d, k, diagonal)\nsample = getSample(num_samples, w, true_means, true_covariances)\nfirst_moms, diagonal_moms, off_diagonals = sampleMoments(sample, k)\npass, (mixing_coefficients, means, covariances) = estimate_parameters(d, k, first_moms, diagonal_moms, off_diagonals, diagonal)","category":"page"},{"location":"#Generate-and-sample-from-Gaussian-Mixture-Models","page":"Introduction","title":"Generate and sample from Gaussian Mixture Models","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"makeCovarianceMatrix\ngenerateGaussians\ngetSample\nsampleMoments\nperfectMoments","category":"page"},{"location":"#GMMParameterEstimation.makeCovarianceMatrix","page":"Introduction","title":"GMMParameterEstimation.makeCovarianceMatrix","text":"makeCovarianceMatrix(d::Integer, diagonal::Bool)\n\nGenerate random dxd covariance matrix.\n\nIf diagonal==true, returns a diagonal covariance matrix.\n\n\n\n\n\n","category":"function"},{"location":"#GMMParameterEstimation.generateGaussians","page":"Introduction","title":"GMMParameterEstimation.generateGaussians","text":"generateGaussians(d::Integer, k::Integer, diagonal::Bool)\n\nGenerate means and covariances for k Gaussians with dimension d.\n\ndiagonal should be true for spherical case, and false for dense covariance matrices.\n\n\n\n\n\n","category":"function"},{"location":"#GMMParameterEstimation.getSample","page":"Introduction","title":"GMMParameterEstimation.getSample","text":"getSample(numb::Integer, w::Vector{Float64}, means::Matrix{Float64}, covariances::Array{Float64, 3})\n\nGenerate a Gaussian mixture model sample with numb entries, mixing coefficients w, means means, and covariances covariances.\n\n\n\n\n\n","category":"function"},{"location":"#GMMParameterEstimation.sampleMoments","page":"Introduction","title":"GMMParameterEstimation.sampleMoments","text":"sampleMoments(sample::Matrix{Float64}, k; diagonal = false)\n\nUse the sample to compute the moments necessary for parameter estimation using method of moments.\n\nReturns moments 0 to 3k for the first dimension, moments 1 through 2k+1 for the other dimensions as a matrix, and a dictionary with indices and moments for the off-diagonal system if diagonal is false.\n\n\n\n\n\n","category":"function"},{"location":"#GMMParameterEstimation.perfectMoments","page":"Introduction","title":"GMMParameterEstimation.perfectMoments","text":"perfectMoments(d, k, w, true_means, true_covariances)\n\nUse the given parameters to compute the exact moments necessary for parameter estimation.\n\nReturns moments 0 to 3k for the first dimension, moments 1 through 2k+1 for the other dimensions as a matrix, and a dictionary with indices and moments for the off-diagonal system.\n\n\n\n\n\n","category":"function"},{"location":"#Build-the-polynomial-systems","page":"Introduction","title":"Build the polynomial systems","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"build1DSystem\nselectSol\ntensorPower\nconvert_indexing\nmixedMomentSystem","category":"page"},{"location":"#GMMParameterEstimation.build1DSystem","page":"Introduction","title":"GMMParameterEstimation.build1DSystem","text":"build1DSystem(k::Integer, m::Integer)\n\nBuild the polynomial system for a mixture of 1D Gaussians where 'm' is the highest desired moment.\n\nIf a is given, use a as the mixing coefficients, otherwise leave them as unknowns.\n\n\n\n\n\nbuild1DSystem(k::Integer, m::Integer, a::Union{Vector{Float64}, Vector{Variable}})\n\nBuild the polynomial system for a mixture of 1D Gaussians where 'm' is the highest desired moment.\n\nIf a is given, use a as the mixing coefficients, otherwise leave them as unknowns.\n\n\n\n\n\n","category":"function"},{"location":"#GMMParameterEstimation.selectSol","page":"Introduction","title":"GMMParameterEstimation.selectSol","text":"selectSol(k::Integer, solution::Result, polynomial::Expression, moment::Number)\n\nSelect a k mixture solution from solution accounting for polynomial and moment.\n\nSort out a k mixture statistically significant solutions from solution, and return the one closest to moment when polynomial is evaluated at those values.\n\n\n\n\n\n","category":"function"},{"location":"#GMMParameterEstimation.tensorPower","page":"Introduction","title":"GMMParameterEstimation.tensorPower","text":"tensorPower(tensor, power::Integer)\n\nCompute the power tensor power of tensor.\n\n\n\n\n\n","category":"function"},{"location":"#GMMParameterEstimation.convert_indexing","page":"Introduction","title":"GMMParameterEstimation.convert_indexing","text":"convert_indexing(moment_i, d)\n\nConvert the d dimensional multivariate moment_i index to the corresponding tensor moment index.\n\n\n\n\n\n","category":"function"},{"location":"#GMMParameterEstimation.mixedMomentSystem","page":"Introduction","title":"GMMParameterEstimation.mixedMomentSystem","text":"mixedMomentSystem(d, k, mixing, ms, vs)\n\nBuild a linear system for finding the off-diagonal covariances entries.\n\nFor a d dimensional Gaussian k-mixture model with mixing coefficients mixing, means ms, and covariances vs where the diagonal entries have been filled in and the off diagonals are variables.\n\n\n\n\n\n","category":"function"},{"location":"#Index","page":"Introduction","title":"Index","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"","category":"page"}]
}
