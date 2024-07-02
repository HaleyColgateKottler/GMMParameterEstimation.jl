 
# Performance Metrics

The following scripts were used to check errors and runtime.

## Error Calculations

Since parameter sets are identified up to permutation, the following finds the permutation that best reduces the error of the mixing coefficients and applies it to the means and covariances to compute their respective errors.

```julia
function computeError(w, true_means, true_covariances, mixing_coefficients, means, covariances, diagonal)
    k = size(w)[1]
    d = size(true_means)[2]

    basis = 1:k

    minimum_weight_error = (norm(w - mixing_coefficients), basis)
    for i in 2:factorial(k)
        permutation = nthperm(basis, i)
        mixed_weights = Array{Float64}(undef, size(w))
        for j in 1:k
            mixed_weights[j, 1:end] = mixing_coefficients[permutation[j], 1:end]
        end
        weight_error = norm(mixed_weights - w)
        (weight_error < minimum_weight_error[1]) && (minimum_weight_error = (weight_error, permutation))
    end

    permutation = minimum_weight_error[2]

    final_mixing_coefficients = Array{Float64}(undef, size(mixing_coefficients))
    final_means = Array{Float64}(undef, size(means))

    if diagonal
        final_covariances = []
        for j in 1:k
            final_mixing_coefficients[j] = mixing_coefficients[permutation[j]]
            final_means[j, 1:end] = means[permutation[j], 1:end]
            push!(final_covariances, covariances[permutation[j]][1:end, 1:end])
        end
    else
    final_covariances = Array{Union{Float64}, 3}(undef, (k,d,d))
        for j in 1:k
            final_mixing_coefficients[j] = mixing_coefficients[permutation[j]]
            final_means[j, 1:end] = means[permutation[j], 1:end]
            final_covariances[j, 1:end, 1:end] = covariances[permutation[j], 1:end, 1:end]
        end
    end

    mixing_error = norm(final_mixing_coefficients - w)
    means_error = norm(final_means - true_means)
    covariance_error = norm(final_covariances - true_covariances)

    return (mixing_error, means_error, covariance_error)
end
```

This is similar, but for Algorithm 2 from [Estimating Gaussian Mixtures Using Sparse Polynomial Moment Systems](https://arxiv.org/abs/2106.15675) all but the means are known so it merely permutes the means to find the minimum error permutation and then returns that error.

```julia
function computeErrorAlg2(true_mean, est_mean)
    k = size(true_mean)[1]
    d = size(true_mean)[2]

    basis = 1:k

    minimum_error = norm(true_mean - est_mean)
    for i in 2:factorial(k)
        permutation = nthperm(basis, i)
        mixed_means = Array{Float64}(undef, size(true_mean))
        for j in 1:k
            mixed_means[j, 1:end] = est_mean[permutation[j], 1:end]
        end
        mean_error = norm(mixed_means - true_mean)
        (mean_error < minimum_error) && (minimum_error = mean_error)
    end

    return minimum_error
end
```
