 
# Performance Metrics

The following scripts were used to check errors and runtime.

## Error Calculations

Since parameter sets are identified up to permutation, the following finds the permutation that best reduces the error of the mixing coefficients and applies it to the means and covariances to compute their respective errors.

```julia
function computeError(w, true_means, true_covariances, mixing_coefficients, means, covariances, diagonal)
    k, d = size(true_means)

    weight_errors = [norm(mixing_coefficients[nthperm(1:k, i), :] - w) for i in 1:factorial(k)]
    minimum_weight_error, best_permutation = findmin(weight_errors)

    final_mixing_coefficients = mixing_coefficients[nthperm(1:k, best_permutation), :]
    final_means = means[nthperm(1:k, best_permutation), :]

    if diagonal
        final_covariances = [covariances[i][1:end, 1:end] for i in nthperm(1:k, best_permutation)]
    else
        final_covariances = covariances[nthperm(1:k, best_permutation), :, :]
    end

    return (norm(final_mixing_coefficients - w), norm(final_means - true_means), norm(final_covariances - true_covariances))
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
