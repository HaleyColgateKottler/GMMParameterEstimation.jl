 
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

## Tests

The following tests performance while adding specified levels of noise to the denoised moments. This relies on the above listed `computeError` function.

```julia
function testNoise(d, k, diagonal, noise_levels, reps, known_mixing_coeffs)
    passes = []
    mix_errs = []
    mean_errs = []
    covar_errs = []
    timings = []

    for noise in noise_levels
        passing = 0
        mixing_error = 0
        means_error = 0
        covariance_error = 0
        times = 0

        for i in 1:reps
            w, true_means, true_covariances = generateGaussians(d, k, diagonal)
            if diagonal
                true_first, true_diag = diagonalPerfectMoments(d, k, w, true_means, true_covariances)
                num_moments = 3*k + (d-1)*(2*k+1)
            else
                true_first, true_diag, true_others = densePerfectMoments(d, k, w, true_means, true_covariances)
                num_moments = 3*k + (d-1)*(2*k+1) + k*(d^2-d)/2
            end

            if noise>0
                randomness = randn(Int64(num_moments))
                randomness = noise/norm(randomness) * randomness

                noisy_first = true_first + [0, randomness[1:3*k]...]
                noisy_diag = true_diag + reshape(randomness[3*k+1:3*k + (d-1)*(2*k+1)], (d-1,2*k+1))

                if !diagonal
                    noisy_others = Dict{Vector{Int64}, Float64}()
                    orig = []
                    new = []
                    counter = 3*k + (d-1)*(2*k+1) + 1
                    for (key, moment) in true_others
                        push!(orig, moment)
                        push!(new, moment + randomness[counter])
                        noisy_others[key] = moment + randomness[counter]
                        counter += 1
                    end
                end
            else
                noisy_first = true_first
                noisy_diag = true_diag
                if !diagonal
                    noisy_others = true_others
                end
            end

            pass = false
            if diagonal
                if known_mixing_coeffs
                    timing = @elapsed begin
                        pass, (mixing_coefficients, means, covariances) = estimate_parameters(d, k, w, noisy_first, noisy_diag)
                    end
                else
                    timing = @elapsed begin
                        pass, (mixing_coefficients, means, covariances) = estimate_parameters(d, k, noisy_first, noisy_diag)
                    end
                end
            else
                if known_mixing_coeffs
                    timing = @elapsed begin
                        pass, (mixing_coefficients, means, covariances) = estimate_parameters(d, k, w, noisy_first, noisy_diag, noisy_others)
                    end
                else
                    timing = @elapsed begin
                        pass, (mixing_coefficients, means, covariances) = estimate_parameters(d, k, noisy_first, noisy_diag, noisy_others)
                    end
                end
            end

            if pass == true
                passing += 1
                (mix, mean, cov) = computeError(w, true_means, true_covariances, mixing_coefficients, means, covariances, diagonal)
                mixing_error += mix
                means_error += mean
                covariance_error += cov
                times += timing
            end
        end
        push!(passes, passing)
        if passing > 0
            push!(mix_errs, mixing_error/passing)
            push!(mean_errs, means_error/passing)
            push!(covar_errs, covariance_error/passing)
            push!(timings, times/passing)
        else
            push!(mix_errs, nothing)
            push!(mean_errs, nothing)
            push!(covar_errs, nothing)
            push!(timings, nothing)
        end
        open("noise_test_d" * string(d) * "_k" * string(k) * "_diag" * string(diagonal) * "_knoww" * string(known_mixing_coeffs) * ".txt", "w") do file
            write(file, "reps: " * string(reps) * "\nnoise levels: " * string(noise_levels) * "\npasses: " * string(passes) * "\naverage mixing coefficient error: " * string(mix_errs) * "\naverage mean error: " * string(mean_errs) * "\naverage covariance error: " * string(covar_errs) * "\naverage time: " * string(timings) * "\n")
        end
        println("check")
    end
end
```

Example of use:

```julia
k = 3
d = 5
diag = true
known = false
testNoise(d, k, diag, [.1, .01, .001, .0001, 0], 10, known)
```

The following tests performance with 100 samples per parameter, cycling over dimensions to find candidate mixing coefficients if the solver is unsuccessful with prior mixing coefficient candidates.  This relies on the above listed `computeError` function.

```julia
function testSampleAlg2(d, k, reps)
    num_params = k*d
    num_samples = 200*num_params

    passing = 0
    means_error = 0
    times = 0

    w = (1/k)*ones(k)

    for i in 1:reps
        shared_cov = makeCovarianceMatrix(d, false)
        true_mean = randn(k,d)

        variances = Array{Float64, 3}(undef, (k,d,d))
        for i in 1:k
            variances[i, 1:end, 1:end] = copy(shared_cov)
        end

        sample = getSample(num_samples, w, true_mean, variances)
        variances = []

        m1, m2 = equalMixCovarianceKnown_moments(k, sample)

        timing = @elapsed begin
            pass, est_mean = estimate_parameters(k, shared_cov, m1, m2)
        end
        if pass == true
            passing += 1
            means_error += computeErrorAlg2(true_mean, est_mean)
            times += timing
        end
    end

    if passing > 0
        means_error = means_error/passing
        times = times/passing
    else
        means_error = nothing
        times = nothing
    end

    open("alg2_sample_test_d" * string(d) * "_k" * string(k) * ".txt", "w") do file
        write(file, "reps: " * string(reps) * "\npasses: " * string(passing) * "\naverage mean error: " * string(means_error) * "\naverage time: " * string(times) * "\n")
    end
end
```

Example of use:

```julia
k = 2
diagonal = true
d = 10
reps = 100
testSampleCycle(d, k, diagonal, reps)
```

The following tests performance of Algorithm 2 from [Estimating Gaussian Mixtures Using Sparse Polynomial Moment Systems](https://arxiv.org/abs/2106.15675) for denoised moments. This relies on the above listed `computeErrorAlg2` function.

```julia
function testPerfectAlg2(d, k, reps)
    passing = 0
    means_error = 0
    times = 0

    for i in 1:reps
        shared_cov = makeCovarianceMatrix(d, false)
        true_mean = randn(k,d)
        m1, m2 = equalMixCovarianceKnown_moments(k, true_mean, shared_cov)

        timing = @elapsed begin
            pass, est_mean = estimate_parameters(k, shared_cov, m1, m2)
        end
        if pass == true
            passing += 1
            means_error += computeErrorAlg2(true_mean, est_mean)
            times += timing
        end
    end

    if passing > 0
        means_error = means_error/passing
        times = times/passing
    else
        means_error = nothing
        times = nothing
    end

    open("alg2_perfect_test_d" * string(d) * "_k" * string(k) * ".txt", "w") do file
        write(file, "reps: " * string(reps) * "\npasses: " * string(passing) * "\naverage mean error: " * string(means_error) * "\naverage time: " * string(times) * "\n")
    end
end
```

The following tests performance of Algorithm 2 from [Estimating Gaussian Mixtures Using Sparse Polynomial Moment Systems](https://arxiv.org/abs/2106.15675) for sample moments with 200 samples per parameter. This relies on the above listed `computeErrorAlg2` function.

```julia
function testSampleAlg2(d, k, reps)
    num_params = k*d
    num_samples = 200*num_params

    passing = 0
    means_error = 0
    times = 0

    w = (1/k)*ones(k)

    for i in 1:reps
        shared_cov = makeCovarianceMatrix(d, false)
        true_mean = randn(k,d)

        variances = Array{Float64, 3}(undef, (k,d,d))
        for i in 1:k
            variances[i, 1:end, 1:end] = copy(shared_cov)
        end

        sample = getSample(num_samples, w, true_mean, variances)
        variances = []

        m1, m2 = equalMixCovarianceKnown_moments(k, sample)

        timing = @elapsed begin
            pass, est_mean = estimate_parameters(k, shared_cov, m1, m2)
        end
        if pass == true
            passing += 1
            means_error += computeErrorAlg2(true_mean, est_mean)
            times += timing
        end
    end

    if passing > 0
        means_error = means_error/passing
        times = times/passing
    else
        means_error = nothing
        times = nothing
    end

    open("alg2_sample_test_d" * string(d) * "_k" * string(k) * ".txt", "w") do file
        write(file, "reps: " * string(reps) * "\npasses: " * string(passing) * "\naverage mean error: " * string(means_error) * "\naverage time: " * string(times) * "\n")
    end
end
```

