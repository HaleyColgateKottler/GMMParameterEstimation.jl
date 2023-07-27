# using HomotopyContinuation
# using Distributions
# using LinearAlgebra
# using Combinatorics
# using JLD2

"""
    makeCovarianceMatrix(d::Integer, diagonal::Bool)

Generate random `d`x`d` covariance matrix.

If `diagonal`==true, returns a diagonal covariance matrix.
"""
function makeCovarianceMatrix(d::Integer, diagonal::Bool)
    if diagonal
        covars = randn(d)
        covars = covars .* covars
        covars = covars/norm(covars)
        covars = Diagonal(covars)
    else
        covars = randn(d,d)
        covars = covars * covars'
        covars = covars/norm(covars)
    end
    @assert isposdef(covars)
    return covars
end

"""
    generateGaussians(d::Integer, k::Integer, diagonal::Bool)

Generate means and covariances for `k` Gaussians with dimension `d`.

`diagonal` should be true for spherical case, and false for dense covariance matrices.
"""
function generateGaussians(d::Integer, k::Integer, diagonal::Bool)
    w::Vector{Float64} = abs.(randn(k))
    w /= sum(w)
    means = randn(k, d)
    if diagonal
        variances = []
        for i in 1:k
            push!(variances, makeCovarianceMatrix(d, diagonal))
        end
    else
        variances = Array{Float64, 3}(undef, (k,d,d))
        for i in 1:k
            variances[i, 1:end, 1:end] = makeCovarianceMatrix(d, diagonal)
        end
    end
    return (w, means, variances)
end

"""
    getSample(numb::Integer, w::Vector{Float64}, means::Matrix{Float64}, covariances::Vector)

Generate a Gaussian mixture model sample with `numb` entries, mixing coefficients `w`, means `means`, and covariances `covariances`.
"""
function getSample(numb::Integer, w::Vector{Float64}, means::Matrix{Float64}, covariances::Vector)
    k = size(w)[1]
    gaussians = [(means[i, 1:end], Matrix(covariances[i][1:end, 1:end])) for i in 1:k]
    m = MixtureModel(MvNormal, gaussians, w)
    r = rand(m, numb)
    return r
end

"""
    getSample(numb::Integer, w::Vector{Float64}, means::Matrix{Float64}, covariances::Array{Float64, 3})

Generate a Gaussian mixture model sample with `numb` entries, mixing coefficients `w`, means `means`, and covariances `covariances`.
"""
function getSample(numb::Integer, w::Vector{Float64}, means::Matrix{Float64}, covariances::Array{Float64, 3})
    k = size(w)[1]
    gaussians = [(means[i, 1:end], Matrix(covariances[i, 1:end, 1:end])) for i in 1:k]
    m = MixtureModel(MvNormal, gaussians, w)
    r = rand(m, numb)
    return r
end

"""
    densePerfectMoments(d, k, w, true_means, true_covariances; method = "low")

Use the given parameters to compute the exact moments necessary for parameter estimation with dense covariance matrices.

Returns moments 0 to 3k for the first dimension, moments 1 through 2k+1 for the other dimensions as a matrix, and a dictionary with indices and moments for the off-diagonal system.
"""
function densePerfectMoments(d, k, w, true_means, true_covariances; method = "low")
    @var s[1:k] y[1:k] a[1:k]
    (system, polynomial) = build1DSystem(k, 3*k+1)
    # Compute the moments for step 1
    true_params = append!(copy(w), [true_covariances[i,1,1] for i in 1:k], [true_means[i,1] for i in 1:k])
    first_moms = [p([a; s; y] => true_params) for p in system]

    (system_real, polynomial_real) = build1DSystem(k, 2*k+1, w)
    diagonal_moms = Matrix{Float64}(undef, (d-1,2*k+1))
    for i in 2:d
        true_params = append!([true_covariances[j,i,i] for j in 1:k], [true_means[j,i] for j in 1:k])
        all_moments = append!([p([s; y] => true_params) for p in system_real[2:end]], polynomial_real([s; y] => true_params))
        if typeof(all_moments[1]) != Float64
            diagonal_moms[i-1, 1:end] = to_number.(expand.(all_moments))
        else
            diagonal_moms[i-1, 1:end] = all_moments
        end
    end

    @var vs[1:k, 1:d, 1:d]
    covariances::Array{Union{Variable, Float64}} = reshape([vs...], (k, d, d))
    for dimension in 1:d
        for factor in 1:k
            covariances[factor, dimension, dimension] = true_covariances[factor,dimension, dimension]
        end
    end

    if method == "low"
        true_mixed_system = tensorMixedMomentSystem(d, k, w, true_means, covariances)
    elseif method == "k"
        for i1 in 1:k
            for i2 in 1:d
                for i3 in i2:d
                    covariances[i1,i3,i2] = covariances[i1, i2, i3]
                end
            end
        end

        indexes = Dict{String, Array}()
        for i in 2:k + 1
            indexes[string(i)] = []
        end

        target = k*(d^2-d)/2
        for i in 1:d-1
            for j in i+1:d
                n = 2
                temp = Int64.(zeros(d))
                temp[i] = 1
                temp[j] = 1
                push!(indexes["2"], temp)
                for n in 3:(k+1)
                    temp1 = Int64.(zeros(d))
                    temp1[i] = n-1
                    temp1[j] = 1
                    push!(indexes[string(n)], temp1)
                end
            end
        end

        true_mixed_system = Dict{Vector{Int64}, Expression}()
        for n in 2:k + 1
            for i in 1:k
                for j in 0:floor((n)/2)
                    coefficient = w[i]*binomial(n, Int64(2*j))*factorial(Int64(2*j))/(factorial(Int64(j))*(2^j))
                    means = tensorPower(true_means[i, 1:end], Int64(n - 2*j))
                    variances = tensorPower(covariances[i, 1:end, 1:end], Int64(j))
                    partial_tensor = coefficient .* reshape(kron(means, variances), Integer.(tuple(d * ones(n)...)))

                    # symmetrize the tensors
                    for key in indexes[string(n)]
                        ind = convert_indexing(key, d)
                        polynomial = 0*w[1]
                        for item in collect(Combinatorics.permutations(ind))
                            polynomial += partial_tensor[item...]
                        end
                        if typeof(polynomial) == Float64
                            polynomial = polynomial/factorial(n)
                        else
                            polynomial = expand(polynomial/factorial(n))
                        end
                        if key in keys(true_mixed_system)
                            true_mixed_system[key] += polynomial
                        else
                            true_mixed_system[key] = polynomial
                        end
                    end
                end
            end
        end
    end

    indexes = Dict{Vector{Int64}, Float64}()

    for (key, polynomial) in true_mixed_system
        sample_moment = true_mixed_system[key](vs=>true_covariances)
        indexes[key] = sample_moment
    end
    return (first_moms, diagonal_moms, indexes)
end

"""
    diagonalPerfectMoments(d, k, w, true_means, true_covariances)

Use the given parameters to compute the exact moments necessary for parameter estimation with diagonal covariance matrices.

Returns moments 0 to 3k for the first dimension, and moments 1 through 2k+1 for the other dimensions as a matrix.
"""
function diagonalPerfectMoments(d, k, w, true_means, true_covariances)
    @var s[1:k] y[1:k] a[1:k]
    (system, polynomial) = build1DSystem(k, 3*k+1)
    # Compute the moments for step 1
    true_params = append!(copy(w), [true_covariances[i][1,1] for i in 1:k], [true_means[i,1] for i in 1:k])
    first_moms = [p([a; s; y] => true_params) for p in system]

    (system_real, polynomial_real) = build1DSystem(k, 2*k+1, w)
    diagonal_moms = Matrix{Float64}(undef, (d-1,2*k+1))
    for i in 2:d
        true_params = append!([true_covariances[j][i,i] for j in 1:k], [true_means[j,i] for j in 1:k])
        all_moments = append!([p([s; y] => true_params) for p in system_real[2:end]], polynomial_real([s; y] => true_params))
        if typeof(all_moments[1]) != Float64
            diagonal_moms[i-1, 1:end] = to_number.(expand.(all_moments))
        else
            diagonal_moms[i-1, 1:end] = all_moments
        end
    end

    return (first_moms, diagonal_moms)
end

"""
    equalMixCovarianceKnown_moments(k, mean, shared_cov)

Use the given parameters to compute the exact moments necessary for parameter estimation with equal mixing coefficients and shared known covariances.

Returns moments 0 to `k` for the first dimension, and moments ```math m\\_{je\\_1+e\\_i}``` for j in 0 to `k`-1 and i in 2 to d as a matrix where d is the dimension, i varies across rows, and j varies down columns.
"""
function equalMixCovarianceKnown_moments(k::Integer, mean::Matrix{Float64}, shared_cov::Matrix{Float64})
    d = size(shared_cov)[1]
    @var ms[1:k,1:d]

    @var s[1:k], y[1:k] #y means, s sigma
    first_system = build1DSystem(k, k+1, ones(k)/k)[1][2:end]
    first_moms = evaluate(first_system, s => repeat([shared_cov[1,1]],k))
    first_moms = evaluate(first_moms, y => mean[1:k,1])

    mean_systems = Array{Union{Expression, Float64}}(undef, (k,d-1))

    for i in 1:d-1
        mean_systems[1,i] = expand(sum(ms[1:k,i+1])/k)
    end

    for n in 2:k #order of tensor moment
        for i in 1:k
            for j in 0:floor((n)/2)
                coefficient = (1/k)*binomial(n, Int64(2*j))*factorial(Int64(2*j))/(factorial(Int64(j))*(2^j))
                means = tensorPower(ms[i, 1:end], Int64(n - 2*j))
                variances = tensorPower(shared_cov, Int64(j))
                partial_tensor = coefficient .* reshape(kron(means, variances), Integer.(tuple(d * ones(n)...)))

                for l in 1:d-1
                    key = Int64.(zeros(d))
                    key[1] = n-1
                    key[l+1] = 1

                    ind = convert_indexing(key, d)
                    polynomial = 0*ms[1,1]
                    for item in collect(Combinatorics.permutations(ind))
                        polynomial += partial_tensor[item...]
                    end
                    if typeof(polynomial) == Float64
                        polynomial = polynomial/factorial(n)
                    else
                        polynomial = expand(polynomial/factorial(n))
                    end

                    if isassigned(mean_systems, (l-1)*(k) + n)
                        mean_systems[n,l] += polynomial
                    else
                        mean_systems[n,l] = polynomial
                    end
                end
            end
        end
    end

    second_moms = zeros(k,d-1)
    for i in 1:d-1
        system = mean_systems[1:end, i]
        for j in 1:k
            second_moms[j,i] = system[j](ms=>mean)
        end
    end

    # solve the systems and put them into ms
    return (first_moms, second_moms)
end

"""
    cycle_moments(d, k, w, means, covars, diagonal)

Calculate 0 through 3`k`+1 denoised moments for every dimension.

Used as input for cycling over the dimensions to find candidate mixing coefficients.
"""
function cycle_moments(d, k, w, means, covars, diagonal)
    moments = Array{Float64}(undef, (d, 3*k+1))
    @var s[1:k] y[1:k] a[1:k]
    (system, polynomial) = build1DSystem(k, 3*k+1)

    if diagonal
        for i in 1:d
            true_params = append!(copy(w), [covars[j][i,i] for j in 1:k], [means[j,i] for j in 1:k])
            all_moments = [p([a; s; y] => true_params) for p in system]
            if typeof(all_moments[1]) != Float64
                moments[i, 1:end] = to_number.(expand.(all_moments))
            else
                moments[i, 1:end] = all_moments
            end
        end
    else
        for i in 1:d
            true_params = append!(copy(w), [covars[j,i,i] for j in 1:k], [means[j,i] for j in 1:k])
            all_moments = [p([a; s; y] => true_params) for p in system]
            if typeof(all_moments[1]) != Float64
                moments[i, 1:end] = to_number.(expand.(all_moments))
            else
                moments[i, 1:end] = all_moments
            end
        end
    end
    return moments
end

"""
    cycle_moments(k, sample, diagonal, style)

Calculate 0 through 3`k`+1 sample moments for every dimension.

Used as input for cycling over the dimensions to find candidate mixing coefficients.
"""
function cycle_moments(k, sample, diagonal, style)
    (d, sample_size) = size(sample)
    single_dim_moments = Array{Float64}(undef, (d, 3*k+1))

    for i in 1:d
        for j in 0:(3*k)
            single_dim_moments[i,j+1] = mean(sample[i,n]^j for n in 1:sample_size)
        end
    end

    if diagonal
        return single_dim_moments
    else
        indexes = Dict{Vector{Int64}, Float64}()

        if style == "low"
            for i in 1:(d-1)
                for j in (i+1):d
                    key = repeat([0], d)
                    key[i] = 1
                    key[j] = 1

                    mom = mean(sample[i,n]*sample[j,n] for n in 1:sample_size)

                    indexes[key] = mom

                    second_lim = (k%2 == 0) ? Int64(k/2) : Int64(floor(k/2))+1

                    for t in 2:second_lim
                        key = repeat([0], d)
                        key[i] = t
                        key[j] = 1

                        mom = mean((sample[i,n]^t)*sample[j,n] for n in 1:sample_size)

                        indexes[key] = mom

                        key = repeat([0], d)
                        key[j] = t
                        key[i] = 1

                        mom = mean((sample[j,n]^t)*sample[i,n] for n in 1:sample_size)

                        indexes[key] = mom
                    end

                    if k % 2 == 0
                        key = repeat([0], d)
                        key[i] = second_lim + 1
                        key[j] = 1

                        mom = mean((sample[i,n]^(second_lim + 1))*sample[j,n] for n in 1:sample_size)

                        indexes[key] = mom
                    end
                end
            end
        elseif style == "k"
            for i in 1:(d-1)
                for j in (i+1):d
                    for t in 1:k
                        key = repeat([0], d)
                        key[i] = t
                        key[j] = 1

                        mom = mean((sample[i,n]^t)*sample[j,n] for n in 1:sample_size)

                        indexes[key] = mom
                    end
                end
            end
        else
            println("style must be low or k")
        end
        return (single_dim_moments, indexes)
    end
end

"""
    equalMixCovarianceKnown_moments(k, sample)

Use the given parameters to compute the sample moments necessary for parameter estimation with equal mixing coefficients and shared known covariances.

Returns moments 0 to `k` for the first dimension, and moments m_{je_1+e_i} for j in 0 to `k`-1 and i in 2 to d as a matrix where d is the dimension, i varies across rows, and j varies down columns.
"""
function equalMixCovarianceKnown_moments(k::Integer, sample::Matrix{Float64})
    (d, sample_size) = size(sample)
    @var ms[1:k,1:d]

    first_moms = Vector{Float64}(undef, k)
    for j in 1:k
        first_moms[j] = mean(sample[1,i]^j for i in 1:sample_size)
    end

    second_moms = zeros(k,d-1)
    for j in 1:k
        second_moms[1,j] = mean(sample[j,i] for i in 1:sample_size)
    end

    for n in 2:k #order of tensor moment
        for i in 1:k
            for j in 0:floor((n)/2)
                for l in 1:d-1
                    key = Int64.(zeros(d))
                    key[1] = n-1
                    key[l+1] = 1

                    sample_moment = 0
                    for j in 1:sample_size
                        temp_moment = 1
                        for i in 1:d
                            temp_moment *= sample[i, j]^(key[i])
                        end
                        sample_moment += temp_moment
                    end
                    sample_moment = sample_moment/sample_size

                    second_moms[n,l] = sample_moment
                end
            end
        end
    end

    return (first_moms, second_moms)
end

"""
    sampleMoments(sample::Matrix{Float64}, k; diagonal = false)

Use the sample to compute the moments necessary for parameter estimation using method of moments with general covariance matrices and mixing coefficients.

Returns moments 0 to 3k for the first dimension, moments 1 through 2k+1 for the other dimensions as a matrix, and a dictionary with indices and moments for the off-diagonal system if `diagonal` is false.
"""
function sampleMoments(sample::Matrix{Float64}, k; diagonal = false)
    (d, sample_size) = size(sample)
    first_moms = Vector{Float64}(undef, 3*k+1)
    first_moms[1] = 1.0
    for j in 1:3*k
        first_moms[j+1] = mean(sample[1,i]^j for i in 1:sample_size)
    end
    diagonal_moms = Matrix{Float64}(undef, (d-1,2*k+1))
    for j in 1:2*k+1
        diagonal_moms[1:end, j] = mean.(sample[i, 1:end].^j for i in 2:d)
    end
    if diagonal
        return (first_moms, diagonal_moms)
    else
        indexes = Dict{Vector{Int64}, Float64}()

        target = k*(d^2-d)/2
        for i in 1:d-1
            for j in i+1:d
                n = 2
                temp = Int64.(zeros(d))
                temp[i] = 1
                temp[j] = 1

                sample_moment = 0
                for j in 1:sample_size
                    temp_moment = 1
                    for i in 1:d
                        temp_moment *= sample[i, j]^(temp[i])
                    end
                    sample_moment += temp_moment
                end
                sample_moment = sample_moment/sample_size
                indexes[temp] = sample_moment

                for n in 3:Int64(ceil((k+1)/2))
                    temp1 = Int64.(zeros(d))
                    temp1[i] = n-1
                    temp1[j] = 1
                    sample_moment1 = 0
                    for j in 1:sample_size
                        temp_moment1 = 1
                        for i in 1:d
                            temp_moment1 *= sample[i, j]^(temp1[i])
                        end
                        sample_moment1 += temp_moment1
                    end
                    sample_moment1 = sample_moment1/sample_size
                    indexes[temp1] = sample_moment1

                    temp2 = Int64.(zeros(d))
                    temp2[j] = n-1
                    temp2[i] = 1
                    sample_moment2 = 0
                    for j in 1:sample_size
                        temp_moment2 = 1
                        for i in 1:d
                            temp_moment2 *= sample[i, j]^(temp2[i])
                        end
                        sample_moment2 += temp_moment2
                    end
                    sample_moment2 = sample_moment2/sample_size
                    indexes[temp2] = sample_moment2
                end

                n = Int64(ceil((k+1)/2)) + 1
                temp = Int64.(zeros(d))
                temp[j] = 1
                temp[i] = ceil((k+1)/2)

                sample_moment = 0
                for j in 1:sample_size
                    temp_moment = 1
                    for i in 1:d
                        temp_moment *= sample[i, j]^(temp[i])
                    end
                    sample_moment += temp_moment
                end
                sample_moment = sample_moment/sample_size
                indexes[temp] = sample_moment

                if k % 2 != 0
                    temp = Int64.(zeros(d))
                    temp[j] = (k+1)/2
                    temp[i] = 1
                    sample_moment = 0
                    for j in 1:sample_size
                        temp_moment = 1
                        for i in 1:d
                            temp_moment *= sample[i, j]^(temp[i])
                        end
                        sample_moment += temp_moment
                    end
                    sample_moment = sample_moment/sample_size
                    indexes[temp] = sample_moment
                end
            end
        end
        return (first_moms, diagonal_moms, indexes)
    end
end
