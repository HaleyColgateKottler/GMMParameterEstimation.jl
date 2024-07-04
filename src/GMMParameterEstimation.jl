module GMMParameterEstimation

using HomotopyContinuation
using Distributions
using LinearAlgebra
using Combinatorics
using JLD2
using GMMParameterEstimation
include("momentGeneration.jl")

export makeCovarianceMatrix, generateGaussians, getSample, build1DSystem, tensorMixedMomentSystem, estimate_parameters, sampleMoments, generalPerfectMoments, diagonalPerfectMoments, equalMixCovarianceKnown_moments, estimate_parameters_weights_known

"""
    checkInputs(d::Integer, k::Integer, first::Vector{Float64}, second::Matrix{Float64}, last::Dict{Vector{Int64}, Float64}, method)

Returns `true` if the inputs are the right format for `estimate_parameters` and an error otherwise.
"""
function checkInputs(d::Integer, k::Integer, first::Vector{Float64}, second::Matrix{Float64}, last::Dict{Vector{Int64}, Float64}, method)
    !(method in ["recursive", "tensor"]) && error("method must be recursive or tensor")
    (length(first) != 3k+1) && error("first must have length 3k+1")
    (size(second) != (d-1,2k+1)) && error("second must be d-1 x 2k+1")
    (length(last) != k*d*(d-1)/2) && error("last must include kd(d-1)/2 entries")
    for key in keys(last)
        (count(x -> x>0, key) != 2) && error("moment keys in last should be of the form te_{i}+e_{j} for i != j, t<= floor((k-1)/2)+1, see docs for more detail")
        (count(x -> x==1, key) < 1) && error("moment keys in last should be of the form te_{i}+e_{j} for i != j, t<= floor((k-1)/2)+1, see docs for more detail")
    end

    return true
end

"""
    checkInputsKnown(d::Integer, k::Integer, w::Vector{Float64}, first::Matrix{Float64}, last::Dict{Vector{Int64}, Float64}, method)
"""
function checkInputsKnown(d::Integer, k::Integer, w::Vector{Float64}, first::Matrix{Float64}, last::Dict{Vector{Int64}, Float64}, method)
    (length(w) != k) && error("w must be length k")
    (size(first) != (d,2k+1)) && error("first must be d x 2k+1")

    (length(last) != k*d*(d-1)/2) && error("last must include kd(d-1)/2 entries")
    for key in keys(last)
        (count(x -> x>0, key) != 2) && error("moment keys in last should be of the form te_{i}+e_{j} for i != j, t<= floor((k-1)/2)+1, see docs for more detail")
        (count(x -> x==1, key) < 1) && error("moment keys in last should be of the form te_{i}+e_{j} for i != j, t<= floor((k-1)/2)+1, see docs for more detail")
    end
    return true
end

"""
    checkInputs(d::Integer, k::Integer, first::Vector{Float64}, second::Matrix{Float64})
"""
function checkInputs(d::Integer, k::Integer, first::Vector{Float64}, second::Matrix{Float64})
    (length(first) != 3k+1) && error("first must have length 3k+1")
    (size(second) != (d-1,2k+1)) && error("second must be d-1 x 2k+1")

    return true
end

"""
    checkInputsKnown(d::Integer, k::Integer, first::Vector{Float64}, second::Matrix{Float64})
"""
function checkInputsKnown(d::Integer, k::Integer, w::Vector{Float64}, second::Matrix{Float64})
    (length(w) != k) && error("w must be length k")
    (size(second) != (d, 2k+1)) && error("moments must be d x 2k+1")

    return true
end

"""
    checkInputs(d::Integer, k::Integer, shared_cov::Matrix{Float64}, first::Vector{Float64}, second::Matrix{Float64}, method)
"""
function checkInputs(d::Integer, k::Integer, shared_cov::Matrix{Float64}, first::Vector{Float64}, second::Matrix{Float64}, method)
    !(method in ["recursive", "tensor"]) && error("method must be recursive or tensor")
    !isposdef(shared_cov) && error("shared_cov must be positive definite")
    (length(first) != k) && error("first must have length k+1")
    (size(second) != (k,d-1)) && error("second must be k x d-1")

    return true
end

"""
    build1DSystem(k::Integer, m::Integer)

Build the polynomial system for a mixture of 1D Gaussians where 'm'-1 is the highest desired moment and the mixing coefficients are unknown.
"""
function build1DSystem(k::Integer, m::Integer)
    @var a[1:k]
    return build1DSystem(k, m, a)
end

"""
    build1DSystem(k::Integer, m::Integer, a::Union{Vector{Float64}, Vector{Variable}})

Build the polynomial system for a mixture of 1D Gaussians where 'm'-1 is the highest desired moment, and `a` is a vector of the mixing coefficients.
"""
function build1DSystem(k::Integer, m::Integer, a::Union{Vector{Float64}, Vector{Variable}})
    @var s[1:k] y[1:k] t x

    bases = [1, x]
    for i in 2:m
        push!(bases, expand(x*bases[i]+(i-1)*t*bases[i-1]))
    end

    system::Vector{Expression} = []
    for i in 1:length(bases)-1
        push!(system, a'*[bases[i]([t,x]=>[s[j],y[j]]) for j in 1:k])
    end
    target = a'*[bases[end]([t,x]=>[s[j],y[j]]) for j in 1:k]
    return (system, target)
end

"""
    selectSol(k::Integer, solution::Result, polynomial::Expression, moment::Number)

Select a `k` mixture solution from `solution` accounting for `polynomial` and `moment`.

Sort out a `k` mixture statistically significant solutions from `solution`, and return the one closest to `moment` when `polynomial` is evaluated at those values.
"""
function selectSol(k::Integer, solution::Result, polynomial::Expression, moment::Number)
    @var s[1:k] y[1:k]
    stat_significant = [];

    vars = append!(s,y)
    stat_significant = filter(r -> all(r[1:k] .> 0), real_solutions(solution))
    size(stat_significant)[1] == 0 && (return false)
    # Chose the best statistically meaningful solution
    best_sol = [];
    for i in stat_significant
        t = polynomial(vars => i)
        append!(best_sol, norm(moment - t))
    end # Create list of differences between moment and polynomial(statistically significant solutions)

    id = findall(x->x==minimum(best_sol), best_sol); # Get the indices of the vals with minimum norm
    return stat_significant[id][1] # pull the best one
end

"""
    tensorPower(tensor, power::Integer)

Compute the `power` tensor power of `tensor`.
"""
function tensorPower(tensor, power::Integer)
    if power == 0
        return 1
    elseif power == 1
        return tensor
    else
        base::Array = tensor
        for i in 2:power
            base = kron(base, tensor)
        end
        return base
    end
end

"""
    convert_indexing(moment_i, d)

Convert the `d` dimensional multivariate `moment_i` index to the corresponding tensor moment index.
"""
function convert_indexing(moment_i, d)
    indexing::Array{Int64} = [repeat([1], moment_i[1])...]
    for w in 2:d
        push!(indexing, repeat([w], moment_i[w])...)
    end
    return indexing
end

"""
    tensorMixedMomentSystem(d, k, mixing, ms, vs)

Build a linear system for finding the off-diagonal covariances entries using tensor moments.

For a `d` dimensional Gaussian `k`-mixture model with mixing coefficients `mixing`, means `ms`, and covariances `vs` where the diagonal entries have been filled in and the off diagonals are variables.
"""
function tensorMixedMomentSystem(d, k, mixing, ms, vs)
    # Force symmetry of covariance matrix
    for i1 in 1:k
        for i2 in 1:d
            for i3 in i2:d
                vs[i1,i3,i2] = vs[i1, i2, i3]
            end
        end
    end

    indexes = Dict{String, Array}()
    for i in 2:Int64(ceil((k+1)/2)) + 1
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
            for n in 3:Int64(ceil((k+1)/2))
                temp1 = Int64.(zeros(d))
                temp1[i] = n-1
                temp1[j] = 1
                temp2 = Int64.(zeros(d))
                temp2[j] = n-1
                temp2[i] = 1
                push!(indexes[string(n)], temp1, temp2)
            end
            n = Int64(ceil((k+1)/2)) + 1
            temp = Int64.(zeros(d))
            temp[j] = 1
            temp[i] = ceil((k+1)/2)
            push!(indexes[string(n)], temp)
            if k % 2 != 0
                temp = Int64.(zeros(d))
                temp[j] = (k+1)/2
                temp[i] = 1
                push!(indexes[string(n)], temp)
            end
        end
    end

    indexed_system = Dict{Vector{Int64}, Expression}()
    for n in 2:Int64(ceil((k+1)/2)) + 1
        for i in 1:k
            for j in 0:floor((n)/2)
                coefficient = mixing[i]*binomial(n, Int64(2*j))*factorial(Int64(2*j))/(factorial(Int64(j))*(2^j))
                means = tensorPower(ms[i, 1:end], Int64(n - 2*j))
                variances = tensorPower(vs[i, 1:end, 1:end], Int64(j))
                partial_tensor = coefficient .* reshape(kron(means, variances), Integer.(tuple(d * ones(n)...)))

                # symmetrize the tensors
                for key in indexes[string(n)]
                    ind = convert_indexing(key, d)
                    polynomial = 0*mixing[1]
                    for item in collect(Combinatorics.permutations(ind))
                        polynomial += partial_tensor[item...]
                    end
                    if typeof(polynomial) == Float64
                        polynomial = polynomial/factorial(n)
                    else
                        polynomial = expand(polynomial/factorial(n))
                    end
                    if key in keys(indexed_system)
                        indexed_system[key] += polynomial
                    else
                        indexed_system[key] = polynomial
                    end
                end
            end
        end
    end
    return indexed_system
end

"""
    tensorOffDiagonalSolve(d, k, mixing_coefficients, means, covariances, last)

Use tensor moments to build and solve a system for the off-diagonal covariance entries.
"""
function tensorOffDiagonalSolve(d, k, mixing_coefficients, means, covariances, last)
    mixed_system = tensorMixedMomentSystem(d, k, mixing_coefficients, means, covariances)
    final_system::Vector{Expression} = []
    target_vector = Vector{Float64}()
    @var vs[1:k, 1:d, 1:d]

    for (key, polynomial) in mixed_system
        constant = polynomial(vs => zeros(size(vs)))
        sample_moment = last[key]
        push!(final_system, polynomial)
        push!(target_vector, sample_moment-constant)
    end

    remaining_vars = variables(final_system)
    matrix_system = jacobian(System(final_system), zeros(size(final_system)[1]))
    last_covars = [matrix_system\target_vector]
    final_system = []
    for i2 in 1:d
        for i3 in i2:d
            for i1 in 1:k
                if typeof(covariances[i1, i2, i3]) != Float64
                    covariances[i1, i2, i3] = covariances[i1, i2, i3](remaining_vars => last_covars[1])
                    covariances[i1, i3, i2] = covariances[i1, i2, i3]
                end
            end
        end
    end
    return(covariances)
end

"""
    kOrder_offDiagonalSolve(d, k, mixing_coefficients, means, covariances, last)

Use a recursive system to build and solve a system for the off-diagonal covariance entries using moments ``m_{te_i+e_j}`` for t in [k] and i<j
"""
function kOrder_offDiagonalSolve(d, k, mixing_coefficients, means, covariances, last)
    @var s[1:k] y[1:k] a[1:k] vs[1:k, 1:d, 1:d]
    base_moments = build1DSystem(1, k+1)[1]

    for i in 1:(d-1)
        i_params = [1, covariances[1,i,i], means[1,i]]
        i_moments = [[p([a[1];s[1];y[1]]=> (i_params)) for p in base_moments]]

        for ind in 2:k
            i_params = [1, covariances[ind,i,i], means[ind,i]]
            push!(i_moments, [p([a[1];s[1];y[1]]=> (i_params)) for p in base_moments])
        end

        for j in (i+1):d
            block_system = [mixing_coefficients[1]*(means[1,j]*i_moments[1][t+1] + t*covariances[1,i,j]*i_moments[1][t]) for t in 1:k]

            for l in 2:k
                block_system = block_system + [mixing_coefficients[l]*(means[l,j]*i_moments[l][t+1] + t*covariances[l,i,j]*i_moments[l][t]) for t in 1:k]
            end

            target_vector = Vector{Float64}()

            for t in 1:k
                constant = block_system[t](vs => zeros(size(vs)))
                key = repeat([0], d)
                key[i] = t
                key[j] = 1
                sample_moment = last[key]
                push!(target_vector, sample_moment-constant)
            end

            remaining_vars = variables(block_system)
            matrix_system = jacobian(System(block_system), zeros(size(block_system)[1]))
            last_covars = [matrix_system\target_vector]
            for l in 1:k
                covariances[l, i, j] = covariances[l, i, j](remaining_vars => last_covars[1])
                covariances[l, j, i] = covariances[l, i, j]
            end
        end
    end
    return covariances
end

"""
    lowOrder_offDiagonalSolve(d, k, mixing_coefficients, means, covariances, last)

Use a recursive system to build and solve a system for the off-diagonal covariance entries using minimal order moments ``m_{te_i+e_j}`` and ``m_{e_i+te_j}``
"""
function lowOrder_offDiagonalSolve(d, k, mixing_coefficients, means, covariances, last)
    @var s[1:k] y[1:k] a[1:k] vs[1:k, 1:d, 1:d]
    base_moments = build1DSystem(1, Int64(floor(k/2)+2))[1]

    for i in 1:(d-1)
        i_params = [1, covariances[1,i,i], means[1,i]]
        i_moments = [[p([a[1];s[1];y[1]]=> (i_params)) for p in base_moments]]

        for ind in 2:k
            i_params = [1, covariances[ind,i,i], means[ind,i]]
            push!(i_moments, [p([a[1];s[1];y[1]]=> (i_params)) for p in base_moments])
        end

        for j in (i+1):d
            j_params = [1, covariances[1,j,j], means[1,j]]
            j_moments = [[p([a[1];s[1];y[1]]=> (j_params)) for p in base_moments]]

            for ind in 2:k
                j_params = [1, covariances[ind,j,j], means[ind,j]]
                push!(j_moments, [p([a[1];s[1];y[1]]=> (j_params)) for p in base_moments])
            end

            block_system = [mixing_coefficients[1]*(means[1,j]*i_moments[1][t+1] + t*covariances[1,i,j]*i_moments[1][t]) for t in 1:Int64(floor(k/2)+1)]

            for l in 2:k
                block_system = block_system + [mixing_coefficients[l]*(means[l,j]*i_moments[l][t+1] + t*covariances[l,i,j]*i_moments[l][t]) for t in 1:Int64(floor(k/2)+1)]
            end

            second_lim = (k%2 == 0) ? Int64(k/2) : Int64(floor(k/2)+1)

            append!(block_system, [mixing_coefficients[1]*(means[1,i]*j_moments[1][t+1] + t*covariances[1,i,j]*j_moments[1][t]) for t in 2:second_lim])
            for l in 2:k
                block_system[Int64(floor(k/2)+2):end] = block_system[Int64(floor(k/2)+2):end] + [mixing_coefficients[l]*(means[l,i]*j_moments[l][t+1] + t*covariances[l,i,j]*j_moments[l][t]) for t in 2:second_lim]
            end

            target_vector = Vector{Float64}()

            constant = block_system[1](vs => zeros(size(vs)))
            key = repeat([0], d)
            key[i] = 1
            key[j] = 1
            sample_moment = last[key]
            target_vector = [sample_moment - constant]

            for t in 2:Int64(floor(k/2)+1)
                constant = block_system[t](vs => zeros(size(vs)))
                key = repeat([0], d)
                key[i] = t
                key[j] = 1
                sample_moment = last[key]
                push!(target_vector, sample_moment-constant)
            end

            for t in 2:second_lim
                constant = block_system[t + Int64(floor(k/2))](vs => zeros(size(vs)))
                key = repeat([0], d)
                key[j] = t
                key[i] = 1
                sample_moment = last[key]
                push!(target_vector, sample_moment-constant)
            end

            remaining_vars = variables(block_system)
            matrix_system = jacobian(System(block_system), zeros(size(block_system)[1]))
            last_covars = [matrix_system\target_vector]
            for l in 1:k
                covariances[l, i, j] = covariances[l, i, j](remaining_vars => last_covars[1])
                covariances[l, j, i] = covariances[l, i, j]
            end
        end
    end
    return covariances
end

"""
   recursiveOffDiagonalSolve(d, k, mixing_coefficients, means, covariances, last)

Use a recursive system to build and solve a system for the off-diagonal covariance entries.
"""
function recursiveOffDiagonalSolve(d, k, mixing_coefficients, means, covariances, last)
    low = true
    for key in keys(last)
        if maximum(key) == k
            low = false
        end
    end

    if low
        return lowOrder_offDiagonalSolve(d, k, mixing_coefficients, means, covariances, last)
    else
        return kOrder_offDiagonalSolve(d, k, mixing_coefficients, means, covariances, last)
    end
end

"""
    estimate_parameters_weights_known(d::Integer, k::Integer, w::Array{Float64}, moments::Matrix{Float64})

Compute an estimate for the parameters of a `d`-dimensional Gaussian `k`-mixture model from the moments.

For the known mixing coefficient diagonal covariance matrix case, `w` should be a vector of the mixing coefficients, and `moments` should be a matrix of moments 1 through 2k+1 for all dimensions.
"""
function estimate_parameters_weights_known(d::Integer, k::Integer, w::Array{Float64}, moments::Matrix{Float64})
    target2 = Int64(doublefactorial(2*k-1)*factorial(k)) # Number of solutions to look for in step 3
    checkInputsKnown(d::Integer, k::Integer, w::Array{Float64}, moments::Matrix{Float64})

    @var m[0:2*k] s[1:k] y[1:k] a[1:k]
    means = Array{Float64}(undef, (k,d))
    covariances = []
    for i in 1:k
        push!(covariances, Diagonal{Float64}(undef, d))
    end

    # Build 1D system
    (system_i, polynomial_i) = build1DSystem(k, 2*k+1, w)

    temp_start = append!(randn(2*k) + im*randn(2*k))
    temp_moments = [p([s;y]=>(temp_start)) for p in system_i[2:end]]
    R1 =  monodromy_solve(system_i[2:end] - m[1:2*k], temp_start, temp_moments, parameters = m[1:2*k], target_solutions_count = target2, show_progress=false)

    for i in 1:d
        all_moments = moments[i, 1:end]

        solution = solve(system_i[2:end] - m[1:2*k], solutions(R1); parameters=m[1:2*k], start_parameters=temp_moments, target_parameters=all_moments[1:2*k], show_progress=false)

        # Choose the statistically significant solution closest to the next moment
        best_sol_i = selectSol(k, solution, polynomial_i, all_moments[end])
        if best_sol_i == false
            return (2, (nothing,nothing,nothing))
        end

        for j in 1:k
            covariances[j][i, i] = best_sol_i[j]
            means[j, i] = best_sol_i[k+j]
        end
    end

    solution = []
    binomials = []
    true_params = []
    all_moments = []
    system_i = []
    polynomial_i = 0
    best_sol_i = []
    gaussians = []

    return(0, (w, means, covariances))
end

"""
    estimate_parameters(d::Integer, k::Integer, first::Vector{Float64}, second::Matrix{Float64})

Compute an estimate for the parameters of a `d`-dimensional Gaussian `k`-mixture model from the moments.

For the unknown mixing coefficient diagonal covariance matrix case, `first` should be a list of moments 0 through 3k for the first dimension, and `second` should be a matrix of moments 1 through 2k+1 for the remaining dimensions.
"""
function estimate_parameters(d::Integer, k::Integer, first::Vector{Float64}, second::Matrix{Float64})
    target2 = Int64(doublefactorial(2*k-1)*factorial(k)) # Number of solutions to look for in step 3
    checkInputs(d::Integer, k::Integer, first::Vector{Float64}, second::Matrix{Float64})

    # Build the system of equations for step 1
    # m is the parameter for the moments, s gives the variances, y gives the means, and a gives the mixing coefficients
    @var m[0:3*k-1] s[1:k] y[1:k] a[1:k]
    (system, polynomial) = build1DSystem(k, 3*k)

    if k in [2,3,4]
        temp_moments = load(pkgdir(GMMParameterEstimation) * "/src/sys1_k" * string(k) * ".jld2", "moments")
        R1_sols = load(pkgdir(GMMParameterEstimation) * "/src/sys1_k" * string(k) * ".jld2", "sols")
    else
        target1 = target2
        relabeling = (GroupActions(v -> map(p -> (v[1:k][p]...,v[k+1:2*k][p]...,v[2*k+1:3k][p]...),SymmetricGroup(k))))

        temp_start = append!(randn(3*k) + im*randn(3*k))
        temp_moments = [p([a;s;y]=>(temp_start)) for p in system]

        R1 =  monodromy_solve(system - m[1:3*k], temp_start, temp_moments, parameters = m[1:3*k], target_solutions_count = target1, group_action = relabeling, show_progress=false)

        R1_sols = solutions(R1)
        relabeling = ()
        temp_start = []
    end

    vars = append!(a,s,y)
    # Parameter homotopy from random parameters to real parameters
    solution1 = solve(system - m[1:3*k], R1_sols; parameters=m[1:3*k], start_parameters=temp_moments, target_parameters=first[1:3*k], show_progress=false)

    system = []
    temp_moments = []
    R1_sols = []

    # Check for statistically significant solutions
    # Return the one closest to the given moment, and the number of statistically significant solutions
    # Filter out the statistically meaningful solutions (up to symmetry)
    # Check positive mixing coefficients
    pos_mixing  = filter(r -> all(r[1:k] .> 0), real_solutions(solution1));

    solution1 = []
    num_pos_mix = size(pos_mixing)[1]
    if num_pos_mix == 0
        best_sol = 1
        num_sols = 0
    else
        # Check positive variances
        stat_significant1 = filter(r -> all(r[k+1:2*k] .> 0), pos_mixing);
        num_sols = size(stat_significant1)[1]
        if num_sols == 0
            best_sol = 1
        else
            best_sols1 = [];
            # Create list of differences between moment and polynomial(statistically significant solutions)
            for i in 1:size(stat_significant1)[1]
                t = polynomial(vars => stat_significant1[i])
                append!(best_sols1, norm(first[end] - t))
            end

            # Chose the best statistically meaningful solution
            id = findall(x->x==minimum(best_sols1), best_sols1); # Get the indices of the values with minimum norm
            best_sols1 = []
            best_sol = stat_significant1[id][1]
        end
        stat_significant1 = []
    end
    polynomial = 0
    pos_mixing = []

    # If there aren't any statistically significant solutions, return false
    if best_sol == 1
        return (best_sol, (nothing,nothing,nothing))
    end

    # Separate out the mixing coefficients, variances, and means
    mixing_coefficients = best_sol[1:k]

    means = Array{Float64}(undef, (k,d))
    covariances = []
    for i in 1:k
        push!(covariances, Diagonal{Float64}(undef, d))
    end

    for i in 1:k
        means[i, 1] = best_sol[2*k+i]
        covariances[i][1, 1] = best_sol[k+i]
    end
    best_sol = []

    higher_dim_sols = estimate_parameters_weights_known(d-1, k, mixing_coefficients, second)
    if higher_dim_sols[1] > 0
        return (higher_dim_sols)
    else
        perm = Vector{Int}(undef, k)
        for (i, x) in enumerate(mixing_coefficients)
            perm[i] = findfirst(y -> y == x, higher_dim_sols[2][1])
        end
        means[1:end, 2:end] = higher_dim_sols[2][2][perm, 1:end]
        for i in 1:k
            for j in 2:d
                covariances[i][j,j] = higher_dim_sols[2][3][perm[i]][j-1,j-1]
            end
        end
    end

    return(0, (mixing_coefficients, means, covariances))
end

"""
    estimate_parameters_weights_known(d::Integer, k::Integer, w::Array{Float64}, first::Matrix{Float64}, last::Dict{Vector{Int64}, Float64}; method = "recursive")

Compute an estimate for the parameters of a `d`-dimensional Gaussian `k`-mixture model from the moments.

For the known mixing coefficient general covariance matrix case, `w` should be a vector of the mixing coefficients `first` should be a list of moments 0 through 3k for the first dimension, `second` should be a matrix of moments 1 through 2k+1 for the remaining dimensions, and `last` should be a dictionary of the indices as lists of integers and the corresponding moments.
"""
function estimate_parameters_weights_known(d::Integer, k::Integer, w::Array{Float64}, first::Matrix{Float64}, last::Dict{Vector{Int64}, Float64}; method = "recursive")
    target2 = Int64(doublefactorial(2*k-1)*factorial(k)) # Number of solutions to look for in step 3
    checkInputsKnown(d::Integer, k::Integer, w::Array{Float64}, first::Matrix{Float64}, last::Dict{Vector{Int64}, Float64}, method)

    diag_solve = estimate_parameters_weights_known(d, k, w, first)
    if diag_solve[1] > 0
        return(diag_solve)
    end
    perm = Vector{Int}(undef, k)
    for (i, x) in enumerate(w)
        perm[i] = findfirst(y -> y == x, diag_solve[2][1])
    end

    means = diag_solve[2][2][perm, 1:end]

    @var m[0:2*k] s[1:k] y[1:k] a[1:k]
    @var vs[1:k, 1:d, 1:d] ms[1:k, 1:d]

    covariances::Array{Union{Variable, Float64}} = reshape([vs...], (k, d, d))
    for i in 1:k
        for j in 1:d
            covariances[i, j, j] = diag_solve[2][3][perm[i]][j,j]
        end
    end

    if d>1
        if method == "tensor"
            covariances = tensorOffDiagonalSolve(d, k, w, means, covariances, last)
        elseif method == "recursive"
            covariances = recursiveOffDiagonalSolve(d, k, w, means, covariances, last)
        end
    end

    for i in 1:k
        if !isposdef(convert(Matrix{Float64}, covariances[i, 1:end, 1:end]))
            return(3, (nothing,nothing,nothing))
        end
    end

    return(0, (w, means, covariances))
end

"""
    estimate_parameters(d::Integer, k::Integer, first::Vector{Float64}, second::Matrix{Float64}, last::Dict{Vector{Int64}, Float64}; method = "recursive")

Compute an estimate for the parameters of a `d`-dimensional Gaussian `k`-mixture model from the moments.

For the unknown mixing coefficient general covariance matrix case, `first` should be a list of moments 0 through 3k for the first dimension, `second` should be a matrix of moments 1 through 2k+1 for the remaining dimensions, and `last` should be a dictionary of the indices as lists of integers and the corresponding moments.
"""
function estimate_parameters(d::Integer, k::Integer, first::Vector{Float64}, second::Matrix{Float64}, last::Dict{Vector{Int64}, Float64}; method = "recursive")
    checkInputs(d::Integer, k::Integer, first::Vector{Float64}, second::Matrix{Float64}, last::Dict{Vector{Int64}, Float64}, method)

    target2 = Int64(doublefactorial(2*k-1)*factorial(k)) # Number of solutions to look for in step 3

    # Build the system of equations for step 1
    # m is the parameter for the moments, s gives the variances, y gives the means, and a gives the mixing coefficients
    @var m[0:3*k-1] s[1:k] y[1:k] a[1:k]
    (system, polynomial) = build1DSystem(k, 3*k)

    if k in [2,3,4]
        temp_moments = load(pkgdir(GMMParameterEstimation) * "/src/sys1_k" * string(k) * ".jld2", "moments")
        R1_sols = load(pkgdir(GMMParameterEstimation) * "/src/sys1_k" * string(k) * ".jld2", "sols")
    else
        target1 = target2

        relabeling = (GroupActions(v -> map(p -> (v[1:k][p]...,v[k+1:2*k][p]...,v[2*k+1:3k][p]...),SymmetricGroup(k))))

        temp_start = append!(randn(3*k) + im*randn(3*k))
        temp_moments = [p([a;s;y]=>(temp_start)) for p in system]

        R1 =  monodromy_solve(system - m[1:3*k], temp_start, temp_moments, parameters = m[1:3*k], target_solutions_count = target1, group_action = relabeling, show_progress=false)

        R1_sols = solutions(R1)
        relabeling = ()
        temp_start = []
    end

    vars = append!(a,s,y)
    # Parameter homotopy from random parameters to real parameters
    solution1 = solve(system - m[1:3*k], R1_sols; parameters=m[1:3*k], start_parameters=temp_moments, target_parameters=first[1:3*k], show_progress=false)

    system = []
    temp_moments = []
    R1_sols = []

    # Check for statistically significant solutions
    # Return the one closest to the given moment, and the number of statistically significant solutions
    # Filter out the statistically meaningful solutions (up to symmetry)
    # Check positive mixing coefficients
    pos_mixing  = filter(r -> all(r[1:k] .> 0), real_solutions(solution1));
    solution1 = []
    num_pos_mix = size(pos_mixing)[1]
    if num_pos_mix == 0
        best_sol = 1
        num_sols = 0
    else
        # Check positive variances
        stat_significant1 = filter(r -> all(r[k+1:2*k] .> 0), pos_mixing);
        num_sols = size(stat_significant1)[1]
        if num_sols == 0
            best_sol = 1
        else
            best_sols1 = [];
            # Create list of differences between moment and polynomial(statistically significant solutions)
            for i in 1:size(stat_significant1)[1]
                t = polynomial(vars => stat_significant1[i])
                append!(best_sols1, norm(first[end] - t))
            end

            # Chose the best statistically meaningful solution
            id = findall(x->x==minimum(best_sols1), best_sols1); # Get the indices of the values with minimum norm
            best_sols1 = []
            best_sol = stat_significant1[id][1]
        end
        stat_significant1 = []
    end
    polynomial = 0
    pos_mixing = []

    # If there aren't any statistically significant solutions, return false
    if best_sol == 1
        return (best_sol, (nothing,nothing,nothing))
    end

    # Separate out the mixing coefficients, variances, and means
    mixing_coefficients = best_sol[1:k]

    @var vs[1:k, 1:d, 1:d] ms[1:k, 1:d]

    means::Array{Union{Variable, Float64}} = reshape([ms...], (k, d))
    covariances::Array{Union{Variable, Float64}} = reshape([vs...], (k, d, d))

    for i in 1:k
        means[i, 1] = best_sol[2*k+i]
        covariances[i, 1, 1] = best_sol[k+i]
    end
    best_sol = []

    higher_dim_sols = estimate_parameters_weights_known(d, k, mixing_coefficients, vcat(transpose(first[2:2*k+2]), second), last; method = method)
    if higher_dim_sols[1] == 0
        for i in 1:k
            if !isposdef(convert(Matrix{Float64}, higher_dim_sols[2][3][i, 1:end, 1:end]))
                return(3, (nothing,nothing,nothing))
            end
        end
    end

    return(higher_dim_sols)
end

"""
    estimate_parameters(k::Integer, shared_cov::Matrix{Float64}, first::Vector{Float64}, second::Matrix{Float64}; method = "recursive")

Compute an estimate for the means of a Gaussian `k`-mixture model with equal mixing coefficients and known shared covariances from the moments.

The shared covariance matrix `shared_cov` will determine the dimension. Then `first` should be a list of moments 0 through k for the first dimension, `second` should be a matrix of moments ``m_{je_1+e_i}`` for j in 0 to `k`-1 and i in 2 to d as a matrix where d is the dimension, i varies across rows, and j varies down columns.
"""
function estimate_parameters(k::Integer, shared_cov::Matrix{Float64}, first::Vector{Float64}, second::Matrix{Float64}; method = "recursive")
    d = size(shared_cov)[1]
    checkInputs(d::Integer, k::Integer, shared_cov::Matrix{Float64}, first::Vector{Float64}, second::Matrix{Float64}, method)

    @var ms[1:k, 1:d]
    meansEst::Array{Union{Variable, Float64}} = reshape([ms...], (k, d))
    @var s[1:k], y[1:k] #y means, s sigma
    first_system = build1DSystem(k, k+1, ones(k)/k)[1][2:end]
    first_system = evaluate(first_system, s => repeat([shared_cov[1,1]],k))

    sol1 = solve(first_system-first, show_progress = false)
    real_sols = real_solutions(sol1)

    size(real_sols)[1] == 0 && (return (false, nothing))

    meansEst[1:k,1] = real_solutions(sol1)[1]
    mean_systems = Array{Expression}(undef, (k,d-1))

    if method == "tensor"
        mean_systems = Array{Expression}(undef, (k,d-1))

        for i in 1:d-1
            mean_systems[1,i] = expand(sum(ms[1:k,i+1])/k)
        end

        for n in 2:k #order of tensor moment
            for i in 1:k
                for j in 0:floor((n)/2)
                    coefficient = (1/k)*binomial(n, Int64(2*j))*factorial(Int64(2*j))/(factorial(Int64(j))*(2^j))
                    means = tensorPower(meansEst[i, 1:end], Int64(n - 2*j))
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
        for i in 1:d-1
            system = mean_systems[1:end, i]

            target_vector = Vector{Float64}()
            for j in 1:k
                constant = system[j](ms=>zeros(size(ms)))
                sample_moment = second[j, i]
                push!(target_vector, sample_moment - constant)
            end
            vars = variables(system)
            matrix_system = jacobian(System(system), zeros(size(system)[1]))
            solution = [matrix_system\target_vector]
            meansEst[1:k, i+1] = solution[1]
        end
    elseif method == "recursive"
        @var s[1:k] y[1:k] a[1:k]
        base_moments = build1DSystem(1, k)[1]

        i_params = [1, shared_cov[1,1], meansEst[1,1]]
        i_moments = [[p([a[1];s[1];y[1]]=> (i_params)) for p in base_moments]]

        for ind in 2:k
            i_params = [1, shared_cov[1,1], meansEst[ind,1]]
            push!(i_moments, [p([a[1];s[1];y[1]]=> (i_params)) for p in base_moments])
        end

        for j in 2:d
            block_system = [(1/k)*(meansEst[1,j]*i_moments[1][t+1] + t*shared_cov[1,j]*i_moments[1][t]) for t in 1:(k-1)]

            for l in 2:k
                block_system = block_system + [(1/k)*(meansEst[l,j]*i_moments[l][t+1] + t*shared_cov[1,j]*i_moments[l][t]) for t in 1:(k-1)]
            end

            push!(block_system, 1/k .* sum(meansEst[t, j] for t in 1:k))


            target_vector = Vector{Float64}()

            for t in 1:(k-1)
                constant = block_system[t](ms => zeros(size(ms)))
                sample_moment = second[t+1, j-1]
                push!(target_vector, sample_moment-constant)
            end

            constant = block_system[end](ms => zeros(size(ms)))
            sample_moment = second[1, j-1]
            push!(target_vector, sample_moment-constant)

            remaining_vars = variables(block_system)
            matrix_system = jacobian(System(block_system), zeros(size(block_system)[1]))
            last_means = [matrix_system\target_vector]
            for l in 1:k
                meansEst[l, j] = meansEst[l, j](remaining_vars => last_means[1])
            end
        end
    end

    return(true, meansEst)
end
end
