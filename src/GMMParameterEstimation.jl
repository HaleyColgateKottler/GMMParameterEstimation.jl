module GMMParameterEstimation
using HomotopyContinuation
using Distributions
using LinearAlgebra
using Combinatorics
using JLD2

export makeCovarianceMatrix, generateGaussians, getSample, build1DSystem, selectSol, tensorPower, convert_indexing, mixedMomentSystem, estimate_parameters, sampleMoments, densePerfectMoments, diagonalPerfectMoments, cycle_for_weights, dimension_cycle, moments_for_cycle, equalMixCovarianceKnown_moments

"""
    makeCovarianceMatrix(d::Integer, diagonal::Bool)

Generate random `d`x`d` covariance matrix.

If `diagonal`==true, returns a diagonal covariance matrix.
"""
function makeCovarianceMatrix(d::Integer, diagonal::Bool)
    if diagonal
        covars = randn(d)
        covars = covars .* covars
        covars = Diagonal(covars)
    else
        covars = randn(d,d)
        covars = covars * covars'
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
    densePerfectMoments(d, k, w, true_means, true_covariances)
    
Use the given parameters to compute the exact moments necessary for parameter estimation with dense covariance matrices.

Returns moments 0 to 3k for the first dimension, moments 1 through 2k+1 for the other dimensions as a matrix, and a dictionary with indices and moments for the off-diagonal system.
"""
function densePerfectMoments(d, k, w, true_means, true_covariances)
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
    
    true_mixed_system = mixedMomentSystem(d, k, w, true_means, covariances)
    
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

Returns moments 0 to `k` for the first dimension, and moments m_{je_1+e_i} for j in 0 to `k`-1 and i in 2 to d as a matrix where d is the dimension, i varies across rows, and j varies down columns.
"""
function equalMixCovarianceKnown_moments(k, mean, shared_cov)
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
                temp[i] = 1
                temp[j] = ceil((k+1)/2)
                
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
                    temp[i] = (k+1)/2
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
                end
            end
        end
        return (first_moms, diagonal_moms, indexes)
    end
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
    mixedMomentSystem(d, k, mixing, ms, vs)

Build a linear system for finding the off-diagonal covariances entries.

For a `d` dimensional Gaussian `k`-mixture model with mixing coefficients `mixing`, means `ms`, and covariances `vs` where the diagonal entries have been filled in and the off diagonals are variables.
"""
function mixedMomentSystem(d, k, mixing, ms, vs)
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
    estimate_parameters(d::Integer, k::Integer, first::Vector{Float64}, second::Matrix{Float64}, last::Dict{Vector{Int64}, Float64})

Compute an estimate for the parameters of a `d`-dimensional Gaussian `k`-mixture model from the moments.

For the unknown mixing coefficient dense covariance matrix case, `first` should be a list of moments 0 through 3k for the first dimension, `second` should be a matrix of moments 1 through 2k+1 for the remaining dimensions, and `last` should be a dictionary of the indices as lists of integers and the corresponding moments.
"""
function estimate_parameters(d::Integer, k::Integer, first::Vector{Float64}, second::Matrix{Float64}, last::Dict{Vector{Int64}, Float64})
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
        best_sol = "No all positive mixing coefficient solutions"
        num_sols = 0
    else    
        # Check positive variances
        stat_significant1 = filter(r -> all(r[k+1:2*k] .> 0), pos_mixing);
        num_sols = size(stat_significant1)[1]
        if num_sols == 0
            best_sol = "No all positive variance solutions in 1st dimension"
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
    if typeof(best_sol) == String
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

    # Build 1D system for other dimensions
    (system_i, polynomial_i) = build1DSystem(k, 2*k+1, mixing_coefficients)
        
    temp_start = append!(randn(2*k) + im*randn(2*k))
    temp_moments = [p([s;y]=>(temp_start)) for p in system_i[2:end]]
    R1 =  monodromy_solve(system_i[2:end] - m[1:2*k], temp_start, temp_moments, parameters = m[1:2*k], target_solutions_count = target2, show_progress=false)
    
    for i in 2:d        
        # Compute the relevant moments
        all_moments = second[i-1, 1:end]
                
        # Parameter homotopy from random parameters to real parameters
        solution = solve(system_i[2:end] - m[1:2*k], solutions(R1); parameters=m[1:2*k], start_parameters=temp_moments, target_parameters=all_moments[1:2*k], show_progress=false)
                
        # Choose the statistically significant solution closest to the next moment
        best_sol_i = selectSol(k, solution, polynomial_i, all_moments[end])
        if best_sol_i == false
            return ("No all positive variance solutions in "*string(i)*"th dimension", (nothing,nothing,nothing))
        end 
        for j in 1:k
            covariances[j, i, i] = best_sol_i[j]
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
    
    if d>1
        mixed_system1 = mixedMomentSystem(d, k, mixing_coefficients, means, covariances)
        final_system::Vector{Expression} = []
        target_vector = Vector{Float64}()
        @var mixing[1:k]
        
        for (key, polynomial) in mixed_system1
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
    end
    return(true, (mixing_coefficients, means, covariances))
end

"""
    estimate_parameters(d::Integer, k::Integer, w::Array{Float64}, first::Vector{Float64}, second::Matrix{Float64}, last::Dict{Vector{Int64}, Float64})

Compute an estimate for the parameters of a `d`-dimensional Gaussian `k`-mixture model from the moments.

For the known mixing coefficient dense covariance matrix case, `w` should be a vector of the mixing coefficients `first` should be a list of moments 0 through 3k for the first dimension, `second` should be a matrix of moments 1 through 2k+1 for the remaining dimensions, and `last` should be a dictionary of the indices as lists of integers and the corresponding moments.
"""
function estimate_parameters(d::Integer, k::Integer, w::Array{Float64}, first::Vector{Float64}, second::Matrix{Float64}, last::Dict{Vector{Int64}, Float64})
    target2 = Int64(doublefactorial(2*k-1)*factorial(k)) # Number of solutions to look for in step 3
    
    @var m[0:2*k] s[1:k] y[1:k] a[1:k]
    @var vs[1:k, 1:d, 1:d] ms[1:k, 1:d]
    
    means::Array{Union{Variable, Float64}} = reshape([ms...], (k, d))
    covariances::Array{Union{Variable, Float64}} = reshape([vs...], (k, d, d))

    # Build 1D system
    (system_i, polynomial_i) = build1DSystem(k, 2*k+1, w)
        
    temp_start = append!(randn(2*k) + im*randn(2*k))
    temp_moments = [p([s;y]=>(temp_start)) for p in system_i[2:end]]
    R1 =  monodromy_solve(system_i[2:end] - m[1:2*k], temp_start, temp_moments, parameters = m[1:2*k], target_solutions_count = target2, show_progress=false)
    
    for i in 1:d        
        if i == 1
            all_moments = first[2:2*k+2]
        else
            all_moments = second[i-1, 1:end]
        end
        # Solve via the binomial start system
        
        solution = solve(system_i[2:end] - m[1:2*k], solutions(R1); parameters=m[1:2*k], start_parameters=temp_moments, target_parameters=all_moments[1:2*k], show_progress=false)
        
        # Choose the statistically significant solution closest to the next moment
        best_sol_i = selectSol(k, solution, polynomial_i, all_moments[end])
        if best_sol_i == false
            return ("No all positive variance solutions in "*string(i)*"th dimension", (nothing,nothing,nothing))
        end 

        for j in 1:k
            covariances[j, i, i] = best_sol_i[j]
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
    
    if d>1
        mixed_system1 = mixedMomentSystem(d, k, w, means, covariances)
        
        final_system::Vector{Expression} = []
        target_vector = Vector{Float64}()
        @var mixing[1:k]
        
        for (key, polynomial) in mixed_system1
            constant = polynomial(vs => zeros(size(vs)))
            sample_moment = last[key]
            push!(final_system, polynomial)
            push!(target_vector, sample_moment-constant)
        end 
        
        remaining_vars = variables(final_system)
        matrix_system = jacobian(System(final_system), zeros(size(final_system)[1]))
        final_system = []
        last_covars = [matrix_system\target_vector]        
        matrix_system = []
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
    end
    return(true, (w, means, covariances))
end

"""
    estimate_parameters(d::Integer, k::Integer, first::Vector{Float64}, second::Matrix{Float64})

Compute an estimate for the parameters of a `d`-dimensional Gaussian `k`-mixture model from the moments.

For the unknown mixing coefficient diagonal covariance matrix case, `first` should be a list of moments 0 through 3k for the first dimension, and `second` should be a matrix of moments 1 through 2k+1 for the remaining dimensions.
"""
function estimate_parameters(d::Integer, k::Integer, first::Vector{Float64}, second::Matrix{Float64})
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
        best_sol = "No all positive mixing coefficient solutions"
        num_sols = 0
    else    
        # Check positive variances
        stat_significant1 = filter(r -> all(r[k+1:2*k] .> 0), pos_mixing);
        num_sols = size(stat_significant1)[1]
        if num_sols == 0
            best_sol = "No all positive variance solutions in 1st dimension"
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
    if typeof(best_sol) == String
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

    # Build 1D system for other dimensions
    (system_i, polynomial_i) = build1DSystem(k, 2*k+1, mixing_coefficients)
        
    temp_start = append!(randn(2*k) + im*randn(2*k))
    temp_moments = [p([s;y]=>(temp_start)) for p in system_i[2:end]]
    R1 =  monodromy_solve(system_i[2:end] - m[1:2*k], temp_start, temp_moments, parameters = m[1:2*k], target_solutions_count = target2, show_progress=false)
    
    for i in 2:d        
        # Compute the relevant moments
        all_moments = second[i-1, 1:end]
                
        # Parameter homotopy from random parameters to real parameters
        solution = solve(system_i[2:end] - m[1:2*k], solutions(R1); parameters=m[1:2*k], start_parameters=temp_moments, target_parameters=all_moments[1:2*k], show_progress=false)
                
        # Choose the statistically significant solution closest to the next moment
        best_sol_i = selectSol(k, solution, polynomial_i, all_moments[end])
        if best_sol_i == false
            return ("No all positive variance solutions in "*string(i)*"th dimension", (nothing,nothing,nothing))
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
    return(true, (mixing_coefficients, means, covariances))
end

"""
    estimate_parameters(d::Integer, k::Integer, w::Array{Float64}, first::Vector{Float64}, second::Matrix{Float64})

Compute an estimate for the parameters of a `d`-dimensional Gaussian `k`-mixture model from the moments.

For the known mixing coefficient diagonal covariance matrix case, `w` should be a vector of the mixing coefficients `first` should be a list of moments 0 through 3k for the first dimension, and `second` should be a matrix of moments 1 through 2k+1 for the remaining dimensions..
"""
function estimate_parameters(d::Integer, k::Integer, w::Array{Float64}, first::Vector{Float64}, second::Matrix{Float64})
    target2 = Int64(doublefactorial(2*k-1)*factorial(k)) # Number of solutions to look for in step 3
    
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
        if i == 1
            all_moments = first[2:2*k+2]
        else
            all_moments = second[i-1, 1:end]
        end
        # Solve via the binomial start system
        
        solution = solve(system_i[2:end] - m[1:2*k], solutions(R1); parameters=m[1:2*k], start_parameters=temp_moments, target_parameters=all_moments[1:2*k], show_progress=false)
        
        # Choose the statistically significant solution closest to the next moment
        best_sol_i = selectSol(k, solution, polynomial_i, all_moments[end])
        if best_sol_i == false
            return ("No all positive variance solutions in "*string(i)*"th dimension", (nothing,nothing,nothing))
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

    return(true, (w, means, covariances))
end

"""
    estimate_parameters(d::Integer, first::Vector{Float64}, second::Matrix{Float64})

Compute an estimate for the parameters of a `d`-dimensional Gaussian model from the moments.

For the unknown mixing coefficient diagonal covariance matrix case, `first` should be a list of moments 0 through 3 for the first dimension, and `second` should be a matrix of moments 1 through 3 for the remaining dimensions.
"""
function estimate_parameters(d::Integer, first::Vector{Float64}, second::Matrix{Float64})
    means = Array{Float64}(undef, (1,d))
    covariances = [Diagonal{Float64}(undef, d)]
    
    means[1,1] = first[2]
    covariances[1][1,1] = first[3] - first[2]^2
    
    for i in 1:d-1
        means[1, i+1] = second[i, 1]
        covariances[1][i+1, i+1] = second[i, 2] - second[i, 1]^2
    end
    return (means, covariances)
end

"""
    estimate_parameters(d::Integer, first::Vector{Float64}, second::Matrix{Float64}, last::Dict{Vector{Int64}, Float64})

Compute an estimate for the parameters of a `d`-dimensional Gaussian `k`-mixture model from the moments.

For the unknown mixing coefficient dense covariance matrix case, `first` should be a list of moments 0 through 3 for the first dimension, `second` should be a matrix of moments 1 through 3 for the remaining dimensions, and `last` should be a dictionary of the indices as lists of integers and the corresponding moments.
"""
function estimate_parameters(d::Integer, first::Vector{Float64}, second::Matrix{Float64}, third::Dict{Vector{Int64}, Float64})
    @var vs[1:k, 1:d, 1:d]
    means = Array{Float64}(undef, (1,d))
    covariances::Array{Union{Variable, Float64}} = reshape([vs...], (1, d, d))
    
    means[1, 1] = first[2]
    covariances[1, 1, 1] = first[3] - first[2]^2
    
    for i in 1:d-1
        means[1, i+1] = second[i, 1]
        covariances[1, i+1, i+1] = second[i, 2] - second[i, 1]^2
    end
    for (key, moment) in third
        indexing = convert_indexing(key, d)
        covar = moment/3 - means[1, indexing[1]] * means[1,indexing[2]]
        covariances[1, indexing[1], indexing[2]] = covar
        covariances[1, indexing[2], indexing[1]] = covar        
    end
    return (means, covariances)
end

"""
    estimate_parameters(k::Integer, shared_cov::Matrix{Float64}, first_moms::Vector{Float64}, second_moms::Matrix{Float64})

Compute an estimate for the means of a Gaussian `k`-mixture model with equal mixing coefficients and known shared covariances from the moments.

The shared covariance matrix `shared_cov` will determine the dimension. Then `first_moms` should be a list of moments 0 through k for the first dimension, `second_moms` should be a matrix of moments m_{je_1+e_i} for j in 0 to `k`-1 and i in 2 to d as a matrix where d is the dimension, i varies across rows, and j varies down columns.
"""
function estimate_parameters(k::Integer, shared_cov::Matrix{Float64}, first_moms::Vector{Float64}, second_moms::Matrix{Float64})
    d = size(shared_cov)[1]
    
    @var ms[1:k, 1:d]    
    meansEst::Array{Union{Variable, Float64}} = reshape([ms...], (k, d))
    @var s[1:k], y[1:k] #y means, s sigma
    first_system = build1DSystem(k, k+1, ones(k)/k)[1][2:end]
    first_system = evaluate(first_system, s => repeat([shared_cov[1,1]],k))
    
    sol1 = solve(first_system-first_moms, show_progress = false)
    meansEst[1:k,1] = real_solutions(sol1)[1]
    
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
            sample_moment = second_moms[j, i]
            push!(target_vector, sample_moment - constant)
        end
        
        vars = variables(system)
        matrix_system = jacobian(System(system), zeros(size(system)[1]))
        solution = [matrix_system\target_vector]
        meansEst[1:k, i+1] = solution[1]
    end
    
    return meansEst
end

"""
    cycle_for_weights(k::Integer, sample; start_dim = 1)

Starting with `start_dim` cycle over the dimensions of `sample` until candidate mixing coefficients are found.
"""
function cycle_for_weights(k::Integer, sample; start_dim = 1)
    (d, sample_size) = size(sample)
    mixing_coeffs = false
    while (mixing_coeffs == false) & (start_dim <= d)
        first_moms = Vector{Float64}(undef, 3*k+1)
        first_moms[1] = 1.0
        for j in 1:3*k
            first_moms[j+1] = mean(sample[start_dim, i]^j for i in 1:sample_size)
        end
        
        first_pass, (weights, means, covar) = estimate_parameters(1, k, first_moms, zeros((2,2)))
        
        if first_pass == true
            mixing_coeffs = weights
        end
        start_dim += 1
    end
    return(mixing_coeffs, start_dim)
end
    
    
"""
    moments_for_cycle(d, k, w, means, covars, diagonal)

Calculate 0 through 3`k`+1 denoised moments for every dimension.

Used as input for cycling over the dimensions to find candidate mixing coefficients.
"""
function moments_for_cycle(d, k, w, means, covars, diagonal)
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
    dimension_cycle(k::Integer, sample::Matrix{Float64}, diagonal)

Cycle over the dimensions of the `sample` to find candidate mixing coefficients, then solve for parameters based on those.

This will take longer than estimate\\_parameters since it does multiple tries.  Will try each dimension to attempt to find mixing coefficients, and if found will try to solve for parameters.  Returns `pass` = false if no dimension results in mixing coefficients that allow for a solution.
"""
function dimension_cycle(k::Integer, sample::Matrix{Float64}, diagonal::Bool)
    (d, sample_size) = size(sample)
    first_moms, diagonal_moms, indexes = sampleMoments(sample, k)
    start_dim = 1
        
    stop = false
    while (stop != true) & (start_dim <= d)
        mixing_coeffs, start_dim = cycle_for_weights(k, sample; start_dim = start_dim)
            
        if mixing_coeffs != false
            if diagonal
                pass, (weights, means, covariances) = estimate_parameters(d, k, mixing_coeffs, first_moms, diagonal_moms)
            else
                pass, (weights, means, covariances) = estimate_parameters(d, k, mixing_coeffs, first_moms, diagonal_moms, indexes)
            end
        else
            pass, (weights, means, covariances) = false, (nothing, nothing, nothing)
        end
        if pass == true
            stop = true
        end
    end
    return pass, (weights, means, covariances)
end

"""
    dimension_cycle(d::Integer, k::Integer, cycle_moments::Array{Float64}, diagonal::Bool)

Cycle over the dimensions of `cycle_moments` to find candidate mixing coefficients, then solve for parameters based on those.

This will take longer than estimate\\_parameters since it does multiple tries.  Will try each dimension to attempt to find mixing coefficients, and if found will try to solve for parameters.  Returns `pass` = false if no dimension results in mixing coefficients that allow for a solution.  `cycle_moments` should be an array of the 0 through 3`k` moments for each dimension. If no `indexes` is given, assumes diagonal covariance matrices.
"""
function dimension_cycle(d::Integer, k::Integer, cycle_moments::Array{Float64})    
    first_dim = 1
    stop = false
    first_pass = false
    while (first_dim <= d) & (stop == false)
        first_moms = cycle_moments[first_dim, 1:end]
        
        first_pass, (weights, means, covar) = estimate_parameters(1, k, first_moms, zeros((2,2)))
        if first_pass == true
            pass, (weights, means, covariances) = estimate_parameters(d, k, weights, cycle_moments[1,1:end], cycle_moments[2:end, 2:2*k+1])
        end
        
        if pass == true
            stop = true
        end
        first_dim += 1
        println(pass)
    end
    if first_pass != true
        pass, (weights, means, covariances) = false, (nothing, nothing, nothing)
    end
    return pass, (weights, means, covariances)
end

function dimension_cycle(d::Integer, k::Integer, cycle_moments::Array{Float64}, indexes::Dict{Vector{Int64}, Float64})    
    first_dim = 1
    stop = false
    first_pass = false
    while (first_dim <= d) & (stop == false)
        first_moms = cycle_moments[first_dim, 1:end]
        
        first_pass, (weights, means, covar) = estimate_parameters(1, k, first_moms, zeros((2,2)))
        if first_pass == true
            pass, (weights, means, covariances) = estimate_parameters(d, k, weights, cycle_moments[1,1:end], cycle_moments[2:end, 2:2*k+1], indexes)
        end
        
        if pass == true
            stop = true
        end
        first_dim += 1
    end
    if first_pass != true
        pass, (weights, means, covariances) = false, (nothing, nothing, nothing)
    end
    return pass, (weights, means, covariances)
end
end
