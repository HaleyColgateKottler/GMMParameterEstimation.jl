module GMMParameterEstimation
using HomotopyContinuation
using Distributions
using LinearAlgebra
using Combinatorics

export makeCovarianceMatrix, generateGaussians, get1Dmoments, getSample, build1DSystem, selectSol, tensorPower, convert_indexing, mixedMomentSystem, unknown_coefficients, known_coefficients, estimate_parameters

"""
    makeCovarianceMatrix(d::Integer, diagonal::Bool)

Generate random `d`x`d` covariance matrix.

If `diagonal`==true, returns a diagonal covariance matrix.
"""
function makeCovarianceMatrix(d::Integer, diagonal::Bool)
    covars = randn(d,d)
    covars = covars * covars'
    diagonal == true && (covars = Diagonal(covars))
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
    variances = Array{Float64, 3}(undef, (k,d,d))
    for i in 1:k
        variances[i, 1:end, 1:end] = makeCovarianceMatrix(d, diagonal)
    end
    return (w, means, variances)
end

"""
    get1Dmoments(sample::Matrix{Float64}, dimension::Integer, m::Integer)

Compute the 1D sample moments 0 through `m`, for the given `dimension` of `sample`.
"""
function get1Dmoments(sample::Matrix{Float64}, dimension::Integer, m::Integer)
    d1moments = [1.0]
    for j in 1:m
        append!(d1moments, mean(sample[dimension,i]^j for i in 1:size(sample)[2]))
    end
    return d1moments
end

"""
    getSample(numb::Integer, w::Vector{Float64}, means::Matrix{Float64}, covariances::Array{Float64, 3})

Generate a Gaussian mixture model sample with `numb` entries, mixing coefficients `w`, means `means`, and covariances `covariances`.
"""
function getSample(numb::Integer, w::Vector{Float64}, means::Matrix{Float64}, covariances::Array{Float64, 3})
    k = size(w)[1]
    gaussians = [(means[i, 1:end], covariances[i, 1:end, 1:end]) for i in 1:k]
    m = MixtureModel(MvNormal, gaussians, w)
    r = rand(m, numb)
    return r
end

"""
    build1DSystem(k::Integer, m::Integer)

Build the polynomial system for a mixture of 1D Gaussians where 'm' is the highest desired moment.

If `a` is given, use `a` as the mixing coefficients, otherwise leave them as unknowns.
"""
function build1DSystem(k::Integer, m::Integer)
    @var a[1:k]
    return build1DSystem(k, m, a)
end

"""
    build1DSystem(k::Integer, m::Integer, a::Union{Vector{Float64}, Vector{Variable}})

Build the polynomial system for a mixture of 1D Gaussians where 'm' is the highest desired moment.

If `a` is given, use `a` as the mixing coefficients, otherwise leave them as unknowns.
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
            temp[i] = 1
            temp[j] = ceil((k+1)/2)
            push!(indexes[string(n)], temp)
            if k % 2 != 0
                temp = Int64.(zeros(d))
                temp[i] = (k+1)/2
                temp[j] = 1
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


# number of solutions for use in homotopy continuations
const target_numbers = Dict{String, Tuple{Int64, Int64}}("4"=>(10350,2520), "3"=>(225, 90), "2"=>(9,6), "1"=>(2, 1))

"""
    unknown_coefficients(d::Integer, k::Integer, w::Array{Float64}, true_means::Array{Float64,2}, true_covariances::Array{Float64,3}, diagonal::Bool)

Compute parameters and build and solve times for the perfect moment, unknown mixing coefficients case.

Assuming a `d` dimensional Gaussian `k`-mixture model with mixing coefficients `w`, means `true_means`, and covariances `true_covariances`.
"""
function unknown_coefficients(d::Integer, k::Integer, w::Array{Float64}, true_means::Array{Float64,2}, true_covariances::Array{Float64,3}, diagonal::Bool)
    
    build_time = 0
    solve_time = 0
    
    build_time1 = @elapsed begin
        target1, target2 = target_numbers[string(k)] # Number of solutions to look for in steps 1 and 3 respectively
        
        # Build the system of equations for step 1
        # m is the parameter for the moments, s gives the variances, y gives the means, and a gives the mixing coefficients
        @var m[0:3*k-1] s[1:k] y[1:k] a[1:k]

        (system, polynomial) = build1DSystem(k, 3*k)
        # Compute the moments for step 1
        true_params = append!(copy(w), [true_covariances[i,1,1] for i in 1:k], [true_means[i,1] for i in 1:k])
        all_moments = [p([a; s; y] => true_params) for p in system]
    
        # Define the relabeling group action
        relabeling = (GroupActions(v -> map(p -> (v[1:k][p]...,v[k+1:2*k][p]...,v[2*k+1:3k][p]...),SymmetricGroup(k))))
    
        # Generate random complex parameters for initial solution for monodromy method
        temp_start = append!(randn(3*k) + im*randn(3*k))
        temp_moments = [p([a;s;y]=>(temp_start)) for p in system]
    end
    build_time += build_time1
    
    solve_time1 = @elapsed begin
        # Monodromy solve system with random complex parameters
        R1 =  monodromy_solve(system - m[1:3*k], temp_start, temp_moments, parameters = m[1:3*k], target_solutions_count = target1, group_action = relabeling, show_progress=false)
        relabeling = nothing
        vars = append!(a,s,y)
        # Parameter homotopy from random parameters to real parameters
        solution1 = solve(system - m[1:3*k], solutions(R1); parameters=m[1:3*k], start_parameters=temp_moments, target_parameters=all_moments[1:3*k], show_progress=false)
        R1 = []
        system = []
        temp_start = []
        temp_moments = []
        
        # Check for statistically significant solutions
        # Return the one closest to the given moment, and the number of statistically significant solutions
        # Filter out the statistically meaningful solutions (up to symmetry)
        # Check positive mixing coefficients
        pos_mixing  = filter(r -> all(r[1:k] .> 0), real_solutions(solution1)); 
        solution1 = []
        num_pos_mix = size(pos_mixing)[1]
        if num_pos_mix == 0
            best_sol = false
            num_sols = 0
        else    
            # Check positive variances
            stat_significant1 = filter(r -> all(r[k+1:2*k] .> 0), pos_mixing);
            num_sols = size(stat_significant1)[1]
            if num_sols == 0
                best_sol = false
            else
                best_sols1 = [];
                # Create list of differences between moment and polynomial(statistically significant solutions)
                for i in 1:size(stat_significant1)[1]
                    t = polynomial(vars => stat_significant1[i])
                    append!(best_sols1, norm(polynomial(vars => true_params) - t))
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
        if best_sol == false
            return (false, (nothing,nothing,nothing), (nothing, nothing))
        end
        
        # Separate out the mixing coefficients, variances, and means
        mixing_coefficients = best_sol[1:k]
        means = zeros(k,d)
        means[1:k, 1] = best_sol[2*k+1:end]
        if diagonal == true
            covariances = zeros(k,d)
            covariances[1:k, 1] = best_sol[k+1:2*k]
        else
            @var vs[1:k, 1:d, 1:d]
            covariances::Array{Union{Variable, Float64}} = reshape([vs...], (k, d, d))
            covariances[1:k, 1, 1] = best_sol[k+1:2*k]
        end
        best_sol = []
    end
    solve_time += solve_time1
    
    build_time2 = @elapsed begin
        # Build 1D system for other dimensions
        (system_i, polynomial_i) = build1DSystem(k, 2*k+1, mixing_coefficients)
        (system_real, polynomial_real) = build1DSystem(k, 2*k+1, w)
        temp_start = append!(randn(2*k) + im*randn(2*k))
        temp_moments = [p([s;y]=>(temp_start)) for p in system_i[2:end]]
        R1 =  monodromy_solve(system_i[2:end] - m[1:2*k], temp_start, temp_moments, parameters = m[1:2*k], target_solutions_count = target2, show_progress=false)
    end
    build_time += build_time2
    
    for i in 2:d
        solve_time2 = @elapsed begin
        # Compute the relevant moments
        true_params = append!([true_covariances[j,i,i] for j in 1:k], [true_means[j,i] for j in 1:k])
        all_moments = [p([s; y] => true_params) for p in system_real[2:end]]
        
        # Parameter homotopy from random parameters to real parameters
        solution = solve(system_i[2:end] - m[1:2*k], solutions(R1); parameters=m[1:2*k], start_parameters=temp_moments, target_parameters=all_moments[1:2*k], show_progress=false)
        
        # Choose the statistically significant solution closest to the next moment
        best_sol_i = selectSol(k, solution, polynomial_i, polynomial_real([s; y] => true_params))
        if best_sol_i == false
            return (false, (nothing,nothing,nothing), (nothing, nothing))
        end 

        means[1:k, i] = best_sol_i[k+1:end]
        if diagonal == true
            covariances[1:k, i] = best_sol_i[1:k]
        else
            covariances[1:k, i, i] = best_sol_i[1:k]
        end
        end
        solve_time += solve_time2
    end
        
    solution = []
    binomials = []
    true_params = []
    all_moments = []
    system_i = []
    polynomial_i = 0
    best_sol_i = []
    gaussians = []
    
    if (diagonal == false) && (d>1)
        build_time3 = @elapsed begin
        mixed_system1 = mixedMomentSystem(d, k, mixing_coefficients, means, covariances)
        true_mixed_system = mixedMomentSystem(d, k, w, true_means, true_covariances)
        final_system::Vector{Expression} = []
        target_vector = Vector{Float64}()
        @var mixing[1:k]
        
        for (key, polynomial) in mixed_system1
            constant = polynomial(vs => zeros(size(vs)))
            sample_moment = true_mixed_system[key](vs=>true_covariances)
            push!(final_system, polynomial)
            push!(target_vector, sample_moment-constant)
        end 
        remaining_vars = variables(final_system)
        matrix_system = jacobian(System(final_system), zeros(size(final_system)[1]))
        end
        build_time += build_time3
        
        solve_time3 = @elapsed begin
        last_covars = [matrix_system\target_vector]
        final_system = []
        matrix_system = []
        target_vector = []
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
        solve_time += solve_time3
    end
    return (true, (mixing_coefficients, means, covariances), (build_time, solve_time))
end
   
"""
    known_coefficients(d::Integer, k::Integer, w::Array{Float64}, true_means::Array{Float64,2}, true_covariances::Array{Float64,3}, diagonal::Bool)

Compute parameters for the perfect moment, known mixing coefficients case.

Assuming a `d` dimensional Gaussian `k`-mixture model with mixing coefficients `w`, means `true_means`, and covariances `true_covariances`.
"""
function known_coefficients(d::Integer, k::Integer, w::Array{Float64}, true_means::Array{Float64,2}, true_covariances::Array{Float64,3}, diagonal::Bool)
    target1, target2 = target_numbers[string(k)] # Number of solutions to look for in steps 1 and 3 respectively
        
    # Build the system of equations for step 1
    # m is the parameter for the moments, s gives the variances, y gives the means, and a gives the mixing coefficients
    @var m[0:3*k-1] s[1:k] y[1:k] a[1:k]
    
    @var vs[1:k, 1:d, 1:d] ms[1:k, 1:d]
    
    means::Array{Union{Variable, Float64}} = reshape([ms...], (k, d))
    covariances::Array{Union{Variable, Float64}} = reshape([vs...], (k, d, d))
    
    # Build 1D system for other dimensions
    (system_i, polynomial_i) = build1DSystem(k, 2*k+1, w)
    temp_start = append!(randn(2*k) + im*randn(2*k))
    temp_moments = [p([s;y]=>(temp_start)) for p in system_i[2:end]]
    R1 =  monodromy_solve(system_i[2:end] - m[1:2*k], temp_start, temp_moments, parameters = m[1:2*k], target_solutions_count = target2, show_progress=false)
        
    for i in 1:d
        i%100 == 0 && println(i)
        
        # Compute the relevant moments
        true_params = append!([true_covariances[j,i,i] for j in 1:k], [true_means[j,i] for j in 1:k])
        all_moments = [p([s; y] => true_params) for p in system_i]
        
        # Parameter homotopy from random parameters to real parameters
        solution = solve(system_i[2:end] - m[1:2*k], solutions(R1); parameters=m[1:2*k], start_parameters=temp_moments, target_parameters=all_moments[1:2*k], show_progress=false)
        
        # Choose the statistically significant solution closest to the next moment
        best_sol_i = selectSol(k, solution, polynomial_i, polynomial_i([s; y] => true_params))
        if best_sol_i == false
            return (false, (nothing,nothing,nothing))
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
    
    if (diagonal == false) && (d>1)
        mixed_system1 = mixedMomentSystem(d, k, w, means, covariances)
        true_mixed_system = mixedMomentSystem(d, k, w, true_means, true_covariances)
        final_system::Vector{Expression} = []
        target_vector = Vector{Float64}()
        @var mixing[1:k]
        
        for (key, polynomial) in mixed_system1
            constant = polynomial(vs => zeros(size(vs)))
            sample_moment = true_mixed_system[key](vs=>true_covariances)
            push!(final_system, polynomial)
            push!(target_vector, sample_moment-constant)
        end 
        remaining_vars = variables(final_system)
        matrix_system = jacobian(System(final_system), zeros(size(final_system)[1]))
        last_covars = [matrix_system\target_vector]
        final_system = []
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
    return (true, (w, means, covariances))
end

"""
    estimate_parameters(d::Integer, k::Integer, sample::Array{Float64}, diagonal::Bool)

Compute an estimate for the parameters of a `d`-dimensional Gaussian `k`-mixture model from a sample.

If `diagonal` is true, the covariance matrices are assumed to be diagonal. If `w` is provided it is taken as the mixing coefficients, otherwise those are computed as well. The sample should be a d x sample-size array.
"""
function estimate_parameters(d::Integer, k::Integer, sample::Array{Float64}, diagonal::Bool)
    target1, target2 = target_numbers[string(k)] # Number of solutions to look for in steps 1 and 3 respectively
        
    # Build the system of equations for step 1
    # m is the parameter for the moments, s gives the variances, y gives the means, and a gives the mixing coefficients
    @var m[0:3*k-1] s[1:k] y[1:k] a[1:k]
    (system, polynomial) = build1DSystem(k, 3*k)
    
    # Compute the moments for step 1
    all_moments = get1Dmoments(sample, 1, 3*k)
 
    # Define the relabeling group action
    relabeling = (GroupActions(v -> map(p -> (v[1:k][p]...,v[k+1:2*k][p]...,v[2*k+1:3k][p]...),SymmetricGroup(k))))
 
    # Solve the system using the first 3k moments for target1 solutions, 
    # then pick the statistically significant solution that best matches moment 3k+1    
    
    # Generate random complex parameters for initial solution for monodromy method
    temp_start = append!(randn(3*k) + im*randn(3*k))
    temp_moments = [p([a;s;y]=>(temp_start)) for p in system]

    # Monodromy solve system with random complex parameters
    R1 =  monodromy_solve(system - m[1:3*k], temp_start, temp_moments, parameters = m[1:3*k], target_solutions_count = target1, group_action = relabeling, show_progress=false)
        
    relabeling = nothing
     vars = append!(a,s,y)
    # Parameter homotopy from random parameters to real parameters
    solution1 = solve(system - m[1:3*k], solutions(R1); parameters=m[1:3*k], start_parameters=temp_moments, target_parameters=all_moments[1:3*k], show_progress=false)
    
    R1 = []
    system = []
    temp_start = []
    temp_moments = []
    
    # Check for statistically significant solutions
    # Return the one closest to the given moment, and the number of statistically significant solutions
    # Filter out the statistically meaningful solutions (up to symmetry)
    # Check positive mixing coefficients
    pos_mixing  = filter(r -> all(r[1:k] .> 0), real_solutions(solution1)); 
    solution1 = []
    num_pos_mix = size(pos_mixing)[1]
    if num_pos_mix == 0
        best_sol = false
        num_sols = 0
    else    
        # Check positive variances
        stat_significant1 = filter(r -> all(r[k+1:2*k] .> 0), pos_mixing);
        num_sols = size(stat_significant1)[1]
        if num_sols == 0
            best_sol = false
        else
            best_sols1 = [];
            # Create list of differences between moment and polynomial(statistically significant solutions)
            for i in 1:size(stat_significant1)[1]
                t = polynomial(vars => stat_significant1[i])
                append!(best_sols1, norm(all_moments[end] - t))
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
    if best_sol == false
        return (false, (nothing,nothing,nothing))
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
        i%100 == 0 && println(i)
        
        # Compute the relevant moments
        all_moments = get1Dmoments(sample, i, 2*k+1)
        
        # Parameter homotopy from random parameters to real parameters
        solution = solve(system_i[2:end] - m[1:2*k], solutions(R1); parameters=m[1:2*k], start_parameters=temp_moments, target_parameters=all_moments[2:2*k+1], show_progress=false)
                
        # Choose the statistically significant solution closest to the next moment
        best_sol_i = selectSol(k, solution, polynomial_i, all_moments[end])
        if best_sol_i == false
            return (false, (nothing,nothing,nothing))
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
    
    if (diagonal == false) && (d>1)
        mixed_system1 = mixedMomentSystem(d, k, mixing_coefficients, means, covariances)
        
        final_system::Vector{Expression} = []
        target_vector = Vector{Float64}()
        @var mixing[1:k]
        
        sample_size = size(sample)[2]
        for (key, polynomial) in mixed_system1
            constant = polynomial(vs => zeros(size(vs)))
            sample_moment = 0
            for j in 1:sample_size
                temp_moment = 1
                for i in 1:d
                    temp_moment *= sample[i, j]^(key[i])
                end
                sample_moment += temp_moment
            end
            sample_moment = sample_moment/sample_size
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
    estimate_parameters(d::Integer, k::Integer, sample::Array{Float64}, diagonal::Bool, w::Array{Float64})

Compute an estimate for the parameters of a `d`-dimensional Gaussian `k`-mixture model from a sample.

If `diagonal` is true, the covariance matrices are assumed to be diagonal. If `w` is provided it is taken as the mixing coefficients, otherwise those are computed as well. The sample should be a d x sample-size array.
"""
function estimate_parameters(d::Integer, k::Integer, sample::Array{Float64}, diagonal::Bool, w::Array{Float64})
    target1, target2 = target_numbers[string(k)] # Number of solutions to look for in steps 1 and 3 respectively
    
    @var m[0:3*k-1] s[1:k] y[1:k] a[1:k]
    @var vs[1:k, 1:d, 1:d] ms[1:k, 1:d]
    
    means::Array{Union{Variable, Float64}} = reshape([ms...], (k, d))
    covariances::Array{Union{Variable, Float64}} = reshape([vs...], (k, d, d))

    # Build 1D system
    (system_i, polynomial_i) = build1DSystem(k, 2*k+1, w)
    temp_start = append!(randn(2*k) + im*randn(2*k))
    temp_moments = [p([y;s]=>(temp_start)) for p in system_i[2:end]]
    R1 =  monodromy_solve(system_i[2:end] - m[1:2*k], temp_start, temp_moments, parameters = m[1:2*k], target_solutions_count = target2, show_progress=false)
    
    for i in 1:d        
        # Compute the relevant moments
        all_moments = get1Dmoments(sample, i, 2*k+1)
        # Solve via the binomial start system
        
        solution = solve(system_i[2:end] - m[1:2*k], solutions(R1); parameters=m[1:2*k], start_parameters=temp_moments, target_parameters=all_moments[2:2*k+1], show_progress=false)
        
        # Choose the statistically significant solution closest to the next moment
        best_sol_i = selectSol(k, solution, polynomial_i, all_moments[end])
        if best_sol_i == false
            return (false, (nothing,nothing,nothing))
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
    
    if (diagonal == false) && (d>1)
        mixed_system1 = mixedMomentSystem(d, k, w, means, covariances)
        
        final_system::Vector{Expression} = []
        target_vector = Vector{Float64}()
        @var mixing[1:k]
        
        sample_size = size(sample)[2]
        for (key, polynomial) in mixed_system1
            constant = polynomial(vs => zeros(size(vs)))
            sample_moment = 0
            for j in 1:sample_size
                temp_moment = 1
                for i in 1:d
                    temp_moment *= sample[i, j]^(key[i])
                end
                sample_moment += temp_moment
            end
            sample_moment = sample_moment/sample_size
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
end
