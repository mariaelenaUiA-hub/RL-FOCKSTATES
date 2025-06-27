####################################################################################################################
###############
#= Constants =#
###############
# Physical constants in SI units

const h = 6.62607015e-34          # Planck constant (J·s)
const hbar = 1.054571817e−34;     # Reduced Planck constant (J·s)
const kb = 1.380649e-23           # Boltzmann constant (J/K)

####################################################################################################################




####################################################################################################################
#####################################
#= QM SYSTEMS WITHIN QUANTUMOPTICS =#
#####################################

#= Struct to define the different qm systems -> agebra operators, basis =#
struct harmonic_oscillator{ho_basis}
    basis::ho_basis
    a::Operator{ho_basis, ho_basis, <:AbstractMatrix} 
    ad::Operator{ho_basis, ho_basis, <:AbstractMatrix} 
    n::Operator{ho_basis, ho_basis, <:AbstractMatrix} 
    Id::Operator{ho_basis, ho_basis, <:AbstractMatrix}
end

struct qubit
    basis::SpinBasis{1//2, Int64}
    σm::Operator{SpinBasis{1//2, Int64}, SpinBasis{1//2, Int64}, <:AbstractMatrix}
    σp::Operator{SpinBasis{1//2, Int64}, SpinBasis{1//2, Int64}, <:AbstractMatrix}
    σx::Operator{SpinBasis{1//2, Int64}, SpinBasis{1//2, Int64},<:AbstractMatrix}
    σy::Operator{SpinBasis{1//2, Int64}, SpinBasis{1//2, Int64}, <:AbstractMatrix}
    σz::Operator{SpinBasis{1//2, Int64}, SpinBasis{1//2, Int64}, <:AbstractMatrix}
    Id::Operator{SpinBasis{1//2, Int64}, SpinBasis{1//2, Int64}, <:AbstractMatrix}
end

struct qubit_ho{ho_basis}
    zI::Operator{CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, <:AbstractMatrix}
    xI::Operator{CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, <:AbstractMatrix}
    mI::Operator{CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, <:AbstractMatrix}
    pI::Operator{CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, <:AbstractMatrix}
    II::Operator{CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, <:AbstractMatrix}
    pa::Operator{CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, <:AbstractMatrix}
    mad::Operator{CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, <:AbstractMatrix}

    Ia::Operator{CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, <:AbstractMatrix}
    Iad::Operator{CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, <:AbstractMatrix}

    n_mech::Operator{CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, <:AbstractMatrix}
    n_qubit::Operator{CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, <:AbstractMatrix}
end



#Built-in function to fill the structs =#
function Harmonic_oscillator(N_particle::Int64, type_basis::Symbol)
    basis_Dict = Dict(
        :FockBasis => FockBasis
    )

    basis_type = basis_Dict[type_basis]
    basis = basis_type(N_particle)

    return harmonic_oscillator(basis,
    destroy(basis), 
    create(basis), 
    number(basis),
    one(basis)
    )

end

function Qubit(spin)
    basis = SpinBasis(spin)

    return qubit(basis,
    sigmam(basis),
    sigmap(basis),
    sigmax(basis),
    sigmay(basis),
    sigmaz(basis),
    one(basis)
    )
    
end


function Qubit_HO(N_mech, type_basis_mech::Symbol, type_basis_qubit)
    qub = Qubit(type_basis_qubit)
    mech_res = Harmonic_oscillator(N_mech, type_basis_mech)
    
    return qub, mech_res,
            qubit_ho(
            tensor(qub.σz, mech_res.Id),
            tensor(qub.σx, mech_res.Id),
            tensor(qub.σm, mech_res.Id),
            tensor(qub.σp, mech_res.Id),
            tensor(qub.Id, mech_res.Id),
            tensor(qub.σm, mech_res.a),
            tensor(qub.σp, mech_res.ad),
            tensor(qub.Id, mech_res.a),
            tensor(qub.Id, mech_res.ad),
            tensor(qub.Id, mech_res.n),
            0.5 * (tensor(qub.Id, mech_res.Id) + tensor(qub.σz, mech_res.Id))
           )
end


######################################################################################################

######################################################################################################
######################
#= SPECIAL MATRICES =#
######################

#= Gell-Mann matrices =#
#= Pauli matrices for N=2 =#
function gellmann_operators(N::Int)
    matrices = Operator[]
    
    # Symmetric Matrices
    for j in 1:N
        for k in j+1:N
            M = zeros(ComplexF64, N, N)
            M[j, k] = 1.0
            M[k, j] = 1.0
            push!(matrices, Operator(basis, M))
        end
    end

    # Antisymmetric Matrices
    for j in 1:N
        for k in j+1:N
            M = zeros(ComplexF64, N, N)
            M[j, k] = -im
            M[k, j] = im
            push!(matrices, Operator(basis, M))
        end
    end

    # Diagonal, Traceless Matrices
    for d in 1:(N - 1)
        M = zeros(ComplexF64, N, N)
        coeff = sqrt(2 / (d * (d + 1)))
        for j in 1:d
            M[j, j] = 1.0
        end
        M[d + 1, d + 1] = -d
        M *= coeff
        push!(matrices, Operator(basis, M))
    end

    return matrices
end

function pauli_operators()
    return gellmann_operator(2)
end

function rand_hermitian_orthonormal_basis(d, bases)
    n = d^2  # Dimension of the space of Hermitian matrices
    mats = Matrix{ComplexF64}[]  # Array to hold the random Hermitian matrices

    # Step 1: Generate n random Hermitian matrices
    for i in 1:n
        A = randn(ComplexF64, d, d)           # Random complex matrix
        H = (A + A') / 2                      # Make it Hermitian (A' is conjugate transpose in Julia)
        push!(mats, H)
    end

    # Step 2: Flatten matrices into real vectors by separating real and imaginary parts
    # This converts each Hermitian matrix into a 2*d^2-dimensional real vector
    vecs = zeros(Float64, 2*d*d, n)
    for (i, H) in enumerate(mats)
        vecs[1:d*d, i] = vec(H) |> real      # Real parts flattened
        vecs[d*d+1:end, i] = vec(H) |> imag  # Imaginary parts flattened
    end

    # Step 3: Orthonormalize the vectors using QR decomposition
    Q, R = qr(vecs)

    # Step 4: Map the orthonormal vectors back to Hermitian matrices
    basis = Matrix{ComplexF64}[]
    half = d*d
    for i in 1:n
        real_part = reshape(Q[1:half, i], d, d)
        imag_part = reshape(Q[half+1:end, i], d, d)
        H = real_part + im*imag_part
        # Force Hermiticity to correct numerical errors
        H = (H + H') / 2
        push!(basis, H)
    end

    return [Operator(bases, H) for H in basis]
end

#########################################################################################################


#########################################################################################################
########################################
#= QM PROBLEMS DYNAMICS and EXPECTATION VALUES =#
########################################
#= Expectation value of operators on states =#
function expectation_value(operator, states)
    return real(expect(operator, states))
end


#= Plot the expectation values =#
function plot_expectation(tspan, operator, expectation_data)
    ev_scatter=scatter(x=tspan, y=expectation_data)

    plot([ev_scatter],
    Layout(
        title = "Expectation value of operator "*operator,
        xaxis_title = "t (μs)",
        yaxis_title = "<" * operator * ">"
        )
    )
end

#= Merging expectation and plots =#
function expectation_and_plot(tspan, operator, operator_str, states)
    data = expectation_value(operator, states)
    plot_expectation(tspan, operator_str, data)
end

#= expectation and plot of two operators =#
function expectation_and_plot_comparison(tspan, operator1, operator1_str, operator2, operator2_str, states)
    data1 = expectation_value(operator1, states)
    data2 = expectation_value(operator2, states)

    e1_scatter=scatter(x=tspan, y=data1)
    e2_scatter=scatter(x=tspan, y=data2)

    plt = plot([e1_scatter, e2_scatter],
    Layout(
        xaxis_title = "t (μs)",
        yaxis_title = "<n>",
        showlegend = false
        )
    )
    display(plt)
    return data1, data2, plt
end


#= Run the dynamical evolution of states, depending if it is unitary or dissipative =#
function dynamic_evolution(time, ψ0, dynamics_input, type_dynamics::Symbol)
    tspan = time[1]:time[2]:time[end]
    dt = time[2]

    # Define a mapping from symbol to function
    dynamics_map = Dict(
        :schroedinger => timeevolution.schroedinger,
        :schroedinger_dynamic => timeevolution.schroedinger_dynamic,
        :master => timeevolution.master,
        :master_dynamic => timeevolution.master_dynamic
    )

    # Get the appropriate function
    evolve_func = dynamics_map[type_dynamics]

    return evolve_func(tspan, ψ0, dynamics_input;
    adaptive=false, 
    dt=dt,
    reltol=1e-9,
    abstol=1e-9
    )
end

function Quantum_solver_ODE(prob)
    sol = solve(prob, Tsit5())

    return sol.t, sol.u
end

########################################################################################################


###############
#= QM PULSES =#
###############
#= pi-pulse =#
function π_pulse_shape(t, t0, duration, eps=1e-12)
    δt = t - t0
    if 0.0 <= δt < duration
        s = sin(pi * δt / duration)^2
        return s / (s + eps^2)
    else
        return 0.0
    end
end

##################
#= INFIDELITIES =#
##################

#= mixed-state infidelity =#
function qo_infidelity(ρ, σ)
    return 1.0 - min(real(QuantumOptics.fidelity(ρ, σ)), 1)
end

function in_qo_infidelity(states, target_state)
    return [qo_infidelity(matrix, target_state) for matrix in states]
end



##############################################################################################################################################
################################
#= SAMPLED DATASET GENERATION =#
################################

#= This function needs to be recalled to generate the training/testing dataset for the NN =#
function dataset_generation(problem_settings , dataset_settings) #ok
    type_of_dataset, dim_dataset, dim_parameters_space, parameters_range, p, n_samples, typeofdynamics, t0, initial_state = problem_settings

    D = Dict(
        :FL_1step => [FL_NN_inputs, FL_NN_outputs],
        :FL_1step_v2 => [FL_NN_inputs_v2, FL_NN_outputs]
    )
    input_func, output_func = D[type_of_dataset]

    #Outputs
    outputs = output_func(p, parameters_range, dim_parameters_space, n_samples)

    #Inputs
    inputs = input_func(t0, initial_state, dataset_settings, outputs, n_samples, typeofdynamics)


    #NN_input and NN_output preparation
    n_training, n_input = dim_dataset
    dataset_matrix = Float64.(reduce(hcat, dataset_creation(inputs, outputs))')

    return  dataset_matrix[1:n_training, 1:n_input], 
            dataset_matrix[1:n_training, (n_input + 1):end],
            dataset_matrix[(n_training+1):end, 1:n_input],
            dataset_matrix[(n_training+1):end, (n_input + 1):end]

end


#= Final call for dataset dataset_creation =#
function dataset_creation(input_data, output_data) #ok
    len_dataset = length(input_data)
    dim_input = length(input_data[1])
    dim_output = length(output_data[1])

    total_dim = dim_input + dim_output

    dataset = Vector{Vector{Float64}}(undef, len_dataset)

    for i in 1:len_dataset
        row = Vector{Float64}(undef, total_dim)

        for j in 1:dim_input
            row[j] = input_data[i][j]
        end
        for k in 1:dim_output
            row[dim_input + k] = output_data[i][k]
        end
        dataset[i] = row
    end
    return dataset
end


###################################################################################################
###################################
#= PARAMETERS SPACE FOR SAMPLING =#
###################################
function twoD_parameter_space(p, parameters_range, dim_parameters_space)
    para1 = LinRange(parameters_range[1][1], parameters_range[1][2], dim_parameters_space[1])
    para2 = logrange(parameters_range[2][1], parameters_range[2][2], dim_parameters_space[2])

    parameters_space = vec([(x, y) for x in para1, y in para2])
    prob = vec([p(x,y) for (x,y) in parameters_space])

    return parameters_space, prob
end
#####################################################################################################


##################################################################################################
##################
#= MISCELLANOUS =#
##################
function recomposition(vector)
    exit = ComplexF64[]
    N = length(vector)

    for k in 1:2:N
        push!(exit, vector[k] + im*vector[k+1])
    end

    return exit
end
function to_real_vec(vector)
    real_imag = Vector{Float64}(undef, 2 * length(vector))
    for i in eachindex(vector)
        real_imag[2i - 1] = real(vector[i])
        real_imag[2i]     = imag(vector[i])
    end
    return real_imag
end



