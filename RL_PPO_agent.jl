module RLagent    
    
    using Flux 
    using Flux: Chain, Dense, relu, softmax, glorot_uniform 
    using Distributions 
    using LinearAlgebra
    using Random 

    using ..QuantumConstants 
    using ..QuantumSystemOperators 
    using ..B_spline_basis


    function create_actor_network(N_input::Int, hidden_dims::Vector{Int})

         """
        create_actor_network(N_input::Int, hidden_dims::Vector{Int})

            Creates the Actor neural network (Policy Network)
            Input is the target state representation
            Output is the means (mu) for the B-spline coefficients of the two drives

        Args:
            N_input:     dimension of the network input 
            hidden_dims: dimensions of the hidden layers

        Returns:
            Flux.jl Chain object representing the Actor network

    """

        N_output =  2 * BS_n_coeffs

        layers = []

        #FIRST HIDDEN LAYER 
        push!(layers, Dense(N_input, hidden_dims[1], relu; init = glorot_uniform))

        #HIDDEN LAYERS 
        for i in 1:(length(hidden_dims)-1)
            push!(layers,Dense(hidden_dims[i], hidden_dims[i+1],relu; init= glorot_uniform ))
        end

        #OUTPUT 
        push!(layers, Dense(hidden_dims[end], N_output; init = glorot_uniform))

        return Chain(layers...)

    end



    function create_critic_network(N_input::Int, hidden_dims::Vector{Int})

            """
        create_critic_network(N_input::Int, hidden_dims::Vector{Int})

            Creates the Value Network - Critic
            Input is the current state of the qs
            Output is the estimated value of V(s)

        Args:
                N_input:     dimension of the flatten vector of the state
                hidden_dims: dimensions of the hidden layers

        Returns:
                 Flux.jl Chain object representing the Critic network

    """

        layers = []

        push!(layers, Dense(N_input, hidden_dims[1], relu, init = glorot_uniform))
                
        #HIDDEN LAYERS 
        for i in 1:(length(hidden_dims)-1)
            push!(layers,Dense(hidden_dims[i], hidden_dims[i+1],relu; init= glorot_uniform ))
        end

        #OUTPUT 
        push!(layers, Dense(hidden_dims[end], 1 ; init = glorot_uniform))

        return Chain(layers...)
    end

end