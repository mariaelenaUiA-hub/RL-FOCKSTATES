module RLagent    
    
    using Flux 
    using Flux: Chain, Dense, relu, softmax, glorot_uniform 
    using Distributions 
    using LinearAlgebra
    using Random 


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
            N_input:     dimension of the flatten vector of the current state
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




    function sample_action_coeffs(mu_coeffs::Vector{Float32}, sigma_action::Float32)



    """
        sample_action_coeffs(mu_coeffs::Vector{Float32}, sigma_action::Float32)

        Sampling of the coeffs of the B-Spline from a Gaussian distribution

        Args:
            mu_coeffs:vector of the mu for each coefficient, Actor output
            sigma_action: we can start with this fixed

        Returns:
            Base B-Spline coefficients
    """
        
        # 'Normal.' con il punto è una forma di broadcasting in Julia
        #crea un array di oggetti Normal, uno per ogni elemento di mu_coeffs
        
        dist = Normal.(mu_coeffs, sigma_action)
        
        #campiona un valore da ciascuna di queste distribuzioni normali
        #'rand.' con il punto esegue il campionamento per ogni distribuzione nell'array 'dist'

        coeffs_sampled = rand.(dist)
        
        
        return Float32.(coeffs_sampled)
    end





    function log_prob_coeffs(mu_coeffs::Vector{Float32}, sigma_action::Float32, coeffs_sampled::Vector{Float32})


    """
        log_prob_coeffs(mu_coeffs::Vector{Float32}, sigma_action::Float32, coeffs_sampled::Vector{Float32})

    Args:
        mu_coeffs: Vettore delle medie (mu) per ogni coefficiente, prodotto dall'Actor
        sigma_action: La deviazione standard usata per il campionamento 
        coeffs_sampled: Il vettore di coefficienti effettivamente campionati.

    Returns:
        La probabilità logaritmica totale (somma delle log-probabilità individuali).
    """

        sigma_action = max(sigma_action, 1.0f-6) 
        
        dist = Normal.(mu_coeffs, sigma_action)
        
        return sum(logpdf.(dist, coeffs_sampled))
    end


    export create_actor_network, create_critic_network, sample_action_coeffs, log_prob_coeffs

end