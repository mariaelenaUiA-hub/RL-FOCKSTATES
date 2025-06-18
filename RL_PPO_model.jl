
module RLmodel

    using DifferentialEquations 
    using LinearAlgebra       

    using ..QuantumConstants         
    using ..QuantumSystemOperators  
    using ..BSplineUtilities         



    struct QuantumEnv
        current_state_vector::Vector{ComplexF32} 
        target_state_vector::Vector{ComplexF32}  
        system_params::Dict                       
        Δt::Float32                               
        # Aggiungi qui i coefficienti B-spline e il set di funzioni B-spline se l'azione dell'Actor li definisce
        # in un unico passaggio (controllo feedforward parametrizzato).
        # Nella fase 1, l'action_t è passata direttamente alla funzione schrodinger_equation!.
        # Per la Fase 1, non abbiamo bisogno di memorizzare i coeffs B-spline qui nella struct.
    end


    function step_env(env::QuantumEnv, action_t::Vector{Float32})
        
        p_ode = (action_t = action_t, system_params = env.system_params)
        
        
        ode_prob = ODEProblem(schrodinger_equation!, env.current_state_vector, (0.0f0, env.Δt), p_ode)
        
        
        sol = solve(ode_prob, Tsit5(), dt=env.Δt) 

        next_state_vector = sol.u[end] # Prendi lo stato alla fine dell'intervallo

    
        next_state_vector = next_state_vector / norm(next_state_vector)

        
        reward_exponent_theta = 8 
        reward_val = calculate_reward(next_state_vector, env.target_state_vector, reward_exponent_theta)
        
        
        env.current_state_vector[:] = next_state_vector 

        is_done = false 
        info_dict = Dict("fidelity" => abs2(dot(env.target_state_vector, next_state_vector)))

        return next_state_vector, reward_val, is_done, info_dict
    end

end