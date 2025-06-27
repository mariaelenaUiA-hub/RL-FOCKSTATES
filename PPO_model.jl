
using ReinforcementLearning, Flux, StableRNGs, Plots, LinearAlgebra, DifferentialEquations, QuantumOptics
using ReinforcementLearningBase 


include("QuantumEnvironment.jl")
using .QuantumEnvironment

include("QuantumConstants.jl")
include("QuantumSystemOperators.jl")
using .QuantumConstants
using .QuantumSystemOperators


Base.@kwdef mutable struct QuantumEnv <: AbstractEnv
        
        
        env_params::NamedTuple
        H_ops_full::NamedTuple 
        
        
        initial_state_vector::Vector{ComplexF32}
        current_state_vector::Vector{ComplexF32}
        target_state_vector::Vector{ComplexF32}
        
        
        Δt::Float32
        max_episode_steps::Int
        episode_length_counter::Int = 0
        
        
        action_space::ContinuousSpace{Vector{Float32}}
        observation_space::ContinuousSpace{Vector{Float32}}
        state::Vector{Float32}
        reward::Float32
        done::Bool
    end


RLBase.action_space(env::QuantumEnv) = env.action_space
RLBase.state_space(env::QuantumEnv) = env.observation_space
RLBase.reward(env::QuantumEnv) = env.reward
RLBase.is_terminated(env::QuantumEnv) = env.done
RLBase.state(env::QuantumEnv) = env.state



function RLBase.reset!(env::QuantumEnv)
    env.current_state_vector = deepcopy(env.initial_state_vector)
    env.episode_length_counter = 0
    env.done = false
    env.reward = 0.0f0
    env.state = vcat(real.(env.current_state_vector), imag.(env.current_state_vector), real.(env.target_state_vector), imag.(env.target_state_vector))
    nothing
end

function (env::QuantumEnv)(action_t::Vector{Float32})
    
    
    p_ode = (
        action_t = action_t,
        system_params = env.env_params.system_params,
        H_ops_full = env.H_ops_full 
    )

    u0_ode = Vector{Float64}(undef, 2 * length(env.current_state_vector))
    for i in eachindex(env.current_state_vector)
        u0_ode[2i - 1] = real(env.current_state_vector[i])
        u0_ode[2i]     = imag(env.current_state_vector[i])
    end
   
    ode_prob = ODEProblem(schrodinger_equation!, u0_ode, (0.0, Float64(env.env_params.Δt)), p_ode)
    sol = solve(ode_prob, Tsit5(), save_everystep=false, save_start=false)

    next_state = normalize(reinterpret(ComplexF32, sol.u[end]))

    
    env.reward = calculate_reward(next_state, env.target_state_vector)
    
    
    env.current_state_vector = next_state
    env.episode_length_counter += 1
    env.done = (env.episode_length_counter >= env.env_params.max_episode_steps)
    env.state = vcat(real.(env.current_state_vector), imag.(env.current_state_vector), real.(env.target_state_vector), imag.(env.target_state_vector))
    
    
end