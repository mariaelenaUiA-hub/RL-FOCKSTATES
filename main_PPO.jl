using Random
using Flux
using StableRNGs
using Distributions
using OrdinaryDiffEq
using OrdinaryDiffEqTsit5
using ReinforcementLearningBase

include("RL_Environment.jl")
include("PPO_agent.jl")

using .RLEnvironment
using .PPOpolicy

rng = StableRNG(42)



env = QuantumEnv()
s = real.(env.current_state.data)
state_dim = length(s)

action_dim = 2 


actor  = PPOpolicy.RLEnvironment.Actor(state_dim, action_dim)
critic = PPOpolicy.RLEnvironment.Critic(state_dim)
policy = PPOPolicy(actor,critic)

action, log_prob, value = select_action(policy,s)


action = Float64.(action)


RLBase.reset!(env)

println("Stato iniziale:")
println(env.current_state)

for t in 1:env.max_steps
    # Stato corrente
    state = real.(env.current_state.data)

    
    state = Float64.(state)

    # Azione dalla policy
    action, log_prob, value = select_action(policy, state)
    action = Float64.(action)

    # Applica azione
    reward, done =  RLEnvironment.step!(env, action)

    println("Step $t | Reward = $reward | Done = $done")

    if done
        println("Episodio terminato dopo $t passi.")
        break
    end
end

println("Stato finale:")
println(env.current_state)



action = Float64.(action)

@show typeof(action[1])

Δ, Ω = action

Δ