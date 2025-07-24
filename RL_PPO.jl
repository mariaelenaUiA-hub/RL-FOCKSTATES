using ReinforcementLearning, ReinforcementLearningCore
using .ReinforcementLearningBase
using Statistics , Flux , Functors , Flux.Optimise, StableRNGs ,  Random, Zygote, Distributions, LinearAlgebra, QuantumOptics, DifferentialEquations, DiffEqFlux, OrdinaryDiffEq
using Flux: Chain
using Flux: Dense
using IntervalSets: ClosedInterval 

include("QM_library.jl")

N_cut_off = 6
N_mech    = 1
g         = 258.0
wm        = 5.9614e6


qub, mech, HBAR_qubit = Qubit_HO(N_cut_off, :FockBasis, 1//2)

basis_system = HBAR_qubit.Ia.basis_l

#Matrices for time evolution
Iad = HBAR_qubit.Iad.data
mI = HBAR_qubit.mI.data
Ia = HBAR_qubit.Ia.data
pI = HBAR_qubit.pI.data
zI = HBAR_qubit.zI.data
xI = HBAR_qubit.xI.data

H_JC = g * (Iad * mI  + Ia * pI) 



function SE_Fock_dynamics(du::Vector{Float64}, u::Vector{Float64}, p, t) 
    Δ, Ω = p[1],p[2]

    
    #= Hamiltonians of the problem =#
    
    H0  =( Δ / 2.0 ) * zI
    H_drive = Ω * xI

    H_tot = H0 + H_JC + H_drive


    ψ = u[1:2:end] + im * u[2:2:end]
    dψ = -1im * H_tot * ψ

    for i in eachindex(dψ)
        du[2i-1] = real(dψ[i])
        du[2i] = imag(dψ[i])
    end
    return nothing

end

function SE_Fock_problem!(tspan, p, ψ0)
    return ODEProblem(SE_Fock_dynamics, to_real_vec(ψ0), tspan, p)
end




function Quantum_solver_ODE(tspan, p, ψ0)
    prob = SE_Fock_problem!(tspan, p, ψ0)
    sol = solve(prob, Tsit5())


    return sol.t, sol.u
end



# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# RL ENVIRONMENT
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────



mutable struct QuantumEnv <: RLBase.AbstractEnv
    operators::qubit_ho
    target_state::Ket
    current_state::Ket
    t_span::Tuple{Float64, Float64}
    max_steps::Int
    current_step::Int
    reward::Float64
    done::Bool
end

function QuantumEnv(N_cut_off::Int)
    
    qub, mech, ops = Qubit_HO(N_cut_off, :FockBasis, 1//2)
    target = tensor(spindown(qub.basis), fockstate(mech.basis, N_mech))
    
    #initial_state =  tensor(spindown(qub.basis), fockstate(mech.basis, 0))
    initial_state = tensor(spinup(qub.basis), fockstate(mech.basis, 0))
    t0 = 0
    t_step = 1e-3

    t_span =(t0, t_step)

    return QuantumEnv(ops, target, initial_state, t_span, max_steps, 0, 0.0, false)
end

function RLBase.action_space(env::QuantumEnv)
    low_bound = -10e4
    high_bound = 10e4
    return  [ClosedInterval(low_bound, high_bound), ClosedInterval(low_bound, high_bound)]
end

function RLBase.state_space(env::QuantumEnv)
    dims = 2 * 2 * (N_cut_off + 1) 
    return  [ClosedInterval(-1.0, 1.0) for _ in 1:dims ]
end

RLBase.state(env::QuantumEnv) = Float64.(to_real_vec(env.current_state.data))
#RLBase.state(env::QuantumEnv) = to_real_vec(env.current_state.data)
RLBase.is_terminated(env::QuantumEnv) = env.current_step >= env.max_steps


function RLBase.reset!(env::QuantumEnv)
    qub_basis = env.operators.zI.basis_l.bases[1] 
    mech_basis = env.operators.zI.basis_l.bases[2]
    env.current_state = tensor(spindown(qub_basis), fockstate(mech_basis, 0))
    env.current_step = 0
    env.reward = 0.0
    env.done = false 
    return RLBase.state(env)
    
end


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# RL-PPO-STRUCTURES
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

struct Actor
    chain::Chain
end

Flux.@layer Actor

function Actor(state_dim::Int, action_dim::Int)
    chain = Chain(
        Dense(state_dim, 256, tanh),
        Dense(256, 128, tanh),
        Dense(128, 64, tanh),
        Dense(64, 32, tanh),
        Dense(32, action_dim * 2)
    )
    Actor(chain)
end

function (actor::Actor)(state)
    x = actor.chain(state)

    action_dim = size(x, 1) ÷ 2

    if ndims(x) == 1
    # Input is a single state vector
    μ = x[1:action_dim]
    log_σ = x[action_dim + 1 : end]

    else
        # Input is a batch of states (matrix)
        
        μ = x[1:action_dim, :]
        log_σ = x[action_dim + 1 : end, :]
    end


    

    σ = exp.(log_σ)
    
        
    # This will now work for both cases
    return Normal.(μ, σ)
end


struct Critic
    chain::Chain
end
Flux.@layer Critic

function Critic(state_dim::Int)
    chain = Chain(
        Dense(state_dim, 256, tanh),
        Dense(256, 128, tanh),
        Dense(128, 64, tanh),
        Dense(64, 32, tanh),
        Dense(32, 1) 
    )
    Critic(chain)
end
function (critic::Critic)(state)
    v = critic.chain(state) # Get the output from the neural network

    
    if ndims(v) > 1
        return vec(v)
    else
        return first(v)
    end
end


function step!(env::QuantumEnv, a::Vector{Float64})

    env.current_step += 1

    old_fid_amplitude= abs(env.target_state' * env.current_state)
    old_fid = old_fid_amplitude^2

    low_bound = -10e2
    high_bound = 10e2

    # Applica il clipping ai valori di Delta e Omega
    Δ = clamp(a[1], low_bound, high_bound)
    Ω = clamp(a[2], low_bound, high_bound)



    

    

    #Δ, Ω = a[1], a[2]
    p = (Δ, Ω)

    u0 = to_real_vec(env.current_state.data)
    
    prob = ODEProblem(SE_Fock_dynamics, u0, env.t_span, p)
    sol = solve(prob, Tsit5(), reltol=1e-6, abstol=1e-9,save_everystep=false, save_start=false)

    final_state = sol.u[end]

    env.current_state = Ket(env.current_state.basis, recomposition(final_state))

    normalize!(env.current_state)



    new_fid_amplitude = abs(env.target_state' * env.current_state)
    new_fidelity = new_fid_amplitude^2
    new_fidelity = clamp(new_fidelity, 0.0, 1.0)

    delta_fidelity = new_fidelity - old_fid




    # Parametri reward shaping
    fidelity_weight = 5000.0
    delta_weight = 2000.0
    success_bonus = 100000.0
    success_threshold = 0.80

    progress = env.current_step / env.max_steps
    delta_weight = delta_weight * (1.0 - progress)
    fidelity_weight = fidelity_weight * (1.0 - progress)
    
  
    
    

    done = false
    
    

    fid_cutoff = 0.001
    reward = 0.0

    if new_fidelity >= fid_cutoff
        reward += new_fidelity^2 * fidelity_weight
        if delta_fidelity > 0
            reward += delta_fidelity * delta_weight
        end
    else
        
        reward -= 0.05
    end

    if new_fidelity > success_threshold
        reward += success_bonus
        done = true
    end

    if env.current_step >= env.max_steps
        done = true
    end




    return reward, done
end

struct PPOPolicy
    actor::Actor
    critic::Critic
end

Functors.@functor PPOPolicy

mutable struct PPOBuffer
    states::Vector{Vector{Float64}}
    actions::Vector{Vector{Float64}}
    rewards::Vector{Float64}
    dones::Vector{Bool}
    log_probs::Vector{Float64}
    values::Vector{Float64}
end

function PPOBuffer()
    PPOBuffer([], [], [], [], [], [])
end

function Base.empty!(buffer::PPOBuffer)
    empty!(buffer.states)
    empty!(buffer.actions)
    empty!(buffer.rewards)
    empty!(buffer.dones)
    empty!(buffer.log_probs)
    empty!(buffer.values)
end

# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Agent and Policy PPO
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

mutable struct PPOAgent
    policy::PPOPolicy
    actor_optimizer
    critic_optimizer
    gamma::Float64
    lambda::Float64
    clip_range::Float64
    entropy_loss_weight::Float64
    critic_loss_weight::Float64
    max_grad_norm::Float64
    n_rollout::Int
    n_env::Int
    n_update_epochs::Int
    mini_batch_size::Int
    rng::StableRNGs.StableRNG
    buffer::PPOBuffer
end


Functors.@functor PPOAgent (policy,)


function PPOAgent(
    actor::Actor,
    critic::Critic;
    actor_optimizer = Adam(1e-3),
    critic_optimizer = Adam(1e-3),
    gamma::Real = 0.99,
    lambda::Real = 0.95,
    clip_range::Real = 0.2,
    entropy_loss_weight::Real = 0.01,
    critic_loss_weight::Real = 0.5,
    max_grad_norm::Real = 0.5,
    n_rollout::Int = 2048,
    n_env::Int = 2,
    n_update_epochs::Int = 10,
    mini_batch_size::Int = 64,
    rng = StableRNG(123),
)
    policy = PPOPolicy(actor, critic)
    buffer = PPOBuffer()
    
    return PPOAgent(
        policy,
        actor_optimizer,
        critic_optimizer,
        gamma,
        lambda,
        clip_range,
        entropy_loss_weight,
        critic_loss_weight,
        max_grad_norm,
        n_rollout,
        n_env,
        n_update_epochs,
        mini_batch_size,
        rng,
        buffer
    )
end

# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Core PPO Functions
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

function store_transition!(agent::PPOAgent, state, action, reward, done, logp, value)
    push!(agent.buffer.states, state)
    push!(agent.buffer.actions, action)
    push!(agent.buffer.rewards, reward)
    push!(agent.buffer.dones, done)
    push!(agent.buffer.log_probs, logp)
    push!(agent.buffer.values, value)
end

function select_action(agent::PPOAgent, state::Vector{Float64})
    dist = agent.policy.actor(state)
    action = [rand(d) for d in dist] # Campiona un'azione per ogni distribuzione
    log_prob = sum(logpdf.(dist, action))
    value = agent.policy.critic(state)
    return action, log_prob, value
end

function ready_to_update(agent::PPOAgent)
    return length(agent.buffer.rewards) >= agent.n_rollout
end


function update!(agent::PPOAgent; next_value = 0.0)
    
    all_values = [agent.buffer.values; next_value]

    update_policy!(
        agent,
        all_values
    )
    empty!(agent.buffer)
end

# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# PPO Logic
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

function compute_gae(rewards::Vector{Float64}, values::Vector{Float64}, dones::Vector{Bool}; γ=0.99, λ=0.95)
    T = length(rewards)
    advantages = zeros(Float64, T)
    gae = 0.0
    for t in T:-1:1
        
        delta = rewards[t] + γ * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + γ * λ * (1 - dones[t]) * gae
        advantages[t] = gae
    end
    return advantages
end

function update_policy!(agent::PPOAgent, all_values::Vector{Float64})
    
    
    advantages = compute_gae(agent.buffer.rewards, all_values, agent.buffer.dones; γ=agent.gamma, λ=agent.lambda)
    returns = advantages .+ agent.buffer.values # returns = A_t + V_t

    
    adv_mean, adv_std = mean(advantages), std(advantages) + 1e-8
    norm_adv = (advantages .- adv_mean) ./ adv_std

    data = (
        states=agent.buffer.states, 
        actions=agent.buffer.actions, 
        returns=returns, 
        advantages=norm_adv, 
        log_probs=agent.buffer.log_probs
    )

    
    for _ in 1:agent.n_update_epochs
        
        indices = randperm(agent.rng, length(data.states))
        
        for batch_indices in Iterators.partition(indices, agent.mini_batch_size)
            
            
            batch_states = hcat(data.states[batch_indices]...)
            batch_actions = hcat(data.actions[batch_indices]...)
            batch_returns = data.returns[batch_indices]
            batch_advantages = data.advantages[batch_indices]
            batch_old_log_probs = data.log_probs[batch_indices]

            
            ps_actor = Flux.params(agent.policy.actor)
            grads_actor = Flux.gradient(ps_actor) do
                dist = agent.policy.actor(batch_states)
                log_probs = sum(logpdf.(dist, batch_actions), dims=1)[:]
                
                ratios = exp.(log_probs .- batch_old_log_probs)
                
                surr1 = ratios .* batch_advantages
                surr2 = clamp.(ratios, 1 - agent.clip_range, 1 + agent.clip_range) .* batch_advantages
                
                actor_loss = -mean(min.(surr1, surr2))
                entropy_loss = -mean(sum(entropy.(dist), dims=1))
                
                return actor_loss + agent.entropy_loss_weight * entropy_loss
            end
            
            
            ps_critic = Flux.params(agent.policy.critic)
            grads_critic = Flux.gradient(ps_critic) do
                values_pred = agent.policy.critic(batch_states)[:]
                return Flux.mse(values_pred, batch_returns) * agent.critic_loss_weight
            end

            
            Flux.Optimisers.update!(agent.actor_optimizer, ps_actor, grads_actor)
            Flux.Optimisers.update!(agent.critic_optimizer, ps_critic, grads_critic)
        end
    end


end


#────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
