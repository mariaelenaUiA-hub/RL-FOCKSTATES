module PPOpolicy

    using ReinforcementLearning
    using ReinforcementLearningCore
    using Statistics
    using Flux
    using Functors
    using Flux.Optimise
    using StableRNGs
    using Random
    using Zygote
    using Distributions


    include("RL_Environment.jl")
    using .RLEnvironment

       

    


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

    function store!(buffer::PPOBuffer, s, a, r, done, log_p, v)
        push!(buffer.states, s)
        push!(buffer.actions, a)
        push!(buffer.rewards, r)
        push!(buffer.dones, done)
        push!(buffer.log_probs, log_p)
        push!(buffer.values, v)
    end


    function select_action(policy::PPOPolicy, state::Vector{Float64})
        x = state
        dist = policy.actor(x)        # restituisce [Normal(μ₁, σ₁), Normal(μ₂, σ₂)]
        action = [rand(d) for d in dist]  # genera [Δ, Ω]
        log_prob = sum(logpdf.(dist, action))
        value = policy.critic(x)

        return action, log_prob, value
    end


    function compute_gae(rewards, values, dones; γ=0.99, λ=0.95)
        T = length(rewards)
        advantages = zeros(Float64, T)
        gae = 0.0
        for t = T:-1:1
            delta = rewards[t] + (1 - dones[t]) * γ * (t < T ? values[t+1] : 0.0) - values[t]
            gae = delta + (1 - dones[t]) * γ * λ * gae
            advantages[t] = gae
        end
        return advantages
    end

    """
    Aggiornamento della policy PPO
    """
    function update_policy!(policy::PPOPolicy, buffer::PPOBuffer;
                            γ=0.99, λ=0.95, clip_eps=0.2, lr=3e-4, epochs=10, batch_size=64)

        # Calcolo dei vantaggi
        advantages = compute_gae(buffer.rewards, buffer.values, buffer.dones; γ=γ, λ=λ)
        returns = [adv + v for (adv, v) in zip(advantages, buffer.values)]

        # Normalizza il vantaggio
        adv_mean, adv_std = mean(advantages), std(advantages) + 1e-8
        norm_adv = (advantages .- adv_mean) ./ adv_std

        # Dataset per update
        data = [(buffer.states[i], buffer.actions[i], returns[i], norm_adv[i], buffer.log_probs[i])
                for i in eachindex(buffer.states)]

        opt = Flux.setup(Adam(lr), policy)

        for _ in 1:epochs
            for batch in Iterators.partition(data, batch_size)
                grads = Flux.gradient(policy) do p
                    loss = 0.0
                    for (s, a, R, A, old_logp) in batch
                        dist = policy.actor(s)
                        logp = sum(logpdf.(dist, a))
                        ratio = exp(logp - old_logp)
                        surr1 = ratio * A
                        surr2 = clamp(ratio, 1 - clip_eps, 1 + clip_eps) * A
                        critic_value = policy.critic(s)
                        loss += -min(surr1, surr2) + 0.5 * (critic_value - R)^2
                    end
                    return loss / length(batch)
                end
                Flux.update!(opt, policy, grads)
            end
        end
    end


    export PPOPolicy, PPOBuffer, select_action, update_policy!
end