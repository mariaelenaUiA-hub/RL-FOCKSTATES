using Revise 
using Base.Threads

using Statistics
N_cut_off = 6;
N_mech    = 1;
g         = 258.0;
global ωm = 5.9614e6;



include("RL_PPO_n_env.jl")

# --- PPO Hyperparameters 
BATCH_SIZE = 256

N_UPDATE_EPOCHS = 8
GAMMA = 0.995 
LAMBDA = 0.95
CLIP_RANGE = 0.2 #provare 0.3 forse??
ENTROPY_LOSS_WEIGHT = 0.02 
CRITIC_LOSS_WEIGHT = 0.5 #era 0.5
MAX_GRAD_NORM = 0.5 
LR_ACTOR = 1e-4 # Learning rate for the actor network #MI RACCOMANDO MARI, I DUE LR MAI DIVERSI TANTO!!
LR_CRITIC =1e-3 # Learning rate for the critic network
N_ROLLOUT = 8 * 800
N_ENV = 8
n_envs = N_ENV
# --- 

function create_envs(N_env::Int, N_cut_off::Int)
    [QuantumEnv(N_cut_off) for _ in 1:N_env]
end

function reset_envs!(envs::Vector{QuantumEnv})
    states = Vector{Any}(undef, length(envs))
    Threads.@threads for i in 1:length(envs)
        states[i] = RLBase.reset!(envs[i])
    end
    return states
end


env = QuantumEnv(N_cut_off)

state_dim = length(RLBase.state_space(env)) 
action_dim = length(RLBase.action_space(env)) 



rng = StableRNG(123) 






actor  = Actor(state_dim, action_dim) 
critic = Critic(state_dim) 
policy = PPOPolicy(actor, critic) 

actor = Flux.f64(actor)
critic = Flux.f64(critic)



agent = PPOAgent(
    actor,
    critic;
    actor_optimizer = Flux.Optimise.Adam(LR_ACTOR),
    critic_optimizer = Flux.Optimise.Adam(LR_CRITIC),
    gamma = GAMMA,
    lambda = LAMBDA,
    clip_range = CLIP_RANGE,
    entropy_loss_weight= ENTROPY_LOSS_WEIGHT ,
    n_rollout = N_ROLLOUT,
    n_env = N_ENV,
    n_update_epochs = N_UPDATE_EPOCHS,
    mini_batch_size = BATCH_SIZE,
    rng = rng
    
)



using Flux.Optimisers: OptimiserChain, ClipNorm, Adam
agent.actor_optimizer  = OptimiserChain(ClipNorm(0.5), Adam(LR_ACTOR))
agent.critic_optimizer = OptimiserChain(ClipNorm(0.5), Adam(LR_CRITIC))

reset_opt_states!(agent)


envs = create_envs(N_ENV,N_cut_off)



function main_training_loop_parallel(envs::Vector{QuantumEnv}, agent::PPOAgent, num_episodes::Int)
    n_env = length(envs)

    episode_rewards    = Float64[]
    episode_fidelities = Float64[]

    best_fidelity = 0.0
    best_actions  = Vector{Any}()

    println("Starting PPO training for $num_episodes episodes on $n_env environments...")

    for episode in 1:num_episodes
        # Reset di tutti gli ambienti
        states          = [RLBase.reset!(env) for env in envs]
        done_flags      = falses(n_env)
        rewards_this_ep = zeros(Float64, n_env)
        actions_this_ep = [Any[] for _ in 1:n_env]

        # --- loop sugli step ---
        while !all(done_flags)
            # Seleziona azione + logp + value (una sola volta per env)
            actions      = Vector{Vector{Float64}}(undef, n_env)
            log_probs    = Vector{Float64}(undef, n_env)
            values_state = Vector{Float64}(undef, n_env)

            for i in 1:n_env
                if !done_flags[i]
                    a, lp, v = select_action(agent, states[i])
                    actions[i]      = a
                    log_probs[i]    = lp
                    values_state[i] = v
                else
                    actions[i]      = [0.0, 0.0]
                    log_probs[i]    = 0.0
                    values_state[i] = 0.0
                end
            end

            # Step parallelo degli ambienti
            new_states, rewards, dones = step_envs!(envs, actions)

            # Salva transizioni nel buffer usando gli stessi (a, lp, v)
            for i in 1:n_env
                if !done_flags[i]
                    store_transition!(agent, states[i], actions[i], rewards[i], dones[i],
                                      log_probs[i], values_state[i]; env_id=i)
                    push!(actions_this_ep[i], actions[i])
                    rewards_this_ep[i] += rewards[i]
                    done_flags[i] = dones[i]
                end
            end

            # Aggiorna stati
            states = new_states

            # Update policy quando il buffer è pieno
            if ready_to_update(agent)
                # bootstrap value per ogni env (0 se done)
                bootstrap = [done_flags[i] ? 0.0 : agent.policy.critic(states[i]) for i in 1:n_env]
                update!(agent; bootstrap_values_by_env=bootstrap)
            end
        end # while

        # Flush finale se rimane roba nel buffer (tutti gli env sono done → bootstrap=0)
        if !isempty(agent.buffer.rewards)
            update!(agent; bootstrap_values_by_env=zeros(n_env))
        end

        # Calcola fidelities finali
        fidelities    = [abs2(env.target_state' * env.current_state) for env in envs]
        avg_fidelity  = mean(fidelities)
        max_fidelity, idx = findmax(fidelities)
        best_actions_episode = actions_this_ep[idx]

        # Track best globale
        if max_fidelity > best_fidelity
            best_fidelity = max_fidelity
            best_actions  = copy(best_actions_episode)
        end

        push!(episode_rewards,   mean(rewards_this_ep))
        push!(episode_fidelities, avg_fidelity)

        if episode % 10 == 0 && length(episode_rewards) ≥ 10
            recent_rewards    = episode_rewards[end-9:end]
            recent_fidelities = episode_fidelities[end-9:end]
            corr_val = Statistics.cor(recent_rewards, recent_fidelities)
            println("Episode $episode/$num_episodes | AvgR: $(round(mean(recent_rewards), digits=4)) | AvgF: $(round(mean(recent_fidelities), digits=4)) | Corr(R,F)=$(round(corr_val, digits=4))")
        else
            println("Episode $episode | AvgR: $(round(mean(rewards_this_ep), digits=4)) | AvgF: $(round(avg_fidelity, digits=4)) | BestF(ep): $(round(max_fidelity, digits=4))")
        end
    end

    println("Training finished! Best Fidelity = $best_fidelity")
    return episode_rewards, episode_fidelities, best_actions
end



num_episodes = 10000
envs = create_envs(N_ENV, N_cut_off)
all_rewards, all_fidelities, best_actions = main_training_loop_parallel(envs, agent, num_episodes)




function evolution_step_from_action(a::Tuple{<:Real,<:Real}, ψ0::Ket, tspan::Tuple{Float64,Float64})
    a1, a2 = a
    Δ_max = 5e4      # 50 MHz (in kHz)
    Ω_max = 8.33e2     # ≈520 kHz (o 1.04e3 se xI=σx/2)
    # NB: durante il training a era già in [-1,1] (tanh-squashed)
    # e in step! veniva rimappata con tanh(.) di nuovo:
    ωq = ωm + Δ_max * tanh(a1)      # kHz
    Ω  = Ω_max * tanh(a2)           # kHz

    Δ = ωq - ωm
    

    H0      = dense((Δ/2) * HBAR_qubit.zI)
    H_drive = dense(Ω      * HBAR_qubit.xI)
    H_JC    = dense(g * (HBAR_qubit.Iad * HBAR_qubit.mI + HBAR_qubit.Ia * HBAR_qubit.pI))

    # Hamiltoniana (time-independent qui, ma la firma accetta H(t,ψ))
    H(t, ψ) = H0 + H_drive + H_JC

    ts, ψt = timeevolution.schroedinger_dynamic(tspan, ψ0, H)
    exp_mech = expect(HBAR_qubit.n_mech,  ψt)
    exp_qub  = expect(HBAR_qubit.n_qubit, ψt)
    return real.(exp_mech), real.(exp_qub), ψt
end


# ----------------- rollout deterministico con best_actions -----------------
t0 = 0.0
# usa lo STESSO Δt dell'env usato in training (es. 0.3 μs):
Δt = 0.3e-2     # <— metti il tuo valore reale se diverso

ψ0 = tensor(spindown(qub.basis), fockstate(mech.basis, 0))

solution     = Ket[ψ0]
exp_values   = Float64[0.0]
exp_values_q = Float64[0.0]

for step in eachindex(best_actions)
    p = best_actions[step]
    # assicura tupla (a1,a2)
    a = (p[1],p[2])

    t1 = t0 + Δt
    tspan = (t0, t1)

    exp_m, exp_q, ψt = evolution_step_from_action(a, ψ0, tspan)

    # accumula (salta il primo punto per evitare duplicati in tempo)
    append!(solution, ψt[2:end])
    append!(exp_values,   exp_m[2:end])
    append!(exp_values_q, exp_q[2:end])

    t0 = t1
    ψ0 = ψt[end]
end

# ----------------- fidelity finale contro target -----------------
target = tensor(spindown(qub.basis), fockstate(mech.basis, N_mech))
# usa dot sui vettori dati (gestisce la coniugazione in prima arg)
fid = abs2(dot(target.data, solution[end].data))
println("Fidelity finale = ", fid)

# ----------------- plot attese n_mech e n_qubit -----------------
plot_mech = plot(exp_values;
    label="⟨n_mech⟩",
    xlabel="step",
    ylabel="value",
    title="Mechanics occupancy")

plot_q = plot(exp_values_q;
    label="⟨n_qubit⟩",
    xlabel="step",
    ylabel="value",
    title="Qubit excitation")

display(plot(plot_mech, plot_q, layout=(2,1), size=(800,600)))




