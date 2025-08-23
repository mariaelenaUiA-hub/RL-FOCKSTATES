using Revise 

using Statistics
N_cut_off = 6;
N_mech    = 1;
g         = 258.0;
global ωm = 5.9614e6;


#Matrices for time evolution
Iad = HBAR_qubit.Iad.data
mI = HBAR_qubit.mI.data
Ia = HBAR_qubit.Ia.data
 pI = HBAR_qubit.pI.data
 zI = HBAR_qubit.zI.data
 xI = HBAR_qubit.xI.data

include("RL_PPO.jl")

# --- PPO Hyperparameters 
BATCH_SIZE = 512
max_steps = 100   #era 256
N_UPDATE_EPOCHS = 13
GAMMA = 0.99 
LAMBDA = 0.97
CLIP_RANGE = 0.2 #provare 0.3 forse??
ENTROPY_LOSS_WEIGHT = 0.05 
CRITIC_LOSS_WEIGHT = 0.5 #era 0.5
MAX_GRAD_NORM = 0.5 
LR_ACTOR = 3e-4 # Learning rate for the actor network #MI RACCOMANDO MARI, I DUE LR MAI DIVERSI TANTO!!
LR_CRITIC =3e-4 # Learning rate for the critic network
N_ROLLOUT = 1024
N_ENV = 1

# --- 



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

println(agent)



total_rewards = [] 

episode_fidelities = [] 
episode_rewards = [] 

function main_training_loop(env::QuantumEnv, agent::PPOAgent, num_episodes::Int)

    episode_rewards = Float64[]
    episode_fidelities = Float64[]
    total_steps = 0

    best_fidelity = 0.0
    best_actions = []




    println("Starting PPO training for $(num_episodes) episodes...")

    for episode in 1:num_episodes
        current_state = RLBase.reset!(env)
        done = false
        episode_reward = 0.0
        episode_fidelity = 0.0 # To store final fidelity of the episode

        current_episode_actions = []


        while !done
            # Select action
            action, log_prob, value = select_action(agent, current_state)

            push!(current_episode_actions, action)


            # Step the environment
            reward, done = step!(env, action)

            # Store transition
            store_transition!(agent, current_state, action, reward, done, log_prob, value)

            # Update current state and episode reward
            current_state = RLBase.state(env) # Get the new state after the step
            episode_reward += reward
            total_steps += 1

           
            if ready_to_update(agent)
                # Calculate next_value for GAE if the episode isn't truly done
                next_value = done ? 0.0 : agent.policy.critic(current_state)
                update!(agent; next_value=next_value)
                #println("--- Policy Updated at step $(total_steps) ---")
            end

            
            if done
                # Record final fidelity for the episode
                episode_fidelity = abs(env.target_state' * env.current_state)^2


                if episode_fidelity > best_fidelity
                    best_fidelity = episode_fidelity
                    best_actions = copy(current_episode_actions)
                    
                end
                break
            end
        end

        push!(episode_rewards, episode_reward)
        push!(episode_fidelities, episode_fidelity)



        if episode % 10 == 0 && length(episode_rewards) >= 10
            recent_rewards = episode_rewards[end-9:end]
            recent_fidelities = episode_fidelities[end-9:end]
            corr_val = Statistics.cor(recent_rewards, recent_fidelities)
            println("Corr Reward-Fidelity (last 10): ", round(corr_val, digits=4))
        end

        # Plot reward e fidelity ogni 50 episodi
        if episode % 50 == 0
            plot(episode_rewards, label="Reward", color=:blue)
            plot!(episode_fidelities .* maximum(episode_rewards), label="Fidelity (scaled)", color=:red)
            gui()  # oppure display()
        end
        # Print episode summary
        if episode % 10 == 0 || episode == num_episodes
            avg_reward = mean(episode_rewards[max(1, end-9):end]) # Average over last 10 episodes
            avg_fidelity = mean(episode_fidelities[max(1, end-9):end])
            println("Episode $(episode)/$(num_episodes) | Total Steps: $(total_steps) | Episode Reward: $(round(episode_reward, digits=4)) | Final Fidelity: $(round(episode_fidelity, digits=4)) | Avg Reward (last 10): $(round(avg_reward, digits=4)) | Avg Fidelity (last 10): $(round(avg_fidelity, digits=4))")
        end
    end

    println("Training finished!")
    return episode_rewards, episode_fidelities,best_actions
end


num_training_episodes = 500

#Run the training
rewards_history, fidelities_history, best_actions_list = main_training_loop(env, agent, num_training_episodes)




if !isempty(best_actions_list)  
    println("Numero di step di controllo: ", length(best_actions_list))
    println("Parametri di controllo [Δ, Ω] per ogni step:")
    for (i, action) in enumerate(best_actions_list)
        println("Step $i: Δ = $(action[1]), Ω = $(action[2])")
    end
else
    println("Nessuna strategia con fedeltà > 0 è stata trovata.")
end


# # Flux.save("ppo_quantum_policy.bson", agent.policy)


 using Plots



plot(fidelities_history,
     label="Fidelity",
     xlabel="Episode",
     ylabel="Fidelity",
     title="Fidelities vs Episodes")


     
  gr()

actions_matrix = hcat(best_actions_list...)'

plot_delta = plot(actions_matrix[:, 1],
    title = "Detuning (Δ)",
    xlabel = "Step", 
    ylabel = " Δ",
    legend = false,
    color = :blue,
    linewidth = 2,
    marker = :circle
)


plot_omega = plot(actions_matrix[:, 2],
    title = " Drive (Ω)",
    xlabel = "Step",
    ylabel = " Ω",
    legend = false,
    color = :red,
    linewidth = 2,
    marker = :circle
)


combined_plot = plot(plot_delta, plot_omega, 
                     layout = (2, 1), 
                     size = (800, 600) 
)


display(combined_plot)












 


function H()

    H_tot = H0 + H_JC_ + H_drive
    return H_tot
end

function evolution_SE(p, ψ0, tspan)
    Δ, Ω = p[1],p[2]

    H0  = ( Δ / 2.0 ) * HBAR_qubit.zI
    H_drive = Ω * HBAR_qubit.xI
    H_JC_ = g * (HBAR_qubit.Iad * HBAR_qubit.mI  + HBAR_qubit.Ia * HBAR_qubit.pI)

    H(t, ψ) = H0 + H_drive + H_JC_


    _ , psi_t = timeevolution.schroedinger_dynamic(tspan, ψ0, H)
    exp_val = expect(HBAR_qubit.n_mech, psi_t)
    exp_val_q = expect(HBAR_qubit.n_qubit, psi_t)

    return real.(exp_val), real.(exp_val_q), psi_t

end





t0 = 0.0

ψ0 = tensor(spindown(qub.basis), fockstate(mech.basis, 0))
Δt = 1e-3

solution     = [ψ0]
exp_values   = [0]
exp_values_q = [0]

for step in eachindex(best_actions_list)

    p = best_actions_list[step]
    t1 = t0 + Δt
    tspan = (t0,t1)


    
    exp_val, exp_val_q, ψt  = evolution_SE( p, ψ0, tspan )
     
    solution     = vcat(solution, ψt[2:end])
    exp_values   = vcat(exp_values,exp_val[2:end])
    exp_values_q = vcat(exp_values_q,exp_val_q[2:end])
    

    t0 = t1
    ψ0 = ψt[end]



end

exp_values
solution
exp_values_q

target = tensor(spindown(qub.basis), fockstate(mech.basis, N_mech))

abs2(real(dagger(solution[end]) * target))    



plot_mech = plot(exp_values,
     label="exp values n",
     xlabel="step",
     ylabel="n",
     title="")

 exp_values[end]


plot_q = plot(exp_values_q,
     label="exp values q",
     xlabel="step",
     ylabel="q",
     title="")
  

exp_values_q[end]


combined_plot_2 = plot(plot_mech, plot_q, 
                     layout = (2, 1), 
                     size = (800, 600) 
)


display(combined_plot_2)