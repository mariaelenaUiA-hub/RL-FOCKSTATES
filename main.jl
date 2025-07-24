using Revise 


include("RL_PPO.jl")

# --- PPO Hyperparameters 
BATCH_SIZE = 4096
N_TRANSITIONS_PER_ENV = 400   #era 256
N_UPDATE_EPOCHS = 5
GAMMA = 0.99 
LAMBDA = 0.95
CLIP_RANGE = 0.1 #provare 0.3 forse??
ENTROPY_LOSS_WEIGHT = 0.1 
CRITIC_LOSS_WEIGHT = 0.5 #era 0.5
MAX_GRAD_NORM = 0.5 
LR_ACTOR = 3e-4 # Learning rate for the actor network #MI RACCOMANDO MARI, I DUE LR MAI DIVERSI TANTO!!
LR_CRITIC = 3e-4 # Learning rate for the critic network
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
    n_update_epochs = N_UPDATE_EPOCHS,
    mini_batch_size = BATCH_SIZE,
    rng = rng
    
)

println(agent)



total_rewards = [] 

const MAX_STEPS_PER_EPISODE = 400

function main_training_loop(env::QuantumEnv, agent::PPOAgent, num_episodes::Int)
    # Get state and action dimensions for initializing actor/critic
    #state_dim = length(RLBase.state(env))
    #action_dim = length(RLBase.action_space(env))

    # Initialize Actor and Critic if not already done
    # (This assumes the agent is passed uninitialized or needs re-initialization for a new run)
    # If agent is already initialized, these lines would be redundant or need to be handled differently.
    # For a fresh start, you'd typically do:
    # actor = Actor(state_dim, action_dim)
    # critic = Critic(state_dim)
    # agent = PPOAgent(actor, critic; n_rollout=2048, n_update_epochs=10, mini_batch_size=64)

    # Initialize variables for tracking training progress
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


num_training_episodes = 10000

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

tspan = (0,Δt)

evolution_SE(best_actions_list[1],ψ0, tspan)



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



plot(exp_values,
     label="exp values n",
     xlabel="step",
     ylabel="n",
     title="")

exp_values[end]


plot(exp_values_q,
     label="exp values q",
     xlabel="step",
     ylabel="q",
     title="")
  

function main_training_loop_dual_env(env1::QuantumEnv, env2::QuantumEnv, agent::PPOAgent, num_episodes::Int)
    episode_rewards = Float64[]
    episode_fidelities = Float64[]
    total_global_steps_taken = 0 # Rename to avoid confusion with env.current_step

    best_fidelity = 0.0
    best_actions = []

    println("Starting PPO training for $(num_episodes) episodes with two environments...")

    for episode_num in 1:num_episodes # This loop now truly represents distinct episodes

        # --- IMPORTANT: RESET EPISODE-SPECIFIC ACCUMULATORS HERE ---
        episode_reward_env1 = 0.0
        episode_fidelity_env1 = 0.0 # This will store final fidelity for logging
        current_episode_actions1 = []

        episode_reward_env2 = 0.0
        episode_fidelity_env2 = 0.0 # This will store final fidelity for logging
        current_episode_actions2 = []
        # -----------------------------------------------------------

        # Reset environments for a NEW EPISODE
        current_state1 = RLBase.reset!(env1)
        current_state2 = RLBase.reset!(env2)
        done1 = false
        done2 = false

        # Inner loop: Steps within the current episode for both environments
        # We need to run environments independently until *both* are done.
        # This can be tricky with fixed-length `env.max_steps`.
        # Simplest approach for now is to run for env.max_steps, and let `done` handle early termination.

        # Store transitions for this episode's update
        # You're using `store_transition!` which likely manages a fixed-size buffer
        # This is okay, but we need to ensure the update happens based on the agent's logic.

        # Let's use a simpler loop for clarity, assuming `env.max_steps` is the max per env per "episode"
        for step_in_episode = 1:env1.max_steps # Assuming both envs have the same max_steps
            
            # If an environment already finished, it remains done for the rest of this episode's steps,
            # but its reward isn't further accumulated (if handled correctly).
            # However, you still need to provide an action for the agent to learn.
            # A common pattern is to "mask" done environments or let them take "no-op" actions.
            # For simplicity here, we'll continue stepping them.

            # Select actions for both environments
            # Note: You might want to sample actions only if the env is NOT done.
            # However, PPO's update mechanism usually expects a full rollout.
            # The `done` flag in `store_transition!` handles this.
            action1, log_prob1, value1 = select_action(agent, current_state1)
            action2, log_prob2, value2 = select_action(agent, current_state2)
            
            action1 = convert(Vector{Float64}, action1)
            action2 = convert(Vector{Float64}, action2)

            # Store actions only if the environment is still active
            if !done1
                push!(current_episode_actions1, action1)
            end
            if !done2
                push!(current_episode_actions2, action2)
            end

            # Step both environments (only if not done, or allow step! to handle it)
            # Your `step!` function inherently handles `env.current_step >= env.max_steps`
            # and sets `done`, so this is fine.
            reward1, new_done1 = step!(env1, action1)
            reward2, new_done2 = step!(env2, action2)

            # Accumulate reward only if environment was not done before this step
            if !done1
                episode_reward_env1 += reward1
            end
            if !done2
                episode_reward_env2 += reward2
            end

            # Update done flags
            done1 = new_done1
            done2 = new_done2

            # Store transitions for both environments
            # Crucially, `done` flag here tells the agent if this transition is terminal.
            store_transition!(agent, current_state1, action1, reward1, done1, log_prob1, value1)
            store_transition!(agent, current_state2, action2, reward2, done2, log_prob2, value2)

            # Update current states
            current_state1 = RLBase.state(env1)
            current_state2 = RLBase.state(env2)

            total_global_steps_taken += 2 # Two actual steps in total per inner loop iteration

            # If both environments are done, break the inner loop (current episode for both is finished)
            if done1 && done2
                break
            end
        end # End of inner loop (steps within an episode)

        # --- AFTER EACH COMPLETE EPISODE (or when max steps reached for it) ---
        # Final fidelity for current episode
        episode_fidelity_env1 = abs(env1.target_state' * env1.current_state)^2
        episode_fidelity_env2 = abs(env2.target_state' * env2.current_state)^2

        # Update best fidelity/actions if needed (outside the inner loop)
        if episode_fidelity_env1 > best_fidelity
            best_fidelity = episode_fidelity_env1
            best_actions = copy(current_episode_actions1)
        end
        if episode_fidelity_env2 > best_fidelity
            best_fidelity = episode_fidelity_env2
            best_actions = copy(current_episode_actions2)
        end

        # Push the total rewards and final fidelities for this *completed* episode
        # You need to decide how to handle episode_rewards for two environments.
        # Option 1: Store them separately.
        # Option 2: Store an average or sum if you want one list.
        # For logging, you're currently storing each separately when `done` is true,
        # but the `episode_rewards` and `episode_fidelities` list will have mixed results.
        # Let's refine the logging.
        push!(episode_rewards, episode_reward_env1)
        push!(episode_rewards, episode_reward_env2) # Add both
        push!(episode_fidelities, episode_fidelity_env1)
        push!(episode_fidelities, episode_fidelity_env2) # Add both

        # Update agent if enough data is collected (usually done outside the step loop for PPO)
        # This block might be better placed here, after a full episode (or a batch of episodes) is run
        # but still within the outer episode_num loop, if your `ready_to_update` logic uses buffer size.
        if ready_to_update(agent)
            # next_value calculation for GAE:
            # When an episode ends naturally, the last state has a value of 0.0 (done=true).
            # If the episode is truncated (max_steps reached but not success),
            # the value of the final state is estimated by the critic.
            # Your `update!` function should internally handle GAE calculation based on the `done` flags
            # stored in `store_transition!`. So, you don't necessarily need to pass `next_value` here
            # unless `update!` requires it for its internal logic on the *last* state of the *entire batch*.
            # For simplicity, if your update is on the collected transitions, it will use the `done` flag there.
            update!(agent) # Call update without passing next_value if it's handled internally
        end

        # Print episode summary at the end of EACH logical episode
        if length(episode_rewards) >= 2 # Ensure at least one pair of env results
            # To average over last 10 *completed episodes*, need to adjust indexing
            # If episode_rewards stores pairs (env1, env2), then need `end-19:end` for last 10 pairs.
            avg_reward_last_10 = mean(episode_rewards[max(1, end-19):end]) # Assuming 2 rewards per logical episode
            avg_fidelity_last_10 = mean(episode_fidelities[max(1, end-19):end])
            
            println("Episode $(episode_num)/$(num_episodes) | Total Global Steps: $(total_global_steps_taken) | Last Rewards: $(round(episode_reward_env1, digits=4)), $(round(episode_reward_env2, digits=4)) | Last Fidelities: $(round(episode_fidelity_env1, digits=4)), $(round(episode_fidelity_env2, digits=4)) | Avg Reward (last 10 completed, overall): $(round(avg_reward_last_10, digits=4)) | Avg Fidelity (last 10 completed, overall): $(round(avg_fidelity_last_10, digits=4))")
        end

        # Reset the environments *once* at the beginning of the next episode loop iteration.
        # The `RLBase.reset!` at the top of the outer loop will handle this.
    end # End of outer loop (episodes)

    println("Training finished!")
    return episode_rewards, episode_fidelities, best_actions
end


empty!(agent.buffer) # Svuota il buffer dell'esperienza
RLBase.reset!(env1) 
RLBase.reset!(env2) 

env1 = QuantumEnv(N_cut_off) # Your first environment
env2 = QuantumEnv(N_cut_off) # Your second environment

# Initialize your agent as before
state_dim = length(RLBase.state(env1)) # Or env2, should be the same
action_dim = length(RLBase.action_space(env1))

actor = Actor(state_dim, action_dim)
critic = Critic(state_dim)

agent = PPOAgent(
    actor,
    critic;
    actor_optimizer = Flux.Optimise.Adam(LR_ACTOR),
    critic_optimizer = Flux.Optimise.Adam(LR_CRITIC),
    gamma = GAMMA,
    lambda = LAMBDA,
    clip_range = CLIP_RANGE,
    max_grad_norm = 1.0,
    n_rollout=4096,
    n_env = 2,
    n_update_epochs = N_UPDATE_EPOCHS,
    mini_batch_size = BATCH_SIZE,
    rng = rng
    
    
)





num_training_episodes = 10000

rewards_history, fidelities_history, best_actions_list = main_training_loop_dual_env(env1, env2, agent, num_training_episodes)

if !isempty(best_actions_list)
    println("Numero di step di controllo: ", length(best_actions_list))
    println("Parametri di controllo [Δ, Ω] per ogni step:")
    for (i, action) in enumerate(best_actions_list)
        println("Step $i: Δ = $(action[1]), Ω = $(action[2])")
    end
    
    # The message "Nessuna strategia con fedeltà > 0 è stata trovata."
    # should NOT be here, as we clearly found one if best_actions_list is not empty.
    println("\nLa migliore strategia trovata ha raggiunto una fedeltà di: $(round(best_fidelity, digits=4))") # Optional: print the best fidelity
else # If best_actions_list IS empty
    println("Nessuna strategia con fedeltà > 0 è stata trovata.")
end