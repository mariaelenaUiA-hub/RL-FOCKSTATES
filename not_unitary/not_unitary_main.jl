using Flux.Optimisers: OptimiserChain, ClipNorm, Adam
using Revise
using Base.Threads
using Plots
using JLD2
plotlyjs()
using Statistics
using StableRNGs
using Flux



N_cut_off = 6;
N_mech    = 1;
g         = 258.0;
global ωm = 5.9614e6;

include("not_unitary_RL_PPO.jl")




# --- PPO Hyperparameters 
BATCH_SIZE = 64;
LAST_BUMP_EP = Ref(0)
THR_LADDER = [ 0.80,0.90,0.905,0.91,0.915,0.920,0.925,0.930,0.940,0.950,0.955,0.970,0.980,0.985,0.990,0.991,0.992,0.995,0.996,0.997,0.998,0.999,0.9992,0.9993,0.9994,0.9995,0.9996,0.9997,0.9998,0.9999];
THR_IDX      = Ref(1) ;
SUCCESS_THR  = Ref(THR_LADDER[THR_IDX[]]);
N_UPDATE_EPOCHS = 4;
GAMMA = 0.99 ;
LAMBDA = 0.95;
CLIP_RANGE = 0.2 #provare 0.3 forse??;
ENTROPY_LOSS_WEIGHT = 0.02 ;
CRITIC_LOSS_WEIGHT = 0.5 #era 0.5;
MAX_GRAD_NORM = 0.5 ;
LR_ACTOR = 0.5e-5; # Learning rate for the actor network #MI RACCOMANDO MARI, I DUE LR MAI DIVERSI TANTO!!
LR_CRITIC = 0.5e-5 ;# Learning rate for the critic network

N_ENV = 2;
N_ROLLOUT = N_ENV* 500
n_envs = N_ENV;
# --- 

function create_envs(N_env::Int, N_cut::Int)
    [QuantumEnv(N_cut) for _ in 1:N_env]
end

function reset_envs!(envs::Vector{QuantumEnv})
    states = Vector{Any}(undef, length(envs))
    Threads.@threads for i in 1:length(envs)
        states[i] = RLBase.reset!(envs[i])
    end
    return states
end

env = QuantumEnv(N_cut_off) ;

state_dim  = length(RLBase.state_space(env))   ;  # = 2*d^2
action_dim = length(RLBase.action_space(env))   ; # = 2

rng = StableRNG(123)

actor  = Actor(state_dim, action_dim) |> Flux.f64
critic = Critic(state_dim)            |> Flux.f64

agent = PPOAgent(
    actor,
    critic;
    actor_optimizer  = Flux.Optimise.Adam(LR_ACTOR),
    critic_optimizer = Flux.Optimise.Adam(LR_CRITIC),
    gamma = GAMMA,
    lambda = LAMBDA,
    clip_range = CLIP_RANGE,
    entropy_loss_weight = ENTROPY_LOSS_WEIGHT,
    critic_loss_weight = CRITIC_LOSS_WEIGHT,
    max_grad_norm = MAX_GRAD_NORM,
    n_rollout = N_ROLLOUT,
    n_env = N_ENV,
    n_update_epochs = N_UPDATE_EPOCHS,
    mini_batch_size = BATCH_SIZE,
    rng = rng
);

# clip dei gradienti (come nel tuo)
agent.actor_optimizer  = OptimiserChain(ClipNorm(MAX_GRAD_NORM), Adam(LR_ACTOR));
agent.critic_optimizer = OptimiserChain(ClipNorm(MAX_GRAD_NORM), Adam(LR_CRITIC));
reset_opt_states!(agent);

envs = create_envs(N_ENV, N_cut_off);
function main_training_loop_parallel(envs::Vector{QuantumEnv}, agent::PPOAgent, num_episodes::Int)
    n_env = length(envs)

    episode_rewards    = Float64[]
    episode_fidelities = Float64[]
    sr_hist            = Float64[]
    best_fidelity = 0.0
    best_actions  = Vector{Any}()

    println("Starting PPO (Lindblad) for $num_episodes episodes on $n_env envs...")

    for episode in 1:num_episodes
        states          = [RLBase.reset!(env) for env in envs]
        done_flags      = falses(n_env)
        rewards_this_ep = zeros(Float64, n_env)
        actions_this_ep = [Any[] for _ in 1:n_env]

        while !all(done_flags)
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

            # steppa solo gli attivi
            active_idx = findall(i -> !done_flags[i], 1:n_env)
            if !isempty(active_idx)
                envs_active    = envs[active_idx]
                actions_active = [actions[i] for i in active_idx]

                ns_act, r_act, d_act = step_envs!(envs_active, actions_active)

                new_states = copy(states)
                rewards    = zeros(Float64, n_env)
                dones      = copy(done_flags)

                @inbounds for (j,k) in enumerate(active_idx)
                    new_states[k] = ns_act[j]
                    rewards[k]    = r_act[j]
                    dones[k]      = d_act[j]
                end
            else
                new_states = states
                rewards    = zeros(Float64, n_env)
                dones      = done_flags
            end

            for i in 1:n_env
                if !done_flags[i]
                    store_transition!(agent, states[i], actions[i], rewards[i], dones[i],
                                      log_probs[i], values_state[i]; env_id=i)
                    push!(actions_this_ep[i], actions[i])
                    rewards_this_ep[i] += rewards[i]
                    done_flags[i] = dones[i]
                end
            end

            states = new_states

            if ready_to_update(agent)
                bootstrap = [done_flags[i] ? 0.0 : agent.policy.critic(states[i]) for i in 1:n_env]
                update!(agent; bootstrap_values_by_env=bootstrap)
            end
        end

        # flush finale
        if !isempty(agent.buffer.rewards)
            update!(agent; bootstrap_values_by_env=zeros(n_env))
        end

        # metriche episodio (Lindblad): F = ⟨ψ_t| ρ |ψ_t⟩
        fidelities = [clamp(real(expect(env.target_proj, env.current_state)), 0.0, 1.0) for env in envs]
        avg_fidelity  = mean(fidelities)
        max_fidelity, idx = findmax(fidelities)
        best_actions_episode = actions_this_ep[idx]

        if max_fidelity > best_fidelity
            best_fidelity = max_fidelity
            best_actions  = copy(best_actions_episode)
        end

        sr = count(>=(SUCCESS_THR[]), fidelities) / n_env
        push!(sr_hist, sr)
        push!(episode_rewards,    mean(rewards_this_ep))
        push!(episode_fidelities, avg_fidelity)

        maybe_bump_threshold!(episode_fidelities, sr_hist, agent; episode=episode)

        if episode % 10 == 0 && length(episode_rewards) >= 10
            recent_rewards    = episode_rewards[end-9:end]
            recent_fidelities = episode_fidelities[end-9:end]
            corr_val = Statistics.cor(recent_rewards, recent_fidelities)

            println("Ep $episode | AvgR=$(round(mean(recent_rewards);    digits=4)) | " *
                    "AvgF=$(round(mean(recent_fidelities); digits=4)) | " *
                    "Corr=$(round(corr_val;               digits=4)) | " *
                    "SR=$(round(sr;                       digits=2)) | Thr=$(SUCCESS_THR[]) | " *
                    "EntW=$(round(agent.entropy_loss_weight; digits=4))")
        else
            println("Ep $episode | AvgR=$(round(mean(rewards_this_ep); digits=4)) | " *
                    "AvgF=$(round(avg_fidelity;        digits=4)) | " *
                    "BestF(ep): $(round(max_fidelity;  digits=4)) | " *
                    "SR=$(round(sr;                    digits=2)) | Thr=$(SUCCESS_THR[]) | " *
                    "EntW=$(round(agent.entropy_loss_weight; digits=4))")
        end
    end

    println("Training finished! Best Fidelity = $best_fidelity")
    return episode_rewards, episode_fidelities, best_actions
end


num_episodes = 500;
envs = create_envs(N_ENV, N_cut_off);
all_rewards, all_fidelities, best_actions = main_training_loop_parallel(envs, agent, num_episodes)
