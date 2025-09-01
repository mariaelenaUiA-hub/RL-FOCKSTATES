using Revise 
using Base.Threads
plotlyjs()
using Statistics
N_cut_off = 6;
N_mech    = 1;
g         = 258.0;
global ωm = 5.9614e6;

include("RL_PPO_n_env.jl")

# --- PPO Hyperparameters 
BATCH_SIZE = 64;
LAST_BUMP_EP = Ref(0)
THR_LADDER = [ 0.80,0.90,0.905,0.91,0.915,0.920,0.925,0.930,0.940,0.950,0.955,0.970,0.980,0.985,0.990,0.991,0.992,0.995,0.996,0.997,0.998,0.999,0.9999];
THR_IDX      = Ref(1) ;
SUCCESS_THR  = Ref(THR_LADDER[THR_IDX[]]);
N_UPDATE_EPOCHS = 4;
GAMMA = 0.99 ;
LAMBDA = 0.95;
CLIP_RANGE = 0.1 #provare 0.3 forse??;
ENTROPY_LOSS_WEIGHT = 0.02 ;
CRITIC_LOSS_WEIGHT = 0.5 #era 0.5;
MAX_GRAD_NORM = 0.5 ;
LR_ACTOR = 0.5e-4; # Learning rate for the actor network #MI RACCOMANDO MARI, I DUE LR MAI DIVERSI TANTO!!
LR_CRITIC = 0.5e-4 ;# Learning rate for the critic network
N_ROLLOUT = 8* 500
N_ENV = 8;
n_envs = N_ENV;
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


env = QuantumEnv(N_cut_off);

state_dim = length(RLBase.state_space(env)) ;
action_dim = length(RLBase.action_space(env)) ;

rng = StableRNG(123) ;

actor  = Actor(state_dim, action_dim); 
critic = Critic(state_dim); 
policy = PPOPolicy(actor, critic) ;

actor = Flux.f64(actor);
critic = Flux.f64(critic);

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
    
);

using Flux.Optimisers: OptimiserChain, ClipNorm, Adam
agent.actor_optimizer  = OptimiserChain(ClipNorm(0.5), Adam(LR_ACTOR));
agent.critic_optimizer = OptimiserChain(ClipNorm(0.5), Adam(LR_CRITIC));

reset_opt_states!(agent);
envs = create_envs(N_ENV,N_cut_off);

function main_training_loop_parallel(envs::Vector{QuantumEnv}, agent::PPOAgent, num_episodes::Int)
    n_env = length(envs)

    episode_rewards    = Float64[]
    episode_fidelities = Float64[]
    sr_hist            = Float64[]  
    best_fid_global = 0.0        # tiene traccia dello SR per episodio

    best_fidelity = 0.0
    best_actions  = Vector{Any}()

    println("Starting PPO training for $num_episodes episodes on $n_env environments...")

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
                    # placeholder: non verrà usato perché non steppiamo i done
                    actions[i]      = [0.0, 0.0]
                    log_probs[i]    = 0.0
                    values_state[i] = 0.0
                end
            end

            # ======= NUOVO: steppa solo gli env attivi (non-done) =======
            active_idx = findall(i -> !done_flags[i], 1:n_env)
            if !isempty(active_idx)
                envs_active    = envs[active_idx]
                actions_active = [actions[i] for i in active_idx]

                ns_act, r_act, d_act = step_envs!(envs_active, actions_active)

                # ricostruisci vettori full senza toccare i done
                new_states = copy(states)
                rewards    = zeros(Float64, n_env)
                dones      = copy(done_flags)

                @inbounds for (j, k) in enumerate(active_idx)
                    new_states[k] = ns_act[j]
                    rewards[k]    = r_act[j]
                    dones[k]      = d_act[j]
                end
            else
                # tutti done: mantieni stato/reward e uscirai dal while
                new_states = states
                rewards    = zeros(Float64, n_env)
                dones      = done_flags
            end
            # ======= FINE BLOCCO =======

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

        # flush finale se restano transizioni nel buffer
        if !isempty(agent.buffer.rewards)
            update!(agent; bootstrap_values_by_env=zeros(n_env))
        end

        # metriche episodio
        fidelities    = [abs2(env.target_state' * env.current_state) for env in envs]
        avg_fidelity  = mean(fidelities)
        max_fidelity, idx = findmax(fidelities)
        best_actions_episode = actions_this_ep[idx]

        if max_fidelity > best_fidelity
            best_fidelity = max_fidelity
            best_actions  = copy(best_actions_episode)
        end

        # Success rate rispetto alla soglia corrente
        sr = count(>=(SUCCESS_THR[]), fidelities) / n_env
        push!(sr_hist, sr)

        push!(episode_rewards,    mean(rewards_this_ep))
        push!(episode_fidelities, avg_fidelity)
        ep_best = maximum(episode_fidelities)
        best_fid_global = max(best_fid_global, ep_best)
        # logica di bump invariata
        maybe_bump_threshold!(episode_fidelities, sr_hist, agent;
                            best_fid=best_fid_global,
                            episode=episode,
                            
                               
                                
                                
                            )
        if episode % 10 == 0 && length(episode_rewards) >= 10
            recent_rewards    = episode_rewards[end-9:end]
            recent_fidelities = episode_fidelities[end-9:end]
            corr_val = Statistics.cor(recent_rewards, recent_fidelities)

            println("Episode $episode/$num_episodes | AvgR: $(round(mean(recent_rewards),  digits=4)) | " *
                    "AvgF: $(round(mean(recent_fidelities), digits=4)) | Corr(R,F)=$(round(corr_val, digits=4)) | " *
                    "SR: $(round(sr, digits=2)) | Thr=$(SUCCESS_THR[]) | EntW=$(round(agent.entropy_loss_weight, digits=4))")
        else
            println("Episode $episode | AvgR: $(round(mean(rewards_this_ep), digits=4)) | " *
                    "AvgF: $(round(avg_fidelity, digits=4)) | BestF(ep): $(round(max_fidelity, digits=4)) | " *
                    "SR: $(round(sr, digits=2)) | Thr=$(SUCCESS_THR[]) | EntW=$(round(agent.entropy_loss_weight, digits=4))")
        end
    end

    println("Training finished! Best Fidelity = $best_fidelity")
    return episode_rewards, episode_fidelities, best_actions
end


num_episodes = 3000;
envs = create_envs(N_ENV, N_cut_off);
all_rewards, all_fidelities, best_actions = main_training_loop_parallel(envs, agent, num_episodes)


using JLD2

@save "plots & data/results_2_0.9986.jld2" all_rewards all_fidelities best_actions

function evolution_step_from_action(a::Tuple{<:Real,<:Real}, ψ0::Ket, tspan::Tuple{Float64,Float64})
    a1, a2 = a
    a1 = clamp(a[1], -1.0, 1.0)
    a2 = clamp(a[2], -1.0, 1.0)
    
    Δ_max = 10e4      
    Ω_max = 10e3      # ≈520 kHz (o 1.04e3 se xI=σx/2)
   
    Δ  = Δ_max *a1  
    Ω  = Ω_max *a2          # kHz

    
    

    H0      = dense((Δ/2.0) * HBAR_qubit.zI)
    H_drive = dense(Ω      * HBAR_qubit.xI)
    H_JC    = dense(g * (HBAR_qubit.Iad * HBAR_qubit.mI + HBAR_qubit.Ia * HBAR_qubit.pI))

    # Hamiltoniana (time-independent qui, ma la firma accetta H(t,ψ))
    H(t, ψ) =  2 * π * 1e-3 * (H0 + H_drive + H_JC)

    ts, ψt = timeevolution.schroedinger_dynamic(tspan, ψ0, H )
    exp_mech = expect(HBAR_qubit.n_mech,  ψt)
    exp_qub  = expect(HBAR_qubit.n_qubit, ψt)
    return real.(exp_mech), real.(exp_qub), ψt
end


# ----------------- rollout deterministico con best_actions -----------------
t0 = 0.0;

Δt = 0.3e-2  ;  

ψ0 = tensor(spindown(qub.basis), fockstate(mech.basis, 0));

solution     = Ket[ψ0];
exp_values   = Float64[0.0];
exp_values_q = Float64[0.0];

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

var = real(expect(HBAR_qubit.n_mech*HBAR_qubit.n_mech,solution[end]) - expect(HBAR_qubit.n_mech,solution[end])^2)


target = tensor(spindown(qub.basis), fockstate(mech.basis, N_mech));

fid = abs2(dot(target.data, solution[end].data));
println("Fidelity finale = ", fid);

plot_mech = plot(exp_values;
    label="⟨n_mech⟩",
    xlabel="step",
    ylabel="value",
    title="Mechanics occupancy");

plot_q = plot(exp_values_q;
    label="⟨n_qubit⟩",
    xlabel="step",
    ylabel="value",
    title="Qubit excitation");

plot_tot = display(plot(plot_mech, plot_q, layout=(2,1), size=(1000,800)))
savefig("plots & data/plot_2.pdf")
savefig(plot_mech,"plots & data/plot_mech_2.pdf")
savefig(plot_q,"plots & data/plot_q_2.pdf")

 
steps = length(exp_values_q)


p = plot( exp_values;
     label="⟨n_mech⟩",
     xlabel="step",
     ylabel="value",
     title="Occupancy",
     legend=:outertopright,
      size=(1200,800),
      legendtitle="Fidelity = 0.99037490",
      grid=true);

plot!(p, exp_values_q; label="⟨n_qubit⟩")

savefig(p, "plots & data/occupancy_2.pdf")





exp_values[end]


exp_values_q[end]


function plot_best_controls(best_actions::Vector; ωm, Δ_max, Ω_max)
    isempty(best_actions) && (@warn "best_actions è vuoto"; return nothing)
    a_mat = hcat([Float64.(vec(a)) for a in best_actions]...)
    T = size(a_mat, 2)
    steps = 1:T

    a1 = a_mat[1, :]                # in [-1,1]
    a2 = a_mat[2, :]
    Δ = Δ_max .* a1         # kHz
    Ω = Ω_max .* a2                # kHz

    p = plot(layout=(2,1), link=:x, size=(1200,800))
    plot!(p[1], steps, Δ, xlabel="step", ylabel="Δ [Hz]", legend=false, grid=true, framestyle=:box)
    plot!(p[2], steps, Ω,  xlabel="step", ylabel="Ω  [Hz]", legend=false, grid=true, framestyle=:box)
    display(p)
    return p
end



Δ_max = 10e4   ;   
Ω_max = 10e3    ;   


plot_best_actions = plot_best_controls(best_actions; ωm=ωm, Δ_max=Δ_max, Ω_max=Ω_max)

savefig(plot_best_actions,"plots & data/best_actions_1.pdf")

a = 1-fid




plot_f = plot(all_fidelities;
    label="Fidelity",
    title="Fidelity")


