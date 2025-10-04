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
THR_LADDER = [ 0.45,0.50,0.55, 0.60,0.65,0.70 ,0.75,0.76,0.78,0.80,0.83,0.85,0.86,0.87,0.88,0.89,0.90,0.905,0.91,0.915,0.920,0.925,0.930,0.940,0.950,0.955,0.96,0.97,0.975,0.980,0.985,0.990,0.991,0.992,0.995,0.996,0.997,0.998,0.999,0.9992,0.9993,0.9994,0.9995,0.9996,0.9997,0.9998,0.9999];
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

N_ENV = 8;
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

actor  = Actor(state_dim, action_dim) |> Flux.f64;
critic = Critic(state_dim)            |> Flux.f64;

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
                    actions[i]      = Vector{Float64}(a)   # forza il tipo corretto
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

        # --- metriche episodio ---
        # Caso 1: target salvato come density matrix (consigliato)
        fidelities = [clamp(real(QuantumOpticsBase.fidelity(env.current_state, env.target_state)), 0.0, 1.0) for env in envs]

        # Caso 2 (alternativo): se hai `env.target_ket`
        # fidelities = [clamp(real(expect(env.current_state, env.target_ket)), 0.0, 1.0) for env in envs]

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


num_episodes = 1000;
envs = create_envs(N_ENV, N_cut_off);
all_rewards, all_fidelities, best_actions = main_training_loop_parallel(envs, agent, num_episodes)






@save "not_unitary/plots & data//results2.jld2" all_rewards all_fidelities best_actions   








qub, mech, ops = Qubit_HO(N_cut_off, :FockBasis, 1//2);

"""
Simula una sequenza di azioni usando la stessa fisica/parametrizzazione di `step!`.

Ritorna:
- ρ_solution :: Vector{Operator}    # traiettoria degli stati (incluso solo il primo stato iniziale una volta)
- exp_values :: Vector{Float64}     # ⟨n_mech⟩ lungo la traiettoria (senza il punto duplicato)
- exp_values_q :: Vector{Float64}   # ⟨n_qubit⟩ lungo la traiettoria (senza il punto duplicato)
"""
function simulate_with_actions_step!(best_actions::Vector,
                                     ψ_init::Ket,
                                     ops;
                                     g::Real,
                                     γm::Real,
                                     κϕ::Real,
                                     κ::Real,
                                     nthm::Real,
                                     Δt::Float64 = 0.3e-5)

    # stato iniziale e tempo
    ρ  = dm(ψ_init)
    t0 = 0.0

    # output
    ρ_solution   = Operator[]   # include lo stato iniziale una sola volta
    exp_values   = Float64[]     # ⟨n_mech⟩
    exp_values_q = Float64[]     # ⟨n_qubit⟩

    
    n_mech_op  = ops.Iad * ops.Ia        # a†a sul modo meccanico (⊗ I_qubit)
    n_qubit_op = ops.pI  * ops.mI        # σ⁺σ⁻ = |1⟩⟨1| sul qubit (⊗ I_osc)

    for a in best_actions
        # --- controlli come in step! ---
        a1 = Float64((a[1]+1)/2)
        a2 = Float64(a[2])

        Δ_max = 1e4
        Ω_max = 1e3
        Δ = Δ_max * a1
        Ω = Ω_max * a2

        # Hamiltoniana: JC + drive X/Z con le stesse scalature
        H_JC = g/Δ_max * (ops.Iad * ops.mI + ops.Ia * ops.pI)

        Ω_(t) = Ω / Δ_max
        Δ_(t) = Δ / Δ_max

        Ht = LazySum([Ω_(0.0), Δ_(0.0)], [ops.xI, ops.zI])
        function Hamiltonian(t, ψ)
            Ht.factors[1] = Ω_(t)
            Ht.factors[2] = Δ_(t) / 2
            return H_JC + Ht
        end

        # dissipatori identici a step!
        diss = [sqrt(κϕ/2)  * ops.zI, sqrt(κ) * ops.mI, sqrt(γm * (nthm+1))*ops.Ia, sqrt(γm*(nthm))*ops.Iad]
         #diss_dag = dagger.([sqrt(γm * (nthm+1))*ops.Ia, sqrt(γm*(nthm))*ops.Iad, sqrt(κϕ/2)  * ops.zI, sqrt(κ) * ops.mI])
        diss_dag = dagger.(diss)
        

        dynamics_input = (t, ψ) -> (Hamiltonian(t, ψ), diss, diss_dag)

        # evoluzione su [t0, t1]
        t1 = t0 + Δt
        t, sol = timeevolution.master_dynamic((t0, t1), ρ, dynamics_input;
                                              adaptive = true, reltol = 1e-9, abstol = 1e-13)

        ρ_series = sol

        # --- FIX 1: salta il primo punto (duplicato del finale precedente) ---
        @inbounds begin
            for i in Base.Iterators.drop(eachindex(ρ_series), 1)  # salta il primo
                push!(ρ_solution, ρ_series[i])
                push!(exp_values,   real(expect(n_mech_op, ρ_series[i])))
                push!(exp_values_q, real(expect(n_qubit_op, ρ_series[i])))
            end
        end
        
        ρ  = ρ_series[end] / tr(ρ_series[end])
        t0 = t1
    end

    return ρ_solution, exp_values, exp_values_q
end



Δ_max =  1e4
κϕ =  20.0  / Δ_max
κ  =  19.0    / Δ_max
γm =  15/ Δ_max



Teq   = kb / (2 * pi * 1.054571817e−34) * 1e-3 * 10e-3
nthm  = 1 / (exp(ωm / Teq) - 1)



# stati/basi per costruire ρ0 e target
ψ_init = tensor(spindown(qub.basis), fockstate(mech.basis, 0));
ρ_sol, n_mech_traj, n_qubit_traj = simulate_with_actions_step!(best_actions,
                                     ψ_init,
                                     ops;
                                     g,
                                     γm,
                                     κϕ,
                                     κ,
                                     nthm,
                                     Δt=3e-5*Δ_max);


ψ_target = tensor(spindown(qub.basis), fockstate(mech.basis, N_mech));


# varianza del numero meccanico all'ultimo stato
var = real(expect(ops.n_mech*ops.n_mech, ρ_sol[end]) - expect(ops.n_mech, ρ_sol[end])^2)

# fedeltà finale rispetto al target puro (usa pure–mixed)
final_fid = clamp(real(expect(ρ_sol[end], ψ_target)), 0.0, 1.0)
println("Fidelity finale = ", final_fid)

# plotting
plot_mech = plot(n_mech_traj;   label="⟨n_mech⟩",  xlabel="step", ylabel="value", title="Mechanics occupancy");
plot_q    = plot(n_qubit_traj; label="⟨n_qubit⟩", xlabel="step", ylabel="value", title="Qubit excitation");

display(plot(plot_mech, plot_q, layout=(2,1), size=(1000,800)))

savefig("unitary/plots & data/plot_4.pdf")
savefig(plot_mech,"unitary/plots & data/plot_mech_4.pdf")
savefig(plot_q,"unitary/plots & data/plot_q_4.pdf")

p = plot(n_mech_traj; label="⟨n_mech⟩", xlabel="step", ylabel="value",
         title="Occupancy", legend=:outertopright, size=(1200,800),
         legendtitle="Fidelity finale = $(round(final_fid; digits=10))", grid=true);
plot!(p, n_qubit_traj; label="⟨n_qubit⟩")
savefig(p, "unitary/plots & data/occupancy_4.pdf")

# ultimi valori
last_n_mech  = n_mech_traj[end]
last_n_qubit = n_qubit_traj[end]


# limiti (in kHz)
Δ_max =  1e4
Ω_max =  1e3





function plot_best_controls(best_actions::Vector; ωm, Δ_max, Ω_max)
    isempty(best_actions) && (@warn "best_actions è vuoto"; return nothing)
    a_mat = hcat([Float64.(vec(a)) for a in best_actions]...)
    T = size(a_mat, 2)
    steps = 1:T

    a1 = float(a_mat[1, :])   
    a1 = (a1 .+1 )./2             # in [-1,1]
    a2 = float(a_mat[2, :])
    Δ = Δ_max               # kHz
    Ω = Ω_max               # kHz

    p = plot(layout=(2,1), link=:x, size=(1200,800))
    plot!(p[1], steps, a1, xlabel="step", ylabel="Δ [Hz]", legend=false, grid=true, framestyle=:box)
    plot!(p[2], steps, a2,  xlabel="step", ylabel="Ω  [Hz]", legend=false, grid=true, framestyle=:box)
    display(p)
    return p
end





plot_best_actions = plot_best_controls(best_actions; ωm=ωm, Δ_max=Δ_max, Ω_max=Ω_max)

savefig(plot_best_actions,"plots & data/best_actions_1.pdf")

a = 1-final_fid




plot_f = plot(all_fidelities;
    label="Fidelity",
    title="Fidelity")


