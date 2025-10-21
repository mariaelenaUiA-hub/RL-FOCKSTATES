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


global ωm = 5.9614e6;

include("not_unitary_RL_PPO.jl")


# --- PPO Hyperparameters 
BATCH_SIZE = 64;
LAST_BUMP_EP = Ref(0);
BEST_MED10   = Ref(0.0) ;  
LAST_DOWN_EP = Ref(0)   ;  
THR_LADDER = [0.55,0.56,0.57,0.58,0.59,0.60,0.61,0.62,0.63,0.65,0.70,0.73,0.75,0.76,0.78,0.80,0.83,0.85,0.86,0.87,0.88,0.89,0.90,0.905,0.91,0.915,0.920,0.925,0.930,0.940,0.945,0.950,0.955,0.96,0.965,0.97,0.975,0.980,0.985,0.990,0.991,0.992,0.995,0.996,0.997,0.998,0.999,0.9992,0.9993,0.9994,0.9995,0.9996,0.9997,0.9998,0.9999];
THR_IDX      = Ref(1) ;
SUCCESS_THR  = Ref(THR_LADDER[THR_IDX[]]);
N_UPDATE_EPOCHS = 4;
GAMMA = 0.995;
LAMBDA = 0.95;
CLIP_RANGE = 0.2 #provare 0.3 forse??;
ENTROPY_LOSS_WEIGHT = 0.02 ;
CRITIC_LOSS_WEIGHT = 0.5 #era 0.5;
MAX_GRAD_NORM = 0.5 ;
LR_ACTOR = 0.5e-4 #0.5e-4; # Learning rate for the actor network #MI RACCOMANDO MARI, I DUE LR MAI DIVERSI TANTO!!
LR_CRITIC = 0.5e-4 #0.5e-4 ;# Learning rate for the critic network

N_ENV = 8;
N_ROLLOUT = 1024
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

using BSON: @save


function main_training_loop_parallel(envs::Vector{QuantumEnv}, agent::PPOAgent, num_episodes::Int;
                                     save_path::String="best_agent.bson")
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
                    actions[i]      = Vector{Float64}(a)
                    log_probs[i]    = lp
                    values_state[i] = v
                else
                    actions[i]      = [0.0, 0.0]
                    log_probs[i]    = 0.0
                    values_state[i] = 0.0
                end
            end

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

        if !isempty(agent.buffer.rewards)
            update!(agent; bootstrap_values_by_env=zeros(n_env))
        end

        fidelities = [clamp(real(QuantumOpticsBase.fidelity(env.current_state, env.target_state)), 0.0, 1.0)
                      for env in envs]

        avg_fidelity  = mean(fidelities)
        max_fidelity, idx = findmax(fidelities)
        best_actions_episode = actions_this_ep[idx]

        if max_fidelity > best_fidelity
            best_fidelity = max_fidelity
            best_actions  = copy(best_actions_episode)
            @save save_path agent best_fidelity episode
            println("✨ New best fidelity = $(round(best_fidelity; digits=5)) saved to $(save_path)")
        end

        sr = count(>=(SUCCESS_THR[]), fidelities) / n_env
        push!(sr_hist, sr)
        push!(episode_rewards, mean(rewards_this_ep))
        push!(episode_fidelities, avg_fidelity)

        maybe_bump_threshold!(episode_fidelities, sr_hist, agent; episode=episode)

        if episode % 20 == 0
            println("Ep $episode | AvgF=$(round(avg_fidelity; digits=4)) | BestF=$(round(best_fidelity; digits=4)) | SR=$(round(sr; digits=2))")
        end
    end

    println("\n✅ Training finished! Best Fidelity = $(round(best_fidelity; digits=5))")
    println("Best policy saved to $(save_path)")

    return (episode_rewards, episode_fidelities, best_actions, best_fidelity)
end


num_episodes = 350;
envs = create_envs(N_ENV, N_cut_off);
all_rewards, all_fidelities, best_actions = main_training_loop_parallel(envs, agent, num_episodes; save_path="not_unitary/plots&data/g2e3/best_agent_1_g2e3.bson")



#popopo






@save "not_unitary/plots&data/g2e3/results_1.JLD2" all_rewards all_fidelities best_actions   




Δ_max =  1e5
g  = 358*2*pi
#g=1e5
g  =  g/Δ_max
κϕ =  0.25 / Δ_max
κ  =  19 / Δ_max
γm =  0.025 / Δ_max

kb = 1.3806488e-23
hbar_= 1.054571817e-34

Teq   = 1e-2
nthm  = 1 / (exp((ωm*1e3*hbar_) / (Teq*kb)) - 1)




qub, mech, ops = Qubit_HO(N_cut_off, :FockBasis, 1//2);


function simulate_with_actions_step!(best_actions::Vector,
                                     ψ_init::Ket,
                                     ops;
                                     g::Real,
                                     γm::Real,
                                     κϕ::Real,
                                     κ::Real,
                                     nthm::Real,
                                     Δt::Float64)

    # stato iniziale e tempo
    ρ  = dm(ψ_init)
    t0 = 0.0

    
    n_mech_op  = ops.Iad * ops.Ia        # a†a sul modo meccanico (⊗ I_qubit)
    n_qubit_op = ops.pI  * ops.mI   

    
    ρ_solution   = Operator[ρ]   # include lo stato iniziale una sola volta
    exp_values   =  [real(expect( n_mech_op, ρ))]     # ⟨n_mech⟩
    exp_values_q = [real(expect(  n_qubit_op, ρ))]    # ⟨n_qubit⟩
     # σ⁺σ⁻ = |1⟩⟨1| sul qubit (⊗ I_osc)

    for a in best_actions
        # --- controlli come in step! ---
        #a1 = Float64((a[1]+1)*0.55 .- 0.1)
        a1=(a[1]+1)/2
        a2 = Float64(a[2])


        Δ = a1
        Ω = a2

        # Hamiltoniana: JC + drive X/Z con le stesse scalature
        H_JC = g * (ops.Iad * ops.mI + ops.Ia * ops.pI)

        Ω_(t) = Ω 
        Δ_(t) = Δ 

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
                                     Δt=5e-2);


ψ_target = tensor(spindown(qub.basis), fockstate(mech.basis, N_mech));



var = real(expect(ops.n_mech*ops.n_mech, ρ_sol[end]) - expect(ops.n_mech, ρ_sol[end])^2)
V = sqrt(var)


final_fid = real(QuantumOpticsBase.fidelity(ρ_sol[end], dm(ψ_target)))
println("Fidelity finale = ", final_fid)


p = plot(n_mech_traj; label="⟨n_mech⟩", xlabel="step",
          legend=:outertopright, size=(1200,800),
         legendtitle="Best Fidelity = $(round(final_fid; digits=5))", grid=true,ylims=(0,1.1),
         marker=:circle,       
        markersize=0.5,         
        line=:solid,          
        linewidth=3,
        framestyle=:box);
plot!(p, n_qubit_traj; label="⟨n_qubit⟩",
         marker=:circle,       
        markersize=0.5,         
        line=:solid,         
        linewidth=3,
        framestyle=:box)
savefig(p, "not_unitary/plots&data/g8e4/plot_1.pdf")

using Plots, Statistics
using PlotThemes

function plot_populations(n_mech_traj, n_qubit_traj;
                          legend_title = "",
                          markers_every::Int = 25,
                          plot_size::Tuple{Int,Int} = (1200,800),
                          dpi::Int = 180,
                          show_means_in_legend::Bool = true)

    steps = 1:length(n_mech_traj)
    @assert length(n_mech_traj) == length(n_qubit_traj)

    # figura
    p = plot(steps, n_mech_traj;
             label="⟨n_mech⟩",
             xlabel="step", ylabel="population",
             legend=:outertopright, legendtitle=legend_title,
             size=plot_size, dpi=dpi, palette=:tab10,
             grid=true, gridalpha=0.25, framestyle=:box,
             lw=2.6, linealpha=0.95, ylims=(0,1.05))

    plot!(p, steps, n_qubit_traj;
          label="⟨n_qubit⟩", lw=2.6, linealpha=0.95)

    # marker radi
    idx = 1:max(markers_every,1):lastindex(steps)
    scatter!(p, steps[idx], n_mech_traj[idx]; ms=3, markerstrokewidth=0, label=false)
    scatter!(p, steps[idx], n_qubit_traj[idx]; ms=3, markerstrokewidth=0, label=false)

    # riempimento tra le curve (molto leggero)
    plot!(p, steps, n_mech_traj;
      lw=0, c=1, fillrange=0, fillalpha=0.15, label=false)

    plot!(p, steps, n_qubit_traj;
        lw=0, c=2, fillrange=0, fillalpha=0.15, label=false)

    # incrocio (interpolazione lineare del primo cambio di segno)
    d = n_mech_traj .- n_qubit_traj
    ix = findfirst(diff(sign.(d)) .!= 0)
    if ix !== nothing
        x1, x2 = steps[ix], steps[ix+1]
        y1, y2 = d[ix], d[ix+1]
        xc = x1 - y1*(x2 - x1)/(y2 - y1)  # stima continua
        vline!(p, [xc]; lw=1, linealpha=0.4, label=false)
    end

    # solo le medie in legenda (senza aggiungere nuove linee)
    if show_means_in_legend
        m1, m2 = mean(n_mech_traj), mean(n_qubit_traj)
        plot!(p, [NaN], [NaN]; label="mean mech = $(round(m1,digits=3))")
        plot!(p, [NaN], [NaN]; label="mean qubit = $(round(m2,digits=3))")
    end

    return p
end

p = plot_populations(n_mech_traj, n_qubit_traj;
     legend_title = "Best Fidelity = $(round(final_fid; digits=5))",
     markers_every = 30)


savefig(p, "not_unitary/plots&data/g2e3/plot_1.pdf")





# ultimi valori
last_n_mech  = n_mech_traj[end]
last_n_qubit = n_qubit_traj[end]

length(best_actions)





using Plots
using Statistics

function plot_best_controls(best_actions::Vector;
                            markers_every::Int = 10,
                            movavg_window::Int = 0,
                            plot_size::Tuple{Int,Int} = (1200, 800),
                            dpi::Int = 180,
                            mean_in_caption::Bool = true)

    # matrice delle azioni
    a_mat = hcat([Float64.(vec(a)) for a in best_actions]...)
    T = Base.size(a_mat, 2)
    steps = 1:T

    a1 = (a_mat[1, :] .+ 1) ./ 2      # in [0,1]
    a2 = a_mat[2, :]

    # marker radi
    idx = 1:max(markers_every,1):T

    # figura
    p = plot(layout=(2,1), link=:x, size=plot_size, dpi=dpi,
             framestyle=:box, legend=false, grid=true,
             gridalpha=0.25, guidefontsize=11, tickfontsize=9)

    # Δ / Δ_max (senza label)
    plot!(p[1], steps, a1; lw=1.8, linealpha=0.85, label=false)
    scatter!(p[1], steps[idx], a1[idx]; marker=:circle, ms=3,
             markeralpha=0.9, markerstrokewidth=0, label=false)
    xlabel!(p[1], "step"); ylabel!(p[1], "Δ / Δ_max")

    # Ω / Δ_max (senza label)
    plot!(p[2], steps, a2; lw=1.8, linealpha=0.85, label=false)
    scatter!(p[2], steps[idx], a2[idx]; marker=:circle, ms=3,
             markeralpha=0.9, markerstrokewidth=0, label=false)
    xlabel!(p[2], "step"); ylabel!(p[2], "Ω / Δ_max")

    # (opzionale) media mobile, ma non in legenda
    if movavg_window > 1
        movmean(v, w) = [mean(@view v[max(1,i-w+1):i]) for i in eachindex(v)]
        plot!(p[1], steps, movmean(a1, movavg_window); lw=2, color=:black, linealpha=0.5, label=false)
        plot!(p[2], steps, movmean(a2, movavg_window); lw=2, color=:black, linealpha=0.5, label=false)
    end

    # Solo la media in legenda
    #if mean_in_caption
        #m1 = mean(a1); m2 = mean(a2)
        #hline!(p[1], [m1]; lw=2, linealpha=0.7, label="average = $(round(m1, digits=3))")
        #hline!(p[2], [m2]; lw=2, linealpha=0.7, label="average = $(round(m2, digits=3))")
        #plot!(p; legend=true)
    #end

    display(p)
    return p
end



plot_best_actions=plot_best_controls(best_actions; markers_every=12, movavg_window=25)




savefig(plot_best_actions,"not_unitary/plots&data/g1e5/best_actions_1.pdf")

a = 1-final_fid




plot_f = plot(all_fidelities;
    label="Fidelity",
    title="Fidelity")

savefig(plot_f,"not_unitary/plots&data/g1e5/fidelity_1.pdf")
