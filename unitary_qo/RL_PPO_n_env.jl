using ReinforcementLearning, ReinforcementLearningCore
using .ReinforcementLearningBase
using Statistics , Flux , Functors , Flux.Optimise, StableRNGs ,  Random, Zygote, Distributions, LinearAlgebra, QuantumOptics, DifferentialEquations, DiffEqFlux, OrdinaryDiffEq
using Flux: Chain, Dense
using IntervalSets: ClosedInterval 
using Plots

struct qubit
    basis::SpinBasis{1//2, Int64}
    σm::QuantumOpticsBase.Operator{SpinBasis{1//2, Int64}, SpinBasis{1//2, Int64}, <:AbstractMatrix}
    σp::QuantumOpticsBase.Operator{SpinBasis{1//2, Int64}, SpinBasis{1//2, Int64}, <:AbstractMatrix}
    σx::QuantumOpticsBase.Operator{SpinBasis{1//2, Int64}, SpinBasis{1//2, Int64},<:AbstractMatrix}
    σy::QuantumOpticsBase.Operator{SpinBasis{1//2, Int64}, SpinBasis{1//2, Int64}, <:AbstractMatrix}
    σz::QuantumOpticsBase.Operator{SpinBasis{1//2, Int64}, SpinBasis{1//2, Int64}, <:AbstractMatrix}
    Id::QuantumOpticsBase.Operator{SpinBasis{1//2, Int64}, SpinBasis{1//2, Int64}, <:AbstractMatrix}
end

struct qubit_ho{ho_basis}
    zI::QuantumOpticsBase.Operator{CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, <:AbstractMatrix}
    xI::QuantumOpticsBase.Operator{CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, <:AbstractMatrix}
    mI::QuantumOpticsBase.Operator{CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, <:AbstractMatrix}
    pI::QuantumOpticsBase.Operator{CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, <:AbstractMatrix}
    II::QuantumOpticsBase.Operator{CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, <:AbstractMatrix}
    pa::QuantumOpticsBase.Operator{CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, <:AbstractMatrix}
    mad::QuantumOpticsBase.Operator{CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, <:AbstractMatrix}

    Ia::QuantumOpticsBase.Operator{CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, <:AbstractMatrix}
    Iad::QuantumOpticsBase.Operator{CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, <:AbstractMatrix}

    n_mech::QuantumOpticsBase.Operator{CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, <:AbstractMatrix}
    n_qubit::QuantumOpticsBase.Operator{CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, <:AbstractMatrix}
end



struct HOSystem
    basis::FockBasis
    a::Operator
    ad::Operator
    n::Operator
    Id::Operator
end

function Harmonic_oscillator(N_particle::Int64, type_basis::Symbol)
    basis_Dict = Dict(
        :FockBasis =>  QuantumOpticsBase.FockBasis
    )

    basis_type = basis_Dict[type_basis]
    basis = basis_type(N_particle)

    return HOSystem(basis,
    destroy(basis), 
    create(basis), 
    number(basis),
    one(basis)
    )

end

function Qubit(spin)
    basis = QuantumOpticsBase.SpinBasis(spin)

    return qubit(basis,
    sigmam(basis),
    sigmap(basis),
    sigmax(basis),
    sigmay(basis),
    sigmaz(basis),
    one(basis)
    )
    
end


function Qubit_HO(N_mech, type_basis_mech::Symbol, type_basis_qubit)
    qub = Qubit(type_basis_qubit)
    mech_res = Harmonic_oscillator(N_mech, type_basis_mech)
    
    return qub, mech_res,
            qubit_ho(
            tensor(qub.σz, mech_res.Id),
            tensor(qub.σx, mech_res.Id),
            tensor(qub.σm, mech_res.Id),
            tensor(qub.σp, mech_res.Id),
            tensor(qub.Id, mech_res.Id),
            tensor(qub.σm, mech_res.a),
            tensor(qub.σp, mech_res.ad),
            tensor(qub.Id, mech_res.a),
            tensor(qub.Id, mech_res.ad),
            tensor(qub.Id, mech_res.n),
            0.5 * (tensor(qub.Id, mech_res.Id) + tensor(qub.σz, mech_res.Id))
        )
end


function recomposition(vector::Vector{Float64})
    exit_vec = Vector{ComplexF64}(undef, length(vector) ÷ 2)
    N = length(vector)
    
    for k in 1:2:N
        exit_vec[(k+1) ÷ 2] = vector[k] + im * vector[k+1]
    end

    return exit_vec
end

function to_real_vec(vector::Vector{ComplexF64})
    real_imag = Vector{Float64}(undef, 2 * length(vector))
    for i in eachindex(vector)
        real_imag[2i - 1] = real(vector[i])
        real_imag[2i]     = imag(vector[i])
    end
    return real_imag
end

qub, mech, HBAR_qubit = Qubit_HO(N_cut_off, :FockBasis, 1//2)
H_SCALE = 2π * 1e-3 
H_JC = g * (HBAR_qubit.Iad * HBAR_qubit.mI + HBAR_qubit.Ia * HBAR_qubit.pI)
function SE_Fock_hamiltonian(Δ::Real, Ω::Real)
    
    return H_SCALE * (HBAR_qubit.H_JC + (Δ/2) * HBAR_qubit.zI + Ω * HBAR_qubit.xI)
end
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# RL ENVIRONMENT
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────



mutable struct QuantumEnv{TOp,TH} <: RLBase.AbstractEnv
    operators::TOp          # es. l'oggetto "HBAR_qubit" / "qubit_mech" di Qubit_HO
    H_JC::TH                # Hamiltoniano JC precomputato (Operator)
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
    target  = tensor(spindown(qub.basis), fockstate(mech.basis, N_mech))
    ψ0      = tensor(spindown(qub.basis), fockstate(mech.basis, 0))

    # Jaynes–Cummings come Operator (non usare .data)
    H_JC = g * (ops.Iad * ops.mI + ops.Ia * ops.pI)

    # finestra temporale iniziale
    t0 = 0.0
    t_step = 0.3e-2
    t_span = (t0, t0 + t_step)

    max_steps = 500

    return QuantumEnv{typeof(ops), typeof(H_JC)}(
        ops, H_JC, target, ψ0, t_span, max_steps, 0, 0.0, false
    )
end

function RLBase.action_space(envs::Vector{<:QuantumEnv})
    low, high = -1.0, 1.0
    return [ClosedInterval(low, high), ClosedInterval(low, high)]
end

function RLBase.state_space(envs::Vector{<:QuantumEnv})
    d = length(env.current_state.data)          # dimensione di Hilbert
    return [ClosedInterval(-1.0, 1.0) for _ in 1:(2*d)]  # Re/Im
end

RLBase.state(envs::Vector{<:QuantumEnv}) = Float64.(to_real_vec(env.current_state.data))
#RLBase.state(envs::Vector{<:QuantumEnv}) = to_real_vec(env.current_state.data)
RLBase.is_terminated(envs::Vector{<:QuantumEnv}) = env.current_step >= env.max_steps  || env.done


function RLBase.reset!(envs::Vector{<:QuantumEnv})
    # ricostruisci lo stato iniziale dalla base dei tuoi operatori
    qub_basis  = env.operators.zI.basis_l.bases[1]
    mech_basis = env.operators.zI.basis_l.bases[2]
    env.current_state = tensor(spindown(qub_basis), fockstate(mech_basis, 0))

    env.current_step = 0
    env.reward = 0.0
    env.done = false

    # riparti da t=0 con stesso passo temporale
    Δt = env.t_span[2] - env.t_span[1]
    env.t_span = (0.0, Δt)

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
    Dense(64, action_dim * 2)
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
    σ = clamp.(σ, 1e-5, 5.0)  # evita 0 e valori enormi
    
        
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
        Dense(64, 1)
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

function step!(envs::Vector{<:QuantumEnv}, a::Vector{Float64})
    env.current_step += 1
    normalize!(env.current_state)

    old_fid = abs2(env.target_state' * env.current_state)

    # ---- mapping azioni → controlli fisici (kHz) ----
    Δ_max = 10e4  
    Ω_max = 10e3   

    a1 = clamp(a[1], -1.0, 1.0)
    a2 = clamp(a[2], -1.0, 1.0)

    Δ = Δ_max * a1
    Ω  = Ω_max * a2

    
    # ---- integrazione su tempo CONTINUO ----
    t0, t1 = env.t_span
    H_step = SE_Fock_hamiltonian(Δ, Ω)
    tout, ψt = timeevolution.schroedinger([t0, t1], env.current_state, H_step;
                                      reltol=1e-6, abstol=1e-11)
    env.current_state = ψt[end]
    normalize!(env.current_state)

    # finestra successiva
    Δt = t1 - t0
    env.t_span = (t1, t1 + Δt)

    # ---- reward & done ----
    new_fidelity = clamp(abs2(env.target_state' * env.current_state), 0.0, 1.0)
    delta_fidelity = new_fidelity - old_fid

    success_threshold = SUCCESS_THR[] 
   

    

    if env.current_step ≥ env.max_steps || new_fidelity ≥ success_threshold
        reward = 1 * (-log1p(-new_fidelity))  
        env.done = true

    else

        
        #reward =  10*delta_fidelity 
        reward = 20 * delta_fidelity

        

        env.done = false

    end
    
    return reward, env.done
end



function step_envs!(envs::Vector{<:QuantumEnv}, actions::Vector)
    N = length(envs)
    rewards = Vector{Float64}(undef, N)
    dones   = Vector{Bool}(undef, N)
    states  = Vector{Vector{Float64}}(undef, N)
    Threads.@threads for i in 1:N
        r, d = step!(envs[i], actions[i])
        rewards[i] = r
        dones[i]   = d
        states[i]  = RLBase.state(envs[i])
    end
    return states, rewards, dones
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
    env_ids::Vector{Int}
end

function PPOBuffer()
    PPOBuffer([], [], [], [], [], [], Int[])
end

function Base.empty!(buffer::PPOBuffer)
    empty!(buffer.states)
    empty!(buffer.actions)
    empty!(buffer.rewards)
    empty!(buffer.dones)
    empty!(buffer.log_probs)
    empty!(buffer.values)
    empty!(buffer.env_ids)
end

# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Agent and Policy PPO
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

mutable struct PPOAgent
    policy::PPOPolicy
    actor_optimizer
    critic_optimizer
    actor_opt_state     # stato dell'optimizer per l'attore
    critic_opt_state    # stato dell'optimizer per il critico
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

Functors.@functor PPOAgent (policy,)  # va bene così; gli opt_state non sono parametri


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
    n_env::Int = 8,
    n_update_epochs::Int = 10,
    mini_batch_size::Int = 64,
    rng = StableRNG(123),
)
    policy = PPOPolicy(actor, critic)
    buffer = PPOBuffer()

    actor_opt_state  = Flux.setup(actor_optimizer,  policy.actor)
    critic_opt_state = Flux.setup(critic_optimizer, policy.critic)

    return PPOAgent(
        policy,
        actor_optimizer,
        critic_optimizer,
        actor_opt_state,
        critic_opt_state,
        float(gamma),
        float(lambda),
        float(clip_range),
        float(entropy_loss_weight),
        float(critic_loss_weight),
        float(max_grad_norm),
        n_rollout,
        n_env,
        n_update_epochs,
        mini_batch_size,
        rng,
        buffer
    )
end

function reset_opt_states!(agent::PPOAgent)
    agent.actor_opt_state  = Flux.setup(agent.actor_optimizer,  agent.policy.actor)
    agent.critic_opt_state = Flux.setup(agent.critic_optimizer, agent.policy.critic)
    return agent
end


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Core PPO Functions
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

# utility
function _min_len(buf::PPOBuffer)
    minimum((
        length(buf.states), length(buf.actions), length(buf.rewards),
        length(buf.dones), length(buf.log_probs), length(buf.values),
        length(buf.env_ids)
    ))
end

function _trim_to!(buf::PPOBuffer, L::Int)
    buf.states     = buf.states[1:L]
    buf.actions    = buf.actions[1:L]
    buf.rewards    = buf.rewards[1:L]
    buf.dones      = buf.dones[1:L]
    buf.log_probs  = buf.log_probs[1:L]
    buf.values     = buf.values[1:L]
    buf.env_ids    = buf.env_ids[1:L]
end



function store_transition!(agent::PPOAgent, state, action, reward, done, logp, value; env_id::Int)
    push!(agent.buffer.states, state)
    push!(agent.buffer.actions, action)
    push!(agent.buffer.rewards, reward)
    push!(agent.buffer.dones, done)
    push!(agent.buffer.log_probs, logp)
    push!(agent.buffer.values, value)
    push!(agent.buffer.env_ids, env_id)
end


function store_step!(buffer::PPOBuffer, state, action, reward, done, log_prob, value)
    push!(buffer.states, state)
    push!(buffer.actions, action)
    push!(buffer.rewards, reward)
    push!(buffer.dones, done)
    push!(buffer.log_probs, log_prob)
    push!(buffer.values, value)
end

atanh_clamped(x) = 0.5 * log((1 + clamp(x, -1 + 1e-6, 1 - 1e-6)) / (1 - clamp(x, -1 + 1e-6, 1 - 1e-6)))

function sample_squashed(dist)
    u = [rand(d) for d in dist] |> x -> reshape(x, size(dist))
    a = tanh.(u)
    log_base = logpdf.(dist, u)
    if ndims(log_base) == 1
        lp = sum(log_base) - sum(log.(1 .- a.^2 .+ 1e-6))
        return a, lp                # scala
    else
        lp = vec(sum(log_base, dims=1) .- sum(log.(1 .- a.^2 .+ 1e-6), dims=1))
        return a, lp                # vettore per batch
    end
end


function logprob_squashed(dist, a)  # calcola logπ(a) dato a∈[-1,1]
    u = atanh_clamped.(a)
    lp = sum(logpdf.(dist, u), dims=1) .- sum(log.(1 .- a.^2 .+ 1e-6), dims=1)
    return vec(lp)
end


function select_action(agent::PPOAgent, state::Vector{Float64})
    dist = agent.policy.actor(state)
    a, logp = sample_squashed(dist)     # a ∈ [-1,1]
    value = agent.policy.critic(state)
    return collect(a), (logp isa AbstractArray ? logp[1] : logp), value
end






function ready_to_update(agent::PPOAgent)
    return length(agent.buffer.rewards) >= agent.n_rollout
end


function update!(agent::PPOAgent; bootstrap_values_by_env::Vector{Float64})
    L = _min_len(agent.buffer)
    if L == 0
        empty!(agent.buffer); return
    end
    _trim_to!(agent.buffer, L)
    update_policy!(agent, bootstrap_values_by_env)
    empty!(agent.buffer)
end




function clear_buffer!(agent::PPOAgent)
    agent.buffer.states = []
    agent.buffer.actions = []
    agent.buffer.rewards = Float64[]
    agent.buffer.dones = Bool[]
    agent.buffer.values = Float64[]
    agent.buffer.log_probs = Float64[]
end


# ───────────────────────────────────────────────────────────────────────────────────────────────────────────────
# PPO Logic - Versione corretta
# ───────────────────────────────────────────────────────────────────────────────────────────────────────────────


function maybe_bump_threshold!(episode_fidelities::Vector{Float64},
                               sr_hist::Vector{Float64},
                               agent::PPOAgent;
                               best_fid::Float64=0.0,          # opzionale, non usato qui
                               W::Int=10,
                               K::Int=6,
                               sr_target::Union{Nothing,Float64}=nothing,
                               q_level::Float64=0.5,           # quantile robusto (0.5 = mediana)
                               q_factor::Float64=0.95,         # richiede qF ≥ q_factor * soglia
                               hysteresis_margin::Float64=1e-4,# isteresi per evitare ping-pong
                               cooldown::Int=5,
                               episode::Int=0,
                               debug::Bool=true,
                               reset_on_bump::Bool=false)
    # fine ladder
    if THR_IDX[] >= length(THR_LADDER); return nothing; end
    n = length(sr_hist)
    m = length(episode_fidelities)
    if n == 0 || m == 0; return nothing; end

    # rispetto cooldown
    if episode != 0 && (episode - LAST_BUMP_EP[]) < cooldown
        return nothing
    end

    W_eff = min(W, n, m)
    K_eff = min(K, n)

    w_sr = sr_hist[end-W_eff+1:end]
    w_f  = episode_fidelities[end-W_eff+1:end]

    sr_win    = mean(w_sr)
    sr_streak = mean(sr_hist[end-K_eff+1:end])
    curr_thr  = SUCCESS_THR[]

    # statistiche robuste
    qf    = quantile(w_f, q_level)     # es. mediana se q_level=0.5
    medF  = Statistics.median(w_f)

    # target SR dinamico semplice
    dyn_target = sr_target === nothing ? (medF ≥ 0.90 ? 0.60 : (medF ≥ 0.85 ? 0.70 : 0.80)) : sr_target

    # condizione per bump: SR buono e quantile F non troppo sotto soglia
    do_bump = (sr_win ≥ dyn_target || sr_streak ≥ dyn_target) &&
              (qf ≥ q_factor * curr_thr)

    if do_bump
        old_thr = curr_thr
        THR_IDX[] += 1
        THR_IDX[] = min(THR_IDX[], length(THR_LADDER))   # guardia
        SUCCESS_THR[] = THR_LADDER[THR_IDX[]]

        # isteresi: sposta leggermente la soglia effettiva per evitare rimbalzi
        SUCCESS_THR[] = min(SUCCESS_THR[] + hysteresis_margin, 1.0)

        # === ENTROPIA: aggiorna a scalini quando la soglia avanza ===
        thr = SUCCESS_THR[]
        floor = if     thr < 0.95;   0.02      # early: esplora di più
                elseif thr < 0.99;   0.005     # mid
                elseif thr < 0.995;  0.001     # late
                else                 1e-4      # fine-tuning
                end
        decay = (thr < 0.99) ? 0.997 : 0.98    # più rapido sopra 0.99
        agent.entropy_loss_weight = max(agent.entropy_loss_weight * decay, floor)
        if !isfinite(agent.entropy_loss_weight)
            agent.entropy_loss_weight = floor
        end
        # ============================================================

        LAST_BUMP_EP[] = (episode == 0 ? LAST_BUMP_EP[] : episode)

        if debug
            @info "↑ Threshold: $(round(old_thr; digits=3)) → $(round(SUCCESS_THR[]; digits=3))  " *
                  "(SR_win=$(round(sr_win; digits=2)), SR_streak=$(round(sr_streak; digits=2)), " *
                  "q$(Int(round(q_level*100)))F=$(round(qf; digits=3)), medianF=$(round(medF; digits=3)), " *
                  "SR*=$(round(dyn_target; digits=2)), W=$(W_eff))"
        end

        if reset_on_bump
            empty!(sr_hist); empty!(episode_fidelities)
        end

    elseif debug && episode % 5 == 0
        @info "No bump (W=$(W_eff),K=$(K_eff)): SR_win=$(round(sr_win; digits=2)), " *
              "SR_streak=$(round(sr_streak; digits=2)), " *
              "q$(Int(round(q_level*100)))F=$(round(qf; digits=3)) vs $(round(q_factor*curr_thr; digits=3)), " *
              "medianF=$(round(medF; digits=3)), SR*=$(round(dyn_target; digits=2)), Thr=$(round(curr_thr; digits=3))"
    end
    return nothing
end



# Calcolo delle advantage tramite Generalized Advantage Estimation (GAE)
function compute_gae(rewards::Vector{Float64}, values::Vector{Float64}, dones::Vector{Bool}; γ=0.99, λ=0.95)
    T = length(rewards)
    @assert length(values) == T + 1 "Il vettore values deve avere lunghezza T+1"
    @assert length(dones) == T "Il vettore dones deve avere lunghezza T"

    advantages = zeros(Float64, T)
    gae = 0.0
    for t in T:-1:1
        delta = rewards[t] + γ * values[t+1] * (1.0 - dones[t]) - values[t]
        gae = delta + γ * λ * (1.0 - dones[t]) * gae
        advantages[t] = gae
    end
    return advantages
end

function update_policy!(agent::PPOAgent, bootstrap_by_env::Vector{Float64})
    # --- 1) Buffer coerente ---
    T = _min_len(agent.buffer)
    if T == 0; return; end
    _trim_to!(agent.buffer, T)

    r   = agent.buffer.rewards
    v   = agent.buffer.values
    d   = agent.buffer.dones
    eid = agent.buffer.env_ids

    # --- 2) GAE per environment (scan a ritroso) ---
    adv      = zeros(Float64, T)
    last_v   = Dict{Int,Float64}(i => bootstrap_by_env[i] for i in 1:length(bootstrap_by_env))
    last_adv = Dict{Int,Float64}(i => 0.0                for i in 1:length(bootstrap_by_env))

    γ, λ = agent.gamma, agent.lambda
    @inbounds for t in T:-1:1
        e = eid[t]
        next_v = d[t] ? 0.0 : get(last_v, e, 0.0)
        δ = r[t] + γ * next_v - v[t]
        adv[t] = δ + γ * λ * (d[t] ? 0.0 : get(last_adv, e, 0.0))
        last_adv[e] = adv[t]
        last_v[e]   = v[t]
        if d[t]
            last_adv[e] = 0.0
            last_v[e]   = 0.0
        end
    end
    returns = adv .+ v

    # --- 3) Normalizza le advantage ---
    μ, σ = mean(adv), std(adv) + 1e-8
    norm_adv = (adv .- μ) ./ σ

    # --- 4) Dataset ---
    data_states     = agent.buffer.states
    data_actions    = agent.buffer.actions        # in [-1,1]
    data_returns    = returns
    data_advantages = norm_adv
    data_old_logps  = agent.buffer.log_probs

    N = length(data_states)
    if N == 0; return; end

    target_KL = 0.005

    # --- 5) PPO updates ---
    for _ in 1:agent.n_update_epochs
        idx = randperm(agent.rng, N)
        running_kl = 0.0
        num_mb = 0

        for start in 1:agent.mini_batch_size:length(idx)
            stop = min(start + agent.mini_batch_size - 1, length(idx))
            bi = idx[start:stop]

            batch_states     = hcat(data_states[bi]...)
            batch_actions    = hcat(data_actions[bi]...)      # (act_dim, B)
            batch_returns    = data_returns[bi]
            batch_advantages = data_advantages[bi]
            batch_old_logps  = data_old_logps[bi]

            # ----- ACTOR (squashed) — gradienti rispetto al MODELLO -----
            grads_actor_tuple = Flux.gradient(agent.policy.actor) do actor_model
                dist      = actor_model(batch_states)                 # Array{Normal} (act_dim, B)
                new_logps = logprob_squashed(dist, batch_actions)     # logπ(a) con tanh-squash
                ratios    = exp.(new_logps .- batch_old_logps)

                surr1 = ratios .* batch_advantages
                surr2 = clamp.(ratios, 1 - agent.clip_range, 1 + agent.clip_range) .* batch_advantages
                actor_loss = -mean(min.(surr1, surr2))

                # entropia proxy dei Normal non-squashed (regolarizza σ)
                ent = sum(entropy.(dist), dims=1)[:]
                entropy_term = -mean(ent)

                actor_loss + agent.entropy_loss_weight * entropy_term
            end
            grads_actor = first(grads_actor_tuple)

            # Update attore: prende nuovo opt_state e il MODELLO aggiornato
            new_actor_opt_state, new_actor =
                Flux.update!(agent.actor_opt_state, agent.policy.actor, grads_actor)
            agent.actor_opt_state = new_actor_opt_state
            # Ricrea l'intera policy (PPOPolicy è immutabile)
            agent.policy = PPOPolicy(new_actor, agent.policy.critic)

            # ----- CRITIC — gradienti rispetto al MODELLO -----
            grads_critic_tuple = Flux.gradient(agent.policy.critic) do critic_model
                v̂ = critic_model(batch_states)[:]
                Flux.mse(v̂, batch_returns) * agent.critic_loss_weight
            end
            grads_critic = first(grads_critic_tuple)

            new_critic_opt_state, new_critic =
                Flux.update!(agent.critic_opt_state, agent.policy.critic, grads_critic)
            agent.critic_opt_state = new_critic_opt_state
            agent.policy = PPOPolicy(agent.policy.actor, new_critic)

            # ----- KL monitor -----
            dist_after  = agent.policy.actor(batch_states)
            new_logps   = logprob_squashed(dist_after, batch_actions)
            running_kl += mean(batch_old_logps .- new_logps)
            num_mb += 1
        end

        if num_mb > 0 && running_kl / num_mb > target_KL
            break
        end
    end
end









