
    
   module RLEnvironment
    include("QM_library.jl")
    include("QuantumEvolution.jl")
    include("QuantumConstants.jl")

    using .QuantumConstants
    using .QuantumEvolution
    using .QuantumLibrary

    using Distributions
    using Flux
    using IntervalSets: ClosedInterval 
    using LinearAlgebra
    using QuantumOptics
    using ReinforcementLearning
    using .ReinforcementLearningBase
    using DifferentialEquations
    


    mutable struct QuantumEnv <: RLBase.AbstractEnv
        operators::qubit_ho
        target_state::Ket
        current_state::Ket
        
        tspan::Tuple{Float64, Float64}
        max_steps::Int
        current_step::Int
    end



    function QuantumEnv()
        
        qub, mech, ops = Qubit_HO(N_fock_cutoff, :FockBasis, 1//2)
        target = tensor(spindown(qub.basis), fockstate(mech.basis, N_fock_cutoff))
        initial_state = tensor(spindown(qub.basis), fockstate(mech.basis, 0))
        t_step = 1e-8

        return QuantumEnv(ops, target, initial_state, (0.0, t_step), 100, 0)
    end


    function RLBase.action_space(env::QuantumEnv)

        low_bound = 10e5
        high_bound = 10e7

        return  [ClosedInterval(low_bound, high_bound), ClosedInterval(low_bound, high_bound)]
    end

    function RLBase.state_space(env::QuantumEnv)
        dims = 2 * 2 * (N_fock_cutoff + 1) # 2 per qubit, 2 per fock, 2 per real/imag
        
        # Esempio: se lo stato è normalizzato, le componenti sono tra -1 e 1
        # Se non ci sono limiti, usa -Inf..Inf
        
        
        
        return  [ClosedInterval(-1.0, 1.0) for _ in 1:dims ]
    end


    RLBase.state(env::QuantumEnv) = to_real_vec(env.current_state.data)

    RLBase.is_terminated(env::QuantumEnv) = env.current_step >= env.max_steps

   




    function RLBase.reset!(env::QuantumEnv)
        qub_basis = env.operators.zI.basis_l.bases[1]
        mech_basis = env.operators.zI.basis_l.bases[2]
        env.current_state = tensor(spindown(qub_basis), fockstate(mech_basis, 0))
        env.current_step = 0
        
    end

  

    function (env::QuantumEnv)(a)
        env.current_step += 1
        
        Δ, Ω = a[1], a[2]
        
        p = (env.operators, Δ, Ω)

    
        # L'input per ODEProblem dovrebbe essere il vettore di dati nudo
        u0 = env.current_state.data 
        prob = ODEProblem(SE_Fock_dynamics, u0, env.tspan, p)
        sol = solve(prob, Tsit5(), save_everystep=false, save_start=false)

        final_state_data = sol.u[end]

        env.current_state = Ket(env.current_state.basis, final_state_data)
        normalize!(env.current_state)

        fid = abs(env.target_state' * env.current_state)
        calculated_reward = fid^2

        if fid > 1-10e-5
            calculated_reward += 100.0
            env.current_step = env.max_steps 
        end
        
        env.reward = calculated_reward
    end





    struct Actor
        chain::Chain
    end

    Flux.Flux.@functor Actor

    function Actor(state_dim::Int, action_dim::Int)
        chain = Chain(
            Dense(state_dim, 128, tanh),
            Dense(128, 64, tanh),
            Dense(64, action_dim * 2)
        )
        Actor(chain)
    end

    function (actor::Actor)(state)
        x = actor.chain(state)

        action_dim = size(x, 1) ÷ 2
        μ = x[1:action_dim]
        log_σ = x[action_dim + 1 : end]
        σ = exp.(log_σ)
        
        return Normal.(μ, σ)
    end


    struct Critic
        chain::Chain
    end
    Flux.@functor Critic

    function Critic(state_dim::Int)
        chain = Chain(
            Dense(state_dim, 128, tanh),
            Dense(128, 64, tanh),
            Dense(64, 1) 
        )
        Critic(chain)
    end

    function (critic::Critic)(state)
        return critic.chain(state) |> first
    end


    function step!(env::QuantumEnv, action::Vector{Float64})
        Δ, Ω = action
        p = (env.operators, Δ, Ω)

        u0 = env.current_state.data
        prob = ODEProblem(SE_Fock_dynamics, u0, env.tspan, p)
        sol = solve(prob, Tsit5(), save_everystep=false, save_start=false)

        final_state_data = sol.u[end]
        env.current_state = Ket(env.current_state.basis, final_state_data)
        normalize!(env.current_state)

        fid = abs(env.target_state' * env.current_state)
        reward = fid^2

        done = false
        if fid > 1 - 1e-4
            reward += 100.0
            done = true
        end

        
        env.current_step += 1

        if env.current_step >= env.max_steps
            done = true
        end

        return reward, done
    end


    export QuantumEnv, Actor, Critic, step!

end