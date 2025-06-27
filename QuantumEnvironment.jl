
module QuantumEnvironment
    using Flux,  Random, Statistics, QuantumOptics, SparseArrays, StatsBase, LinearAlgebra
    using Zygote, DifferentialEquations, SciMLSensitivity, DiffEqFlux, Optimisers, Compat, PlotlyJS, CSV, DataFrames, BSON
    
    include("QM_library.jl")
    

    function SE_Fock_dynamics(du, u, p, t) #
        Δ, Ω = p

        #= Hamiltonians of the problem =#
        H_JC = g * (Iad * mI  + Ia * pI)
        H0 = (Δ / 2) * zI
        H_drive = Ω * xI

        H_tot = H0 + H_JC + H_drive

        ψ = u[1:2:end] + im * u[2:2:end]
        dψ = -1im * H_tot * ψ

        for i in 1:length(dψ)
            du[2i-1] = real(dψ[i]) 
            du[2i] = imag(dψ[i])
        end

    end

    function SE_Fock_problem!(p)
        return ODEProblem(SE_Fock_dynamics, to_real_vec(ψ0.data), (0.0, 1.0), p)
    end



    function Quantum_solver_ODE(prob)
        sol = solve(prob, Tsit5())

        return sol.t, sol.u

    end



    export Quantum_solver_ODE
end
