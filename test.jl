using Revise 


include("RL_PPO.jl")

function test_fidelity_increase(; Δ = 1e6, Ω = 5e5)
    # Setup
    N_cut_off = 5
    N_mech = 1
    g = 258.0

    qub, mech, ops = Qubit_HO(N_cut_off, :FockBasis, 1//2)

    # Initial: spin down, oscillator ground
    initial_state = tensor(spindown(qub.basis), fockstate(mech.basis, 0))
    # Target: spin down, oscillator excited
    target_state  = tensor(spinup(qub.basis), fockstate(mech.basis, 1))

    # Hamiltonians
    zI = ops.zI.data
    xI = ops.xI.data
    Iad = ops.Iad.data
    mI = ops.mI.data
    Ia = ops.Ia.data
    pI = ops.pI.data
    H_JC = g * (Iad * mI + Ia * pI)

    # Construct total H
    H0 = (Δ / 2.0) * zI
    H_drive = Ω * xI
    H_tot = H0 + H_JC + H_drive

    # Setup time evolution
    tspan = (0.0, 5e-8)
    u0 = to_real_vec(initial_state.data)
    ψ0 = initial_state

    function dynamics(du, u, p, t)
        ψ = u[1:2:end] + im * u[2:2:end]
        dψ = -1im * H_tot * ψ
        for i in eachindex(dψ)
            du[2i - 1] = real(dψ[i])
            du[2i] = imag(dψ[i])
        end
        return nothing
    end

    prob = ODEProblem(dynamics, u0, tspan)
    sol = solve(prob, Tsit5(), reltol=1e-6, abstol=1e-9)

    final_state_vec = recomposition(sol.u[end])
    final_state = Ket(initial_state.basis, final_state_vec)
    normalize!(final_state)

    # Fidelity
    fid_ampl = abs(target_state' * final_state)
    fidelity = fid_ampl^2

    println("Test Result:")
    println("Δ = $(Δ), Ω = $(Ω)")
    println("Fidelity = $(round(fidelity, digits=6))")

    return fidelity
end


test_fidelity_increase(Δ = 5e5, Ω = 5e5)

for Δ in [1e5, 5e5, 1e6]
    for Ω in [1e5, 5e5, 1e6]
        test_fidelity_increase(Δ = Δ, Ω = Ω)
    end
end