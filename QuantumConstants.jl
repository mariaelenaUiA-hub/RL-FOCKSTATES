

module QuantumConstants

    const N_fock_cutoff = 3 #da |0> a |2>
    const N_dim_qs = 2 *N_fock_cutoff #Qubit=2 * Fononi=N_FOCK_CUTOFF

    const hbar = 1.054571817e−34
    const kb = 1.380649e−23


    #frequencies
    const ωm = 5.9614e6 #kHz
    const ωq = 5.9456e6 #kHz
    const g  = 258.0
    const Δ0  = (ωq-ωm)


    #dissipation rates
    const κϕ = 0.25
    const κ  = 19
    const γm = 0.025

    #temperature kHz
    const Teq   = kb / (2 * π * hbar) * 1e-3 * 10e-3
    const szth  = 1 / (exp(ωq / Teq) - 1) - 1/2
    const nmth  = 1 / (exp(ωm / Teq) - 1)

    #drive amplitude
    const n = 0 
    const ω0 = 5e3
    const Tπ   = pi / (2 * ω0)  # Time for one π pulse
    const Tswap0 = pi/ (2 *g)
    const ωstark = (2*g^2)/ Δ0 * (n + 0.5) 

    const T = 2e-6

    export hbar, kb, ωm, ωq, g, Δ0, κϕ, κ, γm, Teq, szth, nmth, n, ω0, Tπ, Tswap0, ωstark, N_fock_cutoff, N_dim_qs
end     