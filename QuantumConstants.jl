

module QuantumConstants

    const N_fock_cutoff = 3 
    const N_dim_qs = 2 *N_fock_cutoff #Qubit=2 * Fononi=N_FOCK_CUTOFF

    const hbar = 1.054571817e−34
    const kb = 1.380649e−23


    #frequencies
    const ωm = 5.9614e6 #kHz
    const g  = 258.0
    
    #dissipation rates
    const κϕ = 0.25
    const κ  = 19
    const γm = 0.025


    const T = 2e-6

    export hbar, kb, ωm, g, κϕ, κ, γm, Teq, szth, nmth, n, ω0, Tπ, Tswap0, ωstark, N_fock_cutoff, N_dim_qs
end     