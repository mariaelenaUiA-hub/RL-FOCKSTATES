module QuantumSystemOperators


    using QuantumOptics 

    
    using ..QuantumConstants 

    const bas_q    = SpinBasis(1//2)
    const bas_hbar = FockBasis(N_fock_cutoff - 1 )

    const bas_qs   = bas_q ⊗ bas_hbar   #bases total : quantum system

  
    #operators for Qubit/SPIN - 1° subsystem
    const sigma_minus_op = sigmam(bas_q)
    const sigma_plus_op  = sigmap(bas_q)
    const sigma_x_op     = sigmax(bas_q)
    const sigma_y_op     = sigmay(bas_q)
    const sigma_z_op     = sigmaz(bas_q)
    

    #operators for HBAR/Fonone - 2° subsystem
    const b_op    = destroy(bas_hbar)
    const bdag_op = create(bas_hbar)
    const n_op    = number(bas_hbar)
    
    
    export bas_hbar, bas_qs, bas_q
    export b_op,bdag_op,sigma_minus_op,sigma_plus_op,sigma_x_op,sigma_y_op,sigma_z_op
    


end