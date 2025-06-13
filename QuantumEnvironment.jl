
module QuantumEnvironment

    using ..QuantumSystemOperators

    using LinearAlgebra   
    using DifferentialEquations

    function schrodinger_equation!(du, u, p, t)
    """

    Parametri `p` (tuple):
    action_t: Vettore Float32 [Re(drive_b), Im(drive_b), Re(drive_q), Im(drive_q)] per l'azione di controllo.
    
    
    """
    action_t = p.action_t 

    ωq  = action_t[1] #qubit frequency, controlled by RL
    Ωx  = action_t[2] #drive amplitude, controlled by RL

    Δ = ωq - ωm

    H_free = Δ //2 * sigma_z_op

    HC = g * (bdag_op * sigma_minus_op + b_op * sigma_plus_op  )

    H_drive = Ωx //2 * sigma_z_op 

    H = H_free + HC + H_drive

    return H

    end 


    export schrodinger_equation!

end


