
module QuantumEnvironment

    using QuantumOptics
    using ..QuantumSystemOperators
    

    using LinearAlgebra   
    using DifferentialEquations

    function schrodinger_equation!(du, u, p,t)
    """

    Parametri `p` (tuple) contiene:

    action_t: Vettore Float32 [Re(drive_b), Im(drive_b), Re(drive_q), Im(drive_q)] per l'azione di controllo.
    system_params: Dizionario con costanti fisiche (omega_m, g)
    
    """

    action_t = p.action_t
    system_params = p.system_params
    

    
    ωm    = system_params[:omega_m] 
    g     = system_params[:g] 

    ωq_t  = action_t[1] #qubit frequency, controlled by RL
    Ωx_t  = action_t[2] #drive amplitude, controlled by RL

    Δ_t = ωq_t - ωm

    H_free = Δ_t /2 * sigma_z_op

    HC = g * (bdag_op * sigma_minus_op + b_op * sigma_plus_op  )

    H_drive = Ωx_t * sigma_x_op 

    H = H_free + HC + H_drive

    u_qo =  StateVector(bas_qs,u)
 
    res = -1im * H * u_qo

    du[:] = res.data

    return 

    end 


    export schrodinger_equation!

end


