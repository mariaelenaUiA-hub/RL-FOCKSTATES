module QuantumLibrary

    using QuantumOptics
    using QuantumOpticsBase


    const h    = 6.62607015e-34          # Planck constant (J·s)
    const hbar = 1.054571817e-34;     # Reduced Planck constant (J·s)
    const kb   = 1.380649e-23           # Boltzmann constant (J/K)



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



    function recomposition(vector)
        exit = ComplexF64[]
        N = length(vector)

        for k in 1:2:N
            push!(exit, vector[k] + im*vector[k+1])
        end

        return exit
    end
    
    function to_real_vec(vector)
        real_imag = Vector{Float32}(undef, 2 * length(vector))
        for i in eachindex(vector)
            real_imag[2i - 1] = real(vector[i])
            real_imag[2i]     = imag(vector[i])
        end
        return real_imag
    end

    export Qubit_HO, Qubit, Harmonic_oscillator, to_real_vec, recomposition, qubit_ho
    

end