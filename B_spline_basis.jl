
module B_spline_basis

    using ..QuantumConstants 


    using LinearAlgebra

    const BS_n_coeffs = 9 
    const BS_degree   = 3 

    const T = QuantumConstants.T


    function generate_knot_vector(degree::Int, num_coeffs::Int, total_time::Float64)
        k = degree
        n = num_coeffs
        T = total_time

        #numero totale di nodi: n + k + 1
        num_knots = n + k + 1
        knots     = zeros(Float64, num_knots)

        #k+1 nodi a 0
        for i in 1:(k + 1)
            knots[i] = 0.0
        end

        #k+1 nodi a T
        for i in (n + 1):(n + k + 1)
            knots[i] = T
        end


        #n-1-k nodi tra 0 e t
        if n - k -1 >0
            num_internal_knots = n - k - 1
            for i in 1:num_internal_knots
                knots[k + 1 + i] = T * (i / (num_internal_knots + 1))

            end
        end
        return knots

    end


    export generate_knot_vector
end