


using JLD2

@load "not_unitary/plots&data/g2e3/results_1.JLD2" all_rewards all_fidelities best_actions


Δ_max =  1e5
g  = 358*2*pi
#g = 2* π * 41
g  =  g/Δ_max
κϕ =  0.25 / Δ_max
κ  =  19 / Δ_max
γm =  0.025 / Δ_max

kb = 1.3806488e-23
hbar_= 1.054571817e-34

Teq   = 1e-2
nthm  = 1 / (exp((ωm*1e3*hbar_) / (Teq*kb)) - 1)

N_cut_off = 5;
N_mech    = 1;


qub, mech, ops = Qubit_HO(N_cut_off, :FockBasis, 1//2);


function simulate_with_actions_step!(best_actions::Vector,
                                     ψ_init::Ket,
                                     ops;
                                     g::Real,
                                     γm::Real,
                                     κϕ::Real,
                                     κ::Real,
                                     nthm::Real,
                                     Δt::Float64)

    # stato iniziale e tempo
    ρ  = dm(ψ_init)
    t0 = 0.0

    
    n_mech_op  = ops.Iad * ops.Ia        # a†a sul modo meccanico (⊗ I_qubit)
    n_qubit_op = ops.pI  * ops.mI   

    
    ρ_solution   = Operator[ρ]   # include lo stato iniziale una sola volta
    exp_values   =  [real(expect( n_mech_op, ρ))]     # ⟨n_mech⟩
    exp_values_q = [real(expect(  n_qubit_op, ρ))]    # ⟨n_qubit⟩
     # σ⁺σ⁻ = |1⟩⟨1| sul qubit (⊗ I_osc)

    for a in best_actions
        # --- controlli come in step! ---
        a1 = Float64((a[1]+1)/2)
        a2 = Float64(a[2])


        Δ = a1
        Ω = a2

        # Hamiltoniana: JC + drive X/Z con le stesse scalature
        H_JC = g * (ops.Iad * ops.mI + ops.Ia * ops.pI)

        Ω_(t) = Ω 
        Δ_(t) = Δ 

        Ht = LazySum([Ω_(0.0), Δ_(0.0)], [ops.xI, ops.zI])
        function Hamiltonian(t, ψ)
            Ht.factors[1] = Ω_(t)
            Ht.factors[2] = Δ_(t) / 2
            return H_JC + Ht
        end

        # dissipatori identici a step!
        diss = [sqrt(κϕ/2)  * ops.zI, sqrt(κ) * ops.mI, sqrt(γm * (nthm+1))*ops.Ia, sqrt(γm*(nthm))*ops.Iad]
         #diss_dag = dagger.([sqrt(γm * (nthm+1))*ops.Ia, sqrt(γm*(nthm))*ops.Iad, sqrt(κϕ/2)  * ops.zI, sqrt(κ) * ops.mI])
        diss_dag = dagger.(diss)
        

        dynamics_input = (t, ψ) -> (Hamiltonian(t, ψ), diss, diss_dag)

        # evoluzione su [t0, t1]
        t1 = t0 + Δt
        t, sol = timeevolution.master_dynamic((t0, t1), ρ, dynamics_input;
                                              adaptive = true, reltol = 1e-9, abstol = 1e-13)

        ρ_series = sol

        # --- FIX 1: salta il primo punto (duplicato del finale precedente) ---
        @inbounds begin
            for i in Base.Iterators.drop(eachindex(ρ_series), 1)  # salta il primo
                push!(ρ_solution, ρ_series[i])
                push!(exp_values,   real(expect(n_mech_op, ρ_series[i])))
                push!(exp_values_q, real(expect(n_qubit_op, ρ_series[i])))
            end
        end
        
        ρ  = ρ_series[end] / tr(ρ_series[end])
        t0 = t1
    end

    return ρ_solution, exp_values, exp_values_q
end






# stati/basi per costruire ρ0 e target
ψ_init = tensor(spindown(qub.basis), fockstate(mech.basis, 0));
ρ_sol, n_mech_traj, n_qubit_traj = simulate_with_actions_step!(best_actions,
                                     ψ_init,
                                     ops;
                                     g,
                                     γm,
                                     κϕ,
                                     κ,
                                     nthm,
                                     Δt=5e-2);


ψ_target = tensor(spindown(qub.basis), fockstate(mech.basis, N_mech));



var = real(expect(ops.n_mech*ops.n_mech, ρ_sol[end]) - expect(ops.n_mech, ρ_sol[end])^2)
V = sqrt(var)


final_fid = real(QuantumOpticsBase.fidelity(ρ_sol[end], dm(ψ_target)))
println("Fidelity finale = ", final_fid)


p = plot(n_mech_traj; label="⟨n_mech⟩", xlabel="step",
          legend=:outertopright, size=(1200,800),
         legendtitle="Best Fidelity = $(round(final_fid; digits=5))", grid=true,ylims=(0,1.1),
         marker=:circle,       
        markersize=0.5,         
        line=:solid,          
        linewidth=3,
        framestyle=:box);
plot!(p, n_qubit_traj; label="⟨n_qubit⟩",
         marker=:circle,       
        markersize=0.5,         
        line=:solid,         
        linewidth=3,
        framestyle=:box)
savefig(p, "not_unitary/plots&data/g2e3/plot_1.pdf")

using Plots, Statistics
using PlotThemes

function plot_populations(n_mech_traj, n_qubit_traj;
                          legend_title = "",
                          markers_every::Int = 25,
                          plot_size::Tuple{Int,Int} = (1200,800),
                          dpi::Int = 180,
                          show_means_in_legend::Bool = true)

    steps = 1:length(n_mech_traj)
    @assert length(n_mech_traj) == length(n_qubit_traj)

    # figura
    p = plot(steps, n_mech_traj;
             label="⟨n_mech⟩",
             xlabel="step", ylabel="population",
             legend=:outertopright, legendtitle=legend_title,
             size=plot_size, dpi=dpi, palette=:tab10,
             grid=true, gridalpha=0.25, framestyle=:box,
             lw=2.6, linealpha=0.95, ylims=(0,1.05))

    plot!(p, steps, n_qubit_traj;
          label="⟨n_qubit⟩", lw=2.6, linealpha=0.95)

    # marker radi
    idx = 1:max(markers_every,1):lastindex(steps)
    scatter!(p, steps[idx], n_mech_traj[idx]; ms=3, markerstrokewidth=0, label=false)
    scatter!(p, steps[idx], n_qubit_traj[idx]; ms=3, markerstrokewidth=0, label=false)

    # riempimento tra le curve (molto leggero)
    plot!(p, steps, n_mech_traj;
      lw=0, c=1, fillrange=0, fillalpha=0.15, label=false)

    plot!(p, steps, n_qubit_traj;
        lw=0, c=2, fillrange=0, fillalpha=0.15, label=false)

    # incrocio (interpolazione lineare del primo cambio di segno)
    d = n_mech_traj .- n_qubit_traj
    ix = findfirst(diff(sign.(d)) .!= 0)
    if ix !== nothing
        x1, x2 = steps[ix], steps[ix+1]
        y1, y2 = d[ix], d[ix+1]
        xc = x1 - y1*(x2 - x1)/(y2 - y1)  # stima continua
        vline!(p, [xc]; lw=1, linealpha=0.4, label=false)
    end

    # solo le medie in legenda (senza aggiungere nuove linee)
    if show_means_in_legend
        m1, m2 = mean(n_mech_traj), mean(n_qubit_traj)
        plot!(p, [NaN], [NaN]; label="mean mech = $(round(m1,digits=3))")
        plot!(p, [NaN], [NaN]; label="mean qubit = $(round(m2,digits=3))")
    end

    return p
end

p = plot_populations(n_mech_traj, n_qubit_traj;
     legend_title = "Best Fidelity = $(round(final_fid; digits=5))",
     markers_every = 30)




# ultimi valori
last_n_mech  = n_mech_traj[end]
last_n_qubit = n_qubit_traj[end]

length(best_actions)





using Plots
using Statistics

function plot_best_controls(best_actions::Vector;
                            markers_every::Int = 10,
                            movavg_window::Int = 0,
                            plot_size::Tuple{Int,Int} = (1200, 800),
                            dpi::Int = 180,
                            mean_in_caption::Bool = true)

    # matrice delle azioni
    a_mat = hcat([Float64.(vec(a)) for a in best_actions]...)
    T = Base.size(a_mat, 2)
    steps = 1:T

    a1 = (a_mat[1, :] .+ 1) ./ 2      # in [0,1]
    a2 = a_mat[2, :]

    # marker radi
    idx = 1:max(markers_every,1):T

    # figura
    p = plot(layout=(2,1), link=:x, size=plot_size, dpi=dpi,
             framestyle=:box, legend=false, grid=true,
             gridalpha=0.25, guidefontsize=11, tickfontsize=9)

    # Δ / Δ_max (senza label)
    plot!(p[1], steps, a1; lw=1.8, linealpha=0.85, label=false)
    scatter!(p[1], steps[idx], a1[idx]; marker=:circle, ms=3,
             markeralpha=0.9, markerstrokewidth=0, label=false)
    xlabel!(p[1], "step"); ylabel!(p[1], "Δ / Δ_max")

    # Ω / Δ_max (senza label)
    plot!(p[2], steps, a2; lw=1.8, linealpha=0.85, label=false)
    scatter!(p[2], steps[idx], a2[idx]; marker=:circle, ms=3,
             markeralpha=0.9, markerstrokewidth=0, label=false)
    xlabel!(p[2], "step"); ylabel!(p[2], "Ω / Δ_max")

    # (opzionale) media mobile, ma non in legenda
    if movavg_window > 1
        movmean(v, w) = [mean(@view v[max(1,i-w+1):i]) for i in eachindex(v)]
        plot!(p[1], steps, movmean(a1, movavg_window); lw=2, color=:black, linealpha=0.5, label=false)
        plot!(p[2], steps, movmean(a2, movavg_window); lw=2, color=:black, linealpha=0.5, label=false)
    end

    # Solo la media in legenda
    #if mean_in_caption
        #m1 = mean(a1); m2 = mean(a2)
        #hline!(p[1], [m1]; lw=2, linealpha=0.7, label="average = $(round(m1, digits=3))")
        #hline!(p[2], [m2]; lw=2, linealpha=0.7, label="average = $(round(m2, digits=3))")
        #plot!(p; legend=true)
    #end

    display(p)
    return p
end



plot_best_actions=plot_best_controls(best_actions; markers_every=12, movavg_window=25)




savefig(plot_best_actions,"not_unitary/plots&data/g2e3/best_actions_1.pdf")

a = 1-final_fid




plot_f = plot(all_fidelities;
    label="Fidelity",
    title="Fidelity")


savefig(plot_f,"not_unitary/plots & data/fidelities_1.pdf")