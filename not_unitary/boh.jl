using Plots

success_threshold = 0.95  # soglia attuale
α = 1/success_threshold                 # coefficiente di "ripidezza"

new_fidelity = range(0, 1, length=500)
w = @. exp(-α * max(0.0, success_threshold - new_fidelity))

plot(
    new_fidelity, w;
    xlabel="new_fidelity (F)",
    ylabel="w",
    title="Peso dinamico w (F))",
    legend=false,
    linewidth=3,
    grid=true,
    framestyle=:box,
    size=(900,500)
)

vline!([success_threshold], color=:red, linestyle=:dash, label="threshold = $(success_threshold)")
