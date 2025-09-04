


using JLD2

@load "plots & data/results.jld2" all_rewards all_fidelities best_actions


best_actions

a1_pos = count(x -> x[1] > 0, best_actions) / length(best_actions)

a2_pos = count(x -> x[2] > 0, best_actions) / length(best_actions)


sum([best_actions[i][1] * best_actions[i][2] for i in 11:length(best_actions)] )
