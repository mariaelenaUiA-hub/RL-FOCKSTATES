
include("RL_Environment.jl")

include("PPO_agent.jl")


using .RLEnvironment
using .PPOpolicy
using ReinforcementLearning
using StableRNGs
using Flux



env = QuantumEnv()

state_dim  = length(RLBase.state_space(env)) 
action_dim = length(RLBase.action_space(env)) 

println("Dimensioni dello spazio degli stati: ", state_dim)
println("Dimensioni dello spazio delle azioni: ", action_dim)

rng        = StableRNG(123)
N_episodes = 50
N_env      = 1


actor  = RLEnvironment.Actor(state_dim, action_dim)
critic = RLEnvironment.Critic(state_dim)


# --- 3. Configurazione dell'Agente PPO ---

const BATCH_SIZE = 128              # Dimensione del batch per l'aggiornamento della rete
const N_TRANSITIONS_PER_ENV = 128   # Numero di transizioni raccolte da ogni ambiente per ogni step di aggiornamento
const N_UPDATE_EPOCHS = 4           # Numero di epoche di aggiornamento per ogni batch raccolto
const GAMMA = 0.99f0                 # Fattore di sconto
const LAMBDA = 0.95f0                # Parametro per GAE (Generalized Advantage Estimation)
const CLIP_RANGE = 0.2f0              # Intervallo di clipping per l'algoritmo PPO
const ENTROPY_LOSS_WEIGHT = 0.01f0    # Peso per il termine di loss dell'entropia
const CRITIC_LOSS_WEIGHT = 0.5f0      # Peso per il termine di loss del critico
const MAX_GRAD_NORM = 0.5f0           # Clipping del gradiente per stabilità
const LR_ACTOR = 3f-4               # Learning rate per l'attore
const LR_CRITIC = 1f-3              # Learning rate per il critico



agent = PPOpolicy.PPOAgent(
    actor,
    critic;
    actor_optimizer = ADAM(LR_ACTOR),
    critic_optimizer = ADAM(LR_CRITIC),
    gamma = GAMMA,
    lambda = LAMBDA,
    clip_range = CLIP_RANGE,
    entropy_loss_weight = ENTROPY_LOSS_WEIGHT,
    critic_loss_weight = CRITIC_LOSS_WEIGHT,
    max_grad_norm = MAX_GRAD_NORM,
    n_rollout = N_TRANSITIONS_PER_ENV,
    n_env = N_env, # Assuming N_env is defined as 1, adjust if using multiple environments
    n_update_epochs = N_UPDATE_EPOCHS,
    mini_batch_size = BATCH_SIZE,
    rng = rng
)




