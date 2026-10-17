import os, sys
import numpy as np
import matplotlib.pyplot as plt
from agent import DQNAgent
from enviroment import CartPoleCustom
from time import time

# ── Hiperparâmetros ──────────────────────────────────────────────────── #
MAX_EPISODES   = 1000
MAX_STEPS      = 500          # CartPole-v1 trunca em 500 passos
SOLVE_SCORE    = 475          # média de 10 episódios para considerar resolvido
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

argumentos = {}
for arg in sys.argv[1:]:
    if '=' in arg:
        chave, valor = arg.split('=', 1)
        argumentos[chave] = valor

def train(num_episodios=MAX_EPISODES):
    #print(f"-------------------------> argumentos: {argumentos}")

    env   = CartPoleCustom(sutton_barto_reward=True)
    agent = DQNAgent()
    
    #agent.load(f"{CHECKPOINT_DIR}/dqn_best")
    
    rewards_history = []
    losses_history  = []
    best_avg        = -np.inf

    for ep in range(1, num_episodios + 1):
        state, _ = env.reset()
        ep_reward = 0
        ep_losses = []

        for _ in range(MAX_STEPS):
            action, orig_act                                = agent.act(state)
            next_state, reward, terminated, truncated, _    = env.step(action)
            #print(f"next_state={next_state}, reward={reward}, terminated={terminated}, truncated={truncated}")
            done                                            = terminated or truncated

            agent.buffer.push(state, action, reward, next_state, float(done))
            loss = agent.learn()
            if loss is not None:
                ep_losses.append(loss)

            state     = next_state
            ep_reward += reward
            if done:
                break

        agent.decay_epsilon()
        agent.maybe_update_target(ep)

        rewards_history.append(ep_reward)
        avg_loss = np.mean(ep_losses) if ep_losses else 0.0
        losses_history.append(avg_loss)

        avg10 = np.mean(rewards_history[-10:])
        print(
            f"Ep {ep:4d} | reward={ep_reward:6.1f} | avg10={avg10:6.1f} "
            f"| ε={agent.epsilon:.3f} | loss={avg_loss:.4f} "
        )

        # Salva o melhor modelo
        if avg10 > best_avg:
            best_avg = avg10
            agent.save(f"{CHECKPOINT_DIR}/dqn_best")

        # Para quando resolver
        if avg10 >= SOLVE_SCORE and ep >= 10:
            print(f"\nResolvido em {ep} episódios!")
            break

    env.close()
    _plot_results(rewards_history, losses_history)

def _plot_results(rewards, losses):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    ax1.plot(rewards, alpha=0.4, label="recompensa por episódio")
    window = min(10, len(rewards))
    avg = np.convolve(rewards, np.ones(window)/window, mode="valid")
    ax1.plot(range(window - 1, len(rewards)), avg, label=f"média {window} ep")
    ax1.axhline(475, linestyle="--", color="green", alpha=0.6, label="meta (475)")
    ax1.set_ylabel("Recompensa")
    ax1.legend()

    ax2.plot(losses, alpha=0.6, color="coral")
    ax2.set_ylabel("Loss (MSE)")
    ax2.set_xlabel("Episódio")

    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    hora_inicio = time()

    if "episodes" in argumentos:
        num_episodios = int(argumentos['episodes'])
    else:
        num_episodios = MAX_EPISODES

    train(num_episodios=num_episodios)
    print(f"Tempo da Execução: {(time() - hora_inicio):.1f} segundos")