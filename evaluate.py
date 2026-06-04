from agent import DQNAgent
from enviroment import CartPoleCustom
from time import time
import sys

argumentos = {}
for arg in sys.argv[1:]:
    if '=' in arg:
        chave, valor = arg.split('=', 1)
        argumentos[chave] = valor

def evaluate(n_episodes: int = 10, render: bool = True):
    env = CartPoleCustom(sutton_barto_reward=True, max_steps=500, render_mode="human")

    agent = DQNAgent()
    agent.load("checkpoints/dqn_best")
    agent.epsilon = 0.0   # sem exploração na avaliação

    hora_inicio = time()
    for ep in range(1, n_episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        while True:
            action, orig_act                        = agent.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward                           += reward
            if terminated or truncated:
                break
        print(f"Episódio {ep}: {total_reward:.0f} pontos")
    env.close()
    print(f"Tempo da Execução: {(time() - hora_inicio):.1f} segundos")

if __name__ == "__main__":
    if "episodes" in argumentos:
        num_episodios = int(argumentos['episodes'])
    else:
        num_episodios = 5
    evaluate(n_episodes=num_episodios)