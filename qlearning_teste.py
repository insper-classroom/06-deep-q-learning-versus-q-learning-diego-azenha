import gymnasium as gym
import numpy as np
from QLAgent import QLearningAgent

def testar(algoritmo="qlearning", episodes=5, max_steps=1000):
    env = gym.make("MountainCar-v0", render_mode="human")

    # Carrega a última Q-table salva
    agent = QLearningAgent(env, 0, 0, 0, 0, 0)
    agent.Q = np.load("qtable_qlearning.npy")

    sucessos = 0
    steps_por_ep = []
    recompensas_totais = []

    for ep in range(episodes):
        obs, _ = env.reset()
        s = agent.transform_state(obs)
        done = False
        soma_recompensas = 0
        truncated = False
        step_count = 0

        for _ in range(max_steps):
            if done or truncated:
                break
            step_count += 1

            # Seleciona ação com base na Q-table (greedy)
            a = np.argmax(agent.Q[s[0], s[1]])
            next_obs, r, done, truncated, _ = env.step(a)
            soma_recompensas += r
            s = agent.transform_state(next_obs)

        steps_por_ep.append(step_count)
        recompensas_totais.append(soma_recompensas)

        print(f"Episódio {ep+1} | Recompensa total: {soma_recompensas:.2f} | Ações executadas: {step_count}")

        if done and not truncated:
            sucessos += 1

    env.close()

    taxa_sucesso = (sucessos / episodes) * 100
    media_acoes = np.mean(steps_por_ep)
    desvio_acoes = np.std(steps_por_ep)

    print("\n--- Resultados Finais ---")
    print(f"Algoritmo: {algoritmo}")
    print(f"Sucessos: {sucessos}/{episodes} ({taxa_sucesso:.2f}%)")
    print(f"Média de ações por episódio: {media_acoes:.2f}")
    print(f"Desvio padrão de ações por episódio: {desvio_acoes:.2f}")

if __name__ == "__main__":
    alg = "qlearning"
    testar(alg, episodes=5, max_steps=1000)
