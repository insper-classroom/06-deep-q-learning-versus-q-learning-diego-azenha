import gymnasium as gym
import numpy as np
from QLAgent import QLearningAgent
import matplotlib.pyplot as plt

# Substitua "plot_qlearning" pela sua função de plot se desejar
# ou deixe como referência a outra library que você criou
# from plot_qlearning import plot_learning_curve


ALPHA = 0.1
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
EPISODES = 1000
MAX_STEPS = 2000
NUM_EXECUCOES = 5  # Número de rodadas de treinamento


def plot_learning_curve(todas_recompensas, nome_alg, alpha, gamma, epsilon, episodes):
    """
    Exemplo simples de plotagem da curva de aprendizado.
    'todas_recompensas' deve ser uma lista de listas,
    onde cada lista interna é a evolução de recompensas
    ao longo dos episódios para uma execução.
    """
    plt.figure()
    for i, recompensas in enumerate(todas_recompensas):
        plt.plot(recompensas, label=f"Execução {i+1}")
    plt.title(f"Curva de Aprendizado - {nome_alg}")
    plt.xlabel("Episódios")
    plt.ylabel("Recompensa")
    plt.legend()
    plt.show()


def treinar(algoritmo="qlearning"):
    env = gym.make("MountainCar-v0")
    nome_arquivo = "qtable_qlearning.npy"

    todas_recompensas = []   # Lista para recompensas de cada execução
    todas_qtables = []       # Armazena as Q-tables de cada treino
    todas_metricas = []      # Armazena métricas de cada execução

    for execucao in range(NUM_EXECUCOES):
        print(f"\nTreinamento {execucao + 1}/{NUM_EXECUCOES} - Algoritmo: {algoritmo}")
        agent = QLearningAgent(env, ALPHA, GAMMA, EPSILON, EPSILON_MIN, EPSILON_DECAY)
        recompensas = []
        passos_por_ep = []

        for ep in range(EPISODES):
            obs, _ = env.reset()
            s = agent.transform_state(obs)
            done = False
            soma_recompensas = 0
            passos = 0

            # Se fosse SARSA, iniciaríamos a ação aqui (ex.: a = agent.select_action(s))
            # Mas para Q-Learning, selecionaremos a ação dentro do loop

            for _ in range(MAX_STEPS):
                if done:
                    break
                passos += 1

                a = agent.select_action(s)
                next_obs, r, done, _, _ = env.step(a)
                s_next = agent.transform_state(next_obs)

                # Atualiza Q-table
                agent.update(s, a, r, s_next, done)

                s = s_next
                soma_recompensas += r

            # Decai epsilon ao final de cada episódio
            agent.decay_epsilon()

            recompensas.append(soma_recompensas)
            passos_por_ep.append(passos)

        todas_recompensas.append(recompensas)
        todas_qtables.append(agent.Q)
        todas_metricas.append({
            "media_recompensas": np.mean(recompensas),
            "desvio_recompensas": np.std(recompensas),
            "media_passos": np.mean(passos_por_ep),
            "desvio_passos": np.std(passos_por_ep)
        })

    # Salva a última Q-table treinada
    np.save(nome_arquivo, todas_qtables[-1])
    env.close()

    return todas_recompensas, todas_metricas


if __name__ == "__main__":
    algoritmo_escolhido = "qlearning"
    todas_recompensas, metricas = treinar(algoritmo_escolhido)

    print("\nResumo das métricas (médias de todas as execuções):")
    print(
        f"Recompensa média final: "
        f"{np.mean([m['media_recompensas'] for m in metricas]):.2f} ± "
        f"{np.std([m['media_recompensas'] for m in metricas]):.2f}"
    )
    print(
        f"Média de passos: "
        f"{np.mean([m['media_passos'] for m in metricas]):.2f} ± "
        f"{np.std([m['media_passos'] for m in metricas]):.2f}"
    )

    # Plota a curva de aprendizado de cada execução
    plot_learning_curve(todas_recompensas, algoritmo_escolhido, ALPHA, GAMMA, EPSILON, EPISODES)
