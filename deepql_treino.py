import gymnasium as gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import csv

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu, linear
from tensorflow.keras.optimizers import Adam

from DeepQLAgent import DeepQLAgent
from deepql_plot import plot_deepql_curve

# Hiperparâmetros principais
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DEC = 0.99
EPISODES = 100             # Você pode subir para 200 ou 300 se quiser
BATCH_SIZE = 32
LEARNING_RATE = 0.001      # Anteriormente estava 0.1, agora 0.001 (inspirado no CartPole e no artigo)
MEMORY_SIZE = 1000
MAX_STEPS = 1000           # Reduzido de 2000 para 1000, agilizando o término do episódio
NUM_EXECUCOES = 5          # Número de rodadas de treinamento

def treinar_deepql():
    """
    Treina o agente Deep Q-Learning no ambiente MountainCar-v0
    por NUM_EXECUCOES e retorna uma lista de listas (recompensas).
    """
    # Criação do ambiente
    env = gym.make('MountainCar-v0')

    # Lista para armazenar as recompensas de cada execução
    todas_recompensas = []

    for execucao in range(NUM_EXECUCOES):
        print(f"\n=== DeepQL Execução {execucao+1}/{NUM_EXECUCOES} ===")

        # Cria a memória (deque)
        memory = deque(maxlen=MEMORY_SIZE)

        # Cria o modelo (rede neural) com duas camadas de 64 neurônios
        model = Sequential()
        model.add(Dense(64, activation=relu, input_dim=env.observation_space.shape[0]))
        model.add(Dense(64, activation=relu))
        model.add(Dense(env.action_space.n, activation=linear))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))

        # Instancia o agente DQL
        dql_agent = DeepQLAgent(
            env=env,
            gamma=GAMMA,
            epsilon=EPSILON,
            epsilon_min=EPSILON_MIN,
            epsilon_dec=EPSILON_DEC,
            episodes=EPISODES,
            batch_size=BATCH_SIZE,
            memory=memory,
            model=model,
            max_steps=MAX_STEPS
        )

        # Executa o treinamento desta execução
        recompensas = dql_agent.train()

        todas_recompensas.append(recompensas)

        # Salva modelo de cada execução
        model.save(f"data/model_mountain_car_exec{execucao+1}.keras")

        # Salva o CSV das recompensas desta execução
        with open(f"results/mountaincar_DQL_rewards_exec{execucao+1}.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Reward"])
            for ep_i, reward in enumerate(recompensas):
                writer.writerow([ep_i, reward])

    env.close()
    return todas_recompensas

if __name__ == "__main__":
    # Executa o treinamento
    recompensas_obtidas = treinar_deepql()

    # Plota as curvas de aprendizado em um único gráfico
    plot_deepql_curve(
        rewards_list=recompensas_obtidas,
        algorithm="Deep Q-Learning - MountainCar",
        gamma=GAMMA,
        epsilon=EPSILON,
        epsilon_min=EPSILON_MIN,
        epsilon_dec=EPSILON_DEC,
        episodes=EPISODES,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )
