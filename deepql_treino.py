import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np
import random
from collections import deque
import csv
import os

from DeepQLAgent import DuelingDeepQLNetwork
from deepql_plot import plot_deepql_curve

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Treinando no dispositivo: {DEVICE}")

# Hiperparâmetros
GAMMA = 0.99
EPSILON = 1
EPSILON_MIN = 0.01
EPSILON_DEC = 0.995
EPISODES = 1000
BATCH_SIZE = 128
LEARNING_RATE = 0.001
MEMORY_SIZE = 50000
MAX_STEPS = 2500
NUM_EXECUCOES = 5

class DeepQLAgent:
    def __init__(self, env, model, optimizer):
        self.env = env
        self.model = model.to(DEVICE)
        self.optimizer = optimizer
        self.criterion = torch.nn.MSELoss()

        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.epsilon_min = EPSILON_MIN
        self.epsilon_dec = EPSILON_DEC
        self.episodes = EPISODES
        self.batch_size = BATCH_SIZE
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.max_steps = MAX_STEPS

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.inference_mode():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values, dim=1).item()

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def experience_replay(self):
        if len(self.memory) < self.batch_size * 4:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(actions, dtype=torch.long, device=DEVICE).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=DEVICE)
        dones = torch.tensor(dones, dtype=torch.float32, device=DEVICE)

        q_values = self.model(states).gather(1, actions).squeeze(dim=1)
        next_q_values = self.model(next_states).max(dim=1)[0]
        targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.criterion(q_values, targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        rewards_all = []
        for episode in range(self.episodes):
            state, _ = self.env.reset()
            total_reward = 0

            for step in range(self.max_steps):
                action = self.select_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)

                reward += 0.1 * (next_state[0] + 1.2) ** 2
                if next_state[0] >= 0.5:
                    reward += 10.0

                self.store_experience(state, action, reward, next_state, done or truncated)
                self.experience_replay()

                state = next_state
                total_reward += reward

                if done or truncated:
                    break

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_dec

            rewards_all.append(total_reward)
            print(f"🎯 Episódio {episode+1}/{self.episodes} — Recompensa: {total_reward:.2f} — epsilon: {self.epsilon:.4f}")

        return rewards_all

def treinar_deepql():
    env = gym.make('MountainCar-v0')
    todas_recompensas = []

    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    for execucao in range(NUM_EXECUCOES):
        print(f"\n🚀 Execução {execucao+1}/{NUM_EXECUCOES} — Iniciando agente...")
        model = DuelingDeepQLNetwork(env.observation_space.shape[0], env.action_space.n)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        agent = DeepQLAgent(env, model, optimizer)

        recompensas = agent.train()
        todas_recompensas.append(recompensas)

        torch.save(model.state_dict(), f"data/model_mountain_car_exec{execucao+1}.pt")
        with open(f"results/mountaincar_DQL_rewards_exec{execucao+1}.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Reward"])
            for ep_i, reward in enumerate(recompensas):
                writer.writerow([ep_i, reward])

    env.close()
    return todas_recompensas

if __name__ == "__main__":
    recompensas_obtidas = treinar_deepql()
    plot_deepql_curve(
        rewards_list=recompensas_obtidas,
        algorithm="Dueling Deep Q-Learning (PyTorch)",
        gamma=GAMMA,
        epsilon=EPSILON,
        epsilon_min=EPSILON_MIN,
        epsilon_dec=EPSILON_DEC,
        episodes=EPISODES,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )
