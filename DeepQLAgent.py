import numpy as np
import random
from collections import deque
from keras.activations import relu, linear
from keras import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import gc
import keras

class DeepQLAgent:
    def __init__(self, env, gamma, epsilon, epsilon_min, epsilon_dec, episodes, batch_size, memory, model, max_steps):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.episodes = episodes
        self.batch_size = batch_size
        self.memory = memory
        self.model = model
        self.max_steps = max_steps

    def select_action(self, state):
        """
        Seleciona a ação com política epsilon-greedy
        """
        if np.random.rand() < self.epsilon:
            return random.randrange(self.env.action_space.n)
        # Usa a rede neural para estimar Q(s)
        action_values = self.model.predict(state, verbose=0)
        return np.argmax(action_values[0])

    def experience(self, state, action, reward, next_state, terminal):
        """
        Armazena (s, a, r, s', done) na memória
        """
        self.memory.append((state, action, reward, next_state, terminal))

    def experience_replay(self):
        """
        Treina o modelo com um batch amostrado aleatoriamente da memória,
        chamado a cada passo (como no CartPole).
        """
        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)

            states = np.array([i[0] for i in batch])
            actions = np.array([i[1] for i in batch])
            rewards = np.array([i[2] for i in batch])
            next_states = np.array([i[3] for i in batch])
            terminals = np.array([i[4] for i in batch])

            # Ajusta formato
            states = np.squeeze(states)
            next_states = np.squeeze(next_states)

            # Prediz Q(s') e pega max Q para cada s'
            next_qs = self.model.predict_on_batch(next_states)
            next_max = np.amax(next_qs, axis=1)

            # y = r + gamma * maxQ(s'), se não terminal
            targets = rewards + self.gamma * next_max * (1 - terminals)

            # Predição Q(s)
            targets_full = self.model.predict_on_batch(states)

            # Atualiza apenas a ação selecionada
            idxs = np.arange(self.batch_size)
            targets_full[idxs, actions] = targets

            # Retropropaga 1 época
            self.model.fit(states, targets_full, epochs=1, verbose=0)

            # Decai epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_dec

    def train(self):
        """
        Loop principal de treinamento:
         - para cada episódio, reset do ambiente
         - coleta experiências passo a passo
         - chama experience_replay() a cada passo
        Retorna lista de recompensas por episódio.
        """
        rewards = []
        for i in range(self.episodes+1):
            state, _ = self.env.reset()
            # Ajusta estado p/ shape (1, state_dim)
            state = np.reshape(state, (1, self.env.observation_space.shape[0]))

            score = 0
            done = False
            steps = 0

            while not done:
                steps += 1
                action = self.select_action(state)
                next_state, reward, terminal, truncated, _ = self.env.step(action)

                # Termina se 'done' ou 'truncated' ou steps > max_steps
                if terminal or truncated or (steps > self.max_steps):
                    done = True

                score += reward
                next_state = np.reshape(next_state, (1, self.env.observation_space.shape[0]))

                # Salva a transição
                self.experience(state, action, reward, next_state, terminal)

                # Chama experience replay a cada passo
                self.experience_replay()

                # Atualiza estado
                state = next_state

            print(f"Episódio: {i+1}/{self.episodes}, Score: {score}")
            rewards.append(score)

            # Coleta de lixo e clear_session() ao fim do episódio
            gc.collect()
            keras.backend.clear_session()

        return rewards
