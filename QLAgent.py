import numpy as np

class QLearningAgent:
    def __init__(self, env, alpha, gamma, epsilon, epsilon_min, epsilon_decay):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Discretização do espaço de estados
        self.num_states = (env.observation_space.high - env.observation_space.low) * np.array([10, 100])
        self.num_states = np.round(self.num_states, 0).astype(int) + 1

        # Inicialização da Q-table
        # Dimensões: [pos_x, vel_x, num_actions]
        self.Q = np.zeros((self.num_states[0], self.num_states[1], env.action_space.n))

    def transform_state(self, state):
        """
        Converte o estado contínuo em índices discretos.
        """
        s = (state - self.env.observation_space.low) * np.array([10, 100])
        return np.round(s, 0).astype(int)

    def select_action(self, s):
        """
        Seleção de ação epsilon-greedy.
        """
        if np.random.rand() > self.epsilon:
            # Exploração da Q-table
            return np.argmax(self.Q[s[0], s[1]])
        else:
            # Exploração aleatória
            return np.random.randint(0, self.env.action_space.n)

    def update(self, s, a, r, s_next, done):
        """
        Atualização do valor de Q segundo a equação de Q-Learning:
        Q(s,a) <- Q(s,a) + alpha * [r + gamma*maxQ(s',_) - Q(s,a)]
        """
        current_q = self.Q[s[0], s[1], a]
        if done:
            target = r
        else:
            target = r + self.gamma * np.max(self.Q[s_next[0], s_next[1]])
        self.Q[s[0], s[1], a] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self):
        """
        Decaimento do epsilon (exploração) após cada episódio.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
