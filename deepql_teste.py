import gymnasium as gym
import numpy as np
from keras.models import load_model
import time

def main():
    # Carrega o ambiente e o modelo
    env = gym.make('MountainCar-v0', render_mode="human")
    model = load_model('data/model_mountain_car.keras')

    episodes = 5
    max_steps = 2000

    for ep in range(episodes):
        state, _ = env.reset()
        state = np.reshape(state, (1, env.observation_space.shape[0]))
        done = False
        score = 0
        steps = 0

        while not done:
            steps += 1
            # Seleciona ação greedily a partir do modelo
            action_values = model.predict(state, verbose=0)
            action = np.argmax(action_values[0])

            next_state, reward, terminal, truncated, _ = env.step(action)
            score += reward

            next_state = np.reshape(next_state, (1, env.observation_space.shape[0]))
            state = next_state

            if terminal or truncated or steps >= max_steps:
                done = True

            # Opcional: controlar velocidade do rendering
            time.sleep(0.01)

        print(f"Episode {ep+1}/{episodes}, Score: {score}")

    env.close()

if __name__ == "__main__":
    main()
