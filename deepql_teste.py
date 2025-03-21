import gymnasium as gym
import torch
import numpy as np
import time

from DeepQLAgent import DeepQLNetwork  # <-- agora importando daqui

def main():
    env = gym.make('MountainCar-v0', render_mode="human")
    model = DeepQLNetwork(env.observation_space.shape[0], env.action_space.n)
    model.load_state_dict(torch.load('data/model_mountain_car_exec1.pt'))
    model.eval()

    episodes = 5
    max_steps = 2000

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        score = 0
        steps = 0

        while not done:
            steps += 1
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_values = model(state_tensor)
            action = torch.argmax(action_values).item()

            next_state, reward, terminal, truncated, _ = env.step(action)
            score += reward
            state = next_state

            if terminal or truncated or steps >= max_steps:
                done = True

            time.sleep(0.01)

        print(f"Episode {ep+1}/{episodes}, Score: {score}")

    env.close()

if __name__ == "__main__":
    main()
