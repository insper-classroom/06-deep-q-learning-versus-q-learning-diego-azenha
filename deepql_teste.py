import gymnasium as gym
import torch
import numpy as np
import time
from DeepQLAgent import DuelingDeepQLNetwork  # <-- agora importando daqui

def testar(algoritmo="deepql", episodes=5, max_steps=1000):
    env = gym.make("MountainCar-v0", render_mode="human")

    # Carrega o modelo treinado
    model = DuelingDeepQLNetwork(env.observation_space.shape[0], env.action_space.n)
    model.load_state_dict(torch.load('data/model_mountain_car_exec1.pt'))
    model.eval()

    sucessos = 0
    steps_por_ep = []
    recompensas_totais = []

    for ep in range(episodes):
        state, _ = env.reset()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        done = False
        soma_recompensas = 0
        truncated = False
        step_count = 0

        for _ in range(max_steps):
            if done or truncated:
                break
            step_count += 1

            with torch.no_grad():
                action_values = model(state_tensor)
            action = torch.argmax(action_values).item()

            next_state, reward, done, truncated, _ = env.step(action)
            soma_recompensas += reward

            # Atualiza o estado para o próximo
            state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

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
    alg = "deepql"
    testar(alg, episodes=5, max_steps=1000)
