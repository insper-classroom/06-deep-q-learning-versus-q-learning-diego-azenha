import numpy as np
import matplotlib.pyplot as plt

def plot_deepql_curve(
    rewards_list,
    algorithm,
    gamma,
    epsilon,
    epsilon_min,
    epsilon_dec,
    episodes,
    batch_size,
    learning_rate
):

    window = 10  # Tamanho da janela para média móvel
    plt.figure(figsize=(8, 5))

    for exec_idx, rewards in enumerate(rewards_list):
        rolling_avg = []
        for i in range(len(rewards)):
            start_idx = max(0, i - window + 1)
            subset = rewards[start_idx : i + 1]
            rolling_avg.append(np.mean(subset))

        # Destaque da primeira curva, demais ficam mais opacas
        alpha_value = 1.0 if exec_idx == 0 else 0.5
        plt.plot(rolling_avg, label=f"Exec {exec_idx+1}", alpha=alpha_value)

    # Título com hiperparâmetros principais
    title = (
        f"{algorithm}\n"
        f"gamma={gamma}, eps={epsilon}, eps_min={epsilon_min}, eps_dec={epsilon_dec},\n"
        f"batch_size={batch_size}, lr={learning_rate}, episodes={episodes}"
    )
    plt.title(title)
    plt.xlabel("Episódios")
    plt.ylabel("Recompensa (média móvel)")
    plt.grid(True)
    plt.legend()
    plt.show()
