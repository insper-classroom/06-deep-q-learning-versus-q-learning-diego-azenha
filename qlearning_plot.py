import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(rewards_list, algorithm, alpha, gamma, epsilon, episodes):
    """
    Plota múltiplas curvas de aprendizado no mesmo gráfico para visualizar a variabilidade.
    """
    window = 10  # Janela da média móvel

    plt.figure(figsize=(8, 5))
    
    for i, rewards in enumerate(rewards_list):
        rolling_avg = []
        for j in range(len(rewards)):
            inicio = max(0, j - window + 1)
            subset = rewards[inicio : j + 1]
            rolling_avg.append(np.mean(subset))

        alpha_value = 1.0 if i == 0 else 0.5  # Primeira execução destacada, outras mais opacas
        plt.plot(rolling_avg, label=f"Execução {i+1}", alpha=alpha_value)

    # Criando o título com hiperparâmetros
    title = (f"Curva de Aprendizado - {algorithm}\n"
             f"α={alpha}, γ={gamma}, ε={epsilon}, Episódios={episodes}")

    plt.xlabel("Episódios")
    plt.ylabel("Recompensa")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()