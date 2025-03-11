########################################################
# main.py
# Arquivo principal que executa todos os passos
# e gera os resultados necessários para cumprir as solicitações.
#
# - Treina e testa Q-Learning (qlearning_treino.py, qlearning_teste.py)
# - Treina e testa Deep Q-Learning (deepql_treino.py, deepql_teste.py)
# - Plota os gráficos e compara Q-Learning vs. Deep Q-Learning
#
########################################################

import os
import numpy as np
import matplotlib.pyplot as plt

# Imports do Q-Learning
from qlearning_treino import treinar as ql_treinar
from qlearning_teste import testar as ql_testar
from qlearning_plot import plot_learning_curve as plot_ql_curve

# Imports do Deep Q-Learning
from deepql_treino import treinar_deepql
from deepql_teste import main as deepql_test
from deepql_plot import plot_deepql_curve

# 1) Configuração e criação de pastas

def criar_pastas_necessarias():
    """
    Cria as pastas 'data' e 'results' (se não existirem)
    para armazenar modelos, gráficos e CSVs.
    """
    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists("results"):
        os.mkdir("results")

# 2) Plot de comparação final entre Q-Learning e DQL

def comparar_qlearning_e_deepql(
    ql_rewards_list,
    dql_rewards_list,
    episodes_ql=500,
    episodes_dql=300
):
    """
    Gera um gráfico de comparação entre as recompensas
    de Q-Learning e Deep Q-Learning em um único plot,
    usando a média móvel de cada execução.
    """

    plt.figure(figsize=(8, 5))
    window = 10

    # Q-Learning

    # 2.1) Calcula média da Q-Learning
    ql_array = np.array(ql_rewards_list)  # shape: (num_execucoes, episodes_ql)
    ql_mean = np.mean(ql_array, axis=0)   # média ao longo das execuções
    ql_rolling = []
    for i in range(len(ql_mean)):
        start_idx = max(0, i - window + 1)
        subset = ql_mean[start_idx : i+1]
        ql_rolling.append(np.mean(subset))

    # 2.2) Calcula média da DQL
    dql_array = np.array(dql_rewards_list)
    dql_mean = np.mean(dql_array, axis=0)
    dql_rolling = []
    for i in range(len(dql_mean)):
        start_idx = max(0, i - window + 1)
        subset = dql_mean[start_idx : i+1]
        dql_rolling.append(np.mean(subset))

    # 2.3) Plot
    plt.plot(ql_rolling, label="Q-Learning (média)", color="blue")
    plt.plot(dql_rolling, label="Deep Q-Learning (média)", color="red")

    plt.title("Comparação Q-Learning vs. Deep Q-Learning (Média de Recompensas)")
    plt.xlabel("Episódios")
    plt.ylabel("Recompensa Média (móvel)")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/comparacao_ql_vs_dql.jpg")
    plt.show()

# 3) Função principal

def main():
    criar_pastas_necessarias()

    # (A) Q-Learning: Treino + Plot + Test

    print("\n========== TREINANDO Q-LEARNING ==========")
    ql_recompensas, ql_metricas = ql_treinar("qlearning")

    print("\n========== TESTANDO Q-LEARNING ==========")
    ql_testar(algoritmo="qlearning", episodes=5, max_steps=1000)


    # (B) Deep Q-Learning: Treino + Plot + Test

    print("\n========== TREINANDO DEEP Q-LEARNING ==========")
    dql_recompensas = treinar_deepql()

    print("\n========== TESTANDO DEEP Q-LEARNING ==========")
    deepql_test()  # Executa o main() do deepql_teste.py


    # (C) Comparação Final: Q-Learning vs Deep Q-Learning

    print("\n========== COMPARANDO Q-LEARNING VS. DEEP Q-LEARNING ==========")
    comparar_qlearning_e_deepql(
        ql_rewards_list=ql_recompensas,
        dql_rewards_list=dql_recompensas,
        episodes_ql=500,
        episodes_dql=300
    )

    print("\n========== TODOS OS PROCESSOS CONCLUIDOS ==========")


if __name__ == "__main__":
    main()
