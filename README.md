# Q-Learning vs Deep Q-Learning no ambiente MountainCar

## Estrutura do código

O código está estruturado da seguinte forma:

- deepql_treino.py: Implementação do Deep Q-Learning (Dueling DQN).
- qlearning_treino.py: Implementação do Q-Learning clássico.
- deepql_plot.py: Função para plotar gráficos de Deep Q-Learning.
- qlearning_plot.py: Função para plotar gráficos de Q-Learning clássico.
- main.py: Arquivo para rodar as comparações entre ambos os algoritmos.
- DeepQLAgent.py: Arquivo com a classe para a rede neural usada no Deep Q-Learning.
- QLAgent.py: Arquivo com a classe para o agente no Q-Learning clássico.


## Q-Learning

### Hiperparâmetros

- Gamma = 0.99
- Epsilon = 1
- Epsilon mínimo = 0.01
- Epsilon decay = 0.995
- Episódios = 500
- Batch size = 128
- Learning rate = 0.1
- Memória de replay = 50000
- Max steps = 2000

### Curva de aprendizado

![treino ql](https://github.com/user-attachments/assets/05c07a5f-dec4-4479-b5b1-e2c3fbe3a4da)


## Deep Q-Learning

### Hiperparâmetros

- Gamma = 0.99
- Epsilon = 1
- Epsilon mínimo = 0.01
- Epsilon decay = 0.995
- Episódios = 1000
- Batch size = 128
- Learning rate = 0.001
- Memória de replay = 50000
- Max steps = 2500

### Curva de aprendizado

![Treino Deep QL](https://github.com/user-attachments/assets/95fb14b6-e9a0-4a4d-81a9-484b2145e7c5)

## Comparação dos algoritmos - resultados dos testes

### Comparação de Métricas de Performance

| **Métrica de Performance**      | **Q-Learning (Clássico)**    | **Deep Q-Learning (Dueling DQN)**   |
|---------------------------------|------------------------------|-------------------------------------|
| **Taxa de Sucesso**                        | xxx           | 5/5 (100%)        |
| **Média de Passos por Episódio**           | xxx           | 160.4             |
| **Desvio Padrão dos Passos por Episódio**  | xxx           | 9.13              |

