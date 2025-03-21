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
- Episódios = 1000
- Batch size = 128
- Learning rate = 0.1
- Memória de replay = 50000
- Max steps = 2000

### Curva de aprendizado

![treino ql](https://github.com/user-attachments/assets/a36726b7-9e4c-471b-b8e9-def6104b82ed)

## Deep Q-Learning

#### Observação

No Deep Q-Learning (Dueling DQN), foi utilizada uma dual network, que separa o cálculo do valor do estado e da vantagem das ações, o que ajuda a melhorar a estabilidade do treinamento. Além disso, foi feita uma alteração no método de recompensa, com a introdução de reward shaping para acelerar a aprendizagem, fornecendo um feedback mais claro e direcionado para o agente. Essas mudanças foram aplicadas para melhorar a convergência e a eficiência do modelo durante o treinamento.

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
| **Taxa de Sucesso**                        | 5/5 (100%)      | 5/5 (100%)        |
| **Média de Passos por Episódio**           | 146.8           | 113.8             |
| **Desvio Padrão dos Passos por Episódio**  | 2.86            | 0.4               |

## Análise dos resultados

O Q-Learning clássico, apesar de ter alcançado uma taxa de sucesso de 100%, apresentou uma maior variabilidade nas recompensas e um número médio maior de passos por episódio. Isso é esperado, dado que o Q-Learning baseia-se em uma tabela de valores que é atualizada de maneira direta e iterativa, o que pode levar a um processo de convergência mais lento e maior flutuação nos resultados. No entanto, a simples atualização da Q-table permite que o Q-Learning alcance uma solução eficiente quando o espaço de estados e ações é relativamente pequeno, como no caso do ambiente MountainCar-v0.

Por outro lado, o Deep Q-Learning (Dueling DQN), embora também tenha alcançado 100% de sucesso, demonstrou uma convergência mais estável com um número menor de passos por episódio e menos variabilidade nas recompensas. O uso de redes neurais permite que o Deep Q-Learning generalize melhor para estados não visitados anteriormente, aproveitando o processo de aprendizado por gradientes. O Dueling DQN, especificamente, separa o valor do estado e a recompensa, o que melhora a estabilidade do aprendizado ao reduzir a superestimação de certas ações. No entanto, o maior tempo de treinamento e a necessidade de mais interações com o ambiente são características esperadas em algoritmos baseados em redes neurais.

Essas diferenças refletem os trade-offs típicos entre métodos tabulados como o Q-Learning e métodos baseados em aproximação de funções como o Deep Q-Learning. Enquanto o Q-Learning é eficaz e rápido para ambientes de tamanho pequeno a médio, o Deep Q-Learning é mais adequado para problemas com espaços de estado e ação grandes ou quando a política exige generalização para estados não vistos. No entanto, o Deep Q-Learning sofre com maior variabilidade nas fases iniciais do treinamento devido à inicialização aleatória dos pesos da rede neural e à necessidade de um processo de replay de experiências.



