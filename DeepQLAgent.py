import torch
import torch.nn as nn
import torch.optim as optim

class DuelingDeepQLNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDeepQLNetwork, self).__init__()

        # Camada comum
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)

        # Stream do valor V(s)
        self.fc_value = nn.Linear(256, 256)
        self.value = nn.Linear(256, 1)

        # Stream da vantagem A(s, a)
        self.fc_advantage = nn.Linear(256, 256)
        self.advantage = nn.Linear(256, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))

        # Caminho do valor
        v = torch.relu(self.fc_value(x))
        v = self.value(v)

        # Caminho da vantagem
        a = torch.relu(self.fc_advantage(x))
        a = self.advantage(a)

        # Q(s, a) = V(s) + (A(s, a) - média(A))
        q_values = v + (a - a.mean(dim=1, keepdim=True))

        return q_values
