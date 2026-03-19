import torch
import torch.nn as nn

class BrainGNN(nn.Module):
    def __init__(self, num_nodes, hidden_dim):
        super(BrainGNN, self).__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, adj):
        h = torch.matmul(adj, x)
        h = torch.relu(self.fc1(h))
        h = self.fc2(h)
        return h
