import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx

from brain_gnn import BrainGNN
from rl_agent import LithiumAgent
from circadian import circadian_cycle

num_nodes = 20
timesteps = 200

G = nx.erdos_renyi_graph(num_nodes, 0.2)
adj = nx.to_numpy_array(G)
adj = torch.tensor(adj, dtype=torch.float32)

model = BrainGNN(num_nodes, hidden_dim=8)
agent = LithiumAgent()

state = torch.randn(num_nodes, 1)
history = []

for t in range(timesteps):
    state = model(state, adj)
    signal = state.detach().numpy()

    s = agent.get_state(signal)
    action = agent.choose_action(s)

    lithium_dose = action * 0.1
    circadian = circadian_cycle(t)

    state = state * (1 - lithium_dose * circadian)

    reward = -np.var(signal)
    next_s = agent.get_state(signal)
    agent.update(s, action, reward, next_s)

    history.append(signal.mean())

plt.plot(history)
plt.title("GNN Bipolar Brain Stabilized by RL + Lithium + Circadian")
plt.xlabel("Time")
plt.ylabel("Neural Activity")
plt.grid()
plt.show()
