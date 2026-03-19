import numpy as np

class LithiumAgent:
    def __init__(self):
        self.q_table = np.zeros((10, 3))
        self.lr = 0.1
        self.gamma = 0.9
        self.epsilon = 0.2

    def get_state(self, signal):
      signal = np.nan_to_num(signal, nan=0.0, posinf=1e6, neginf=-1e6)

      mean_val = signal.mean()
      state = int(np.clip(abs(mean_val) * 2, 0, 9))

      return state

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(3)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        best_next = np.max(self.q_table[next_state])
        self.q_table[state, action] += self.lr * (
            reward + self.gamma * best_next - self.q_table[state, action]
        )
