import numpy as np

def circadian_cycle(t):
    return 0.5 + 0.5 * np.sin(2 * np.pi * t / 24)
