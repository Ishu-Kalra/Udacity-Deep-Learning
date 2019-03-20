import numpy as np
import random
from collections import namedtuple, deque
class ReplayBuffer:
    def __init__(self, bufferSize, batchSize):
        self.memory = deque(maxlen=bufferSize)  # internal memory (deque)
        self.batchSize = batchSize
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "nextState", "done"])

    def add(self, state, action, reward, nextState, done):
        e = self.experience(state, action, reward, nextState, done)
        self.memory.append(e)

    def sample(self, batch_size=128):
        return random.sample(self.memory, k=self.batchSize)

    def __len__(self):
        return len(self.memory)