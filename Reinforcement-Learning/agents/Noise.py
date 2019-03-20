import copy, random
import numpy as np
class OUNoise:
    def __init__(self, size, mu, theta, sigma):
        self.mu=mu*np.ones(size)
        self.theta=theta
        self.sigma=sigma
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu) #Shallow copy

    def sample(self):
        x = self.state
        dx = self.theta*(self.mu-x)+self.sigma*np.random.randn(len(x))
        self.state=x+dx
        return self.state