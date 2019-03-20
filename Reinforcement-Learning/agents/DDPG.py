import numpy as np
from agents.Actor import Actor
from agents.Critic import Critic
from agents.Noise import OUNoise
from agents.Buffer import ReplayBuffer
class DDPG():
    def __init__(self, task):
        self.task = task
        self.stateSize = task.stateSize
        self.actionSize = task.actionSize
        self.actionLow = task.actionLow
        self.actionHigh = task.actionHigh
        self.localActor = Actor(self.stateSize, self.actionSize, self.actionLow, self.actionHigh)
        self.targetActor = Actor(self.stateSize, self.actionSize, self.actionLow, self.actionHigh)
        self.localCritic = Critic(self.stateSize, self.actionSize)
        self.targetCritic = Critic(self.stateSize, self.actionSize)
        self.targetCritic.model.set_weights(self.localCritic.model.get_weights())
        self.targetActor.model.set_weights(self.localActor.model.get_weights())
        self.explorationMu = 0.0
        self.explorationTheta = 0.15
        self.explorationSigma = 0.2
        self.noise = OUNoise(self.actionSize, self.explorationMu, self.explorationTheta, self.explorationSigma)
        self.buffer= 100000
        self.batchSize = 64
        self.memory = ReplayBuffer(self.buffer, self.batchSize)
        self.gamma = 0.99
        self.tau = 0.01

    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last = state
        return state

    def step(self, action, reward, nextState, done):
        self.memory.add(self.last, action, reward, nextState, done)
        if len(self.memory) > self.batchSize:
            experiences = self.memory.sample()
            self.learn(experiences)
        self.last = nextState

    def act(self, state):
        state = np.reshape(state, [-1, self.stateSize])
        action = self.localActor.model.predict(state)[0]
        return list(action + self.noise.sample())

    def learn(self, experiences):
        states = np.vstack([exp.state for exp in experiences if exp is not None])

        actions = np.array([exp.action for exp in experiences if exp is not None]).astype(np.float32).reshape(-1, self.actionSize)

        rewards = np.array([exp.reward for exp in experiences if exp is not None]).astype(np.float32).reshape(-1, 1)

        dones = np.array([exp.done for exp in experiences if exp is not None]).astype(np.uint8).reshape(-1, 1)

        nextStates = np.vstack([exp.nextState for exp in experiences if exp is not None])

        next = self.targetActor.model.predict_on_batch(nextStates)

        QNext = self.targetCritic.model.predict_on_batch([nextStates, next])

        QTargets = rewards + self.gamma * QNext * (1 - dones)

        self.localCritic.model.train_on_batch(x=[states, actions], y=QTargets)

        Gradients = np.reshape(self.localCritic.get_action_gradients([states, actions, 0]), (-1, self.actionSize))

        self.localActor.train([states, Gradients, 1])

        self.soft(self.localCritic.model, self.targetCritic.model)

        self.soft(self.localActor.model, self.targetActor.model)   

    def soft(self, localModel, targetModel):
        local = np.array(localModel.get_weights())
        target = np.array(targetModel.get_weights())
        assert len(local) == len(target)
        new = self.tau * local + (1 - self.tau) * target
        targetModel.set_weights(new)