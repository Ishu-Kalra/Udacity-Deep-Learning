import numpy as np
from keras import layers, models, optimizers
from keras import backend
class Actor:
    def __init__(self, stateSize, actionSize, actionLow, actionHigh):
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.actionLow = actionLow
        self.actionHigh = actionHigh
        self.actionRange = self.actionHigh - self.actionLow
        self.build()

    def build(self):
        total = layers.Input(shape=(self.stateSize,), name='total')
        layer1 = layers.Dense(units=32, activation='relu')(total)
        layer1 = layers.Dense(units=64, activation='relu')(layer1)
        layer1 = layers.Dense(units=32, activation='relu')(layer1)
        rawActions = layers.Dense(units=self.actionSize, activation='sigmoid',
            name='rawActions')(layer1)
        actions = layers.Lambda(lambda x: (x * self.actionRange) + self.actionLow,
            name='actions')(rawActions)
        self.model = models.Model(inputs=total, outputs=actions)
        Gradients = layers.Input(shape=(self.actionSize,))
        loss = backend.mean(-Gradients * actions)
        optimizer = optimizers.Adam(lr = 0.0005)
        updates = optimizer.get_updates(params=self.model.trainable_weights, loss=loss, constraints = [])
        self.train = backend.function(
            inputs=[self.model.input, Gradients, backend.learning_phase()],
            outputs=[],
            updates=updates)