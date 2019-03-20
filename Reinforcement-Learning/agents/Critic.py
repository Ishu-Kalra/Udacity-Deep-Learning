import numpy as np
from keras import layers, models, optimizers
from keras import backend
class Critic:
    def __init__(self, stateSize, actionSize):
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.build()
    def build(self):
        total = layers.Input(shape=(self.stateSize,), name='total')
        actions = layers.Input(shape=(self.actionSize,), name='actions')
        LayerStates = layers.Dense(units=32, activation='relu')(total)
        LayerStates = layers.Dense(units=64, activation='relu')(LayerStates)
        LayerActions = layers.Dense(units=32, activation='relu')(actions)
        LayerActions = layers.Dense(units=64, activation='relu')(LayerActions)
        LayerSum = layers.Add()([LayerStates, LayerActions])
        LayerSum = layers.Activation('relu')(LayerSum)

        Q = layers.Dense(units=1, name='Q')(LayerSum)
        self.model = models.Model(inputs=[total, actions], outputs=Q)
        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss='mse')
        Gradients = backend.gradients(Q, actions)
        self.get_action_gradients = backend.function(
            inputs=[*self.model.input, backend.learning_phase()],
            outputs=Gradients)