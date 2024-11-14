import numpy as np
from pyml.activation import Activation
from .layer import Layer


class ActivationLayer(Layer):
    """ Activation layer class """

    def __init__(self, activation: Activation):
        super().__init__()
        self.activation = activation

    def backward_propagation(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
        return np.multiply(output_error, self.activation.function_derive(self.input))

    def forward_propagation(self) -> np.ndarray:
        return self.activation.function(self.input)
