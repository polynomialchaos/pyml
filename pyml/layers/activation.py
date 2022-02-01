import numpy as np
from .layer import Layer


class ActivationLayer(Layer):
    """Activation layer class."""

    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    def forward_propagation(self):
        return self.activation.function(self.input)

    def backward_propagation(self, output_error, learning_rate):
        return np.multiply(output_error,
                           self.activation.function_derive(self.input))
