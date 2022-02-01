import numpy as np
from .layer import Layer


class ReshapeLayer(Layer):
    """Reshape layer class."""

    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def backward_propagation(self, output_data, learning_rate):
        return np.reshape(output_data, self.input_shape)

    def forward_propagation(self):
        return np.reshape(self.input, self.output_shape)

