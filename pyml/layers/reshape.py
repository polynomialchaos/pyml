import numpy as np
from .layer import Layer


class ReshapeLayer(Layer):
    """Reshape layer class."""

    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward_propagation(self, input_data):
        return np.reshape(input_data, self.output_shape)

    def backward_propagation(self, output_data, learning_rate):
        return np.reshape(output_data, self.input_shape)
