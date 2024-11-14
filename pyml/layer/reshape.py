import numpy as np
from .layer import Layer


class ReshapeLayer(Layer):
    """ Reshape layer class """

    def __init__(self, input_shape: tuple[int], output_shape: tuple[int]):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def backward_propagation(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
        return np.reshape(output_error, self.input_shape)

    def forward_propagation(self) -> np.ndarray:
        return np.reshape(self.input, self.output_shape)
