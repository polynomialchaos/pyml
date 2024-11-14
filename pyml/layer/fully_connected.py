import numpy as np
from .layer import Layer


class FullyConnectedLayer(Layer):
    """ Fully connected layer class """

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def backward_propagation(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
        weights_error = np.dot(output_error, self.input.T)
        input_error = np.dot(self.weights.T, output_error)

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error

    def forward_propagation(self) -> np.ndarray:
        return np.dot(self.weights, self.input) + self.bias
