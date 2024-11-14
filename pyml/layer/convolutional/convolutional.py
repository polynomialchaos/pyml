import numpy as np
from scipy import signal
from ..layer import Layer


class ConvolutionalLayer(Layer):
    """ Convolutional layer class """

    def __init__(self, input_shape, kernel_size, depth):
        super().__init__()
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height -
                             kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def _correlate_2d(self, in1: np.ndarray, in2: np.ndarray) -> np.ndarray:
        return signal.correlate2d(in1, in2, 'valid')

    def _convolve_2d(self, in1: np.ndarray, in2: np.ndarray) -> np.ndarray:
        return signal.correlate2d(in1, in2, 'full')

    def backward_propagation(self, output_error, learning_rate):
        kernel_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernel_gradient[i,
                                j] += self._correlate_2d(self.input[j], output_error[i])
                input_gradient[j] += self._convolve_2d(
                    output_error[i], self.kernels[i, j])

        # update parameters
        self.kernels -= learning_rate * kernel_gradient
        self.biases -= learning_rate * output_error
        return input_gradient

    def forward_propagation(self):
        output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                output[i] += self._correlate_2d(self.input[j],
                                                self.kernels[i, j])

        return output
