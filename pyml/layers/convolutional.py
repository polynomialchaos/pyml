# MIT License
#
# Copyright (c) 2021 Florian
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import numpy as np
from scipy.signal import correlate2d, convolve2d
from .layer import Layer
from .convolve import cross_correlate_2d, convolve_2d
# np.random.seed(0)

class ConvolutionalLayer(Layer):
    """Convolutional layer class."""

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

    def backward_propagation(self, output_error, learning_rate):
        kernel_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernel_gradient[i, j] += correlate2d(self.input[j],
                                                     output_error[i], 'valid')
                input_gradient[j] += convolve2d(output_error[i],
                                                self.kernels[i, j], 'full')
                # kernel_gradient[i, j] += cross_correlate_2d(self.input[j],
                #                                             output_error[i])
                # input_gradient[j] += convolve_2d(output_error[i],
                #                                  self.kernels[i, j],
                #                                  padding=(self.kernels_shape[2] - 1, self.kernels_shape[3] - 1))

        # update parameters
        self.kernels -= learning_rate * kernel_gradient
        self.biases -= learning_rate * output_error
        return input_gradient

    def forward_propagation(self):
        output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                output[i] += correlate2d(self.input[j],
                                         self.kernels[i, j], 'valid')
                # output[i] += cross_correlate_2d(self.input[j],
                #                                 self.kernels[i, j])

        return output
