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
class Network:
    """Neural network class."""

    def __init__(self, loss):
        self.loss = loss
        self.layers = []

    def add_layer(self, layer):
        """Add a layer to the network."""
        self.layers.append(layer)

    def backward(self, output_error, learning_rate):
        """Backward propagation."""
        output = output_error
        for layer in reversed(self.layers):
            output = layer.backward(output, learning_rate)

        return output

    def forward(self, input_data):
        """Forward propagation."""
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)

        return output

    def train(self, x_train, y_train, epochs, learning_rate, do_print=False):
        """Train the neural network."""
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for x, y in zip(x_train, y_train):
                # forward propagation
                output = self.forward(x)

                # compute loss (for display purpose only)
                err += self.loss.function(y, output)

                # backward propagation
                error = self.loss.function_derive(y, output)
                self.backward(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            if do_print:
                print('epoch %d/%d   error=%f' % (i+1, epochs, err))
