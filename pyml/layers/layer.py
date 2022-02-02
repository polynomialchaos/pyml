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
class Layer:
    """Layer (base) class."""

    def __init__(self):
        self.input = None
        self.output = None

    def backward(self, output_error, learning_rate):
        """Handler to call backward propagation."""
        return self.backward_propagation(output_error, learning_rate)

    def backward_propagation(self, output_error, learning_rate):
        """Computes dE/dX for a given output_error=dE/dY
        (and update parameters if any)."""
        raise NotImplementedError

    def forward(self, input_data):
        """Handler to call forward propagation."""
        self.input = input_data
        self.output = self.forward_propagation()
        return self.output

    def forward_propagation(self):
        """Computes the output Y of a layer for a given input X."""
        raise NotImplementedError
