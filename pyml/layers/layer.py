from turtle import forward


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
