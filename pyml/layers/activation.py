from .layer import Layer


class ActivationLayer(Layer):
    """Activation layer class."""

    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation.function(self.input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return self.activation.function_derive(self.input) * output_error
