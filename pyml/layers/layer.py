class Layer:
    """Layer (base) class."""

    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagation(self, input_data):
        """Computes the output Y of a layer for a given input X."""
        raise NotImplementedError

    def backward_propagation(self, output_error, learning_rate):
        """Computes dE/dX for a given output_error=dE/dY
        (and update parameters if any)."""
        raise NotImplementedError
