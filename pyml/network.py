
class Network:
    """Neural network class."""

    def __init__(self, loss):
        self.loss = loss
        self.layers = []

    def add_layer(self, layer):
        """Add a layer to the network."""
        self.layers.append(layer)

    def evaluate(self, input_data):
        """Evaluate the trained neural network."""
        result = []
        for i in input_data:

            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)

            result.append(output)

        return result

    def train(self, x_train, y_train, epochs, learning_rate, print=False):
        """Train the neural network."""
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err += self.loss.function(y_train[j], output)

                # backward propagation
                error = self.loss.function_derive(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            if print:
                print('epoch %d/%d   error=%f' % (i+1, epochs, err))
