import numpy as np
from pyml.layer import Layer
from pyml.loss import Loss


class Network:
    """ Neural network class """

    def __init__(self, loss: Loss):
        self.loss = loss
        self.layers: list[Layer] = []

    def add_layer(self, layer: Layer):
        """ Add a layer to the network """
        self.layers.append(layer)

    def backward(self, output_error: np.ndarray, learning_rate: float):
        """ Backward propagation """
        output = output_error
        for layer in reversed(self.layers):
            output = layer.backward(output, learning_rate)

        return output

    def forward(self, input_data: np.ndarray):
        """ Forward propagation """
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)

        return output

    def train(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int, learning_rate: float, do_print: bool = False) -> float:
        """ Train the neural network """
        samples = len(x_train)

        # training loop
        for epoch in range(epochs):
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
                print(f'{epoch=} / {epochs}:  {err=}')

        return err
