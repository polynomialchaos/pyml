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
from keras.datasets import mnist
from keras.utils import np_utils
from pyml import Network
from pyml.layers import ActivationLayer
from pyml.layers import ConvolutionalLayer, FullyConnectedLayer, ReshapeLayer
from pyml.functions import BinaryCrossEntropyLoss, SigmoidActivation


def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = np_utils.to_categorical(y)
    y = y.reshape(len(y), 2, 1)
    return x, y


# training data
# load MNIST from server,
# limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 100)

# network
network = Network(BinaryCrossEntropyLoss())
network.add_layer(ConvolutionalLayer((1, 28, 28), 3, 5))
network.add_layer(ActivationLayer(SigmoidActivation()))
network.add_layer(ReshapeLayer((5, 26, 26), (5 * 26 * 26, 1)))
network.add_layer(FullyConnectedLayer(5 * 26 * 26, 100))
network.add_layer(ActivationLayer(SigmoidActivation()))
network.add_layer(FullyConnectedLayer(100, 2))
network.add_layer(ActivationLayer(SigmoidActivation()))

# train
network.train(x_train, y_train, epochs=20, learning_rate=0.1, do_print=True)

# test
for x, y in zip(x_test, y_test):
    output = network.forward(x)
    y, y_hat = np.argmax(y), np.argmax(output)
    if y != y_hat:
        print('ok={:}: pred={:} / true={:}'.format(int(y == y_hat), y_hat, y))
