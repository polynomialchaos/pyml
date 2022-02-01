import numpy as np
from pyml import Network
from pyml.layers import FullyConnectedLayer, ActivationLayer
from pyml.functions import MeanSquaredErrorLoss, HyperbolicTangentActivation

# training data
x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = Network(MeanSquaredErrorLoss())
net.add_layer(FullyConnectedLayer(2, 3))
net.add_layer(ActivationLayer(HyperbolicTangentActivation()))
net.add_layer(FullyConnectedLayer(3, 1))
net.add_layer(ActivationLayer(HyperbolicTangentActivation()))

# train
net.train(x_train, y_train, epochs=1000, learning_rate=0.1)

# test
out = net.evaluate(x_train)
print(out)
