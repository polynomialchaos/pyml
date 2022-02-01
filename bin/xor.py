import numpy as np
import matplotlib.pyplot as plt

from pyml import Network
from pyml.layers import FullyConnectedLayer, ActivationLayer
from pyml.functions import MeanSquaredErrorLoss, HyperbolicTangentActivation

# training data
x_train = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
y_train = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

# network
net = Network(MeanSquaredErrorLoss())
net.add_layer(FullyConnectedLayer(2, 3))
net.add_layer(ActivationLayer(HyperbolicTangentActivation()))
net.add_layer(FullyConnectedLayer(3, 1))
net.add_layer(ActivationLayer(HyperbolicTangentActivation()))

# train
net.train(x_train, y_train, epochs=10000, learning_rate=0.1)

# decision boundary plot
points = []
for x in np.linspace(0, 1, 20):
    for y in np.linspace(0, 1, 20):
        z = net.forward([[x], [y]])
        points.append([x, y, z[0, 0]])

points = np.array(points)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2],
           c=points[:, 2], cmap="winter")
plt.show()
