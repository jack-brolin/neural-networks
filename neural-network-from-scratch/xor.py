from utils.layers import FCLayer, Activation
from utils.nn import NeuralNetwork

from utils.loss import Cost

import numpy as np

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = NeuralNetwork()

network.add(FCLayer(input_size=2, output_size=3))
network.add(Activation(activation_func="ReLU"))
network.add(FCLayer(input_size=3, output_size=1))
network.add(Activation(activation_func="Sigmoid"))

# train
network.set_loss(Cost.mse, Cost.mse_prime)
network.fit(X, Y, epochs=1000, learning_rate=0.1)

# test
out = network.predict(X)
print(out)
