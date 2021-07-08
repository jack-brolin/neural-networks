import numpy as np

from utils.activations import Sigmoid, TanH, Softmax, ReLU


class Layer:
    def __init__(self):
        self.layer_input = None
        self.output = None

    def forward(self, layer_input):
        # TODO: return output
        pass

    def backward(self, output_gradient, learning_rate):
        # TODO: update parameters and return input gradient
        pass


class FCLayer(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.rand(output_size, input_size)
        # self.bias = np.zeros((output_size, 1))
        self.bias = np.random.randn(output_size, 1)

    def forward(self, layer_input):
        self.layer_input = layer_input
        return np.dot(self.weights, self.layer_input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weight_gradient = np.dot(output_gradient, self.layer_input.T)
        self.weights -= learning_rate * weight_gradient
        self.bias -= learning_rate * output_gradient
        return np.dot(self.weights.T, output_gradient)


class Activation(Layer):
    def __init__(self, activation_func):
        super().__init__()
        self.activation_func = activation_func

    def forward(self, layer_input):
        self.layer_input = layer_input
        return activation_dict[self.activation_func].activation(self.layer_input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, activation_dict.get(self.activation_func).activation_prime(self.layer_input))


activation_dict = {
    "Sigmoid": Sigmoid,
    "TanH": TanH,
    "Softmax": Softmax,
    "ReLU": ReLU
}
