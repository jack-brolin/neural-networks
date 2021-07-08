import numpy as np


class Sigmoid:
    @staticmethod
    def activation(layer_input):
        return 1 / (1 + np.exp(-layer_input))

    @staticmethod
    def activation_prime(layer_input):
        return 1 / (1 + np.exp(-layer_input)) * (1 - 1 / (1 + np.exp(-layer_input)))


class TanH:
    @staticmethod
    def activation(layer_input):
        return np.tanh(layer_input)

    @staticmethod
    def activation_prime(layer_input):
        return 1 - np.power(np.tanh(layer_input), 2)


class Softmax:
    @staticmethod
    def activation(layer_input):
        e_x = np.exp(layer_input - np.max(layer_input, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    @staticmethod
    def activation_prime(layer_input):
        e_x = np.exp(layer_input - np.max(layer_input, axis=-1, keepdims=True))
        p = e_x / np.sum(e_x, axis=-1, keepdims=True)
        return p * (1 - p)


class ReLU:
    @staticmethod
    def activation(layer_input):
        return np.where(layer_input >= 0, layer_input, 0)

    @staticmethod
    def activation_prime(layer_input):
        return np.where(layer_input >= 0, 1, 0)
