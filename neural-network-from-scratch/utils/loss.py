import numpy as np


class Cost:
    @staticmethod
    def mse(y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    @staticmethod
    def mse_prime(y_true, y_pred):
        return 2 * (y_true - y_pred) / np.size(y_true)
