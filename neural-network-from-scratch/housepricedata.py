from utils.layers import FCLayer, Activation
from utils.nn import NeuralNetwork

from utils.loss import Cost

import pandas as pd
import numpy as np

data = pd.read_csv("./data/housepricedata.csv")
X = np.reshape(data[[
    "LotArea",
    "OverallQual",
    "OverallCond",
    "TotalBsmtSF",
    "FullBath",
    "HalfBath",
    "BedroomAbvGr",
    "TotRmsAbvGrd",
    "Fireplaces",
    "GarageArea"
]].to_numpy(), (data.shape[0], data.shape[1] - 1, 1))
Y = np.reshape(data[["AboveMedianPrice"]].to_numpy(), (data.shape[0], 1, 1))

network = NeuralNetwork()

network.add(FCLayer(input_size=10, output_size=20))
network.add(Activation(activation_func="ReLU"))
network.add(FCLayer(input_size=20, output_size=10))
network.add(Activation(activation_func="ReLU"))
network.add(FCLayer(input_size=10, output_size=8))
network.add(Activation(activation_func="ReLU"))
network.add(FCLayer(input_size=8, output_size=6))
network.add(Activation(activation_func="ReLU"))
network.add(FCLayer(input_size=6, output_size=3))
network.add(Activation(activation_func="ReLU"))
network.add(FCLayer(input_size=3, output_size=1))
network.add(Activation(activation_func="Sigmoid"))

# train
network.set_loss(Cost.mse, Cost.mse_prime)
network.fit(X, Y, epochs=1000, learning_rate=0.1)

# test
out = network.predict(X)
print(out)
