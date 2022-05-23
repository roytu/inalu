import numpy as np
import tensorflow as tf

from inalu.nalu_layers import *
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense

class INALUModel(Sequential):
  def __init__(self):
    super(INALUModel, self).__init__()

    self.hidden1 = Nalui2Layer(2)
    self.hidden2 = Nalui2Layer(1)

  def call(self, x):
    x = self.hidden1(x)
    x = self.hidden2(x)
    return x

if __name__ == "__main__":
    x = np.random.uniform((52550, 10))
    y = np.random.uniform((52550, 1))

    model = INALUModel()
    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["mean_squared_error"])
    model.fit(x, y, epochs=10)
