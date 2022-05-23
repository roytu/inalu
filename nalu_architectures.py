import numpy as np
import tensorflow as tf

from nalu_layers import *
from tensorflow.keras import Model

class INALUModel(Model):
  def __init__(self):
    super(INALUModel, self).__init__()

    self.hidden1 = Nalui2Layer(2)
    self.hidden2 = Nalui2Layer(1)

  def call(self, x):
    x = self.hidden1(x)
    x = self.hidden2(x)
    return x

if __name__ == "__main__":
    x = np.random.uniform(100)

    writer = tf.summary.create_file_writer("logs")

    with writer.as_default():
        model = INALUModel()
        writer.add_graph(model, x)
        writer.flush()

    #with tf.Session() as sess:
    #    writer = tf.summary.FileWriter("output", sess.graph)
    #    print(sess.run(nn))
    #    writer.close()
