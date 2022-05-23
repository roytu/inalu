
import tensorflow as tf
from inalu.nalu_utils import *


class Nalui2Layer(tf.keras.layers.Layer):
    def __init__(self, num_outputs, name="Nalui2", **kwargs):
        super(Nalui2Layer, self).__init__(name=name, **kwargs)
        self.num_outputs = num_outputs

    def build(self, input_shape):
        w_hat1 = self.add_weight(
                "w_hat1",
                shape=(input_shape[-1], self.num_outputs),
                initializer="random_normal",
                trainable=True)

        m_hat1 = self.add_weight(
                "m_hat1",
                shape=(input_shape[-1], self.num_outputs),
                initializer=tf.keras.initializers.RandomNormal(
                    mean=0.5, stddev=0.2, seed=None
                    ),
                trainable=True)

        w_hat2 = self.add_weight(
                "w_hat2",
                shape=(input_shape[-1], self.num_outputs),
                initializer=tf.keras.initializers.RandomNormal(
                    mean=0.88, stddev=0.2, seed=None
                    ),
                trainable=True)

        m_hat2 = self.add_weight(
                "m_hat2",
                shape=(input_shape[-1], self.num_outputs),
                initializer=tf.keras.initializers.RandomNormal(
                    mean=0.5, stddev=0.2, seed=None
                    ),
                trainable=True)

        G1 = self.add_weight(
                "g",
                shape=(self.num_outputs,),
                initializer=tf.keras.initializers.RandomNormal(
                    mean=0.0, stddev=0.2, seed=None
                    ),
                trainable=True)

        self.W1 = tf.tanh(w_hat1) * tf.sigmoid(m_hat1)
        self.W2 = tf.tanh(w_hat2) * tf.sigmoid(m_hat2)
        self.g1 = tf.sigmoid(G1)

    def call(self, inputs):
        # sign
        W1s = tf.reshape(self.W2, [-1])  # flatten W1s to (200)
        W1s = tf.abs(W1s)
        Xs = tf.concat([inputs] * self.W1.shape[1], axis=1)
        Xs = tf.reshape(Xs, shape=[-1, self.W1.shape[0] * self.W1.shape[1]])
        sgn = tf.sign(Xs) * W1s + (1 - W1s)
        sgn = tf.reshape(sgn, shape=[-1, self.W1.shape[1], self.W1.shape[0]])
        ms1 = tf.reduce_prod(sgn, axis=2)
        a1 = tf.matmul(inputs, self.W1)

        m1 = tf.exp(tf.minimum(tf.matmul(tf.math.log(tf.maximum(tf.abs(inputs), 1e-7)), self.W2), 20))  # clipping
        
        out_add = self.g1 * a1
        out_sgn = tf.clip_by_value(ms1, -1, 1)
        out_mul = (1 - self.g1) * m1
        out = out_add + out_mul * out_sgn
        return out

