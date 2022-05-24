
import tensorflow as tf
from inalu.nalu_utils import *


class Nalui2Layer(tf.keras.layers.Layer):
    def __init__(self, num_outputs, name="Nalui2", **kwargs):
        super(Nalui2Layer, self).__init__(name=name, **kwargs)
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.w_hat1 = self.add_weight(
                "w_hat1",
                shape=(input_shape[-1], self.num_outputs),
                initializer=tf.keras.initializers.RandomNormal(
                    mean=0.88, stddev=0.2, seed=None
                    ),
                trainable=True)

        self.m_hat1 = self.add_weight(
                "m_hat1",
                shape=(input_shape[-1], self.num_outputs),
                initializer=tf.keras.initializers.RandomNormal(
                    mean=0.5, stddev=0.2, seed=None
                    ),
                trainable=True)

        self.w_hat2 = self.add_weight(
                "w_hat2",
                shape=(input_shape[-1], self.num_outputs),
                initializer=tf.keras.initializers.RandomNormal(
                    mean=0.88, stddev=0.2, seed=None
                    ),
                trainable=True)

        self.m_hat2 = self.add_weight(
                "m_hat2",
                shape=(input_shape[-1], self.num_outputs),
                initializer=tf.keras.initializers.RandomNormal(
                    mean=0.5, stddev=0.2, seed=None
                    ),
                trainable=True)

        self.G1 = self.add_weight(
                "g",
                shape=(self.num_outputs,),
                initializer=tf.keras.initializers.RandomNormal(
                    mean=0.0, stddev=0.2, seed=None
                    ),
                trainable=True)

    def call(self, inputs, training=None):
        W1 = tf.math.tanh(self.w_hat1) * tf.math.sigmoid(self.m_hat1)
        W2 = tf.tanh(self.w_hat2) * tf.sigmoid(self.m_hat2)
        g1 = tf.sigmoid(self.G1)

        # sign
        W1s = tf.reshape(W2, [-1])  # flatten W1s to (200)
        W1s = tf.abs(W1s)
        Xs = tf.concat([inputs] * W1.shape[1], axis=1)
        Xs = tf.reshape(Xs, shape=[-1, W1.shape[0] * W1.shape[1]])
        sgn = tf.sign(Xs) * W1s + (1 - W1s)
        sgn = tf.reshape(sgn, shape=[-1, W1.shape[1], W1.shape[0]])
        ms1 = tf.reduce_prod(sgn, axis=2)
        a1 = tf.matmul(inputs, W1)

        m1 = tf.exp(tf.minimum(tf.matmul(tf.math.log(tf.maximum(tf.abs(inputs), 1e-7)), W2), 20))  # clipping
        
        out_add = g1 * a1
        out_sgn = tf.clip_by_value(ms1, -1, 1)
        out_mul = (1 - g1) * m1
        out = out_add + out_mul * out_sgn
        return out

