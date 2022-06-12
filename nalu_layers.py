
import tensorflow as tf
from nalu_utils import *


class Nalui2Layer(tf.keras.layers.Layer):
    def __init__(self, num_outputs, name="Nalui2", **kwargs):
        super(Nalui2Layer, self).__init__(name=name, **kwargs)
        self.num_outputs = num_outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_outputs": self.num_outputs
        })
        return config

    def build(self, input_shape):
        self.W_add = self.add_weight(
                "W_add",
                shape=(input_shape[-1], self.num_outputs),
                initializer=tf.keras.initializers.RandomNormal(
                    mean=0.88, stddev=0.2, seed=None
                    ),
                trainable=True)

        self.M_add = self.add_weight(
                "M_add",
                shape=(input_shape[-1], self.num_outputs),
                initializer=tf.keras.initializers.RandomNormal(
                    mean=0.5, stddev=0.2, seed=None
                    ),
                trainable=True)

        self.W_mul = self.add_weight(
                "W_mul",
                shape=(input_shape[-1], self.num_outputs),
                initializer=tf.keras.initializers.RandomNormal(
                    mean=0.88, stddev=0.2, seed=None
                    ),
                trainable=True)

        self.M_mul = self.add_weight(
                "M_mul",
                shape=(input_shape[-1], self.num_outputs),
                initializer=tf.keras.initializers.RandomNormal(
                    mean=0.5, stddev=0.2, seed=None
                    ),
                trainable=True)

        self.g = self.add_weight(
                "g",
                shape=(self.num_outputs,),
                initializer=tf.keras.initializers.RandomNormal(
                    mean=0.0, stddev=0.2, seed=None
                    ),
                trainable=True)

        self.WM_add = tf.Variable(np.zeros((input_shape[-1], self.num_outputs), dtype=np.float32), name="WM_add", trainable=False)
        self.WM_mul = tf.Variable(np.zeros((input_shape[-1], self.num_outputs), dtype=np.float32), name="WM_mul", trainable=False)

    def call(self, inputs, training=None):
        WM_add = tf.math.tanh(self.W_add) * tf.math.sigmoid(self.M_add)
        WM_mul = tf.math.tanh(self.W_mul) * tf.math.sigmoid(self.M_mul)

        self.WM_add.assign(WM_add)
        self.WM_mul.assign(WM_mul)
        g1 = tf.sigmoid(self.g)

        W_nx = WM_add.shape[0]
        W_ny = WM_add.shape[1]

        # sign
        WM_mul_f = tf.abs(tf.reshape(WM_mul, [-1]))
        Xs = tf.concat([inputs] * WM_add.shape[1], axis=1)
        Xs = tf.reshape(Xs, shape=[-1, W_nx * W_ny])
        sgn = tf.sign(Xs) * WM_mul_f + (1 - WM_mul_f)
        sgn = tf.reshape(sgn, shape=[-1, W_ny, W_nx])
        ms1 = tf.reduce_prod(sgn, axis=2)
        a1 = tf.matmul(inputs, WM_add)
        m1 = tf.exp(tf.minimum(tf.matmul(tf.math.log(tf.maximum(tf.abs(inputs), 1e-7)), WM_mul), 20))  # clipping
        
        out_add = g1 * a1
        out_sgn = tf.clip_by_value(ms1, -1, 1)
        out_mul = (1 - g1) * m1
        out = out_add + out_mul * out_sgn

        # Add regularized losses
        def mse_plus_reg_loss(v):
            t = 20
            reg_loss = tf.reduce_mean(tf.maximum(tf.minimum(-v, v) + t, 0) / t)
            return reg_loss

        W_add_loss = mse_plus_reg_loss(self.W_add)
        self.add_loss(W_add_loss)
        self.add_metric(W_add_loss, name="W_add_loss")

        M_add_loss = mse_plus_reg_loss(self.M_add)
        self.add_loss(M_add_loss)
        self.add_metric(M_add_loss, name="M_add_loss")

        W_mul_loss = mse_plus_reg_loss(self.W_mul)
        self.add_loss(W_mul_loss)
        self.add_metric(W_mul_loss, name="W_mul_loss")

        M_mul_loss = mse_plus_reg_loss(self.M_mul)
        self.add_loss(M_mul_loss)
        self.add_metric(M_mul_loss, name="M_mul_loss")

        g_loss = mse_plus_reg_loss(self.g)
        self.add_loss(g_loss)
        self.add_metric(g_loss, name="g_loss")

        return out

    def get_algebraic_repr(self):
        num_inputs = self.W_add.shape[0]
        WM_add = tf.math.tanh(self.W_add) * tf.math.sigmoid(self.M_add)
        WM_mul = tf.math.tanh(self.W_mul) * tf.math.sigmoid(self.M_mul)
        s = ""
        for output_ind in range(self.num_outputs):
            s_add = ""
            s_mul = ""
            gw = self.g[output_ind].numpy()

            if gw > 0.5:
                # Add / Subtract
                for x_ind in range(num_inputs):
                    if WM_add.numpy()[x_ind, output_ind] > 0:
                        # Add
                        s_add += f"+ x[{x_ind}]"
                    else:
                        # Sub
                        s_add += f"- x[{x_ind}]"
            else:
                # Multiply / Divide
                for x_ind in range(num_inputs):
                    if WM_mul.numpy()[x_ind, output_ind] > 0:
                        # Mul
                        s_mul += f"* x[{x_ind}]"
                    else:
                        # Div
                        s_mul += f"/ x[{x_ind}]"
            s += f"y[{output_ind}] = ({s_add}) + ({s_mul})\n"

        return s
