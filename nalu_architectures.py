import numpy as np
import tensorflow as tf

from inalu.nalu_layers import *
from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import Dense

class INALUModel(Sequential):
    def __init__(self, *args, **kwargs):
        super(INALUModel, self).__init__(*args, **kwargs)

        self.add(Nalui2Layer(2, name="hidden1"))
        self.add(Nalui2Layer(1, name="hidden2"))

    #def call(self, inputs, training=False, mask=None):
    #    x = self.hidden1(inputs, training=training)
    #    x = self.hidden2(x, training=training)

    #    return x


    def train_step(self, data):
        x, y = data
        # Run forward pass.
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred)
        #self._validate_target_and_loss(y, loss)
        # Run backwards pass.
        print(self.trainable_variables)
        print(tape.watched_variables())
        import pdb; pdb.set_trace()
        1/0
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return self.compute_metrics(x, y, y_pred, sample_weight)

    #def train_step(self, data):
    #    x, y = data

    #    with tf.GradientTape() as tape:
    #        y_pred = self(x, training=True)
    #        loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

    #    # Compute gradients
    #    trainable_vars = self.trainable_variables
    #    gradients = tape.gradient(loss, trainable_vars)
    #    for v, g in zip(trainable_vars, gradients):
    #        print(f"{v}: {g}")

    #    # Update weights
    #    self.optimizer.apply_gradients(zip(gradients, trainable_vars))

    #    # Update metrics (includes the metric that tracks the loss)
    #    self.compiled_metrics.update_state(y, y_pred)

    #    # Return a dict mapping metric names to current value
    #    return {m.name: m.result() for m in self.metrics}

if __name__ == "__main__":
    x = np.random.random(size=(52550, 10))
    y = np.random.random(size=(52550, 1))

    #model = INALUModel()
    model = Sequential([
            Nalui2Layer(2, name="hidden1"),
            Nalui2Layer(1, name="hidden2")
            ])
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"])

    #model.build(x.shape)
    model.fit(x, y, epochs=10)
    model.summary()
