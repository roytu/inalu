import numpy as np
import tensorflow as tf

from nalu_layers import *
from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D


class INALUModel(Sequential):

    def compute_loss(self, x, y, y_pred, sample_weight):
        loss = tf.reduce_mean(tf.math.squared_difference(y_pred, y))
        loss += tf.add_n(self.losses)
        self.loss_tracker.update_state(loss)
        return loss

    #def train_step(self, data):
    #    x, y = data
    #    # Run forward pass.
    #    with tf.GradientTape() as tape:
    #        y_pred = self(x, training=True)
    #        loss = self.compute_loss(x, y, y_pred)
    #    #self._validate_target_and_loss(y, loss)
    #    # Run backwards pass.
    #    self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
    #    return self.compute_metrics(x, y, y_pred, sample_weight)

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

    should_train = True

    E_gnds = np.load("datasets/E_gnds.npy")
    n_gnds = np.load("datasets/n_gnds.npy")

    SAMPLES = 105100
    k = 3

    # Take the first 8 values only and expand to L+k-1 to
    # reflect periodicity

    n_gnds = n_gnds[:, :8]  # Use only spin-ups (should be the same as spin-downs)
    n_gnds = np.hstack([n_gnds, n_gnds[:, :k-1]])
    #n_gnds = np.expand_dims(n_gnds, axis=2)  # Reshape to fit with Conv1D
    E_gnds = np.expand_dims(E_gnds, axis=1)

    # Split into training and test
    split_val = 0.50  # 0.50 = 50% training, 50% test
    split_n = int(split_val * SAMPLES)

    train_E_gnds = E_gnds[:split_n]
    train_n_gnds = n_gnds[:split_n]
    test_E_gnds = E_gnds[split_n:]
    test_n_gnds = n_gnds[split_n:]

    #x = np.random.random(size=(52550, 10))
    #y = np.random.random(size=(52550, 1))

    #model = INALUModel()
    model = Sequential()
    #model.add(Conv1D(8, (3,), input_shape=(10, 1)))
    #model.add(Activation("relu"))
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(Dense(1))
    #model.add(Nalui2Layer(10, name="hidden1"))
    #model.add(Nalui2Layer(10, name="hidden2"))
    #model.add(Nalui2Layer(10, name="hidden3"))
    #model.add(Nalui2Layer(10, name="hidden4"))
    #model.add(Nalui2Layer(10, name="hidden5"))
    #model.add(Nalui2Layer(1, name="output"))

    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["mean_squared_error"])
    
    logdir = "logs"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    if should_train:
        #model.build(x.shape)
        history = model.fit(train_n_gnds, train_E_gnds, epochs=5, callbacks=[tensorboard_callback])

        # Test results
        results = model.evaluate(test_n_gnds, test_E_gnds)

        # Print metrics
        print("=" * 20)
        print("Metrics:")
        for metric_name, result in zip(model.metrics_names, results):
            print(f"\t{metric_name}: {result}")

        # Save model
        #model.save_weights("model_1cnn_2dense.ckpt")
        #model.save_weights("model_2layer.ckpt")
        model.summary()

        # Algebraic representation
        for i, layer in enumerate(model.layers):
            print(f"Layer {i}:")
            print(layer.get_algebraic_repr())

    else:
        model.build(n_gnds.shape)
        model.load_weights("model_2layer.ckpt")
        model.summary()

        # Test results
        results = model.evaluate(test_n_gnds, test_E_gnds)

        # Print metrics
        print("=" * 20)
        print("Metrics:")
        for metric_name, result in zip(model.metrics_names, results):
            print(f"\t{metric_name}: {result}")

        # Algebraic representation
        for i, layer in enumerate(model.layers):
            print(f"Layer {i}:")
            print(layer.get_algebraic_repr())

