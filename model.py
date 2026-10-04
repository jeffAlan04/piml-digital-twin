import tensorflow as tf
from tensorflow.keras import layers

class EmbeddingLayer(layers.Layer):
    def __init__(
        self,
        d_model,
        num_types: int = 3,
        num_positions: int = 2,
        name="embedding_layer",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        # Hyperparameters
        self.num_types = num_types
        self.num_positions = num_positions
        self.d_model = d_model

        self.projection = layer.Dense(self.d_model, name="embedding_projection")

    def call(self, x):
        types = x[:, :, 0]
        positions = x[:, :, 1]
        states = x[:, :, 2]

        type_onehot = tf.one_hot(tf.cast(types, tf.int32), depth = self.num_types)

        position_onehot = tf.one_hot(tf.cast(positions, tf.int32), depth = self.num_positions)

        states = tf.expand_dims(states, axis = -1)

        concat = tf.concat([type_onehot, position_onehot, states], axis = -1)

        embedded = self.projection(concat)

        return embedded



