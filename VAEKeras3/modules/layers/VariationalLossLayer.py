import tensorflow as tf
from tensorflow.keras import backend as K

class VariationalLossLayer(tf.keras.layers.Layer):
    def __init__(self, loss_weights=[3, 7], **kwargs):
        super(VariationalLossLayer, self).__init__(**kwargs)
        self.k1 = loss_weights[0]
        self.k2 = loss_weights[1]

    def call(self, inputs):
        x, z_mean, z_log_var, y = inputs

        # Reconstruction loss
        r_loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(x, y), axis=(1, 2))

        # KL divergence loss
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)

        # Total loss
        loss = tf.reduce_mean(r_loss * self.k1 + kl_loss * self.k2)

        self.add_loss(loss)

        return y

    def get_config(self):
        return {'loss_weights': [self.k1, self.k2]}

    