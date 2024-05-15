import tensorflow as tf

class SamplingLayer(tf.keras.layers.Layer):
    '''A custom layer that receives (z_mean, z_var) and samples a z vector'''

    def call(self, inputs):
        
        z_mean, z_log_var = inputs
        
        batch_size, latent_dim = tf.shape(z_mean)[0], tf.shape(z_mean)[1]
        
        epsilon = tf.random.normal(shape=(batch_size, latent_dim))

        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon 
        
        return z
