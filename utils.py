import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Layer


class Utils:

    @staticmethod
    def create_hann_window(size):
        w = np.hanning(size)
        window_2d = np.outer(w, w)
        return np.expand_dims(window_2d, axis=-1).astype(np.float32)

    @staticmethod
    def get_grid_coordinates(image_dim, patch_size, min_overlap=64):
        if image_dim <= patch_size:
            return [0]

        n_patches = int(
            np.ceil((image_dim - min_overlap) / (patch_size - min_overlap))
        )
        step = (image_dim - patch_size) / (n_patches - 1)
        return [int(round(i * step)) for i in range(n_patches)]

    @staticmethod
    def layer_threshold(img1, img2, alpha):
        beta = (1.0 - alpha)
        result = cv2.addWeighted(img1, alpha, img2, beta, 0.0)
        return result

@tf.keras.utils.register_keras_serializable(package='Custom', name='SimpleGate')
class SimpleGate(Layer):
    def __init__(self, **kwargs):
        super(SimpleGate, self).__init__(**kwargs)

    def call(self, inputs):
        x1, x2 = tf.split(inputs, 2, axis=-1)
        return x1 * x2

    def get_config(self):
        config = super(SimpleGate, self).get_config()
        return config

@tf.keras.utils.register_keras_serializable(package='Custom', name='LayerScale')
class LayerScale(tf.keras.layers.Layer):
    def __init__(self, init_value=1e-2, **kwargs):
        super(LayerScale, self).__init__(**kwargs)
        self.init_value = init_value

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name='gamma',
            shape=(1, 1, 1, input_shape[-1]),
            initializer=tf.keras.initializers.Constant(self.init_value),
            trainable=True
        )
        super(LayerScale, self).build(input_shape)

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        config = super(LayerScale, self).get_config()
        config.update({'init_value': self.init_value})
        return config