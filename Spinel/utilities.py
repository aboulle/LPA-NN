import tensorflow as tf

class NormalizeL:
    """
    Normalizes and denormalizes label data using either MinMax or Gaussian (standard score) normalization.

    Parameters
    ----------
    y : array-like
        The data used to compute normalization statistics.
    norm : str, optional
        Normalization scheme to use: "MinMax" or "Gauss" (default is "MinMax").

    Methods
    -------
    forward(x)
        Applies normalization to input data x.
    backwards(x)
        Reverts normalized data x back to the original scale.
    """
    def __init__(self, y, norm="MinMax"):
        self.ymin = y.min(axis=0)
        self.ymax = y.max(axis=0)
        self.ymean = y.mean(axis=0)
        self.ystd = y.std(axis=0)
        self.norm = norm

    def forward(self, x):
        print(f"Label normalization scheme: {self.norm}")
        if self.norm == "MinMax":
            x -= self.ymin
            x /= self.ymax - self.ymin
        if self.norm == "Gauss":
            x -= self.ymean
            x /= self.ystd
        return x

    def backwards(self, x):
        print(f"Label normalization scheme: {self.norm}")
        if self.norm == "MinMax":
            x *= self.ymax - self.ymin
            x += self.ymin
        if self.norm == "Gauss":
            x *= self.ystd
            x += self.ymean
        return x


class Log10Layer(tf.keras.layers.Layer):
    """
    Custom Keras layer that computes the base-10 logarithm of its inputs.

    Methods
    -------
    call(inputs)
        Computes log10 of the input tensor.
    get_config()
        Returns the configuration of the layer for serialization.
    """
    def __init__(self, **kwargs):
        super(Log10Layer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.math.log(inputs) / tf.math.log(tf.constant(10.0, dtype=inputs.dtype))

    def get_config(self):
        config = super(Log10Layer, self).get_config()
        return config


class MinMaxScalingLayer(tf.keras.layers.Layer):
    """
    Custom Keras layer that applies Min-Max scaling to its inputs along a specified axis.

    Parameters
    ----------
    axis : int, optional
        The axis along which to compute the min and max values (default is 1).

    Methods
    -------
    call(inputs)
        Applies Min-Max scaling to the input tensor.
    get_config()
        Returns the configuration of the layer for serialization.
    """
    def __init__(self, axis=1, **kwargs):
        super(MinMaxScalingLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        min_val = tf.reduce_min(inputs, axis=self.axis, keepdims=True)
        max_val = tf.reduce_max(inputs, axis=self.axis, keepdims=True)
        scaled = (inputs - min_val) / (max_val - min_val + tf.keras.backend.epsilon())
        return scaled

    def get_config(self):
        config = super(MinMaxScalingLayer, self).get_config()
        config.update({"axis": self.axis})
        return config
