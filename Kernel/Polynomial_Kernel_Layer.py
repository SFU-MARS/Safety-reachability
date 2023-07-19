import tensorflow as tf

class PolynomialKernelLayer(tf.keras.layers.Layer):
    def __init__(self, degree=2, trainable=True, **kwargs):
        super(PolynomialKernelLayer, self).__init__(trainable=trainable, **kwargs)
        self.degree = degree

    def build(self, input_shape):
        self.coefficient = self.add_weight(
            name='coefficient',
            shape=(1,),
            initializer=tf.keras.initializers.Constant(value=1.0),
            trainable=self.trainable,
        )
        super(PolynomialKernelLayer, self).build(input_shape)

    def call(self, inputs):
        return tf.pow(inputs * self.coefficient + 1, self.degree)

    def get_config(self):
        config = super(PolynomialKernelLayer, self).get_config()
        config.update({'degree': self.degree})
        return config
