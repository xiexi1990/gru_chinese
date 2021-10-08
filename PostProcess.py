import tensorflow as tf
import tensorflow.keras as keras
class PostProcess(keras.layers.Layer):
    def __init__(self, M, **kwargs):
        super(PostProcess, self).__init__(**kwargs)
        self.M = M
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.Wgmm = self.add_weight(shape=(input_dim, self.M * 5), initializer='glorot_uniform', name='Wgmm')
        self.bgmm = self.add_weight(shape=(self.M * 5), initializer='zeros', name='bgmm')
        self.Wsoftmax = self.add_weight(shape=(input_dim, 3), initializer='glorot_uniform', name='Wsoftmax')
        self.bsoftmax = self.add_weight(shape=(3), initializer='zeros', name='bsoftmax')
        self.built = True
    def call(self, inputs, **kwargs):
        R5M = tf.matmul(inputs, self.Wgmm) + self.bgmm
        _pi = R5M[:, :, :self.M]
        pi = tf.exp(_pi) / tf.reduce_sum(tf.exp(_pi), axis=-1, keepdims=True)
        mux = R5M[:, :, self.M:self.M * 2]
        muy = R5M[:, :, self.M * 2:self.M * 3]
        sigmax = tf.exp(R5M[:, :, self.M * 3:self.M * 4])
        sigmay = tf.exp(R5M[:, :, self.M * 4:])
        R3 = tf.matmul(inputs, self.Wsoftmax) + self.bsoftmax
        p = tf.exp(R3) / tf.reduce_sum(tf.exp(R3), axis=-1, keepdims=True)
        return tf.concat([pi, mux, muy, sigmax, sigmay, p], axis=-1)
    def get_config(self):
        config = super(PostProcess, self).get_config()
        config.update({"M": self.M})
        return config