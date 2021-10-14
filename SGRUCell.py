import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from keras.layers import AbstractRNNCell
from tensorflow.python.util import nest

def exp_safe(x):
    return tf.clip_by_value(tf.exp(x), clip_value_min=1e-10, clip_value_max=1e10)
   # return tf.exp(x)

class SGRUCell(AbstractRNNCell, keras.layers.Layer):
    def __init__(self, units, nclass, tanh_dim, **kwargs):
        super(SGRUCell, self).__init__(**kwargs)
        self.units = units
        self.nclass = nclass
        self.tanh_dim = tanh_dim
    @property
    def state_size(self):
        return self.units + 5
    @property
    def output_size(self):
        return self.units + 5

    def build(self, input_shape):
        self.Wd = self.add_weight(shape=(2, self.tanh_dim), initializer='glorot_uniform', name='Wd')
        self.bd = self.add_weight(shape=(self.tanh_dim), initializer='zeros', name='bd')
        self.Ws = self.add_weight(shape=(3, self.tanh_dim), initializer='glorot_uniform', name='Ws')
        self.bs = self.add_weight(shape=(self.tanh_dim), initializer='zeros', name='bs')

        self.bias_z = self.add_weight(shape=(self.units), initializer=keras.initializers.constant(0), name='bias_z')

        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units * 4), initializer='orthogonal', name='recurrent_kernel')
        self.kernel_c = self.add_weight(shape=(self.nclass, self.units * 4), initializer='glorot_uniform', name='kernel_c')
        self.kernel_d = self.add_weight(shape=(self.tanh_dim, self.units * 4), initializer='glorot_uniform', name='kernel_d')
        self.kernel_s = self.add_weight(shape=(self.tanh_dim, self.units * 4), initializer='glorot_uniform', name='kernel_s')
        self.bias = self.add_weight(shape=(self.units * 3), initializer='zeros', name='bias')

        self.Wpred = self.add_weight(shape=(self.units, 2), initializer='glorot_uniform', name='Wpred')
        self.bpred = self.add_weight(shape=(2), initializer='zeros', name='bpred')

        self.Wsoftmax = self.add_weight(shape=(self.units, 3), initializer='glorot_uniform', name='Wsoftmax')
        self.bsoftmax = self.add_weight(shape=(3), initializer='zeros', name='bsoftmax')

        self.built = True
    def call(self, inputs, states, training):
        tf.debugging.assert_all_finite(inputs, 'sgrucell inputs ill')
        _h_tm1 = states[0] if nest.is_sequence(states) else states  # previous memory
        h_tm1 = _h_tm1[:, :-5]

        d = _h_tm1[:, -5:-3]
        s = _h_tm1[:, -3:]
        _d = tf.tanh(tf.matmul(d, self.Wd) + self.bd)
        _s = tf.tanh(tf.matmul(s, self.Ws) + self.bs)

        ch = tf.cast(inputs[:, 0], tf.int32)

        _ch = tf.one_hot(ch, self.nclass)

        z = tf.sigmoid(tf.matmul(h_tm1, self.recurrent_kernel[:, :self.units])
                       + tf.matmul(_ch, self.kernel_c[:, :self.units])
                       + tf.matmul(_d, self.kernel_d[:, :self.units])
                       + tf.matmul(_s, self.kernel_s[:, :self.units])
                       + self.bias_z)

        r = tf.sigmoid(tf.matmul(h_tm1, self.recurrent_kernel[:, self.units:self.units * 2])
                       + tf.matmul(_ch, self.kernel_c[:, self.units:self.units * 2])
                       + tf.matmul(_d, self.kernel_d[:, self.units:self.units * 2])
                       + tf.matmul(_s, self.kernel_s[:, self.units:self.units * 2])
                       + self.bias[:self.units])

        hh = tf.tanh(tf.matmul(r * h_tm1, self.recurrent_kernel[:, self.units * 2:self.units * 3])
                       + tf.matmul(_ch, self.kernel_c[:, self.units * 2:self.units * 3])
                       + tf.matmul(_d, self.kernel_d[:, self.units * 2:self.units * 3])
                       + tf.matmul(_s, self.kernel_s[:, self.units * 2:self.units * 3])
                       + self.bias[self.units * 1:self.units * 2])
        h = z * h_tm1 + (1 - z) * hh
        o = tf.tanh(tf.matmul(h, self.recurrent_kernel[:, self.units * 3:])
                       + tf.matmul(_ch, self.kernel_c[:, self.units * 3:])
                       + tf.matmul(_d, self.kernel_d[:, self.units * 3:])
                       + tf.matmul(_s, self.kernel_s[:, self.units * 3:])
                       + self.bias[self.units * 2:])

        xy_pred = tf.matmul(o, self.Wpred) + self.bpred
        R3 = tf.matmul(o, self.Wsoftmax) + self.bsoftmax
        p = exp_safe(R3) / tf.reduce_sum(exp_safe(R3), axis=-1, keepdims=True)

        __pred = tf.concat([xy_pred, p], axis=-1)
        __h = tf.concat([h, __pred], axis=-1)
        __o = tf.concat([o, __pred], axis=-1)

        new_state = [__h] if nest.is_sequence(states) else __h
        return __o, new_state
    def get_config(self):
        config = super(SGRUCell, self).get_config()
        config.update({'units': self.units, 'nclass': self.nclass, 'tanh_dim': self.tanh_dim})
        return config