import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin, AbstractRNNCell
from tensorflow.python.util import nest

class SGRUCell(DropoutRNNCellMixin, keras.layers.Layer):
    def __init__(self, units, in_tanh_dim, nclass, dropout=0., recurrent_dropout=0., **kwargs):
        super(SGRUCell, self).__init__(**kwargs)
        self.units = units
        self.in_tanh_dim = in_tanh_dim
        self.nclass = nclass
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
    @property
    def state_size(self):
        return self.units
    @property
    def output_size(self):
        return self.units
    def build(self, input_shape):
        input_dim = input_shape[-1]
        assert len(input_shape) == 2 and input_dim == 6
        self.Wd = self.add_weight(shape=(2, self.in_tanh_dim), initializer='glorot_uniform', name='Wd')
        self.bd = self.add_weight(shape=(self.in_tanh_dim), initializer='zeros', name='bd')
        self.Ws = self.add_weight(shape=(3, self.in_tanh_dim), initializer='glorot_uniform', name='Ws')
        self.bs = self.add_weight(shape=(self.in_tanh_dim), initializer='zeros', name='bs')
        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units * 4), initializer='orthogonal', name='recurrent_kernel')
        self.kernel_d = self.add_weight(shape=(self.in_tanh_dim, self.units*4), initializer='glorot_uniform', name='kernel_d')
        self.kernel_s = self.add_weight(shape=(self.in_tanh_dim, self.units*4), initializer='glorot_uniform', name='kernel_s')
        self.kernel_c = self.add_weight(shape=(self.nclass, self.units * 4), initializer='glorot_uniform', name='kernel_c')
        self.bias = self.add_weight(shape=(self.units*4), initializer='zeros', name='bias')
        self.built = True
    def call(self, inputs, states, training):
        # if (training == True) and tf.reduce_all(tf.math.is_nan(states)):
        #     raise Exception("SGRUCell: states contains nan")
        # if (training == True) and tf.reduce_all(tf.math.is_nan(inputs)):
        #     tf.print(inputs)
        #     raise Exception("SGRUCell: inputs contains nan")
        assert not tf.reduce_any(tf.math.is_nan(inputs))
        h_tm1 = states[0] if nest.is_sequence(states) else states  # previous memory
        d = inputs[:, 0:2]
        s = inputs[:, 2:5]
        ch = tf.cast(inputs[:, 5], tf.int32)
        _d = tf.tanh(tf.matmul(d, self.Wd) + self.bd)
        _s = tf.tanh(tf.matmul(s, self.Ws) + self.bs)
        _ch = tf.one_hot(ch, self.nclass)
        _d_mask = self.get_dropout_mask_for_cell(_d, training, count=3)
        _s_mask = self.get_dropout_mask_for_cell(_s, training, count=3)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(h_tm1, training, count=3)
        if 0. < self.dropout < 1.:
            _d_z = _d * _d_mask[0]
            _d_r = _d * _d_mask[1]
            _d_h = _d * _d_mask[2]
            _s_z = _s * _s_mask[0]
            _s_r = _s * _s_mask[1]
            _s_h = _s * _s_mask[2]
        else:
            _d_z = _d
            _d_r = _d
            _d_h = _d
            _s_z = _s
            _s_r = _s
            _s_h = _s
        if 0. < self.recurrent_dropout < 1.:
            h_tm1_z = h_tm1 * rec_dp_mask[0]
            h_tm1_r = h_tm1 * rec_dp_mask[1]
            h_tm1_h = h_tm1 * rec_dp_mask[2]
        else:
            h_tm1_z = h_tm1
            h_tm1_r = h_tm1
            h_tm1_h = h_tm1
        z = tf.sigmoid(tf.matmul(h_tm1_z, self.recurrent_kernel[:, :self.units])
                       + tf.matmul(_d_z, self.kernel_d[:, :self.units])
                       + tf.matmul(_s_z, self.kernel_s[:, :self.units])
                       + tf.matmul(_ch, self.kernel_c[:, :self.units])
                       + self.bias[:self.units])
        r = tf.sigmoid(tf.matmul(h_tm1_r, self.recurrent_kernel[:, self.units:self.units * 2])
                       + tf.matmul(_d_r, self.kernel_d[:, self.units:self.units * 2])
                       + tf.matmul(_s_r, self.kernel_s[:, self.units:self.units * 2])
                       + tf.matmul(_ch, self.kernel_c[:, self.units:self.units * 2])
                       + self.bias[self.units:self.units * 2])
        hh = tf.tanh(tf.matmul(r * h_tm1_h, self.recurrent_kernel[:, self.units * 2:self.units * 3])
                       + tf.matmul(_d_h, self.kernel_d[:, self.units * 2:self.units * 3])
                       + tf.matmul(_s_h, self.kernel_s[:, self.units * 2:self.units * 3])
                       + tf.matmul(_ch, self.kernel_c[:, self.units * 2:self.units * 3])
                       + self.bias[self.units * 2:self.units * 3])
        h = z * h_tm1 + (1 - z) * hh
        o = tf.tanh(tf.matmul(h, self.recurrent_kernel[:, self.units * 3:])
                       + tf.matmul(_d_r, self.kernel_d[:, self.units * 3:])
                       + tf.matmul(_s_r, self.kernel_s[:, self.units * 3:])
                       + tf.matmul(_ch, self.kernel_c[:, self.units * 3:])
                       + self.bias[self.units * 3:])
        new_state = [h] if nest.is_sequence(states) else h
        return o, new_state
    def get_config(self):
        config = super(SGRUCell, self).get_config()
        config.update({'units': self.units, 'in_tanh_dim': self.in_tanh_dim, 'nclass': self.nclass, 'dropout': self.dropout, 'recurrent_dropout': self.recurrent_dropout})
        return config