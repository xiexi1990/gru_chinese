from SGRUCell import *
from PostProcess import *

def construct_model(rnn_cell_units, in_tanh_dim, nclass, stateful, M, batch_shape):
    rnn_cell = SGRUCell(units=rnn_cell_units, in_tanh_dim=in_tanh_dim, nclass=nclass, dropout=0., recurrent_dropout=0.)
    rnn_layer = tf.keras.layers.RNN(rnn_cell, return_state=False, return_sequences=True, stateful=stateful, dynamic=True)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(batch_shape=batch_shape),
        rnn_layer,
        PostProcess(M=M, dynamic=True)
      ])
    return model