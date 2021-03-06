import tensorflow as tf
from SGRUCell import SGRUCell
from PostProcess import PostProcess

def construct_model(rnn_cell_units, nclass, M, stateful, batch_shape):
    rnn_cell = SGRUCell(units=rnn_cell_units, nclass=nclass, M=M)
    rnn_layer = tf.keras.layers.RNN(rnn_cell, return_state=False, return_sequences=True, stateful=stateful)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(batch_shape=batch_shape),
        rnn_layer,
        PostProcess(M=M)
      ])
    return model