import pickle
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from model import construct_model
import settings as ss
from Loss import Loss
from SGRUCell import SGRUCell
from PostProcess import PostProcess
from draw_chars import draw_chars

tf.random.set_seed(123)
np.random.seed(1234)

with open(ss.data_path + "x_y_lb_n_" + str(ss.nclass) + "_r_" + str(ss.repeat) + "_dist_" + str(ss.remove_dist_th) + "_ang_" + str(ss.remove_ang_th) + "_drop_" + str(ss.drop) + "_np_" + str(ss.noise_prob) + "_nr_" + str(ss.noise_ratio), 'rb') as f:
    x, y = pickle.load(f)

dataset = tf.data.Dataset.from_generator(lambda: iter(zip(x, y)), output_types=(tf.float32, tf.float32),output_shapes=([None, None, 6], [None, None, 6]))
take_batches = dataset.repeat().shuffle(10000)

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.45)))
tf.compat.v1.keras.backend.set_session(sess)

loss_tracker = keras.metrics.Mean(name="loss")
mae_metric = keras.metrics.MeanAbsoluteError(name="mae")

if True:
    class CustomModel(keras.Model):
        def train_step(self, data):
            x, y = data
       #     print('\n')
            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)

                # print(tf.reduce_any(tf.math.is_nan(y_pred)))
                loss = Loss(y, y_pred)
                #print(tf.reduce_any(tf.math.is_nan(loss)))
       #     _loss = loss.numpy()
       #     print(_loss.min())
       #     print(_loss.max())

            trainable_vars = self.trainable_variables
            _gradients = tape.gradient(loss, trainable_vars)
      #      gradients = tf.clip_by_global_norm(_gradients, clip_norm=10)
            gradients = []
            for i in range(13):
                # if i <= 3 or i >= 9:
                #     _g = tf.clip_by_value(_gradients[i], clip_value_min=-100, clip_value_max=100)
                # else:
                #     _g = tf.clip_by_value(_gradients[i], clip_value_min=-10, clip_value_max=10)
                _g = tf.clip_by_norm(_gradients[i], clip_norm=100)
                gradients.append(_g)

         #   if tf.reduce_any([tf.reduce_any(tf.math.is_nan(g)).numpy() for i, g in enumerate(gradients)]):
         #       print('gradient nan')
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            #print(tf.reduce_any([tf.reduce_any(tf.math.is_nan(v)).numpy() for i, v in enumerate(trainable_vars)]))
            # Compute our own metrics
            loss_tracker.update_state(loss)
          #  mae_metric.update_state(y, y_pred)
            return {"loss": loss_tracker.result()}

        @property
        def metrics(self):
            # We list our `Metric` objects here so that `reset_states()` can be
            # called automatically at the start of each epoch
            # or at the start of `evaluate()`.
            # If you don't implement this property, you have to call
            # `reset_states()` yourself at the time of your choosing.
            return [loss_tracker]


    class CustomCallback(keras.callbacks.Callback):
        def __init__(self, model2, ckp_path):
            self.model2 = model2
            self.ckp_path = ckp_path
        def on_epoch_begin(self, epoch, logs=None):
            pass
            tf.random.set_seed(123 + epoch)
        def on_epoch_end(self, epoch, logs=None):
            self.model2.load_weights(tf.train.latest_checkpoint(self.ckp_path))
            draw_chars(x, self.model2, [0, 1, 2, 3, 4], 50, False, self.ckp_path + 'epoch_' + str(epoch + 1))

    rnn_cell = SGRUCell(units=ss.units, in_tanh_dim=ss.in_tanh_dim, nclass=ss.nclass, dropout=0., recurrent_dropout=0.)
    rnn_layer = tf.keras.layers.RNN(rnn_cell, return_state=False, return_sequences=True, stateful=False)
    post_process_layer = PostProcess(M=ss.M)

    inputs = keras.Input(batch_shape=[None, None, 6])
    gru_out = rnn_layer(inputs)
    outputs = post_process_layer(gru_out)
    model = CustomModel(inputs, outputs)
   # optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))
  #  model.compile(optimizer=optimizer)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ss.checkpoint_path + 'ck_{epoch}', save_weights_only=True)
    custom_callback = CustomCallback(construct_model(ss.units, ss.in_tanh_dim, ss.nclass, True, ss.M, [1, 1, 6]), ss.checkpoint_path)
    model.run_eagerly = False
  #  model.load_weights(tf.train.latest_checkpoint(ss.checkpoint_path))
    model.fit(take_batches, steps_per_epoch=ss.steps_per_epoch, epochs=ss.epochs, initial_epoch=0,
              callbacks=[checkpoint_callback, custom_callback])

if False:
    model = construct_model(units, in_tanh_dim, nclass, False, M, [None, None, 6])
    model.compile(loss=Loss, optimizer=keras.optimizers.Adam())
  #  model.load_weights(tf.train.latest_checkpoint(checkpoint_path))
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path + 'ck_{epoch}', save_weights_only=True)
    model.fit(take_batches, steps_per_epoch=steps_per_epoch, epochs=epochs, initial_epoch=0, callbacks=[checkpoint_callback])

if False:
    def train_step(model, x, y, optimizer):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = Loss(y, y_pred)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return np.mean(loss)

    model = construct_model(units, in_tanh_dim, nclass, False, M, [None, None, 6])

    optimizer = keras.optimizers.Adam()

    print('\n')
    with tf.device("/gpu:0"):
        for i in range(epochs):
            time_begin = time.time()
            loss_sum = 0.0
            for j in range(steps_per_epoch):
                next_batch = next(take_batches.as_numpy_iterator())
                loss_sum += train_step(model, next_batch[0], next_batch[1], optimizer)
                print('\r' + 'epoch = ' + str(i+1) + '/' + str(epochs) + ' step = ' + str(j+1) + '/' + str(steps_per_epoch) + ' loss = ' + str(loss_sum / (j+1)) + ' speed = ' + str((time.time() - time_begin) / (j+1) * 1000) + ' ms/step', end='', flush=True)
            print('\n')


exit()


# class CustomCallback(keras.callbacks.Callback):
#     def __init__(self, model):
#         self.model = model
#
#     def on_epoch_end(self, epoch):
#         y_pred = self.model.predict()
#         print('y predicted: ', y_pred)
