import pickle
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from settings import *
from Loss import Loss
from model import construct_model

tf.random.set_seed(123)
np.random.seed(1234)

with open(data_path + "x_y_lb_n_" + str(nclass) + "_r_" + str(repeat) + "_dist_" + str(remove_dist_th) + "_ang_" + str(remove_ang_th) + "_drop_" + str(drop) + "_np_" + str(noise_prob) + "_nr_" + str(noise_ratio), 'rb') as f:
    x, y = pickle.load(f)

dataset = tf.data.Dataset.from_generator(lambda: iter(zip(x, y)), output_types=(tf.float32, tf.float32),output_shapes=([None, None, 6], [None, None, 6]))
take_batches = dataset.repeat().shuffle(10000)

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.45)))
tf.compat.v1.keras.backend.set_session(sess)

loss_tracker = keras.metrics.Mean(name="loss")
mae_metric = keras.metrics.MeanAbsoluteError(name="mae")

if True:
    model = construct_model(units, in_tanh_dim, nclass, False, M, [None, None, 6])
    model.compile(loss=Loss, optimizer=keras.optimizers.Adam())
    model.load_weights(tf.train.latest_checkpoint(checkpoint_path))
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path + 'ck_{epoch}', save_weights_only=True)

    # for ___i in range(1000):
    #     if ___i % 10 == 0:
    #         print(___i)
    #     a = take_batches.as_numpy_iterator().__next__()
    #     if(tf.reduce_any(tf.math.is_nan(a[0]))):
    #         print("a[0] contains nan")
    #     loss = Loss(a[1], model.predict(a[0]))
    #     if tf.reduce_any(tf.math.is_nan(loss)):
    #         print("loss contains nan")
    #     # for __i in range(len(x)):
    #     #     pred = model.predict(tf.cast(x[__i], tf.float32))
    #     #     if tf.reduce_any(tf.math.is_nan(pred)):
    #     #         print("pred contains nan")
    #     #     loss = Loss(tf.cast(y[__i], tf.float32), pred)
    #     #     if tf.reduce_any(tf.math.is_nan(loss)):
    #     #         print("loss contains nan")
    #
    # exit()


    model.fit(take_batches, steps_per_epoch=steps_per_epoch, epochs=epochs, initial_epoch=3, callbacks=[checkpoint_callback])

else:
    # draw_real_char(5)
    # exit()

    # ii = 0
    # while True:
    #     a = take_batches.as_numpy_iterator().__next__()
    #     los = loss(a[1], model(a[0]))
    #     if(np.sum(los) < 0):
    #         break
    #     ii += 1
    # print(ii)
    # exit()

    model = construct_model(units, in_tanh_dim, nclass, True, M, [1, 1, 6])
    model.load_weights(tf.train.latest_checkpoint(ckdir))
    model.build(tf.TensorShape([1, 1, 6]))
    pass


exit()


# class CustomCallback(keras.callbacks.Callback):
#     def __init__(self, model):
#         self.model = model
#
#     def on_epoch_end(self, epoch):
#         y_pred = self.model.predict()
#         print('y predicted: ', y_pred)
