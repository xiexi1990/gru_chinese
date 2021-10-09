import tensorflow as tf
import numpy as np
from settings import M

def N(x, mu, sigma):
    return tf.exp(-(x - mu)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

def Loss(y, pred):
    # __x = tf.constant([5.0, 0.0], dtype=tf.float32)
    # #if tf.reduce_any(tf.math.is_nan(pred)):
    # if tf.reduce_sum(tf.cast(tf.math.is_nan(__x), tf.int32)):
    #     raise Exception("Loss: test nan")
    assert not tf.reduce_any(tf.math.is_nan(pred))
    pi, mux, muy, sigmax, sigmay = tf.split(pred[:, :, :M * 5], 5, axis=-1)
    p = pred[:, :, M * 5:]
    xtp1 = tf.expand_dims(y[:, :, 0], axis=-1)
    ytp1 = tf.expand_dims(y[:, :, 1], axis=-1)
    stp1 = y[:, :, 2:5]
    w = tf.constant([1, 5, 100], dtype=tf.float32)
    eps = 1e-10
   # clip_max = 1e10
    lPd = tf.math.log(tf.reduce_sum(pi * N(xtp1, mux, sigmax) * N(ytp1, muy, sigmay), axis=-1) + eps)
    lPs = tf.reduce_sum(w * stp1 * tf.math.log(p + eps), axis=-1)
    return - (lPd + lPs)