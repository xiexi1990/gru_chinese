import tensorflow as tf
import numpy as np
import settings as ss

def exp_safe(x):
    return tf.clip_by_value(tf.exp(x), clip_value_min=1e-10, clip_value_max=1e10)
  #  return tf.exp(x)
def log_safe(x):
    return tf.clip_by_value(tf.math.log(x), clip_value_min=-20, clip_value_max=20)
 #   return tf.math.log(x)

def N(x, mu, sigma):
    return exp_safe(-(x - mu)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

def Loss(y, pred):
    # __x = tf.constant([5.0, 0.0], dtype=tf.float32)
    # #if tf.reduce_any(tf.math.is_nan(pred)):
    # if tf.reduce_sum(tf.cast(tf.math.is_nan(__x), tf.int32)):
    #     raise Exception("Loss: test nan")
    tf.debugging.assert_all_finite(pred, 'loss inputs ill')

    pi, mux, muy, sigmax, sigmay = tf.split(pred[:, :, :ss.M * 5], 5, axis=-1)
    p = pred[:, :, ss.M * 5:]
    xtp1 = tf.expand_dims(y[:, :, 0], axis=-1)
    ytp1 = tf.expand_dims(y[:, :, 1], axis=-1)
    stp1 = y[:, :, 2:5]
    w = tf.constant([1, 5, 100], dtype=tf.float32)
  #  eps = 1e-10
  #  eps = 0
   # clip_max = 1e10
    lPd = log_safe(tf.reduce_sum(pi * N(xtp1, mux, sigmax) * N(ytp1, muy, sigmay), axis=-1))
    lPs = tf.reduce_sum(w * stp1 * log_safe(p), axis=-1)
    return - (lPd + lPs)