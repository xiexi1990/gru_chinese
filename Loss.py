import tensorflow as tf
import numpy as np
from settings import M

def N(x, mu, sigma):
    return tf.exp(-(x - mu)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

def Loss(y, pred):
    pi, mux, muy, sigmax, sigmay = tf.split(pred[:, :, :M * 5], 5, axis=-1)
    p = pred[:, :, M * 5:]
    xtp1 = tf.expand_dims(y[:, :, 0], axis=-1)
    ytp1 = tf.expand_dims(y[:, :, 1], axis=-1)
    stp1 = y[:, :, 2:5]
    w = tf.constant([1, 5, 100], dtype=tf.float32)
    lPd = tf.math.log(tf.reduce_sum(pi * N(xtp1, mux, sigmax) * N(ytp1, muy, sigmay), axis=-1))
    lPs = tf.reduce_sum(w * stp1 * tf.math.log(p), axis=-1)
    return - (lPd + lPs)