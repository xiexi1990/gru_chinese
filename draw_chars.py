import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from model import construct_model
from settings import *
import pickle

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.45)))
tf.compat.v1.keras.backend.set_session(sess)

with open(data_path + "x_y_lb_n_" + str(nclass) + "_r_" + str(repeat) + "_dist_" + str(remove_dist_th) + "_ang_" + str(remove_ang_th) + "_drop_" + str(drop) + "_np_" + str(noise_prob) + "_nr_" + str(noise_ratio), 'rb') as f:
    x, y = pickle.load(f)

def draw_real_char(ch):
    fig1 = plt.figure()
    drew = 0
    for _i in range(len(x)):
        for _j in range(np.size(x[_i], 0)):
            if x[_i][_j, 0, 5] == ch:
                drew += 1
                if drew > 10:
                    plt.show()
                    return
                real_plt = fig1.add_subplot(2, 5, drew)
                real_char = x[_i][_j, :, :]
                real_cur_x = 0
                real_cur_y = 0
                for _k in range(1, np.size(real_char, 0)):
                    real_next_x = real_cur_x + real_char[_k, 0]
                    real_next_y = real_cur_y + real_char[_k, 1]
                    if (real_char[_k, 2] == 1):
                        real_plt.plot([real_cur_x, real_next_x], [real_cur_y, real_next_y], color='black')
                    real_cur_x = real_next_x
                    real_cur_y = real_next_y
    plt.show()

def draw_chars(model, classes, maxlen):
    fig = plt.figure()
    ch_cnt = 1
    for ch in classes:
        ch_plt = fig.add_subplot(2, len(classes), ch_cnt)
        real_plt = fig.add_subplot(2, len(classes), ch_cnt + len(classes))
        find = False
        for _i in range(len(x)):
            for _j in range(np.size(x[_i], 0)):
                if x[_i][_j, 0, 5] == ch:
                    find = True
                    break
            if find:
                break
        if not find:
            continue
        real_char = x[_i][_j, :, :]
        real_cur_x = 0
        real_cur_y = 0
        for _k in range(1, np.size(real_char, 0)):
            real_next_x = real_cur_x + real_char[_k, 0]
            real_next_y = real_cur_y + real_char[_k, 1]
            if (real_char[_k, 2] == 1):
                real_plt.plot([real_cur_x, real_next_x], [real_cur_y, real_next_y], color='black')
            real_cur_x = real_next_x
            real_cur_y = real_next_y

        model.reset_states()
        pnt_in = np.array([[[0, 0, 0, 0, 0, ch]]])
        pnt_cnt = 0
        cur_x = 0
        cur_y = 0

        while pnt_cnt < maxlen:
            pred = np.squeeze(model(pnt_in))
            pi, mux, muy, sigmax, sigmay = np.split(pred[:M * 5], 5, axis=-1)
            p = pred[M * 5:]

            sum_choose = True
            N_choose = 0
            if sum_choose:
                r1 = np.random.rand()
                sum = 0
                for i in range(M):
                    sum += pi[i]
                    if sum > r1:
                        N_choose = i
                        break
            else:
                N_choose = np.argmax(pi)
            [x_pred, y_pred] = np.random.multivariate_normal([mux[N_choose], muy[N_choose]],[[np.square(sigmax[N_choose]), 0], [0, np.square(sigmay[N_choose])]])

            s_pred = np.zeros(3)
            S_choose = 0
            if sum_choose:
                r2 = np.random.rand()
                sum = 0
                for i in range(3):
                    sum += p[i]
                    if sum > r2:
                        S_choose = i
                        break
            else:
                S_choose = np.argmax(p)

            s_pred[S_choose] = 1
            next_x = cur_x + x_pred
            next_y = cur_y + y_pred
            if(s_pred[2] == 1):
                break
            if(s_pred[0] == 1):
                ch_plt.plot([cur_x, next_x], [cur_y, next_y], color='black')
            cur_x = next_x
            cur_y = next_y
            pnt_in[0, 0, 0] = x_pred
            pnt_in[0, 0, 1] = y_pred
            pnt_in[0, 0, 2:5] = s_pred
            pnt_cnt += 1
        ch_cnt += 1
    plt.show()

# draw_real_char(2)
# exit()

model = construct_model(units, in_tanh_dim, nclass, True, M, [1, 1, 6])
model.load_weights(tf.train.latest_checkpoint(checkpoint_path))
draw_chars(model, [0, 1, 2, 3, 4], 100)