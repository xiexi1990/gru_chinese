import pickle
import numpy as np
import random
from settings import *

with open(data_path + "dlb_total_" + str(lb_total) + "_dist_" + str(remove_dist_th) + "_ang_" + str(remove_ang_th), "rb") as f:
    lb = pickle.load(f)

random.seed(123)
lb2 = random.sample(list(filter(lambda r: r[0] == 1001, lb)), nclass)
y = []
x = []
i = 0
for i in range(nclass):
    char = lb2[i]
    for j in range(repeat):
        cx = [np.append(np.zeros(5), i)]
        k = 0
        first = True
        while k < np.size(char[2], 0) - 1:
            p = np.append(np.append(char[2][k], char[3][k]), i)
            if random.random() < drop:
                if p[2] == 1:
                    p[2] = 0
                    p[3] = 1
            if random.random() < noise_prob:
                p[0] *= 1 + (random.random() - 0.5) * 2 * noise_ratio
                p[1] *= 1 + (random.random() - 0.5) * 2 * noise_ratio
            if first:
                first = False
                cy = [p]
            else:
                cy = np.append(cy, [p], 0)
            cx = np.append(cx, [p], 0)
            k += 1
        cy = np.append(cy, [np.append(np.append(char[2][k], char[3][k]), i)], 0)
        x.append(cx)
        y.append(cy)

z = sorted(zip(x, y), key=lambda l:np.size(l[0], 0))
result = zip(*z)

x, y = [list(i) for i in result]

batchx = []
batchy = []
i = 0
while i < x.__len__():
    xb = np.expand_dims(x[i], 0)
    yb = np.expand_dims(y[i], 0)
    j = i + 1
    while j < x.__len__():
        if np.size(x[j], 0) == np.size(x[i], 0):
            xb = np.append(xb, np.expand_dims(x[j], 0), 0)
            yb = np.append(yb, np.expand_dims(y[j], 0), 0)
            j += 1
        else:
            break
    i = j
    batchx.append(xb)
    batchy.append(yb)

with open(data_path + "x_y_lb_n_" + str(nclass) + "_r_" + str(repeat) + "_dist_" + str(remove_dist_th) + "_ang_" + str(remove_ang_th) + "_drop_" + str(drop) + "_np_" + str(noise_prob) + "_nr_" + str(noise_ratio), "wb") as f:
    pickle.dump((batchx, batchy), f)
