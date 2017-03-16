import numpy as np
import csv
import numba
import matplotlib.pyplot as plt
import time

file = open('train.csv','r')

dict_by_type = {}
csv_lines = list(csv.reader(file))
for row in csv_lines[1:]:
    try:
        data = list(map(float, row[3:]))
    except ValueError:
        data = []
        for x in row[3:]:
            if x == 'NR':
                data.append(float(0))
            else:
                data.append(float(x))

    try:
        dict_by_type[row[2]] = np.concatenate((dict_by_type[row[2]], data))
    except KeyError:
        dict_by_type[row[2]] = np.array(data)


examples_pool = []
for i in range(9, len(dict_by_type['PM2.5']), 10):
    example = {'y_hat': dict_by_type['PM2.5'][i], 'xs': [1, *dict_by_type['PM2.5'][i-9:i]]}
    examples_pool.append(example)


@numba.jit(nogil=True)
def dL_func(_ws, _examples_pool):

    var_num = len(_ws)
    N = len(_examples_pool)
    dLs = np.array([float(0)] * var_num)
    independent_to_var_i = np.array([])
    for each_example in _examples_pool:
        value = (-2 / N) * (each_example['y_hat'] - np.dot(each_example['xs'], _ws))
        independent_to_var_i = np.concatenate((independent_to_var_i, [value]))
    for var_i in range(var_num):
        dL = 0.0
        for (index, each_example) in enumerate(_examples_pool):
            dL += independent_to_var_i[index] * each_example['xs'][var_i]
        dLs[var_i] = dL
    # print(dLs, 'dL')
    return dLs


# @numba.jit(nogil=True)
# def dL_func(_ws, _examples_pool):
#     var_num = len(_ws)
#     N = len(_examples_pool)
#     dLs = np.array([float(0)] * var_num)
#     for var_i in range(var_num):
#         dL = 0.0
#         for (index, each_example) in enumerate(_examples_pool):
#             dL += (-2 / N) * (each_example['y_hat'] - np.dot(each_example['xs'], _ws)) * each_example['xs'][var_i]
#         dLs[var_i] = dL
#     return dLs


@numba.jit(nogil=True)
def L_func(_ws, _examples_pool):
    e = 0.0
    for each_example in _examples_pool:
        e += (each_example['y_hat'] - np.dot(each_example['xs'], _ws)) ** 2
    return e / len(_examples_pool)

iteration = 100000
learning_rate = 1.0e-4
ws = np.array([[float(0)] * 10] * iteration)
#
# plt.ion()
# fig, ax = plt.subplots()
# plot = ax.scatter([], [])
# ax.set_xlim(0, iteration)
# ax.set_ylim(-3, 3)

for i in range(1, iteration):
    ws[i] = ws[i-1] - learning_rate * dL_func(ws[i-1], examples_pool)
    e = L_func(ws[i], examples_pool)
    print(i, e)
    if e == float('inf'):
        print('diverge!')
        break
#     plot_array = plot.get_offsets()
#     plot_array = np.append(plot_array, np.array([i, ws[i][10]]))
#     plot.set_offsets(plot_array)
#     fig.canvas.draw()
#
# n = np.arange(0, iteration, 1)
# plt.plot(n, ws)
# plt.show()
