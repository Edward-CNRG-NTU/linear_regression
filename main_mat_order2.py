import numpy as np
import csv
import json
import matplotlib.pyplot as plt
import time


def dL_func(_ws, _examples_pool):
    N = _examples_pool['y_hat'].shape[0]
    es = _examples_pool['y_hat'] - np.dot(_examples_pool['xs'], _ws)
    dLs = (-2 / N) * np.dot(np.transpose(_examples_pool['xs']), es)
    return dLs


def L_func(_ws, _examples_pool):
    _N = _examples_pool['y_hat'].shape[0]
    _es = _examples_pool['y_hat'] - np.dot(_examples_pool['xs'], _ws)
    return np.dot(_es, _es) / _N


dict_by_type = {}
csv_lines = list(csv.reader(open('train.csv', 'r')))
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
    # print((max(data)-min(data)))
    # data = (np.array(data) - min(data))/(max(data)-min(data))
    try:
        dict_by_type[row[2]] = np.concatenate((dict_by_type[row[2]], [data]))
    except KeyError:
        dict_by_type[row[2]] = np.array([data])


normal_table = {}
for key in dict_by_type:
    data = dict_by_type[key]
    _min = np.min(data)
    dif = np.max(data) - _min
    if dif == 0.0:
        dif = 1.0
    dict_by_type[key] = (data - _min) / dif
    normal_table[key] = {'min': _min, 'dif': dif}


examples_size = dict_by_type['PM2.5'].shape[0] * (dict_by_type['PM2.5'].shape[1] - 9)
y_hat = np.zeros(examples_size)
xs = np.zeros((examples_size, 9 * 36 + 1))
counter = 0
for day in range(dict_by_type['PM2.5'].shape[0]):
    for hour in range(9, dict_by_type['PM2.5'].shape[1]):
        y_hat[counter] = dict_by_type['PM2.5'][day, hour]
        xs[counter] = [1,
                       *dict_by_type['AMB_TEMP'][day, hour - 9: hour],
                       *dict_by_type['CH4'][day, hour - 9: hour],
                       *dict_by_type['CO'][day, hour - 9: hour],
                       *dict_by_type['NMHC'][day, hour - 9: hour],
                       *dict_by_type['NO'][day, hour - 9: hour],
                       *dict_by_type['NO2'][day, hour - 9: hour],
                       *dict_by_type['NOx'][day, hour - 9: hour],
                       *dict_by_type['O3'][day, hour - 9: hour],
                       *dict_by_type['PM10'][day, hour - 9: hour],
                       *dict_by_type['PM2.5'][day, hour - 9: hour],
                       *dict_by_type['RAINFALL'][day, hour - 9: hour],
                       *dict_by_type['RH'][day, hour - 9: hour],
                       *dict_by_type['SO2'][day, hour - 9: hour],
                       *dict_by_type['THC'][day, hour - 9: hour],
                       *dict_by_type['WD_HR'][day, hour - 9: hour],
                       *dict_by_type['WIND_DIREC'][day, hour - 9: hour],
                       *dict_by_type['WIND_SPEED'][day, hour - 9: hour],
                       *dict_by_type['WS_HR'][day, hour - 9: hour],

                       *np.power(dict_by_type['AMB_TEMP'][day, hour - 9: hour], 2),
                       *np.power(dict_by_type['CH4'][day, hour - 9: hour], 2),
                       *np.power(dict_by_type['CO'][day, hour - 9: hour], 2),
                       *np.power(dict_by_type['NMHC'][day, hour - 9: hour], 2),
                       *np.power(dict_by_type['NO'][day, hour - 9: hour], 2),
                       *np.power(dict_by_type['NO2'][day, hour - 9: hour], 2),
                       *np.power(dict_by_type['NOx'][day, hour - 9: hour], 2),
                       *np.power(dict_by_type['O3'][day, hour - 9: hour], 2),
                       *np.power(dict_by_type['PM10'][day, hour - 9: hour], 2),
                       *np.power(dict_by_type['PM2.5'][day, hour - 9: hour], 2),
                       *np.power(dict_by_type['RAINFALL'][day, hour - 9: hour], 2),
                       *np.power(dict_by_type['RH'][day, hour - 9: hour], 2),
                       *np.power(dict_by_type['SO2'][day, hour - 9: hour], 2),
                       *np.power(dict_by_type['THC'][day, hour - 9: hour], 2),
                       *np.power(dict_by_type['WD_HR'][day, hour - 9: hour], 2),
                       *np.power(dict_by_type['WIND_DIREC'][day, hour - 9: hour], 2),
                       *np.power(dict_by_type['WIND_SPEED'][day, hour - 9: hour], 2),
                       *np.power(dict_by_type['WS_HR'][day, hour - 9: hour], 2)
                       ]
        counter += 1
assert counter == examples_size, 'examples_size doesn''t match!'
examples_pool = {'y_hat': y_hat, 'xs': xs}

#
# examples_size = dict_by_type['PM2.5'].shape[0] - 9
# y_hat = np.zeros(examples_size)
# xs = np.zeros((examples_size, 9*18+1))
# counter = 0
# for i2 in range(9, dict_by_type['PM2.5'].shape[0]):
#     y_hat[counter] = dict_by_type['PM2.5'][i2]
#     xs[counter] = [1,
#                    *dict_by_type['AMB_TEMP'][i2 - 9:i2],
#                    *dict_by_type['CH4'][i2 - 9:i2],
#                    *dict_by_type['CO'][i2 - 9:i2],
#                    *dict_by_type['NMHC'][i2 - 9:i2],
#                    *dict_by_type['NO'][i2 - 9:i2],
#                    *dict_by_type['NO2'][i2 - 9:i2],
#                    *dict_by_type['NOx'][i2 - 9:i2],
#                    *dict_by_type['O3'][i2 - 9:i2],
#                    *dict_by_type['PM10'][i2 - 9:i2],
#                    *dict_by_type['PM2.5'][i2 - 9:i2],
#                    *dict_by_type['RAINFALL'][i2 - 9:i2],
#                    *dict_by_type['RH'][i2 - 9:i2],
#                    *dict_by_type['SO2'][i2 - 9:i2],
#                    *dict_by_type['THC'][i2 - 9:i2],
#                    *dict_by_type['WD_HR'][i2 - 9:i2],
#                    *dict_by_type['WIND_DIREC'][i2 - 9:i2],
#                    *dict_by_type['WIND_SPEED'][i2 - 9:i2],
#                    *dict_by_type['WS_HR'][i2 - 9:i2],
#                    ]
#     counter += 1
# assert counter == examples_size, 'examples_size doesn''t match!'
# examples_pool = {'y_hat': y_hat, 'xs': xs}


iteration = 10000000
learning_rate = 0.001

ws = np.zeros((iteration, 9 * 36 + 1))
es = np.zeros(iteration)
es[0] = float('inf')
for i in range(1, iteration):
    ws[i] = ws[i - 1] - learning_rate * dL_func(ws[i - 1], examples_pool)
    es[i] = L_func(ws[i], examples_pool)*(normal_table['PM2.5']['dif']**2)
    print(i, es[i])
    if es[i] == float('inf'):
        print('diverge!')
        break
    if abs(es[i - 1] - es[i]) <= 1e-15:
        print((es[i - 1] - es[i]), 'steady!')
        break


json.dump({'normal_table': normal_table, 'ws': list(ws[i]), 'es': es[i]}, open('result.txt', 'w'))


print(list(enumerate(ws[i] >= 1e-3)))

plt.figure(1)
plt.subplot(211)
plt.plot(es[:i])
plt.ylabel('Loss function:')
plt.subplot(212)
plt.plot(ws[:i:100, abs(ws[i]) >= 1e-3])
plt.ylabel('weights:')
plt.show()
exit(0)
