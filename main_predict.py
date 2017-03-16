import json
import csv
import numpy as np

json_load = json.load(open('result_half.txt', 'r'))
ws = json_load['ws']
normal_table = json_load['normal_table']

test_dict = {}
for line in csv.reader(open('test_X.csv', 'r')):

    try:
        data = list(map(float, line[2:]))
    except ValueError:
        data = []
        for x in line[2:]:
            if x == 'NR':
                data.append(float(0))
            else:
                data.append(float(x))

    normalized_data = (np.array(data) - normal_table[line[1]]['min']) / normal_table[line[1]]['dif']

    try:
        test_dict[line[0]][line[1]] = normalized_data
    except KeyError:
        test_dict[line[0]] = {line[1]: normalized_data}

outfile = open('output.csv', 'w')
writer = csv.DictWriter(outfile, fieldnames=['id', 'value'])
writer.writeheader()
xs = {}
for id_n in test_dict:
    xs[id_n] = np.array([1,
                         *test_dict[id_n]['AMB_TEMP'],
                         *test_dict[id_n]['CH4'],
                         *test_dict[id_n]['CO'],
                         *test_dict[id_n]['NMHC'],
                         *test_dict[id_n]['NO'],
                         *test_dict[id_n]['NO2'],
                         *test_dict[id_n]['NOx'],
                         *test_dict[id_n]['O3'],
                         *test_dict[id_n]['PM10'],
                         *test_dict[id_n]['PM2.5'],
                         *test_dict[id_n]['RAINFALL'],
                         *test_dict[id_n]['RH'],
                         *test_dict[id_n]['SO2'],
                         *test_dict[id_n]['THC'],
                         *test_dict[id_n]['WD_HR'],
                         *test_dict[id_n]['WIND_DIREC'],
                         *test_dict[id_n]['WIND_SPEED'],
                         *test_dict[id_n]['WS_HR'],

                         *np.power(test_dict[id_n]['AMB_TEMP'], 0.5),
                         *np.power(test_dict[id_n]['CH4'], 0.5),
                         *np.power(test_dict[id_n]['CO'], 0.5),
                         *np.power(test_dict[id_n]['NMHC'], 0.5),
                         *np.power(test_dict[id_n]['NO'], 0.5),
                         *np.power(test_dict[id_n]['NO2'], 0.5),
                         *np.power(test_dict[id_n]['NOx'], 0.5),
                         *np.power(test_dict[id_n]['O3'], 0.5),
                         *np.power(test_dict[id_n]['PM10'], 0.5),
                         *np.power(test_dict[id_n]['PM2.5'], 0.5),
                         *np.power(test_dict[id_n]['RAINFALL'], 0.5),
                         *np.power(test_dict[id_n]['RH'], 0.5),
                         *np.power(test_dict[id_n]['SO2'], 0.5),
                         *np.power(test_dict[id_n]['THC'], 0.5),
                         *np.power(test_dict[id_n]['WD_HR'], 0.5),
                         *np.power(test_dict[id_n]['WIND_DIREC'], 0.5),
                         *np.power(test_dict[id_n]['WIND_SPEED'], 0.5),
                         *np.power(test_dict[id_n]['WS_HR'], 0.5)
                         ])
    y = np.dot(ws, xs[id_n])*normal_table['PM2.5']['dif'] + normal_table['PM2.5']['min']
    writer.writerow({'id': id_n, 'value': y})
    print(id_n, ', ', y)
    # break