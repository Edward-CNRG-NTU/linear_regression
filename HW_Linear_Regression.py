import numpy as np
import csv

file = open('train.csv','r')
dataset = list(csv.reader(file))
data = []
dict_ = {}

# collect data
for row in dataset[1:]:
    try:
        data_tmp = list(map(float, row[3:]))
        data.append(data_tmp)

    except ValueError :
        tmp = []
        for x in row[3:]:
            if x == 'NR':
                tmp.append(float(0))
            else:
                tmp.append(float(x))
        data.append(tmp)

    try:
        dict_[row[2]].append(data_tmp)
    except KeyError:
        dict_[row[2]] = [data_tmp]




def set_data():

    predict_data = predict_data = np.array(dict_['PM2.5'])[:, 9]
    for p in range(10,24):
        predict_data_tmp = np.array(dict_['PM2.5'])[:, p]
        predict_data = np.hstack((predict_data, predict_data_tmp))

    print(np.shape(predict_data))


    data_matrix_AMB_TEMP = data_window(0, 'AMB_TEMP')
    data_matrix_CH4 = data_window(0, 'CH4')
    data_matrix_CO = data_window(0, 'CO')
    data_matrix_NMHC = data_window(0, 'NMHC')
    data_matrix_NO = data_window(0, 'NO')
    data_matrix_NO2 = data_window(0, 'NO2')
    data_matrix_NOx = data_window(0, 'NOx')
    data_matrix_O3 = data_window(0, 'O3')
    data_matrix_PM10 = data_window(0, 'PM10')
    data_matrix_PM25 = data_window(0, 'PM2.5')
    data_matrix_RAINFALL = data_window(0, 'RAINFALL')
    data_matrix_RH = data_window(0, 'RH')
    data_matrix_SO2 = data_window(0, 'SO2')
    data_matrix_THC = data_window(0, 'THC')
    data_matrix_WD_HR = data_window(0, 'WD_HR')
    data_matrix_WIND_DIREC = data_window(0, 'WIND_DIREC')
    data_matrix_WIND_SPEED = data_window(0, 'WIND_SPEED')
    data_matrix_WS_HR = data_window(0, 'WS_HR')
    data_matrix_item = np.hstack((data_matrix_AMB_TEMP, data_matrix_CH4, data_matrix_CO, data_matrix_NMHC
                                      , data_matrix_NO, data_matrix_NO2, data_matrix_NOx, data_matrix_O3,
                                      data_matrix_PM10,
                                      data_matrix_PM25, data_matrix_RAINFALL, data_matrix_RH, data_matrix_SO2,
                                      data_matrix_THC, data_matrix_WD_HR, data_matrix_WIND_DIREC,
                                      data_matrix_WIND_SPEED, data_matrix_WS_HR))




    for start_hour in range(1,15):
        data_matrix_AMB_TEMP = data_window(start_hour, 'AMB_TEMP')
        data_matrix_CH4 = data_window(start_hour, 'CH4')
        data_matrix_CO = data_window(start_hour, 'CO')
        data_matrix_NMHC = data_window(start_hour, 'NMHC')
        data_matrix_NO = data_window(start_hour, 'NO')
        data_matrix_NO2 = data_window(start_hour, 'NO2')
        data_matrix_NOx = data_window(start_hour, 'NOx')
        data_matrix_O3 = data_window(start_hour, 'O3')
        data_matrix_PM10 = data_window(start_hour, 'PM10')
        data_matrix_PM25 = data_window(start_hour, 'PM2.5')
        data_matrix_RAINFALL = data_window(start_hour, 'RAINFALL')
        data_matrix_RH = data_window(start_hour, 'RH')
        data_matrix_SO2 = data_window(start_hour, 'SO2')
        data_matrix_THC = data_window(start_hour, 'THC')
        data_matrix_WD_HR = data_window(start_hour, 'WD_HR')
        data_matrix_WIND_DIREC = data_window(start_hour, 'WIND_DIREC')
        data_matrix_WIND_SPEED = data_window(start_hour, 'WIND_SPEED')
        data_matrix_WS_HR = data_window(start_hour, 'WS_HR')

        data_matrix_item_tmp2 = np.hstack((data_matrix_AMB_TEMP, data_matrix_CH4, data_matrix_CO, data_matrix_NMHC
                                  , data_matrix_NO, data_matrix_NO2, data_matrix_NOx, data_matrix_O3, data_matrix_PM10,
                                  data_matrix_PM25, data_matrix_RAINFALL, data_matrix_RH, data_matrix_SO2,
                                  data_matrix_THC, data_matrix_WD_HR, data_matrix_WIND_DIREC,
                                  data_matrix_WIND_SPEED, data_matrix_WS_HR))

        data_matrix_item = np.vstack((data_matrix_item, data_matrix_item_tmp2))
        print(np.shape(data_matrix_item))

    # print(np.shape(data_matrix_item))
    data_matrix_item = np.hstack((data_matrix_item, (data_matrix_item)**2))
    # print(data_matrix_item[:,[0,162]])
    print(np.shape(data_matrix_item))
    # print(np.shape(data_matrix_item))

    data_matrix_item = np.insert(data_matrix_item, 0, 1, axis=1)
    print(np.shape(data_matrix_item))
    # print(data_matrix_item[:, 0])

    return predict_data, data_matrix_item


def data_window(start_hour, item_name):
    data_matrix = []

    for d in dict_[item_name]:
        data_matrix.append(np.roll(d, 24 - start_hour)[:9])

    return np.array(data_matrix)

def comput_error(ws, points, output_point):

    x = points
    y = output_point
    total_error = np.sum((y - (np.dot(x, ws)))**2)

    return total_error/float(len(points))


def gradient_descent(starting_ws, points, output_point, learning_rate, iteration):
    ws = starting_ws
    for i in range(iteration):
        ws = step_gradient_descent(ws, points, output_point, learning_rate, i)
    return ws


def step_gradient_descent(ws_current, points, output_point, learning_rate, a):
    ws_tmp = np.array([])
    Adagrad = 0
    # select_points = points[0 + 240*start_hour:240+240*start_hour, :]
    # select_output_point = output_point[:, start_hour]

    sum_tmp = (-2) * (output_point - np.dot(points, ws_current))

    for i in range(len(ws_current)):
        T_points = np.transpose(points[:, i])
        ws_dot = np.dot(T_points, sum_tmp)
        ws_tmp = np.append(ws_tmp, ws_dot)
    Adagrad += ws_tmp**2       #
    new_ws = ws_current - learning_rate*ws_tmp*(Adagrad**(-1/2))

    print(a, comput_error(new_ws, points, output_point))

    return new_ws


def run():
    iteration = 100000
    learning_rate = 10e-7

    output_point, points = set_data()
    initial_ws = np.zeros(18*9*2+1)

    # print('start gradient descent  error = {0}'.format(comput_error(initial_ws, points, output_point)))
    new_ws = gradient_descent(initial_ws, points, output_point, learning_rate, iteration)

    print('gradient descent end error = {0}'.format(comput_error(new_ws, points, output_point)))

if __name__ == '__main__':
    run()



