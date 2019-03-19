import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import glob
import os
import csv

def normData(dataX, fmaxes_ = [], fmins_ = ''):
    dataX_norm = dataX
    fmaxes = []
    fmins = []
    for i in range(len(dataX[0, :])):
        Max = max(dataX_norm[:, i]) if len(fmaxes_) == 0 else fmaxes_[i]
        Min = min(dataX_norm[:, i]) if len(fmins_) == 0 else fmins_[i]
        fmaxes.append(Max)
        fmins.append(Min)
        if (Max - Min) > 1.0e-6:
            dataX_norm[:, i] = (dataX_norm[:, i] - Min) / (Max - Min)
        else:
            dataX_norm[:, i] = - Max
    return np.array(dataX_norm), np.array(fmaxes), np.array(fmins)

def day2md(day):
    daysInMonth = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    day_res = 0
    month_res = 0
    day_tmp = round(day) + 1
    for i, days in enumerate(daysInMonth):
        tmp = int(day_tmp - days)
        if tmp <= 0:
            month_res = i + 1
            day_res = day_tmp
            break
        else:
            day_tmp -= days
    return month_res, day_res


dataX_train = []
dataX_test = []
dataXNames = []
dataX_2d_list = []
dataNames = []
dataY_train = []

# load data X train
dataXPath = "data_weather/"
nNum = 0
fNum = 0
for f in glob.glob(os.path.join(dataXPath, "*.txt")):
    # get data
    data_raw = np.loadtxt(f, delimiter=",", dtype=None, skiprows=1)
    #data_feature = data_raw[:334,:]
    data_feature = data_raw[:344, :]
    dataX_2d_list.append(data_feature)
    fNum = len(data_feature[0,:])*len(data_feature[:,0])
    data = data_feature
    nNum += 1
    for val in data:
        dataX_train.append(val)

    # get names
    file = open(f, mode='r')
    data_reader = csv.reader(file,delimiter=",")
    data_raw = [row for row in data_reader]
    dataNames = data_raw[0]

    for day in range(len(data_feature[:,0])):
        dataNamesDays = [name + "_day" + str(day) for name in dataNames]
        dataXNames.extend(dataNamesDays)
dataX_train = np.array(dataX_train)
dataX_train = np.reshape(dataX_train, (nNum,fNum))

isNanIdx = np.argwhere(np.isnan(dataX_train))
isInfIdx = np.argwhere(np.isinf(dataX_train))
print("isNanIdx: ", isNanIdx)
print("isInfIdx: ", isInfIdx)

# load data X test
dataXPath = "data_weather_test/"
nNum = 0
fNum = 0
for f in glob.glob(os.path.join(dataXPath, "*.txt")):
    # get data
    data_raw = np.loadtxt(f, delimiter=",", dtype=None, skiprows=1)
    #data_feature = data_raw[:334,:]
    data_feature = data_raw[:344, :]
    dataX_2d_list.append(data_feature)
    fNum = len(data_feature[0,:])*len(data_feature[:,0])
    data = data_feature
    nNum += 1
    for val in data:
        dataX_test.append(val)

dataX_test = np.array(dataX_test)
dataX_test = np.reshape(dataX_test, (nNum,fNum))

isNanIdx = np.argwhere(np.isnan(dataX_test))
isInfIdx = np.argwhere(np.isinf(dataX_test))
print("isNanIdx: ", isNanIdx)
print("isInfIdx: ", isInfIdx)

# load data Y
dataYPath = "data_sakura/"
for f in glob.glob(os.path.join(dataYPath, "*.txt")):
    data_tgt = np.loadtxt(f, delimiter=",")
    for val in data_tgt:
        dataY_train.append(val)
dataY_train = np.array(dataY_train)

dataX_train, maxes_train, mins_train = normData(dataX_train)
dataX_test, tmp1, tmp2 = normData(dataX_test, maxes_train, mins_train)

# regression by RF
rf = RandomForestRegressor(n_estimators=100,random_state=42)
rf.fit(dataX_train, dataY_train)
dataY_pred = rf.predict(dataX_train)
err_mae = abs(dataY_train - dataY_pred)
ave_mae = abs(dataY_train - np.mean(dataY_train))

print("Train result")
print("err_mae: ", np.mean(err_mae))
print("ave_mae: ", np.mean(ave_mae))
print("importances: ", rf.feature_importances_)
sortedIdx = np.argsort(abs(rf.feature_importances_))
print("sorted idx: ", sortedIdx[::-1])

# getResult
dataY_pred = rf.predict(dataX_test)
print("Prediction result")
for datay in dataY_pred:
    month, day = day2md(datay)
    print("pred, month, day", datay, month, day)
