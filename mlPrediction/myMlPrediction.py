import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import glob
import os
import csv
import matplotlib.cm as cm

def normData(dataX):
    dataX_norm = dataX
    for i in range(len(dataX[0, :])):
        Max = max(dataX_norm[:, i])
        Min = min(dataX_norm[:, i])
        if (Max - Min) > 1.0e-6:
            dataX_norm[:, i] = (dataX_norm[:, i] - Min) / (Max - Min)
        else:
            dataX_norm[:, i] = - Max
    return dataX_norm

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


dataX = []
dataXNames = []
dataX_2d_list = []
dataNames = []
dataNames_clor = []
dataY = []

# load data X
# 4月 ～ 3月のデータ
# 333 .. 2/28
# 347がmax
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
        dataX.append(val)

    # get names
    file = open(f, mode='r')
    data_reader = csv.reader(file,delimiter=",")
    data_raw = [row for row in data_reader]
    dataNames = data_raw[0]

    arr_clr = cm.get_cmap("tab20").colors
    name_clor = [arr_clr[i] for i in range(len(dataNames))]

    for day in range(len(data_feature[:,0])):
        dayFromJan = day + 90 if day < 275 else day - 275
        md_m, md_d = day2md(dayFromJan)
        dataNamesDays = [name + "_day" + str(dayFromJan) + "_" + str(md_m) + "/" + str(md_d) for name in dataNames]
        dataXNames.extend(dataNamesDays)
        dataNames_clor.extend(name_clor)

dataX = np.array(dataX)
dataX = np.reshape(dataX, (nNum,fNum))
# dataX = dataX[:-1,:]
# nNum -= 1

isNanIdx = np.argwhere(np.isnan(dataX))
isInfIdx = np.argwhere(np.isinf(dataX))
print("isNanIdx: ", isNanIdx)
print("isInfIdx: ", isInfIdx)

# load data Y
dataYPath = "data_sakura/"
for f in glob.glob(os.path.join(dataYPath, "*.txt")):
    data_tgt = np.loadtxt(f, delimiter=",")
    for val in data_tgt:
        dataY.append(val)
dataY = np.array(dataY)

## plot 2dlist data
minY, maxY = min(dataY), max(dataY)
dataY_01 = (dataY - minY) / (maxY - minY)
figAll_plt = plt.figure()
figAll_plt.subplots_adjust(wspace=0.4, hspace=0.6)
for i in range(len(dataNames)):
    numcol = 3
    numrow = len(dataNames) / numcol + 1
    ax = figAll_plt.add_subplot(numrow,numcol,i+1)
    for j in range(nNum):
        y = dataX_2d_list[j][:,i]
        x = range(len(y))
        cval = ""
        if dataY_01[j] >= 0.5:
            cval = "#ee0000"
        else:
            cval = "#0000ee"
        ax.plot(x,y,c=cval, alpha=0.7)
        #carr = np.array([dataY_01[j] for nn in range(len(x))])
        #ax.scatter(x, y, c=carr, cmap="jet")
    ttl = dataNames[i]
    ax.set_title(ttl)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
figAll_plt.show()

## calc 1d st
minY, maxY = min(dataY), max(dataY)
dataY_01 = (dataY - minY) / (maxY - minY)
figStat_plt = plt.figure()
figStat_plt.subplots_adjust(wspace=0.4, hspace=0.6)
ftoShow = [0, 1, 3, 5, 6]
for i in range(len(ftoShow)):
    f_mean = []
    f_max = []
    f_min = []
    f_var = []
    for j in range(nNum):
        y = dataX_2d_list[j][:,i]
        f_mean.append(sum(y)/float(len(y)))
        f_max.append(max(y))
        f_min.append(min(y))
        f_var.append(np.std(y))
    f_mean = np.array(f_mean)
    f_max = np.array(f_max)
    f_min = np.array(f_min)
    f_var = np.array(f_var)

    x = range(nNum)
    ax = figStat_plt.add_subplot(len(ftoShow), 4, 4*i + 1)
    ax.scatter(f_mean, dataY, color="#222222", alpha=0.7)
    ttl = dataNames[i] + "_mean"
    ax.set_title(ttl)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax = figStat_plt.add_subplot(len(ftoShow), 4, 4 * i + 2)
    ax.scatter(f_max, dataY, color="#222222", alpha=0.7)
    ttl = dataNames[i] + "_max"
    ax.set_title(ttl)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax = figStat_plt.add_subplot(len(ftoShow), 4, 4 * i + 3)
    ax.scatter(f_min, dataY, color="#222222", alpha=0.7)
    ttl = dataNames[i] + "_min"
    ax.set_title(ttl)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax = figStat_plt.add_subplot(len(ftoShow), 4, 4 * i + 4)
    ax.scatter(f_var, dataY, color="#222222", alpha=0.7)
    ttl = dataNames[i] + "_std"
    ax.set_title(ttl)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
figStat_plt.show()

# data normalization
dataX = normData(dataX)
# dataXminY, maxY = min(dataY), max(dataY)
# dataY = (dataY - minY) / (maxY - minY)

# Split the data into training/testing sets
numTest = 20
dataX_train = dataX[:-numTest]
dataX_test = dataX[-numTest:]
dataY_train = dataY[:-numTest]
dataY_test = dataY[-numTest:]

# regression by RF
rf = RandomForestRegressor(n_estimators=100,random_state=42)
rf.fit(dataX_train, dataY_train)
dataY_pred = rf.predict(dataX_test)
err_mae = abs(dataY_test - dataY_pred)
ave_mae = abs(dataY_test - np.mean(dataY_test))

print("err_mae: ", np.mean(err_mae))
print("ave_mae: ", np.mean(ave_mae))
print("importances: ", rf.feature_importances_)
sortedIdx = np.argsort(abs(rf.feature_importances_))
print("sorted idx: ", sortedIdx[::-1])

numShow = 3
fig = plt.figure()
for i in range(numShow):
    fIdx = sortedIdx[::-1][i]
    ax = fig.add_subplot(1,numShow,i+1)
    ax.scatter(dataX_test[:,fIdx], dataY_test, color="#222222", alpha=0.7)
    dataIdx = np.argsort(dataX_test[:,fIdx])
    x = dataX_test[:,fIdx][dataIdx]
    y = dataY_pred[dataIdx]
    ax.plot(x,y, color='blue', linewidth=3)
    # ttl = dataNames[fIdx] + " " + "{:.2f}".format(regr.coef_[fIdx])
    ttl = dataXNames[fIdx] + " " + "{:.2f}".format(rf.feature_importances_[fIdx])
    ax.set_title(ttl)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
fig.show()

# validation
numFold = 10
mae_arr = []
aveMae_arr = []
coef_list = []
r2score_arr = []
if len(dataX[:,0]) != len(dataY):
    print("err! dataX[:,0] != dataY")
idxes = np.random.permutation(len(dataX[:,0]))
numTsample = int(len(dataX[:,0]) / numFold)
for i in range(numFold):
    idxes_test = idxes[i*numTsample: i*numTsample + numTsample]
    idxes_train = np.hstack((idxes[0:i*numTsample],idxes[i*numTsample + numTsample:]))
    # idxes_test = [ idxes[i*numTsample + j] for j in range(len(numTsample))]
    if i == numFold - 1:
        idxes_test = idxes[i * numTsample:]
        idxes_train = idxes[:i * numTsample]

    dataX_train = dataX[idxes_train,:]
    dataX_test = dataX[idxes_test, :]
    dataY_train = dataY[idxes_train]
    dataY_test = dataY[idxes_test]

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(dataX_train, dataY_train)
    dataY_pred = rf.predict(dataX_test)
    err_mae = np.mean(abs(dataY_test - dataY_pred))
    ave_mae = np.mean(abs(dataY_test - np.mean(dataY_test)))

    mae_arr.append(err_mae)
    aveMae_arr.append(ave_mae)
    coef_list.append(rf.feature_importances_)
    r2score_arr.append(r2_score(dataY_test, dataY_pred))

mae_arr = np.array(mae_arr)
aveMae_arr = np.array(aveMae_arr)
r2score_arr = np.array(r2score_arr)
importance_arr = np.zeros(coef_list[0].shape)
for i in range(len(coef_list)):
    importance_arr += coef_list[i]
importance_arr /= len(coef_list)

# 1 to 10th important features
print("err_mae: ", np.mean(mae_arr))
print("ave_mae: ", np.mean(aveMae_arr))
print('Importance: \n', importance_arr)
sortedIdx = np.argsort(importance_arr)
print("sorted idx: ", sortedIdx[::-1])
for i in range(10):
    print( i, " th important: ",  dataXNames[sortedIdx[::-1][i]])
    print("importance: ", importance_arr[sortedIdx[::-1][i]])


file_log = open("myMLPrediction_Result.txt", mode='w', newline="")
writer_log = csv.writer(file_log)
sumImportance = 0
row = ["fname", "importance", "Sum(importance)"]
writer_log.writerow(row)
for i in range(len(importance_arr)):
    sumImportance += importance_arr[sortedIdx[::-1][i]]
    row = [dataXNames[sortedIdx[::-1][i]], importance_arr[sortedIdx[::-1][i]], sumImportance]
    writer_log.writerow(row)