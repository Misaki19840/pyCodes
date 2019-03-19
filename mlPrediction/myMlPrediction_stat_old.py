import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import glob
import os
import csv

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

dataX = []
dataX_2d_list = []
dataNames = []
dataY = []

# load data X
# 333 .. 2/28
dataXPath = "data_weather/"
nNum = 0
fNum = 0
for f in glob.glob(os.path.join(dataXPath, "*.txt")):
    # get data
    data_raw = np.loadtxt(f, delimiter=",", dtype=None, skiprows=1)
    data_feature = data_raw[:334,:]
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
dataX = np.array(dataX)
dataX = np.reshape(dataX, (nNum,fNum))

# load data Y
dataYName = "data_sakura/sakura_2018_2001.txt"
data_tgt = np.loadtxt(dataYName, delimiter=",")
dataY = data_tgt


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
plt.show()

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
plt.show()

# calc statistics. Use it as feature set.
dataX_stat = []
for j in range(nNum):
    for i in range(len(dataNames)):
        y = dataX_2d_list[j][:,i]
        dataX_stat.append(sum(y)/float(len(y)))
        dataX_stat.append(max(y))
        dataX_stat.append(min(y))
        dataX_stat.append(np.std(y))
dataX_stat = np.array(dataX_stat)
dataX_stat = np.reshape(dataX_stat, (nNum, len(dataNames)*4))
dataX = dataX_stat

# data normalization
dataX = normData(dataX)
# dataXminY, maxY = min(dataY), max(dataY)
# dataY = (dataY - minY) / (maxY - minY)

# Split the data into training/testing sets
numTest = 2
dataX_train = dataX[:-numTest]
dataX_test = dataX[-numTest:]

# Split the targets into training/testing sets
dataY_train = dataY[:-numTest]
dataY_test = dataY[-numTest:]

# data training and get results
regr = linear_model.LinearRegression()
regr.fit(dataX_train, dataY_train)
dataY_pred = regr.predict(dataX_test)

print('Coefficients: \n', regr.coef_)
print("Mean squared error: %.2f"
      % mean_squared_error(dataY_test, dataY_pred))
print('Variance score: %.2f' % r2_score(dataY_test, dataY_pred))

fa = dataY_test - dataY_pred
fa *= fa
fb = dataY_test - ( sum(dataY_test) / len(dataY_test))
fb *= fb
myr2 = 1- sum(fa)/ sum(fb)

sortedIdx = np.argsort(abs(regr.coef_))
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
    ttl = str(fIdx) + " " + "{:.2f}".format(regr.coef_[fIdx])
    ax.set_title(ttl)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
fig.show()
plt.show()