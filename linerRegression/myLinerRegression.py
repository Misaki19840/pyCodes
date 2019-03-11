
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

def normData(dataX):
    dataX_norm = dataX
    for i in range(len(dataX[0, :])):
        Max = max(dataX_norm[:, i])
        Min = min(dataX_norm[:, i])
        dataX_norm[:, i] = (dataX_norm[:, i] - Min) / (Max - Min)
    return dataX_norm

def outlierFilter(dataXorg, upper = 98, lower = 2):
    data = []
    filter = np.array([True for i in range(len(dataXorg[:,0]))])
    for i in range(len(dataXorg[0, :])):
        if i == 1:
            continue
        upper_val = np.percentile(dataXorg[:, i], upper)
        lower_val = np.percentile(dataXorg[:, i], lower)
        filter_new = (dataXorg[:, i] <= upper_val) & (dataXorg[:, i] >= lower_val)
        filter = filter & filter_new
    data = dataXorg[filter,:]
    return data, filter

DataType = "diab"
#DataType = "priceData.txt"
numTest = 20

if DataType is "diab":
    # Load the diabetes dataset
    diabetes = datasets.load_diabetes()

    # diabetes_X = diabetes.data[:, np.newaxis, 4]
    dataNames = diabetes.feature_names
    diabetes_X = diabetes.data

    # dataX = diabetes_X[:,0:5]
    dataX = diabetes_X
    dataY = diabetes.target
else:
    # https://atarimae.biz/archives/18904
    data = np.loadtxt(DataType,delimiter="\t")
    dataNames = ["square",	"age",	"distance"]

    dataX = data[:, 1:len(data)]
    dataY = data[:, 0]

## outlier filter
# dataX, filter = outlierFilter(dataX)
# dataY = dataY[filter]

dataX = normData(dataX)
minY, maxY = min(dataY), max(dataY)
dataY = (dataY - minY) / (maxY - minY)

## insert dummy data
# arrdummy = np.array([0.5 for i in range(len(dataX[:,0]))])
# dataX = np.vstack((dataX.T, arrdummy)).T
# dataNames.extend("a")
#
# arrdummy = np.array([np.random.rand() for i in range(len(dataX[:,0]))])
# dataX = np.vstack((dataX.T, arrdummy)).T
# dataNames.extend("b")

# Split the data into training/testing sets
dataX_train = dataX[:-numTest]
dataX_test = dataX[-numTest:]

# Split the targets into training/testing sets
dataY_train = dataY[:-numTest]
dataY_test = dataY[-numTest:]

## hist data
figAll_hist = plt.figure()
figAll_hist.subplots_adjust(wspace=0.4, hspace=0.3)
for i in range(len(dataX[0,:])):
    numcol = 3
    numrow = len(dataX[0,:]) / numcol if (len(dataX[0,:]) % numcol == 0 ) else len(dataX[0,:]) / numcol + 1
    ax = figAll_hist.add_subplot(int(numrow),numcol,i+1)
    ax.hist(dataX_train[:,i], color='blue')
    ttl = dataNames[i]
    ax.set_title(ttl)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
figAll_hist.show()

## plot data
figAll_train_bfnorm = plt.figure()
figAll_train_bfnorm.subplots_adjust(wspace=0.4, hspace=0.6)
for i in range(len(dataX[0,:])):
    numcol = 3
    numrow = len(dataNames) / numcol + 1
    ax = figAll_train_bfnorm.add_subplot(numrow,numcol,i+1)
    x = dataX_train[:,i]
    ax.scatter(dataX_train[:,i], dataY_train, color="#222222", alpha=0.7)
    dataIdx = np.argsort(dataX_train[:,i])
    x = dataX_train[:,i][dataIdx]
    ttl = dataNames[i]
    ax.set_title(ttl)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
figAll_train_bfnorm.show()


# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(dataX_train, dataY_train)

# Make predictions using the testing set
dataY_pred = regr.predict(dataX_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(dataY_test, dataY_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(dataY_test, dataY_pred))


y_train = dataX_train.dot(regr.coef_) + regr.intercept_

y_pred_train = regr.predict(dataX_train)
figAll_train = plt.figure()
figAll_train.subplots_adjust(wspace=0.4, hspace=0.6)
for i in range(len(dataX[0,:])):
    numcol = 3
    numrow = len(dataNames) / numcol + 1
    ax = figAll_train.add_subplot(numrow,numcol,i+1)
    x = dataX_train[:,i]
    ax.scatter(dataX_train[:,i], dataY_train, color="#222222", alpha=0.7)
    dataIdx = np.argsort(dataX_train[:,i])
    x = dataX_train[:,i][dataIdx]
    y = y_pred_train[dataIdx]
    ax.plot(x,y, color='blue')

    y = y_train[dataIdx]
    ax.plot(x, y, color='red')
    ttl = dataNames[i] + " " + "{:.2f}".format(regr.coef_[i])
    ax.set_title(ttl)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
figAll_train.show()

figAll_test = plt.figure()
figAll_test.subplots_adjust(wspace=0.4, hspace=0.6)
for i in range(len(dataX[0,:])):
    numcol = 3
    numrow = len(dataNames) / numcol + 1
    ax = figAll_test.add_subplot(numrow,numcol,i+1)
    x = dataX_test[:,i]
    ax.scatter(dataX_test[:,i], dataY_test, color="#222222", alpha=0.7)
    dataIdx = np.argsort(dataX_test[:,i])
    x = dataX_test[:,i][dataIdx]
    y = dataY_pred[dataIdx]
    ax.plot(x,y, color='blue')
    ttl = dataNames[i] + " " + "{:.2f}".format(regr.coef_[i])
    ax.set_title(ttl)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
figAll_test.show()

plt.show()

# sortedIdx = np.argsort(abs(regr.coef_))
# print("sorted idx: ", sortedIdx[::-1])
#
# numShow = 3
# fig = plt.figure()
# for i in range(numShow):
#     fIdx = sortedIdx[::-1][i]
#     ax = fig.add_subplot(1,numShow,i+1)
#     ax.scatter(diabetes_X_test[:,fIdx], diabetes_y_test, color='black')
#     dataIdx = np.argsort(diabetes_X_test[:,fIdx])
#     x = diabetes_X_test[:,fIdx][dataIdx]
#     y = dataY_pred[dataIdx]
#     ax.plot(x,y, color='blue', linewidth=3)
#     ttl = dataNames[fIdx] + " " + "{:.2f}".format(regr.coef_[fIdx])
#     ax.set_title(ttl)
#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
# fig.show()
# plt.show()