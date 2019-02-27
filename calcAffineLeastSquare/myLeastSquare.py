import numpy as np
from matplotlib import pyplot as plt
import cv2

# X = (xi, yi, 1)
# X' = A * X
# (xi', yi')

## transfrom 3 points

theta = 60
tx, ty = 5, 3
M = np.array([[np.cos(np.deg2rad(theta)), -np.sin(np.deg2rad(theta)), tx],
                 [np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta)), ty]], float)

# points before transform
X = np.array([[15, 5, 1],
              [25, 10, 1],
              [20, 15, 1]])

# points after transform
X_tfm = M.dot(X.T)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("point")
ax.grid()
ax.set_xlim([0, 30])
ax.set_ylim([0, 30])
for i in range(len(X)):
    ax.plot(X[i][0], X[i][1], marker='.', markersize=10, color='r')
    ax.plot(X_tfm.T[i][0], X_tfm.T[i][1], marker='.', markersize=10, color='b')

fig.show()
fig.savefig("myLeastSquare_transformImg.jpg")
#cv2.waitKey(0)

XX = np.array([[X[0][0], X[0][1], 1, 0, 0, 0],
               [0, 0, 0, X[0][0], X[0][1], 1],
               [X[1][0], X[1][1], 1, 0, 0, 0],
               [0, 0, 0, X[1][0], X[1][1], 1],
               [X[2][0], X[2][1], 1, 0, 0, 0],
               [0, 0, 0, X[2][0], X[2][1], 1]])

XX_tfm = np.array([X_tfm.T[0][0], X_tfm.T[0][1],X_tfm.T[1][0], X_tfm.T[1][1],X_tfm.T[2][0], X_tfm.T[2][1]]).T

M_slv = np.linalg.inv(XX).dot(XX_tfm)

print("M: ")
print(M)
print("M: ")
print(M_slv)

## transform 4 points
X = np.array([[15, 5, 1],
              [25, 10, 1],
              [20, 15, 1],
              [18, 18, 1]])
X_tfm = M.dot(X.T)

XX = []
XX_tfm = []
for i in range(len(X)):
    XX.append([X[i][0], X[i][1], 1, 0, 0, 0])
    XX.append([0, 0, 0, X[i][0], X[i][1], 1])

    XX_tfm.append(X_tfm.T[i][0])
    XX_tfm.append(X_tfm.T[i][1])

XX = np.array(XX)
XX_tfm = np.array(XX_tfm).T

Mat = XX.T.dot(XX)
Mat_inv = np.linalg.inv(Mat)
M_slv = np.linalg.inv(XX.T.dot(XX)).dot(XX.T).dot(XX_tfm)

print("M: ")
print(M)
print("M_slv: ")
print(M_slv)