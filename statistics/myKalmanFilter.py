import numpy as np
import matplotlib.pyplot as plt
import math

# wikipedia 行列の形式で書かれているので参考になる
# https://ja.wikipedia.org/wiki/%E3%82%AB%E3%83%AB%E3%83%9E%E3%83%B3%E3%83%95%E3%82%A3%E3%83%AB%E3%82%BF%E3%83%BC

# obs, estという表記が分かりやすい
# https://qiita.com/IshitaTakeshi/items/740ac7e9b549eee4cc04

# 1次元の場合のカルマンゲインの導出
# http://bufonmake.blogspot.com/2015/05/test.html

# ?? 行列の形式のカルマンゲインk=PHS^-1が分からない。　xが２次元の時、Kの要素はどう書ける？？

tnum = 20
dt = 0.1
acc = 0.1

x_odo = np.array([[0, 0]]).T #状態
x_est = np.array([[0, 0]]).T #状態
P_est = np.array([[0, 0],[0, 0]])
F = np.array([[1, dt],[0, 1]])
B = np.array([[0.5*dt**2, dt]]).T
u = acc
H = np.array([[1, 0],[0, 1]]) #状態→観測空間にする
R = np.array([[0.1, 0],[0, 0.1]])#観測値にのるノイズの共分散行列
Q = np.array([[0.01, 0],[0, 0.01]])#予測値にのるノイズの共分散行列

x_tr = np.array([[0, 0]]).T

x_odo_array = []
x_est_array = []
z_array = []
x_klm_array = []
t_array_0 = []
t_array = []

x_tr_array = []

x_est_array.append(x_est)
x_tr_array.append(x_tr)
t_array_0.append(0)

for i in range(tnum):
    # 真値
    x_tr = F.dot(x_tr) + B*u + np.array([[np.random.normal(0,Q[0,0]), np.random.normal(0,Q[1,1])]]).T
    x_tr_array.append(x_tr)

    # モデルに基づく推定
    x_odo = F.dot(x_odo) + B*u
    x_odo_array.append(x_odo)
    P_est = F.dot(P_est).dot(F.T) + Q #??

    # 観測値
    z = H.dot(x_tr) + np.array([[np.random.normal(0,R[0,0]), np.random.normal(0,R[1,1])]]).T
    z_array.append(z)

    #　誤差
    e = z - H.dot(x_est)

    #S = R + H.dot(P_est).dot(H.T)
    #K = P_est.dot(H.T).dot(np.linalg.inv(S))

    # x_klm = x_est + K.dot(e)
    # P_klm = (np.matrix(np.identity(2)) - K.dot(H)).dot(P_est)

    #x_klm_array.append(x_klm)

    K = np.array([[0.1, 0],[0, 0.1]])
    I = np.array([[1, 0],[0, 1]])
    x_est = (I - K.dot(H)).dot(x_odo) + K.dot(z)
    x_est_array.append(x_est)

    t_array_0.append((i+1)*dt)
    t_array.append((i+1)*dt)

    ''' 更新 '''
    # x_est = x_klm
    # P_est = P_klm

# 位置だけ取り出す
x_tr_array_pos = []
x_est_array_pos = []
for i in range(len(x_est_array)):
    x_est_array_pos.append(x_est_array[i][0,0])
    x_tr_array_pos.append(x_tr_array[i][0,0])

z_array_pos = []
x_klm_array_pos = []
for i in range(len(z_array)):
    z_array_pos.append(z_array[i][0,0])
    # x_klm_array_pos.append(x_klm_array[i][0,0])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(t_array_0,x_tr_array_pos,color='red',marker='o',label='gt')
ax.plot(t_array_0,x_est_array_pos,color='green',marker='o',label='est')
ax.plot(t_array,z_array_pos,color='blue',marker='o',label='obs')
# ax.plot(t_array,x_klm_array_pos,color='orange',marker='o',label='signal filtered')
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.legend(loc="lower right")

fig.show()

a = 1