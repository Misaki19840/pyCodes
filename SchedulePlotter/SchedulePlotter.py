
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz


## year, month
st_y = 2017
st_m = 4
ed_y = 2021
ed_m = 3

## o
shape00_y = [2017,2018,2019]
shape00_m = [5,6,4]
shape00_lb = ["init","mid","fin"]
## □
shape01_y = [2017,2017]
shape01_m = [6,8]
shape01_lb = ["s1","s2"]
## △
shape02_y = [2018,2018,2019]
shape02_m = [5,10,12]
shape02_lb = ["r1","",""]

xlbls = []
current_m = st_m
current_y = st_y
while 1:
    if current_m == 1:
        xlbl = str(current_m) + "\n" + str(current_y)
    elif (current_y == st_y) and (current_m == st_m):
        xlbl = str(current_m) + "\n" + str(current_y)
    else:
        xlbl = str(current_m)
    xlbls.append(xlbl)

    current_m += 1
    if current_m == 13:
        current_m = 1
        current_y += 1

    if (current_y == ed_y) and (current_m == ed_m):
        break


shape00_xpoint = []
for idx, year in enumerate(shape00_y):
    st_point = st_y * 12 + st_m
    tgt_point = year * 12 + shape00_m[idx]
    xpoint = tgt_point - st_point
    shape00_xpoint.append(xpoint)

shape01_xpoint = []
for idx, year in enumerate(shape01_y):
    st_point = st_y * 12 + st_m
    tgt_point = year * 12 + shape01_m[idx]
    xpoint = tgt_point - st_point
    shape01_xpoint.append(xpoint)

shape02_xpoint = []
for idx, year in enumerate(shape02_y):
    st_point = st_y * 12 + st_m
    tgt_point = year * 12 + shape02_m[idx]
    xpoint = tgt_point - st_point
    shape02_xpoint.append(xpoint)

## figsize a4 = (11 inch, 8 inch)
fig = plt.figure(figsize=(11,8))

ax = fig.add_subplot(1,1,1)
ax.set_title("title")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_xticks(np.arange(0, len(xlbls)-1, 1), minor=False)
ax.set_xticklabels(xlbls)
ax.set_xlim([-1,len(xlbls)])
ax.set_yticks(np.linspace(0, 5, 0), minor=False)
ax.set_ylim([0,1])
ax.grid(b=None, which='major', axis='x')

for idx, xp in enumerate(shape00_xpoint):
    ax.plot(xp, 0.2, marker="o", markersize=20, color="blue")
    ax.text(xp, 0.2, shape00_lb[idx])

for idx, xp in enumerate(shape01_xpoint):
    ax.plot(xp, 0.4, marker="s", markersize=20, color="blue")
    ax.text(xp, 0.4, shape01_lb[idx])

for idx, xp in enumerate(shape02_xpoint):
    ax.plot(xp, 0.6, marker="D", markersize=20, color="blue")
    ax.text(xp, 0.6, shape02_lb[idx])

plt.show()

