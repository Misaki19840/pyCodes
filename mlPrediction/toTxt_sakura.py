import matplotlib.pyplot as plt
import numpy as np
import csv
import glob
import os

# 1月はじまり
daysInMonth = np.array([31,28,31,30,31,30,31,31,30,31,30,31])

placeRow = 65
# 65.. Tokyo

placeRows = [65, 64,  67, 42,
             44, 53, 54, 57,
             59, 60, 61]
placeNames = ["tokyo", "shizuoka", "yokohama", "nagano",
              "utsunomiya", "kumagaya", "mito", "nagoya",
              "kofu", "choshi", "tsu"]

dataPath = "data_sakura/"
for f in glob.glob(os.path.join(dataPath, "*.csv")):
    name, ext = os.path.splitext(f)
    fld, filename = name.split("\\")
    file = open(f, mode='r')
    data_reader = csv.reader(file,delimiter=",")
    data_raw = [row for row in data_reader]
    name_raw = data_raw[0]

    for pidx, plc in enumerate(placeRows):
        data_plc = data_raw[plc]
        # 月日を数値に変換
        data = []
        for i in range(2,len(data_plc)-1,2):
            month = int(data_plc[i])
            day = int(data_plc[i+1])
            if month == 2 and day == 29:
                day = 28
            month = month - 1
            dayVal = sum(daysInMonth[:month]) + (day-1)
            data.append(dayVal)

        print("placeName: ", data_plc[0])
        txtName =  fld + "\\" + placeNames[pidx] + "_" + filename + ".txt"
        file_w = open(txtName, mode='w')
        writer = csv.writer(file_w)
        writer.writerow(data)