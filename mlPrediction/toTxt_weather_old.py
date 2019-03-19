import matplotlib.pyplot as plt
import numpy as np
import csv
import glob
import os

ArrIdx = [1,4,10,14,18,22,26,29,31,34,36,42,45,48,51,54,57,61,70,74,]
# 0.. 日付　冗長なのでパス
# 64, 67.. 天気　はラベルが数値化しにくいのでパス
# 7.. 最低気温　地域限定

region2_list = ["choshi", ]
ArrIdx_region2 = [1,4,7,11,15,19,23,26,28,31,33,36,42,45,48,51,54,57,61,70,]

dataPath = "data_weather_test/"
for f in glob.glob(os.path.join(dataPath, "*.csv")):
    name, ext = os.path.splitext(f)
    txtName = name + ".txt"
    file = open(f, mode='r')
    data_reader = csv.reader(file,delimiter=",")
    data_raw = [row for row in data_reader]
    name_raw = data_raw[3]
    data_raw = data_raw[6:]

    print("dataName :", f)

    data = []
    # name = [ name_raw[j]  for j in range(1,len(name_raw),3)]
    data_name = [ name_raw[idx] for idx in ArrIdx ]
    for i in range(len(data_raw)):
        # data_new = [ data_raw[i][j]  for j in range(1,len(data_raw[i]),3)]
        data_new = [data_raw[i][idx] for idx in ArrIdx]
        data.append(data_new)

    # #インデックスごと修正
    # # 0..月日
    # # 0.. 4/1 - 1..3/31
    # fidx = 0
    # dateVal = 0
    # popIdx = -1
    # for i in range(len(data)):
    #     year, month, day = data[i][fidx].split("/")
    #     year = int(year)
    #     month = int(month)
    #     day = int(day)
    #     if (month == 2) and (day == 29):
    #         popIdx = i
    #         continue
    #     data[i][0] = dateVal
    #     dateVal += 1
    # # 2/29を除去
    # if popIdx >= 0:
    #     data.pop(popIdx)

    # 9, 11.. 風向き
    # python list は arr[:,i]という表現が使えない
    fidxArr = [ArrIdx.index(31),ArrIdx.index(36)+15]
    for fidx in fidxArr:
        data_name[fidx] = "N"
        data_name.insert(fidx + 1, "NNE")
        data_name.insert(fidx + 2, "NE")
        data_name.insert(fidx + 3, "NEE")
        data_name.insert(fidx + 4, "E")
        data_name.insert(fidx + 5, "EES")
        data_name.insert(fidx + 6, "ES")
        data_name.insert(fidx + 7, "ESS")
        data_name.insert(fidx + 8, "S")
        data_name.insert(fidx + 9, "SSW")
        data_name.insert(fidx + 10, "SW")
        data_name.insert(fidx + 11, "SWW")
        data_name.insert(fidx + 12, "W")
        data_name.insert(fidx + 13, "WWN")
        data_name.insert(fidx + 14, "WN")
        data_name.insert(fidx + 15, "WNN")
        for i in range(len(data)):
            val = data[i][fidx]
            if "北北東" in data[i][fidx]:
                data[i][fidx] = 0
                data[i].insert(fidx + 1, 1)
                data[i].insert(fidx + 2, 0)
                data[i].insert(fidx + 3, 0)
                data[i].insert(fidx + 4, 0)
                data[i].insert(fidx + 5, 0)
                data[i].insert(fidx + 6, 0)
                data[i].insert(fidx + 7, 0)
                data[i].insert(fidx + 8, 0)
                data[i].insert(fidx + 9, 0)
                data[i].insert(fidx + 10, 0)
                data[i].insert(fidx + 11, 0)
                data[i].insert(fidx + 12, 0)
                data[i].insert(fidx + 13, 0)
                data[i].insert(fidx + 14, 0)
                data[i].insert(fidx + 15, 0)
            elif "東北東" in data[i][fidx]:
                data[i][fidx] = 0
                data[i].insert(fidx + 1, 0)
                data[i].insert(fidx + 2, 0)
                data[i].insert(fidx + 3, 1)
                data[i].insert(fidx + 4, 0)
                data[i].insert(fidx + 5, 0)
                data[i].insert(fidx + 6, 0)
                data[i].insert(fidx + 7, 0)
                data[i].insert(fidx + 8, 0)
                data[i].insert(fidx + 9, 0)
                data[i].insert(fidx + 10, 0)
                data[i].insert(fidx + 11, 0)
                data[i].insert(fidx + 12, 0)
                data[i].insert(fidx + 13, 0)
                data[i].insert(fidx + 14, 0)
                data[i].insert(fidx + 15, 0)
            elif "東南東" in data[i][fidx]:
                data[i][fidx] = 0
                data[i].insert(fidx + 1, 0)
                data[i].insert(fidx + 2, 0)
                data[i].insert(fidx + 3, 0)
                data[i].insert(fidx + 4, 0)
                data[i].insert(fidx + 5, 1)
                data[i].insert(fidx + 6, 0)
                data[i].insert(fidx + 7, 0)
                data[i].insert(fidx + 8, 0)
                data[i].insert(fidx + 9, 0)
                data[i].insert(fidx + 10, 0)
                data[i].insert(fidx + 11, 0)
                data[i].insert(fidx + 12, 0)
                data[i].insert(fidx + 13, 0)
                data[i].insert(fidx + 14, 0)
                data[i].insert(fidx + 15, 0)
            elif "南南東" in data[i][fidx]:
                data[i][fidx] = 0
                data[i].insert(fidx + 1, 1)
                data[i].insert(fidx + 2, 0)
                data[i].insert(fidx + 3, 0)
                data[i].insert(fidx + 4, 0)
                data[i].insert(fidx + 5, 0)
                data[i].insert(fidx + 6, 0)
                data[i].insert(fidx + 7, 1)
                data[i].insert(fidx + 8, 0)
                data[i].insert(fidx + 9, 0)
                data[i].insert(fidx + 10, 0)
                data[i].insert(fidx + 11, 0)
                data[i].insert(fidx + 12, 0)
                data[i].insert(fidx + 13, 0)
                data[i].insert(fidx + 14, 0)
                data[i].insert(fidx + 15, 0)
            elif "南南西" in data[i][fidx]:
                data[i][fidx] = 0
                data[i].insert(fidx + 1, 0)
                data[i].insert(fidx + 2, 0)
                data[i].insert(fidx + 3, 0)
                data[i].insert(fidx + 4, 0)
                data[i].insert(fidx + 5, 0)
                data[i].insert(fidx + 6, 0)
                data[i].insert(fidx + 7, 0)
                data[i].insert(fidx + 8, 0)
                data[i].insert(fidx + 9, 1)
                data[i].insert(fidx + 10, 0)
                data[i].insert(fidx + 11, 0)
                data[i].insert(fidx + 12, 0)
                data[i].insert(fidx + 13, 0)
                data[i].insert(fidx + 14, 0)
                data[i].insert(fidx + 15, 0)
            elif "西南西" in data[i][fidx]:
                data[i][fidx] = 0
                data[i].insert(fidx + 1, 0)
                data[i].insert(fidx + 2, 0)
                data[i].insert(fidx + 3, 0)
                data[i].insert(fidx + 4, 0)
                data[i].insert(fidx + 5, 0)
                data[i].insert(fidx + 6, 0)
                data[i].insert(fidx + 7, 0)
                data[i].insert(fidx + 8, 0)
                data[i].insert(fidx + 9, 0)
                data[i].insert(fidx + 10, 0)
                data[i].insert(fidx + 11, 1)
                data[i].insert(fidx + 12, 0)
                data[i].insert(fidx + 13, 0)
                data[i].insert(fidx + 14, 0)
                data[i].insert(fidx + 15, 0)
            elif "西北西" in data[i][fidx]:
                data[i][fidx] = 0
                data[i].insert(fidx + 1, 0)
                data[i].insert(fidx + 2, 0)
                data[i].insert(fidx + 3, 0)
                data[i].insert(fidx + 4, 0)
                data[i].insert(fidx + 5, 0)
                data[i].insert(fidx + 6, 0)
                data[i].insert(fidx + 7, 0)
                data[i].insert(fidx + 8, 0)
                data[i].insert(fidx + 9, 0)
                data[i].insert(fidx + 10, 0)
                data[i].insert(fidx + 11, 0)
                data[i].insert(fidx + 12, 0)
                data[i].insert(fidx + 13, 1)
                data[i].insert(fidx + 14, 0)
                data[i].insert(fidx + 15, 0)
            elif "北北西" in data[i][fidx]:
                data[i][fidx] = 0
                data[i].insert(fidx + 1, 0)
                data[i].insert(fidx + 2, 0)
                data[i].insert(fidx + 3, 0)
                data[i].insert(fidx + 4, 0)
                data[i].insert(fidx + 5, 0)
                data[i].insert(fidx + 6, 0)
                data[i].insert(fidx + 7, 0)
                data[i].insert(fidx + 8, 0)
                data[i].insert(fidx + 9, 0)
                data[i].insert(fidx + 10, 0)
                data[i].insert(fidx + 11, 0)
                data[i].insert(fidx + 12, 0)
                data[i].insert(fidx + 13, 0)
                data[i].insert(fidx + 14, 0)
                data[i].insert(fidx + 15, 1)
            elif "北東" in data[i][fidx]:
                data[i][fidx] = 0
                data[i].insert(fidx + 1, 0)
                data[i].insert(fidx + 2, 1)
                data[i].insert(fidx + 3, 0)
                data[i].insert(fidx + 4, 0)
                data[i].insert(fidx + 5, 0)
                data[i].insert(fidx + 6, 0)
                data[i].insert(fidx + 7, 0)
                data[i].insert(fidx + 8, 0)
                data[i].insert(fidx + 9, 0)
                data[i].insert(fidx + 10, 0)
                data[i].insert(fidx + 11, 0)
                data[i].insert(fidx + 12, 0)
                data[i].insert(fidx + 13, 0)
                data[i].insert(fidx + 14, 0)
                data[i].insert(fidx + 15, 0)
            elif "南東" in data[i][fidx]:
                data[i][fidx] = 0
                data[i].insert(fidx + 1, 0)
                data[i].insert(fidx + 2, 0)
                data[i].insert(fidx + 3, 0)
                data[i].insert(fidx + 4, 0)
                data[i].insert(fidx + 5, 0)
                data[i].insert(fidx + 6, 1)
                data[i].insert(fidx + 7, 0)
                data[i].insert(fidx + 8, 0)
                data[i].insert(fidx + 9, 0)
                data[i].insert(fidx + 10, 0)
                data[i].insert(fidx + 11, 0)
                data[i].insert(fidx + 12, 0)
                data[i].insert(fidx + 13, 0)
                data[i].insert(fidx + 14, 0)
                data[i].insert(fidx + 15, 0)
            elif "南西" in data[i][fidx]:
                data[i][fidx] = 0
                data[i].insert(fidx + 1, 0)
                data[i].insert(fidx + 2, 0)
                data[i].insert(fidx + 3, 0)
                data[i].insert(fidx + 4, 0)
                data[i].insert(fidx + 5, 0)
                data[i].insert(fidx + 6, 0)
                data[i].insert(fidx + 7, 0)
                data[i].insert(fidx + 8, 0)
                data[i].insert(fidx + 9, 0)
                data[i].insert(fidx + 10, 1)
                data[i].insert(fidx + 11, 0)
                data[i].insert(fidx + 12, 0)
                data[i].insert(fidx + 13, 0)
                data[i].insert(fidx + 14, 0)
                data[i].insert(fidx + 15, 0)
            elif "北西" in data[i][fidx]:
                data[i][fidx] = 0
                data[i].insert(fidx + 1, 0)
                data[i].insert(fidx + 2, 0)
                data[i].insert(fidx + 3, 0)
                data[i].insert(fidx + 4, 0)
                data[i].insert(fidx + 5, 0)
                data[i].insert(fidx + 6, 0)
                data[i].insert(fidx + 7, 0)
                data[i].insert(fidx + 8, 0)
                data[i].insert(fidx + 9, 0)
                data[i].insert(fidx + 10, 0)
                data[i].insert(fidx + 11, 0)
                data[i].insert(fidx + 12, 0)
                data[i].insert(fidx + 13, 0)
                data[i].insert(fidx + 14, 1)
                data[i].insert(fidx + 15, 0)
            elif "北" in data[i][fidx] :
                data[i][fidx] = 1
                data[i].insert(fidx + 1, 0)
                data[i].insert(fidx + 2, 0)
                data[i].insert(fidx + 3, 0)
                data[i].insert(fidx + 4, 0)
                data[i].insert(fidx + 5, 0)
                data[i].insert(fidx + 6, 0)
                data[i].insert(fidx + 7, 0)
                data[i].insert(fidx + 8, 0)
                data[i].insert(fidx + 9, 0)
                data[i].insert(fidx + 10, 0)
                data[i].insert(fidx + 11, 0)
                data[i].insert(fidx + 12, 0)
                data[i].insert(fidx + 13, 0)
                data[i].insert(fidx + 14, 0)
                data[i].insert(fidx + 15, 0)
            elif "東" in data[i][fidx]:
                data[i][fidx] = 0
                data[i].insert(fidx + 1, 1)
                data[i].insert(fidx + 2, 0)
                data[i].insert(fidx + 3, 0)
                data[i].insert(fidx + 4, 1)
                data[i].insert(fidx + 5, 0)
                data[i].insert(fidx + 6, 0)
                data[i].insert(fidx + 7, 0)
                data[i].insert(fidx + 8, 0)
                data[i].insert(fidx + 9, 0)
                data[i].insert(fidx + 10, 0)
                data[i].insert(fidx + 11, 0)
                data[i].insert(fidx + 12, 0)
                data[i].insert(fidx + 13, 0)
                data[i].insert(fidx + 14, 0)
                data[i].insert(fidx + 15, 0)
            elif "南" in data[i][fidx]:
                data[i][fidx] = 0
                data[i].insert(fidx + 1, 0)
                data[i].insert(fidx + 2, 0)
                data[i].insert(fidx + 3, 0)
                data[i].insert(fidx + 4, 0)
                data[i].insert(fidx + 5, 0)
                data[i].insert(fidx + 6, 0)
                data[i].insert(fidx + 7, 0)
                data[i].insert(fidx + 8, 1)
                data[i].insert(fidx + 9, 0)
                data[i].insert(fidx + 10, 0)
                data[i].insert(fidx + 11, 0)
                data[i].insert(fidx + 12, 0)
                data[i].insert(fidx + 13, 0)
                data[i].insert(fidx + 14, 0)
                data[i].insert(fidx + 15, 0)
            elif "西" in data[i][fidx]:
                data[i][fidx] = 0
                data[i].insert(fidx + 1, 0)
                data[i].insert(fidx + 2, 0)
                data[i].insert(fidx + 3, 0)
                data[i].insert(fidx + 4, 0)
                data[i].insert(fidx + 5, 0)
                data[i].insert(fidx + 6, 0)
                data[i].insert(fidx + 7, 0)
                data[i].insert(fidx + 8, 0)
                data[i].insert(fidx + 9, 0)
                data[i].insert(fidx + 10, 0)
                data[i].insert(fidx + 11, 0)
                data[i].insert(fidx + 12, 1)
                data[i].insert(fidx + 13, 0)
                data[i].insert(fidx + 14, 0)
                data[i].insert(fidx + 15, 0)

    # interpolate missing values
    for fidx in range(len(data[0])):
        for i in range(len(data)):
            val = data[i][fidx]
            if val == '':
                if i > 0:
                    data[i][fidx] = data[i-1][fidx]
                else:
                    print("print here is unexpected.")

    # data.insert(0,data_name)
    print("len(data): ", len(data))

    file_w = open(txtName, mode='w', newline="")
    writer = csv.writer(file_w)
    writer.writerow(data_name)
    for i in range(len(data)):
        writer.writerow(data[i])
