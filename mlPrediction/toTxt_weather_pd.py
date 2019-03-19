import matplotlib.pyplot as plt
import numpy as np
import csv
import glob
import os
import pandas as pd

fNameList = [
"平均気温(℃)",
"最高気温(℃)",
"降水量の合計(mm)",
"日照時間(時間)",
"最深積雪(cm)",
"降雪量合計(cm)",
"平均風速(m/s)",
"最大風速(m/s)",
"最大瞬間風速(m/s)",
"平均蒸気圧(hPa)",
"平均湿度(％)",
"最小相対湿度(％)",
"平均現地気圧(hPa)",
"平均海面気圧(hPa)",
"最低海面気圧(hPa)",
"平均雲量(10分比)",
"10分間降水量の最大(mm)",
#"合計全天日射量(MJ/㎡)",
]

ymdName = "年月日"

fNameList_new = [
"Ave(tmp)[cels]",
"Max(tmp)[cels]",
"Sum(rain)[mm]",
"Len(day)[hour]",
"Max(snow)[cm]",
"Sum(snow)[cm]",
"windSpeed[m/s]",
"Max(WS)[m/s]",
"Max(inst WS)[m/s]",
"Ave(stream P)[hPa]",
"Ave(humid)[％]",
"Min(humid)[％]",
"Ave(P)[hPa]",
"Ave(Sea P)[hPa]",
"Min(Sea P)[hPa]",
"Ave(Clowd)[10min]",
"Max(rain10min)[mm]",
#"Sum(sol radiation)[MJ/㎡]",
]

file_log = open("log.txt", mode='w', newline="")
writer_log = csv.writer(file_log)

dataPath = "data_weather_test/"
for f in glob.glob(os.path.join(dataPath, "*.csv")):
    name, ext = os.path.splitext(f)
    pdData_raw = pd.read_csv(f,header=2, encoding="cp932")
    # 2/29をスキップ
    for i, val in enumerate(pdData_raw[ymdName][2:]):
        year, month, day = val.split("/")
        year = int(year)
        month = int(month)
        day = int(day)
        if (month == 2) and (day == 29):
            pdData_raw = pdData_raw.drop(i, axis=0)
    # print("平均気温(℃)", pdData_raw["平均気温(℃)"][2:])
    pdData = pd.DataFrame({
        fNameList_new[0]: pdData_raw[fNameList[0]][2:],
        fNameList_new[1]: pdData_raw[fNameList[1]][2:],
        fNameList_new[2]: pdData_raw[fNameList[2]][2:],
        fNameList_new[3]: pdData_raw[fNameList[3]][2:],
        fNameList_new[4]: pdData_raw[fNameList[4]][2:],
        fNameList_new[5]: pdData_raw[fNameList[5]][2:],
        fNameList_new[6]: pdData_raw[fNameList[6]][2:],
        fNameList_new[7]: pdData_raw[fNameList[7]][2:],
        fNameList_new[8]: pdData_raw[fNameList[8]][2:],
        fNameList_new[9]: pdData_raw[fNameList[9]][2:],
        fNameList_new[10]: pdData_raw[fNameList[10]][2:],
        fNameList_new[11]: pdData_raw[fNameList[11]][2:],
        fNameList_new[12]: pdData_raw[fNameList[12]][2:],
        fNameList_new[13]: pdData_raw[fNameList[13]][2:],
        fNameList_new[14]: pdData_raw[fNameList[14]][2:],
        fNameList_new[15]: pdData_raw[fNameList[15]][2:],
        fNameList_new[16]: pdData_raw[fNameList[16]][2:],
#        fNameList_new[17]: pdData_raw[fNameList[17]][2:],
                           })
    pdData = pdData.reset_index(drop=True)

    # 空白は直前の値で補間
    for fnm_new in fNameList_new:
        for i,val in enumerate(pdData[fnm_new]):
            if np.isnan(val):
                if i > 0:
                    pdData[fnm_new][i] = pdData[fnm_new][i-1]
                else:
                    pdData[fnm_new][i] = pdData[fnm_new][i+1]

    pdDataList = pdData.values.tolist()
    pdHead = pdData.head(1)
    file_w = open(name +".txt", mode='w', newline="")
    writer = csv.writer(file_w)
    writer.writerow(pdHead)
    for i in range(len(pdDataList)):
        writer.writerow(pdDataList[i])
    # utf8で保存できない
    # pdData.to_csv(name +".txt", index=False, encoding='utf-8')

    # log
    print("pdData.shape", pdData.shape)
    #print("pdData.describe()",pdData.describe())
    writer_log.writerow([name])
    writer_log.writerow(pdData.shape)
    writer_log.writerows(pdData.describe().values.tolist())
    pdData_np = np.array(pdDataList)
    isNanIdx = np.argwhere(np.isnan(pdData_np))
    isInfIdx = np.argwhere(np.isinf(pdData_np))
    writer_log.writerow(["isNanIdx: ", isNanIdx])
    writer_log.writerow(["isInfIdx: ", isInfIdx])
