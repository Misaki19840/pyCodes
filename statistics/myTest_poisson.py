import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

## ポアソン分布の検定

## 1ヵ月あたり平均lamda回起きる事象が起こる回数xはポアソン分布Po(lamda)に従う
## x_=1/n *Σxi (i=1,..,n)は中心極限定理によりN(lamda, lamda/n)に従う


# TestType = "OneSided"
TestType = "TwoSided"
n = 12
x_test = 16

lamda = 20
pm =  lamda
pv = lamda

# 95%区間の上限、下限
z_low, z_upp = -1.96, 1.96

# z変換したxの値
z = ( x_test - lamda ) / np.sqrt(lamda/n)

print("x_test: ", x_test)
print("z_low, z_upp: ", z_low, z_upp)
print("z: ", z)

if TestType == "OneSided":
    result = True if (z_low <= z ) else False
else:
    result = True if (z_low <= z) and (z_upp >= z) else False
print("result: ", result)

# p値　統計量より極端な値を取る確率
p_val = norm.cdf(z)
p_val = 1 - p_val if p_val > 0.5 else p_val
print("p_val: ", p_val)
res_005 = True if p_val > 0.05 else False
res_001 = True if p_val > 0.01 else False
print("p_val > 0.05: ", res_005)
print("p_val > 0.01: ", res_001)