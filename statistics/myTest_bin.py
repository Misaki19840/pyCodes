import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

## 二項分布の検定

# TestType = "OneSided"
TestType = "TwoSided"
x_test = 2200

n = 12000
p = 1/6
pm =  n * p
pv = n * p * ( 1 - p )

# 95%区間の上限、下限
z_low, z_upp = -1.96, 1.96

# z変換したxの値
z = ( x_test - pm ) / np.sqrt(pv)

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
