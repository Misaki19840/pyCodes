import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

## ポアソン分布の検定

## 1ヵ月あたり平均lamda回起きる事象が起こる回数xはポアソン分布Po(lamda)に従う
## x_=1/n *Σxi (i=1,..,n)は中心極限定理によりN(lamda, lamda/n)に従う

x_base = np.array([349,261,321,309,323,264,294,328,309,376,350,420])
x = np.array([282,288,303,244,282,276,314,310,299,343,372,381])

n = 12
x_test = x.mean()

lamda = x_base.mean()
pm =  lamda
pv = lamda
print("lamda: ", lamda)

# 95%区間の上限、下限
z_low, z_upp = -1.96, 1.96
# z変換したxの値
z = ( x_test - lamda ) / np.sqrt(lamda/n)
print("x_test: ", x_test)
print("z_low, z_upp: ", z_low, z_upp)
print("z: ", z)

# p値　統計量より極端な値を取る確率
p_val = norm.cdf(z)
p_val = 1 - p_val if p_val > 0.5 else p_val
print("p_val: ", p_val)
res_005 = True if p_val > 0.05 else False
res_001 = True if p_val > 0.01 else False
print("p_val > 0.05: ", res_005)
print("p_val > 0.01: ", res_001)