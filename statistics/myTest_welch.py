import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt

## ウェルチのt検定

# TestType = "OneSided"
TestType = "TwoSided"

sNum1 = 30
pm1 = 5
pv1 = 10
x1 = np.random.normal(pm1, pv1, sNum1)
sm1 = np.mean(x1)
s1 = np.std(x1,ddof=1)

sNum2 = 20
pm2 = 6
pv2 = 15
x2 = np.random.normal(pm2, pv2, sNum2)
sm2 = np.mean(x2)
s2 = np.std(x2,ddof=1)

# t分布での95%区間の上限、下限
df = len(x1) + len(x2) - 2
t_low, t_upp = t.ppf(q=[0.025, 0.975], df=df)

# 統計量
t_val = ( sm1 - sm2 ) / np.sqrt(s1*s1/len(x1) + s2*s2/len(x2))

print("sm1: ", sm1)
print("sm2: ", sm2)
print("t_low, t_upp: ", t_low, t_upp)
print("t_val: ", t_val)

result = True if (t_low <= t_val) and (t_upp >= t_val) else False
print("result: ", result)

# p値　統計量より極端な値を取る確率
p_val = t.cdf(t_val, df=df)
p_val = 1 - p_val if p_val > 0.5 else p_val
print("p_val: ", p_val)
res_005 = True if p_val > 0.05 else False
res_001 = True if p_val > 0.01 else False
print("p_val > 0.05: ", res_005)
print("p_val > 0.01: ", res_001)
