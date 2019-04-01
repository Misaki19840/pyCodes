import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt

## 平均値の検定

sNum = 1000
TestType = "OneSided"
# TestType = "TwoSided"
pm_test = 0.3

x = np.random.randn(sNum)
pm = 0
pv = 1
sm = np.mean(x)
s = np.std(x,ddof=1)

# t分布での95%区間の上限、下限
df = len(x) - 1
t_low, t_upp = t.ppf(q=[0.025, 0.975], df=df)

# pmの95%信頼区間
pm_lower = sm + t_low * np.sqrt( s*s / len(x))
pm_upper = sm + t_upp * np.sqrt( s*s / len(x))

print("t_low, t_upp: ", t_low, t_upp)
print("sm: ", sm)
print("pm: ", pm)
print("pm_low, pm_upp: ", pm_lower, pm_upper)
print("pm_test: ", pm_test)

if TestType == "OneSided":
    result = True if (pm_lower <= pm_test ) else False
else:
    result = True if (pm_lower <= pm_test) and (pm_upper >= pm_test) else False
print("result: ", result)

# p値　統計量より極端な値を取る確率
p_val = t.cdf(pm_test, df=df)
p_val = 1 - p_val if p_val > 0.5 else p_val
print("p_val: ", p_val)
res_005 = True if p_val > 0.05 else False
res_001 = True if p_val > 0.01 else False
print("p_val > 0.05: ", res_005)
print("p_val > 0.01: ", res_001)