import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt

## 適合性の検定

# TestType = "OneSided"
TestType = "TwoSided"

x = np.array([55, 22, 16, 7])
x_true = np.array([40, 30, 20, 10])

# t分布での95%区間の上限、下限
df = len(x)-1
chi2_upp = chi2.ppf(q=[0.95], df=df)

# 統計量
x_diff = (x - x_true)*(x - x_true)/x_true
chi2_val = np.sum(x_diff)

print("chi2_upp: ", chi2_upp)
print("chi2_val: ", chi2_val)

result = True if (chi2_upp >= chi2_val) else False
print("result: ", result)

# p値　統計量より極端な値を取る確率
p_val = chi2.cdf(chi2_val, df=df)
p_val = 1 - p_val if p_val > 0.5 else p_val
print("p_val: ", p_val)
res_005 = True if p_val > 0.05 else False
res_001 = True if p_val > 0.01 else False
print("p_val > 0.05: ", res_005)
print("p_val > 0.01: ", res_001)