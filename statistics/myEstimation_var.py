import numpy as np
from scipy.stats import t
from scipy.stats import chi2
import matplotlib.pyplot as plt

## kai2 = (n - 1)*s^2 / pv　は自由度 n-1のχ2分布に従う
## 95%信頼区間は次のようになる
## (n-1) * s*s / χ2 0.025(n-1) <= pv <= (n-1) * s*s / χ2 0.975(n-1)

DistType = "Norm"
DistType = "Bin"

sNum = 1000

if DistType == "Norm":
    x = np.random.randn(sNum)
    pm = 0
    pv = 1
elif DistType == "Bin":
    n = 100
    p = 0.2
    x = np.random.binomial(n, p, sNum)
    pm =  n * p
    pv = n * p * ( 1 - p )
sm = np.mean(x)
s = np.std(x,ddof=1)

# 95%区間の上限、下限
low, upp = chi2.ppf(q=[0.025, 0.975], df=len(x) - 1)

# pvの95%信頼区間
pv_lower = ( n -1 ) * s*s / upp
pv_upper = ( n -1 ) * s*s / low

print("low, upp: ", low, upp)
print("s*s: ", s*s)
print("pv: ", pv)
print("pv_low, pv_upp: ", pv_lower, pv_upper)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.hist(x, bins=100)
ax.set_title("histgram")
ax.set_xlabel("x")
ax.set_ylabel("frequency")
fig.show()
plt.show()