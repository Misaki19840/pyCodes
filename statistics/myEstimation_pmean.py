import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt

## サンプルの母集団が正規分布に従うとき、
## t = (sm - pm) / sqrt( s^2 / n )　は自由度 t-1のt分布に従う
## 95%信頼区間は次のようになる
## sm + T0.025 * sqrt( s^2/n ) <= pm <= sp + T0.975 * sqrt( s^2/n )

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

# t分布での95%区間の上限、下限
t_low, t_upp = t.ppf(q=[0.025, 0.975], df=len(x) - 1)

# pmの95%信頼区間
pm_lower = sm + t_low * np.sqrt( s*s / len(x))
pm_upper = sm + t_upp * np.sqrt( s*s / len(x))

print("t_low, t_upp: ", t_low, t_upp)
print("sm: ", sm)
print("pm: ", pm)
print("pm_low, pm_upp: ", pm_lower, pm_upper)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.hist(x, bins=100)
ax.set_title("histgram")
ax.set_xlabel("x")
ax.set_ylabel("frequency")
fig.show()
plt.show()