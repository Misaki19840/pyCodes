import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt

## nがある程度大きい時、Bin(n,p)はN(np,np(1-p))に近似できる
## よって、z = ( x - np ) / sqrt( np(1-p) )はN(0,1)に従う
## z = ( x/n - p ) / sqrt( p(1-p)/n )
## = ( sp - p ) / sqrt( p(1-p)/n )と書ける
## 95%信頼区間は次のようになる
## sp - 1.96 * sqrt( p(1-p)/n ) <= p <= sp + 1.96 * sqrt( p(1-p)/n )
## nが大きい時はp~=spなので、以下の式を使う
## sp - 1.96 * sqrt( sp(1-sp)/n ) <= p <= sp + 1.96 * sqrt( sp(1-sp)/n )

sNum = 1000
n = 100
p = 0.2
x = np.random.binomial(n, p, sNum)
pm =  n * p
pv = n * p * ( 1 - p )
sm = np.mean(x)
s = np.std(x,ddof=1)
sp = sm / n

# 95%区間の上限、下限
norm_low, norm_upp = -1.96, 1.96

# pmの95%信頼区間
p_lower = sp + norm_low * np.sqrt( sp*(1-sp) / n)
p_upper = sp + norm_upp * np.sqrt( sp*(1-sp) / n)

print("t_low, t_upp: ", norm_low, norm_upp)
print("sp: ", sp)
print("p: ", p)
print("p_low, p_upp: ", p_lower, p_upper)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.hist(x, bins=100)
ax.set_title("histgram")
ax.set_xlabel("x")
ax.set_ylabel("frequency")
fig.show()
plt.show()