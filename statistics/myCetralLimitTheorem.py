import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.stats import norm
import math

## 中心極限定理
## 平均値が正規分布に従う

DistType = "Norm"
DistType = "Bin"
DistType = "Poisson"

sNum = 10000
numSampleAve = 12

x_true = []
x_distb = []
x_distb_ = []

if DistType == "Norm":
    pm = 0
    pv = 1
    x = np.random.normal(pm, pv, sNum)
    x_ = np.array([np.random.normal(pm, pv, numSampleAve).mean() for i in range(sNum)])
    m_ = x_.mean()
    std_ = x_.std()
elif DistType == "Bin":
    n = 100
    p = 0.2
    x = np.random.binomial(n, p, sNum)
    pm =  n * p
    pv = n * p * ( 1 - p )
    x_ = np.array([np.random.binomial(n, p, numSampleAve).mean() for i in range(sNum)])
    m_ = x_.mean()
    std_ = x_.std()
elif DistType == "Poisson":
    lam = 300
    x = np.random.poisson(lam,sNum)
    pm = lam
    pv = lam
    x_ = np.array([np.random.poisson(lam,numSampleAve).mean() for i in range(sNum)])
    m_ = x_.mean()
    std_ = x_.std()
    #母集団の分布
    x_true = np.arange(x.min(),x.max())
    x_distb = np.array([poisson.pmf(x_true[i], lam) for i in range(len(x_true))])
    #中心極限定理で近似される正規分布
    x_distb_ = np.array([norm.pdf(x_true[i], pm, np.sqrt(pv/numSampleAve)) for i in range(len(x_true))])
sm = np.mean(x)
s = np.std(x,ddof=1)
sstd = np.std(x,ddof=0)


fig = plt.figure()
fig.subplots_adjust(top=0.85)
ax = fig.add_subplot(1,2,1)
ax.hist(x, bins=100, normed = True, label="sample")
ax.set_title("histgram")
ax.set_xlabel("x")
ax.set_ylabel("frequency")
ax.plot(x_true,x_distb,color="#ff0000",label="population")
text = "population" + "\n"
text += "mean: " + "{:.2f}".format(pm) + "\n"
text += "var: " + "{:.2f}".format(pv) + "\n"
text += "\nsample" + "\n"
text += "mean: " + "{:.2f}".format(sm) + "\n"
text += "var: " + "{:.2f}".format(sstd*sstd)
ax.text(0.1, 0.7, text,
        bbox={'facecolor':'#ffffff', 'alpha':0.5, 'pad':10},
        transform=ax.transAxes,)
ax.legend()

ax = fig.add_subplot(1,2,2)
ax.hist(x_, bins=100,  normed = True, label="sample mean")
ax.set_title("histgram")
ax.set_xlabel("x")
ax.set_ylabel("frequency")
ax.set_xlim([x.min(),x.max()])
ax.plot(x_true,x_distb_,color="#ff0000", label="CLT")
text = "CLT" + "\n"
text += "pm: " + "{:.2f}".format(pm) + "\n"
text += "pv/n: " + "{:.2f}".format(pv/numSampleAve) + "\n"
text += "\n"
text += "sample mean" + "\n"
text += "mean: " + "{:.2f}".format(m_) + "\n"
text += "var: " + "{:.2f}".format(std_*std_)
ax.text(0.1, 0.7, text,
        bbox={'facecolor':'#ffffff', 'alpha':0.5, 'pad':10},
        transform=ax.transAxes,)
ax.legend()
fig.show()
plt.show()