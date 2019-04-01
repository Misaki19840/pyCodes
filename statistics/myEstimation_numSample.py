import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt

## sp - 1.96 * sqrt( sp(1-sp)/n ) <= p <= sp + 1.96 * sqrt( sp(1-sp)/n )
## より、信頼区間の幅は
## 2 * 1.96 * sqrt( sp(1-sp)/n )
## 設定したい区間幅をp_widthとすると、
## n >= ( 2 * 1.96 * sqrt( sp * ( 1-sp )) / p_width )^2

sp = 0.1
p_width = 0.05

# 95%区間の上限、下限
norm_low, norm_upp = -1.96, 1.96

# 必要サンプル数
n = 2 * np.abs(norm_upp) * np.sqrt( sp* (1-sp)) / p_width
n = n*n

print("n: ", n)