import numpy as np


# log_10(x+1)
def log10p(x):
    return np.log10(1 + x)


# 10^x - 1
def exp10p(x):
    return 10**x - 1