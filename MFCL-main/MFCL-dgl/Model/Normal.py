"""
Time: 2024.4.13
Author: Yiran Shi
"""

import numpy as np

def average(data):
    return np.sum(data)/len(data)

# standard deviation
def sigma(data,avg):
    sigma_squ=np.sum(np.power((data-avg),2))/len(data)
    return np.power(sigma_squ,0.5)

# Gaussian distribution
def prob(data):
    # print(data)
    ave = average(data)
    sig = sigma(data, ave)
    sqrt_2pi=np.power(2*np.pi,0.5)
    coef=1/(sqrt_2pi*sig)
    powercoef=-1/(2*np.power(sig,2))
    mypow=powercoef*(np.power((data-ave),2))
    return coef*(np.exp(mypow))

