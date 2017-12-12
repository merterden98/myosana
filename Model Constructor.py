# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 09:00:42 2017

@author: Mert
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn; seaborn.set()


data = pd.read_csv('C:\\Users\\Mert\\Desktop\\test_subj1.txt')

emg = np.array(data)
filterr = np.zeros([emg.size])

for i in range(0, emg.size -2, 1):
    if(emg[i-1] < emg[i] and emg[i+1] < emg[i]
        and emg[i] > 150):
        if((emg[i] - emg[i-1]) > 10):
            filterr[i] = 1 * np.log10(emg[i])

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

y_av = movingaverage(filterr, 500)

x = np.arange(0,emg.size,1)
print(x.size)

ax1 = plt.subplot(311)
plt.plot(x,emg)
ax1.set(xlabel = 'time (units)', ylabel='Arduino Analog Output')
plt.title("Subject One")
ax2 = plt.subplot(312, sharex=ax1)
ax2.set(xlabel = 'time (units)', ylabel='Logarithmic Average of Peaks')
plt.plot(x,filterr)
ax3 = plt.subplot(313, sharex=ax1)
plt.plot(x[250:1500],y_av[250:1500])
plt.plot(x[500:1500],y_av[500:1500], 'black') 
plt.plot(x[750:1500],y_av[750:1500], 'green') 
plt.plot(x[1000:1500],y_av[1000:1500], 'red')  
m, b = np.polyfit(x[250:1500], filterr[250:1500], 1)
print(m,b)
plt.plot(x, m*x + b, '--')
ax3.set(xlabel = 'time (units)', ylabel='500 unit MA')

#ax.set(xlim=(0, 10), ylim=(-2, 2),
#       xlabel='x', ylabel='sin(x)',
#       title='A Simple Plot');

plt.show()  
