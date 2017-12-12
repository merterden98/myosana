# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 13:03:36 2017

@author: Mert
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn; seaborn.set()

init = pd.read_csv('C:\\Users\\Mert\\Desktop\\subj1init.txt')
second = pd.read_csv('C:\\Users\\Mert\\Desktop\\subj1x2.txt')

avg = -3.6*(10**-5)
stdev = -1.3*(10**-5)


validate = np.array(second)


validstdev = np.std(validate)
validaverage = np.average(validate)


print(validaverage)


for i in range(0,validate.size):
    if(validate[i] > (validaverage + 2*validstdev) or validate[i] < (validaverage - 2*validstdev)):
        validate[i] = -1

validate = validate[validate > -1]
print(validate.size)

comp = np.array(init[:validate.size])
print(comp.size)
compstdev = np.std(comp)
compaverage = np.average(comp)

filter1 = np.zeros(validate.size)
filter2 = np.zeros(validate.size)

for i in range(1,validate.size):
    if(validate[i-1] < validate[i] and validate[i] < validate[i+1]):
            filter1[i] = 1 * np.log10(validate[i])

for i in range(1,validate.size):
    if(comp[i-1] < comp[i] and comp[i] < comp[i+1]):
            filter2[i] = 1 * np.log10(comp[i])
x = np.arange(0,validate.size,1)      

print(x.size)

#ax3 = plt.subplot(311)
#plt.plot(x,validate)
#plt.plot(x,comp)


#ax1 = plt.subplot(312)
#plt.plot(x,filter1)

#ax2 = plt.subplot(312)
#plt.plot(x,filter2)

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')


comp_av = movingaverage(filter2, 500)
validate_av = movingaverage(filter1, 500)
         
#ax3 = plt.subplot(312)
#plt.plot(x,validate_av)
#plt.plot(x,comp_av)

properExCheck = validate_av

for i in range(0,validate.size):
    properExCheck[i] = (comp_av[i] - validate_av[i])

#print(properExCheck)
m, b = np.polyfit(x, comp_av, 1)
m, c = np.polyfit(x, validate_av, 1)

print(b,c)

ax4 = plt.subplot(313)
plt.plot(x, properExCheck)
plt.plot(x, avg*x + (b+c)/2, '--')


ax4.set(xlabel = 'time (units)', ylabel='500 unit MA')

plotter = np.full(comp_av.size, 0.05)

undertrained = True

for i in range (0, validate.size,10):
   if((properExCheck[i] < (((properExCheck[i]*avg + (b+c)/2)) - 0.02) or 
      (properExCheck[i] > ((properExCheck[i]*avg + (b+c)/2)) + 0.02))):
       if(properExCheck[i] > (properExCheck[i]*avg + (b+c)/2)):
           plt.scatter(x[i], plotter[i], marker='+')
       else:
           if(undertrained):
               plt.scatter(x[i], plotter[i], marker='x')
   else:
       plt.scatter(x[i], plotter[i], marker='o')
       undertrained = False


        







