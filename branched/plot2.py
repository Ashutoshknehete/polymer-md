import numpy as np
import matplotlib.pyplot as plt
from format_plots import format_plots

baseline = 0.68802
colors = format_plots(plt)
mean = ['0.68350', '0.67259', '0.66176', '0.62247', '0.68336', '0.67805', '0.65572', '0.60672', '0.67710', '0.67098', '0.64535', '0.57019']
std = ['0.01385', '0.00933', '0.00786', '0.01349', '0.01111', '0.01076', '0.01210', '0.00982', '0.00924', '0.01008', '0.01464', '0.01202']
mean = [baseline-float(value) for value in mean]
std = [float(value)/np.sqrt(19) for value in std]
copolymer = [24,48,96,192]
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot()
ax.plot(copolymer,mean[0:4], color=colors["gray"], label='n_arms = 1')
plt.scatter(copolymer,mean[0:4],color=colors["gray"],marker="o", s=50)
plt.errorbar(copolymer,mean[0:4],yerr=std[0:4], color=colors["gray"])
ax.plot(copolymer,mean[4:8], color=colors["black"], label='n_arms = 2')
plt.scatter(copolymer,mean[4:8],color=colors["black"],marker="o", s=50)
plt.errorbar(copolymer,mean[4:8],yerr=std[4:8], color=colors["black"])
ax.plot(copolymer,mean[8:12], color=colors["almond"], label='n_arms = 4')
plt.scatter(copolymer,mean[8:12],color=colors["almond"],marker="o", s=50)
plt.errorbar(copolymer,mean[8:12],yerr=std[8:12], color=colors["almond"])
plt.legend()
plt.ylim(0,0.15)
plt.xlabel(r'#copolymers at interface')
plt.ylabel(r'surface pressure')
plt.title('A16B16 graft copolymer at interface')
plt.savefig('replica_2.png')
plt.close()