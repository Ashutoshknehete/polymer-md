import numpy as np
import matplotlib.pyplot as plt
from format_plots import format_plots

baseline = 0.68802
colors = format_plots(plt)
mean = []
std = []
mean = [baseline-float(value) for value in mean]
std = [float(value)/np.sqrt(19) for value in std]
copolymer = [24,48,96,192]
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot()
ax.plot(copolymer,mean[0:4], color=colors["gray"], label='A16B8')
plt.scatter(copolymer,mean[0:4],color=colors["gray"],marker="o", s=50)
plt.errorbar(copolymer,mean[0:4],yerr=std[0:4], color=colors["gray"])
ax.plot(copolymer,mean[4:8], color=colors["black"], label='A16B16')
plt.scatter(copolymer,mean[4:8],color=colors["black"],marker="o", s=50)
plt.errorbar(copolymer,mean[4:8],yerr=std[4:8], color=colors["black"])
ax.plot(copolymer,mean[8:12], color=colors["almond"], label='A16B32')
plt.scatter(copolymer,mean[8:12],color=colors["almond"],marker="o", s=50)
plt.errorbar(copolymer,mean[8:12],yerr=std[8:12], color=colors["almond"])
plt.ylim(0,0.15)
plt.legend()
plt.xlabel(r'#copolymers at interface')
plt.ylabel(r'surface pressure')
plt.title('graft 4-arm copolymer at interface')
plt.savefig('replica_4.png')
plt.close()