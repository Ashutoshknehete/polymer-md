import numpy as np
import matplotlib.pyplot as plt
from format_plots import format_plots

baseline = 0.676401262
colors = format_plots(plt)
mean = ['0.683', '0.681', '0.667', '0.631', '0.684', '0.673', '0.658', '0.620', '0.681', '0.676', '0.661', '0.623']
std = ['0.009', '0.011', '0.013', '0.014', '0.013', '0.011', '0.012', '0.013', '0.014', '0.009', '0.013', '0.012']
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
plt.xlabel(r'#copolymers at interface')
plt.ylabel(r'surface pressure')
plt.title('A16B16 mikto arm copolymer at interface')
plt.savefig('replica_1.png')
plt.close()