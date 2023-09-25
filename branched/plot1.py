import numpy as np
import matplotlib.pyplot as plt
from format_plots import format_plots
from polymerMD.structure import systemspec, systemgen

baseline = 0.68802
colors = format_plots(plt)
mean = ['0.68281', '0.68118', '0.66801', '0.63176', '0.68473', '0.67312', '0.65756', '0.62059', '0.68112', '0.67708', '0.66106', '0.62335']
std = ['0.00898', '0.01065', '0.01321', '0.01350', '0.01274', '0.01075', '0.01141', '0.01325', '0.01380', '0.01081', '0.01245', '0.01137']
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
plt.ylim(0,0.15)
plt.legend()
plt.xlabel(r'#copolymers at interface')
plt.ylabel(r'surface pressure')
plt.title('A16B16 mikto arm copolymer at interface')
plt.savefig('replica_1.png')
plt.close()