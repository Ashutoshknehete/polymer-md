import os
import pickle
import numpy as np
from scipy.signal import find_peaks

from flow import FlowProject
import hoomd
import gsd.hoomd
from signac import JSONDict

from polymerMD.structure import systemspec, systemgen
from polymerMD.simtools import sim_routines
from polymerMD.analysis import trajtools, statistics
import matplotlib.pyplot as plt

from routine_functions import build_system_spec

N_A = 64
M_A = 1024
N_B = 64
M_B = 1024
N_CP = [16,16]
M_CP = 96
n_arms = 4
rho = 0.85
aspect = 0.544
system = build_system_spec(M_A, N_A, M_B, N_B, M_CP, N_CP, n_arms)

print("Computing interfacial tension")
# load log and structure data
dat = gsd.hoomd.open("prod.log.gsd",'rb')
snap = gsd.hoomd.open("prod.gsd", 'rb')[0]
# compute interfacial tension for each frame, determine average and variance
axis=0 # fix the axis! We really don't need to generalize it. Change later if needed
L = snap.configuration.box[axis]
t,gammas = trajtools.interfacial_tension_global(dat,axis,L)
gammas = np.array(gammas)
t = np.squeeze(t)
# compute average interfacial tension and store
avg_gamma = np.average(gammas)
print('bulk_interfacial_tension_average =', avg_gamma)
# compute variance using estimated autocorrelation time
var_gamma = statistics.estimator_variance(gammas)
print('bulk_interfacial_tension_variance =', var_gamma)
# compute the number of independent samples and the average for each sample
samples = statistics.get_independent_samples(gammas,factor=2)
nsamples = np.shape(samples)[0]
print('bulk_interfacial_tension_samples =', nsamples)

print("Computing density profiles for job")
# density profiles of monomers
# in the future we should average this over many frames! but I haven't been recording trajectories...
snap = gsd.hoomd.open("prod.gsd", 'rb')[0] 
profiles = trajtools.density_1D_monomers(snap)

with open("density_1D_monomers.pkl", 'wb') as f:
    pickle.dump(profiles, f)

with open('density_1D_monomers.pkl', 'rb') as f:
    data = pickle.load(f)
keys_list = list(data)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot()
ax.plot(data[keys_list[0]][1][0:100], data[keys_list[0]][0])
ax.plot(data[keys_list[1]][1][0:100], data[keys_list[1]][0])
plt.savefig('density.png')

profiles = trajtools.density_1D_species(snap,system,nBins=100)
with open("density_1D_species.pkl", 'wb') as f:
    pickle.dump(profiles, f)

with open('density_1D_species.pkl', 'rb') as f:
    data = pickle.load(f)
keys_list = list(data)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot()
ax.plot(data[keys_list[0]][1][0:100], data[keys_list[0]][0])
ax.plot(data[keys_list[1]][1][0:100], data[keys_list[1]][0])
ax.plot(data[keys_list[2]][1][0:100], data[keys_list[2]][0])
plt.savefig('density_species.png')
            
print("Computing internal distances for job")
# internal distance curve for all molecules
snap = gsd.hoomd.open("equil.gsd", 'rb')[0] 
n,avgRsq = trajtools.internaldistances_all(snap)
with open("internaldistances_all.pkl", 'wb') as f:
    pickle.dump((n,avgRsq),f)

with open('internaldistances_all.pkl', 'rb') as f:
    data = pickle.load(f)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot()
ax.plot(data[0], np.array(data[1])/np.array(data[0]))
plt.xscale('log')
plt.savefig('internaldist.png')

speciesRsq = trajtools.internaldistances_species(snap,system)
with open("internaldistances_species.pkl", 'wb') as f:
    pickle.dump(speciesRsq, f)
with open('internaldistances_species.pkl', 'rb') as f:
    data = pickle.load(f)
keys_list = list(data)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot()
xaxis = data[keys_list[0]][0]
yaxis = np.array(data[keys_list[0]][1])/np.array(data[keys_list[0]][0])
ax.plot(xaxis, yaxis)
xaxis = data[keys_list[1]][0]
yaxis = np.array(data[keys_list[1]][1])/np.array(data[keys_list[1]][0])
ax.plot(xaxis, yaxis)
plt.xscale('log')
#ax.plot(data[keys_list[1]][1][0:100], data[keys_list[1]][0])
#ax.plot(data[keys_list[2]][1][0:100], data[keys_list[2]][0])
plt.savefig('internaldist_species.png')