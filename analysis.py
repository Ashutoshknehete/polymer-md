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


print("Computing interfacial tension")
# load log and structure data
dat = gsd.hoomd.open("prod.log.gsd",'rb')
snap = gsd.hoomd.open("struct/prod.gsd", 'rb')[0]
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