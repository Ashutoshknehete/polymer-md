import gsd.hoomd
import gsd.pygsd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# utility functions
def read_traj_from_gsd(fname):

    f = gsd.pygsd.GSDFile(open(fname, 'rb'))
    t = gsd.hoomd.HOOMDTrajectory(f)

    return t 

def read_snapshot_from_gsd(fname):
    return read_traj_from_gsd(fname)[-1] # return the last snapshot/frame!

def write_gsd_from_snapshot(snapshot, fname):
    with gsd.hoomd.open(name=fname, mode='wb') as f:
            f.append(snapshot)
    return

def binned_density_1D(coord, box, axis, nBins):
    # given a set of coordinates (and a box that those coordinates should all fall within, centered on origin),
    # compute the binned density along the specified axis!

    lmin = 0 - box[axis]/2
    lmax = 0 + box[axis]/2

    h = np.histogram(coord[:,axis], nBins, range=(lmin,lmax))

    binvol = box[0]*box[1]*box[2] / nBins
    h = (h[0] / binvol, h[1])

    return h

def binned_density_ND(coord, box, N, nBins):

    boxrange = [(0-box[d]/2, 0+box[d]/2) for d in range(N)] # centered on 0. Specific to how I set up my simulations... and maybe HOOMD convention?

    h = np.histogramdd(coord, nBins, boxrange, density=False)
    
    totalbins = np.product(h[0].shape)
    binvol = box[0]*box[1]*box[2] / totalbins
    h = (h[0] / binvol, h[1])

    return h

def count_to_volfrac(hists):

    # takes a dict of numpy histograms, assumed to each be a count of a different species with the same bins,
    # and converts them to volume fractions such that the sum of each histogram sums to 1 at each bin. 
    # Note: only using np histograms for convenience, these are not proper histograms once they've been rescaled
    # differently at each bin like this!! 

    types = list(hists.keys())
    totcount = np.zeros_like(hists[types[0]][0])
    for hist in hists.values():
        totcount += hist[0]
    for type,hist in hists.items():
        hists[type] = (hists[type][0]/totcount, hists[type][1])

    return hists

def integral_ND(dat, x, N):

    # dat should be an N-dimension numpy array that is a sample of the function to be integrated
    # x should be a tuple of the sample points of the data in dat in each dimension
    # N should be the number of dimensions

    if len(x) != N or dat.ndim != N:
        # data not correct dimension 
        raise ValueError("Data passed to integral_ND does not match inputted dimension N.")
    
    # check length of arrays in x. If 1 greater than number of data points in that direction, assume these
    # are bin edges, and drop the last one. 
    for d in range(N):
        if len(x[d]) == dat.shape[d]+1:
            x[d] = x[d][:-1]

    I = np.trapz(dat, x = x[0], axis=0)
    for d in range(1,N):
        I = np.trapz(I, x=x[d], axis=0) # always axis 0 because it keeps getting reduced! 

    return I # should be a scalar now!

def findInterfaceAxis(snapshot, species):

    # finds the most likely axis of the interface
    # computes 1D binned densities of each species in species along each axis
    # the axis with the highest range is likely the one perpendicular to the interface

    return