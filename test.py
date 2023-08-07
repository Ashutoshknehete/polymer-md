import polymerMD.structure.systemspec as systemspec
import polymerMD.structure.systemgen as systemgen
import numpy as np
import gsd.hoomd
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

def write_gsd_from_snapshot(snapshot, fname):
    with gsd.hoomd.open(name=fname, mode='wb') as f:
            f.append(snapshot)
    return

# System parameters
l = 1
L_x = 20
L_y = 20
L_z = 20

# make system
system = systemspec.System()
system.box = [L_x, L_y, L_z]
A = system.addMonomer('A',l)
B = system.addMonomer('B',l)
C = system.addMonomer('C',l)
V = system.addMonomer('V',l)

'''
monomers = [A,B,C]
lengths = [4,4,4]
poly = systemspec.BranchedPolymerSpec.star(monomers,lengths,V)
system.addComponent(poly, 1)
'''

monomers = [A,A,A,B,B]
lengths = [3,3,3,2,2]
poly = systemspec.BranchedPolymerSpec.customgraft(monomers,lengths,V)
system.addComponent(poly, 1)


print(poly.lengths)
print(poly.bondtypes)
print(poly.label)
print(poly.total_vertices) 
print(poly.particletypes)
print(poly.connectivity_at_start_list)
print(poly.connectivity_at_end_list)
print(poly.connectivity_list)
print(poly.connectivity_count)
print(poly.free_chain_ends)
print(poly.junction_nodes)
print(poly.bonds)


snap = systemgen.build_snapshot(system,'random')
if os.path.exists("struct/branched.gsd"):
        os.remove("struct/branched.gsd")

with gsd.hoomd.open(name="struct/branched.gsd", mode='xb') as f:
    f.append(snap)
