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
L_x = 30
L_y = 30
L_z = 30

# make system
system = systemspec.System()
system.box = [L_x, L_y, L_z]
A = system.addMonomer('A',l)
B = system.addMonomer('B',l)
C = system.addMonomer('C',l)
V = system.addMonomer('V',l)

'''
monomers = [A,B,C]
lengths = [2,3,4]
poly = systemspec.BranchedPolymerSpec.star(monomers,lengths,V)
system.addComponent(poly, 1)
'''

'''
monomers = [A,A,A,B,B]
lengths = [3,3,3,2,2]
poly = systemspec.BranchedPolymerSpec.customgraft(monomers,lengths,V)
system.addComponent(poly, 1)
'''

#'''
chain_monomer = A
chain_length = 5
arm_monomer = B
total_arm_length = chain_length+1
n_arms = 3
vertex = A
poly = systemspec.BranchedPolymerSpec.mikto_arm(chain_monomer, chain_length, arm_monomer, total_arm_length, n_arms, vertex)
system.addComponent(poly, 1)
#'''

'''
backbone = A
backbone_length = 11
sidechain = B
n_sidechain = 2
vertex = backbone
total_sidechain_length = backbone_length
poly = systemspec.BranchedPolymerSpec.regulargraft(backbone, backbone_length, sidechain, total_sidechain_length, n_sidechain, vertex)
system.addComponent(poly, 1)
'''

'''
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
'''

snap = systemgen.build_snapshot(system,'random')
if os.path.exists("test.gsd"):
        os.remove("test.gsd")

with gsd.hoomd.open(name="test.gsd", mode='xb') as f:
    f.append(snap)
