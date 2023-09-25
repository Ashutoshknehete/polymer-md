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

### underlying functions for operations to be performed that know nothing about FlowProject API
def compute_box_dimensions(rho, M_A, N_A, M_B, N_B, M_CP, N_CP, aspect):
    n_beads = N_A*M_A + N_B*M_B + M_CP*sum(N_CP)
    volume = n_beads/rho
    L_y = (aspect*volume)**(1/3)
    L_x = L_y/aspect
    L_z = L_y
    return L_x, L_y, L_z

def build_system_spec_graft(M_A, N_A, M_B, N_B, M_CP, N_CP, n_arms):

    # parameter... monomer size, meaning currently unclear
    l = 1
    # make system    
    system = systemspec.System()
    A = system.addMonomer('A',l)
    B = system.addMonomer('B',l)
    poly_A = systemspec.LinearPolymerSpec.linear([A], [N_A])
    poly_B = systemspec.LinearPolymerSpec.linear([B], [N_B])
    system.addComponent(poly_A, M_A)
    system.addComponent(poly_B, M_B)
    
    # copolymer stuff
    backbone = A
    backbone_length = N_CP[0]
    sidechain = B
    n_sidechain = n_arms
    vertex = backbone
    total_sidechain_length = N_CP[1]
    poly = systemspec.BranchedPolymerSpec.regulargraft(backbone, backbone_length, sidechain, total_sidechain_length, n_sidechain, vertex)
    if M_CP != 0:
        system.addComponent(poly, int(M_CP/2))
        system.addComponent(poly, int(M_CP/2))
    
    return system

def build_system_spec_mikto(M_A, N_A, M_B, N_B, M_CP, N_CP, n_arms):

    # parameter... monomer size, meaning currently unclear
    l = 1
    # make system    
    system = systemspec.System()
    A = system.addMonomer('A',l)
    B = system.addMonomer('B',l)
    poly_A = systemspec.LinearPolymerSpec.linear([A], [N_A])
    poly_B = systemspec.LinearPolymerSpec.linear([B], [N_B])
    system.addComponent(poly_A, M_A)
    system.addComponent(poly_B, M_B)
    
    # copolymer stuff 
    chain_monomer = A
    chain_length = N_CP[0] - 1
    arm_monomer = B
    total_arm_length = N_CP[1]
    vertex = chain_monomer
    poly = systemspec.BranchedPolymerSpec.mikto_arm(chain_monomer, chain_length, arm_monomer, total_arm_length, n_arms, vertex)
    if M_CP != 0:
        system.addComponent(poly, int(M_CP/2))
        system.addComponent(poly, int(M_CP/2))
    
    return system

def build_system_spec_linear(M_A, N_A, M_B, N_B, M_CP, N_CP, n_arms=0):

    # parameter... monomer size, meaning currently unclear
    l = 1

    system = systemspec.System()
    A = system.addMonomer('A',l)
    B = system.addMonomer('B',l)
    poly_A = systemspec.LinearPolymerSpec.linear([A], [N_A])
    poly_B = systemspec.LinearPolymerSpec.linear([B], [N_B])
    system.addComponent(poly_A, M_A)
    system.addComponent(poly_B, M_B)
    # copolymer stuff 
    if M_CP != 0 and N_CP[0] != 0:
        nblocks = len(N_CP)
        if 0 in N_CP:
            ValueError("Can't have a zero-length block!")
        cpBlocks = []
        for i in range(nblocks):
            if i%2: # NEEDS TO BE CHANGED TO NOT i%2 IN THE FUTURE. I CREATED BAB POLYMERS UGH
                cpBlocks.append(A)
            else:
                cpBlocks.append(B)
        poly_CP = systemspec.LinearPolymerSpec.linear(cpBlocks, N_CP)
        system.addComponent(poly_CP, int(M_CP/2))
        system.addComponent(poly_CP, int(M_CP/2)) # two groups for two different regions...
    return system

def build_phaseseparated_blend(rho, M_A, N_A, M_B, N_B, M_CP, N_CP, n_arms, aspect, architecture):

    # system size and dimensions
    L_x, L_y, L_z = compute_box_dimensions(rho, M_A, N_A, M_B, N_B, M_CP, N_CP, aspect)

    # phase separated regions
    relative_domain_x = (N_A*M_A)**(1/3)/(N_B*M_B)**(1/3)
    x_A = relative_domain_x/(1+relative_domain_x)*L_x
    x_B = 1/(1+relative_domain_x)*L_x

    reg_A = [x_A,L_y,L_z]
    regcenter_A = np.array([-x_A/2 - x_B/2, 0, 0])
    reg_B = [x_B,L_y,L_z]
    regcenter_B = np.array([0, 0, 0])
    delta = 0.001
    reg_CP = [delta,L_y,L_z]
    regcenter_ABA_1 = np.array([-x_B/2, 0, 0]) # first interface
    regcenter_ABA_2 = np.array([+x_B/2, 0, 0]) # second interface

    regions = [reg_A, reg_B]
    regioncenters = [regcenter_A, regcenter_B]
    if M_CP != 0 and N_CP[0] != 0:
        regions += [reg_CP, reg_CP]
        regioncenters += [regcenter_ABA_1, regcenter_ABA_2]

    # define system and build snapshot of initial guess
    if architecture == "graft":
        system = build_system_spec_graft(M_A, N_A, M_B, N_B, M_CP, N_CP, n_arms)
    if architecture == "mikto":
        system = build_system_spec_mikto(M_A, N_A, M_B, N_B, M_CP, N_CP, n_arms)
    if architecture == "linear":
        system = build_system_spec_linear(M_A, N_A, M_B, N_B, M_CP, N_CP)
    system.box = [L_x, L_y, L_z] # set box
    snap = systemgen.build_snapshot(system,'boxregions',regions,regioncenters)

    return snap

def _relax(snap_initial):
    # simulation devices
    cpu = hoomd.device.CPU()
    #gpu = hoomd.device.GPU()
    # system parameters, set arbitrarily for relaxation
    kT = 1.0
    epsilonAB = 10
    #state_overlap = sim_routines.remove_overlaps(snap_initial, cpu, kT, prefactor_range=[1,120], iterations=10)
    state_relax = sim_routines.relax_overlaps_AB(snap_initial, cpu, epsilonAB, iterations=100000)

    return state_relax

def _equilibrate(snap_initial, kT, epsilonAB, ftraj, iterations=40000000):
    gpu = hoomd.device.GPU()
    #cpu = hoomd.device.CPU()
    state_equil = sim_routines.equilibrate_AB(snap_initial, gpu, epsilonAB, kT, iterations=iterations,ftraj=ftraj)
    return state_equil

def _production_IK(snap_initial, kT, epsilonAB, flog, nbins, fthermo, fedge, iterations=10000000, period=10000):
    gpu = hoomd.device.GPU()
    state_prod = sim_routines.production_IK(snap_initial, gpu, epsilonAB, kT, iterations, period, 
                                                flog=flog, fthermo=fthermo, fedge=fedge, nbins=nbins)
    return state_prod

def _production(snap_initial, kT, epsilonAB, ftraj, flog, iterations=10000000, period=10000):
    gpu = hoomd.device.GPU()
    #cpu = hoomd.device.CPU()
    state_prod = sim_routines.production(snap_initial, gpu, epsilonAB, kT, iterations, period, ftraj=ftraj, flog=flog)
    return state_prod