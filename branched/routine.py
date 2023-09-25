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

from routine_functions import compute_box_dimensions
from routine_functions import build_system_spec_graft
from routine_functions import build_phaseseparated_blend
from routine_functions import _relax
from routine_functions import _equilibrate
from routine_functions import _production_IK
from routine_functions import _production
 
import json

with open('parameters.json', 'r') as file:
    data = json.load(file)
    
n_simulations = len(data["N_A"])

for i in range(n_simulations):
        
        # System parameters
        N_A = data["N_A"][i]
        M_A = data["M_A"][i]
        N_B = data["N_B"][i]
        M_B = data["M_B"][i]
        N_CP = data["N_CP"][i]
        M_CP = data["M_CP"][i]
        n_arms = data["n_arms"][i]
        rho = data["rho"][i]
        aspect = data["aspect"][i]
        architecture = data["architecture"][i]

        snap_random = build_phaseseparated_blend(rho, M_A, N_A, M_B, N_B, M_CP, N_CP, n_arms, aspect, architecture)

        fname_random = architecture+"_random_NA={:04d}_MA={:04d}_NB={:04d}_MB={:04d}_NCP={:04d}{:04d}_MCP={:04d}_narms={:04d}.gsd".format(N_A,M_A,N_B,M_B,N_CP[0],N_CP[1],M_CP,n_arms)
        print(fname_random)
        if os.path.exists(fname_random):
                os.remove(fname_random)
        with gsd.hoomd.open(name=fname_random, mode='xb') as f:
                f.append(snap_random)

        fname_relax = architecture+"_relax_NA={:04d}_MA={:04d}_NB={:04d}_MB={:04d}_NCP={:04d}{:04d}_MCP={:04d}_narms={:04d}.gsd".format(N_A,M_A,N_B,M_B,N_CP[0],N_CP[1],M_CP,n_arms)
        if os.path.exists(fname_relax):
                os.remove(fname_relax)
        state_relax = _relax(snap_random)
        hoomd.write.GSD.write(state=state_relax, filename=fname_relax, mode='xb')

        fname_equil = architecture+"_equil_NA={:04d}_MA={:04d}_NB={:04d}_MB={:04d}_NCP={:04d}{:04d}_MCP={:04d}_narms={:04d}.gsd".format(N_A,M_A,N_B,M_B,N_CP[0],N_CP[1],M_CP,n_arms)
        if os.path.exists(fname_equil):
                os.remove(fname_equil)
        snap_relax = gsd.hoomd.open(fname_relax, mode='rb')[0]
        state_equil = _equilibrate(snap_relax, kT=1, epsilonAB=10, ftraj=None)
        hoomd.write.GSD.write(state=state_equil, filename=fname_equil, mode='xb')

        fname_prod = architecture+"_prod_NA={:04d}_MA={:04d}_NB={:04d}_MB={:04d}_NCP={:04d}{:04d}_MCP={:04d}_narms={:04d}.gsd".format(N_A,M_A,N_B,M_B,N_CP[0],N_CP[1],M_CP,n_arms)
        if os.path.exists(fname_prod):
                os.remove(fname_prod)
        snap_equil = gsd.hoomd.open(fname_equil, mode='rb')[0]
        fname_prod_log = architecture+"_prod_NA={:04d}_MA={:04d}_NB={:04d}_MB={:04d}_NCP={:04d}{:04d}_MCP={:04d}_narms={:04d}.log.gsd".format(N_A,M_A,N_B,M_B,N_CP[0],N_CP[1],M_CP,n_arms)
        state_prod = _production(snap_equil, kT=1, epsilonAB=10, ftraj=None, flog=fname_prod_log)
        hoomd.write.GSD.write(state=state_prod, filename=fname_prod, mode='xb')
