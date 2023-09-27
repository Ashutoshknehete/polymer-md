import signac
import numpy as np

project = signac.init_project()

num_homo = [1024]
length_A = 64
length_B = 64
num_CP = np.arange(32,33,32, dtype=int)
base_CP = [16] #16 is A16B16, 32 is A32B32, 64 is A64B64 for branched. In linear, it is = #monomers in each block
num_arms = [2]
#numblocks = [2,3] #for linear
list_architecture = ["mikto"] # can take inputs "mikto", "graft", or "linear"

rho = 0.85
eps = 50
kT = 1.0
aspect = 0.5443310539518174
num_replicas = 1

for i in range(num_replicas):
    for N_homo in num_homo:
        for shape in list_architecture:
            for M in num_CP:
                for N in base_CP:
                    if shape == 'mikto' or shape == 'graft':
                        for n in num_arms:
                            N_A = N_homo
                            M_A = length_A
                            N_B = N_homo
                            M_B = length_A
                            N_CP = [N,N]
                            M_CP = M
                            n_arms = n
                            architecture = shape
                            sp = {"density": rho, "kT": kT, "epsilon_AB": eps, 
                            "num_A": N_homo, "length_A": length_A, "num_B": N_homo, 
                            "length_B": length_B, "num_CP": M_CP, "length_CP": N_CP, "num_arms": n_arms,
                            "aspect": aspect, "architecture": architecture, "replica": i}
                            job = project.open_job(sp)
                            job.init()
                            print(sp)
                    elif shape == 'linear':
                        for nblock in numblocks:
                            N_CP = [2*N for i in range(nblock)]
                            N_CP[0] = N
                            N_CP[-1] = N
                            M_CP = int(M * 2/(nblock-1)) # M is number of triblocks, M_CP is number of this number of blocks 
                            architecture = shape
                            n_arms = 0
                            sp = {"density": rho, "kT": kT, "epsilon_AB": eps, 
                                "num_A": N_homo, "length_A": length_A, "num_B": N_homo, 
                                "length_B": length_B, "num_CP": M_CP, "length_CP": N_CP, "num_arms": n_arms,
                                "aspect": aspect, "architecture": architecture, "replica": i}
                            job = project.open_job(sp)
                            job.init()
                            print(sp)

'''
# Add bare jobs
for i in range(num_replicas):
   for N_homo in num_homo:
       M_CP = 0
       N_CP = [0,0,0]
       sp = {"density": rho, "kT": kT, "epsilon_AB": eps, 
             "num_A": N_homo, "length_A": length_A, "num_B": N_homo, 
             "length_B": length_B, "num_CP": M_CP, "length_CP": N_CP, "num_arms": 0,
             "aspect": aspect, "architecture": 'linear', "replica": i} # architecture set to linear for consistency, it doesnt matter since NCP = 0
       job = project.open_job(sp)
       job.init()
'''