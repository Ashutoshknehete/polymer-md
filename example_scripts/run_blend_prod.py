import gsd.hoomd
import hoomd
from polymerMD.simtools import sim_routines
from polymerMD.analysis.utility import read_snapshot_from_gsd
import sys
from pathlib import Path

# read blend initial state
if __name__=="__main__":
    f_init = sys.argv[1]
    parentdir = Path(f_init).parent
    idstr = ".".join(str(Path(f_init).stem).split(".")[0:-1])
snap_init = read_snapshot_from_gsd(f_init)

# system parameters
kT = 1.0
epsAB = 5.0

# simulation parameters
cpu = hoomd.device.CPU()
gpu = hoomd.device.GPU()

# production run of A/B homopolymer blend
fname = "struct/{:s}.3_prod.gsd".format(idstr)
#ftraj = "traj/{:s}.3_prod.traj.gsd".format(idstr)
ftraj=None
flog = "dat/{:s}.log.gsd".format(idstr)
iterations = 5000000
period = 2500

print("\nStarting production simulation with FENE/LJ potential and Langevin thermostat on GPU, for iterations = {:d}".format(iterations))
state_prod = sim_routines.production(snap_init, gpu, epsAB, kT, iterations, period=period,
                                        fstruct=fname, ftraj=ftraj, flog=flog)
