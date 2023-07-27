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

def build_system_spec(M_A, N_A, M_B, N_B, M_CP, N_CP):

    # parameter... monomer size, meaning currently unclear
    l = 1

    system = systemspec.System()
    A = system.addMonomer('A',l)
    B = system.addMonomer('B',l)
    poly_A = systemspec.LinearPolymerSpec([A], [N_A])
    poly_B = systemspec.LinearPolymerSpec([B], [N_B])
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
        poly_CP = systemspec.LinearPolymerSpec(cpBlocks, N_CP)
        system.addComponent(poly_CP, int(M_CP/2))
        system.addComponent(poly_CP, int(M_CP/2)) # two groups for two different regions...

    return system

def build_phaseseparated_blend(rho, M_A, N_A, M_B, N_B, M_CP, N_CP, aspect):

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
    system = build_system_spec(M_A, N_A, M_B, N_B, M_CP, N_CP)
    system.box = [L_x, L_y, L_z] # set box
    snap = systemgen.build_snapshot(system,'boxregions',regions,regioncenters)

    return snap

def _relax(snap_initial):
    # simulation devices
    cpu = hoomd.device.CPU()
    gpu = hoomd.device.GPU()
    # system parameters, set arbitrarily for relaxation
    kT = 1.0
    epsilonAB = 5.0
    state_overlap = sim_routines.remove_overlaps(snap_initial, gpu, kT, prefactor_range=[1,120], iterations=100000)
    state_relax = sim_routines.relax_overlaps_AB(state_overlap.get_snapshot(), cpu, epsilonAB, iterations=10000)
    return state_relax

def _equilibrate(snap_initial, kT, epsilonAB, iterations=40000000):
    gpu = hoomd.device.GPU()
    state_equil = sim_routines.equilibrate_AB(snap_initial, gpu, epsilonAB, kT, iterations=iterations)
    return state_equil

def _production_IK(snap_initial, kT, epsilonAB, flog, nbins, fthermo, fedge, iterations=10000000, period=10000):
    gpu = hoomd.device.GPU()
    state_prod = sim_routines.production_IK(snap_initial, gpu, epsilonAB, kT, iterations, period, 
                                                flog=flog, fthermo=fthermo, fedge=fedge, nbins=nbins)
    return state_prod

def _production(snap_initial, kT, epsilonAB, flog, iterations=10000000, period=10000):
    gpu = hoomd.device.GPU()
    state_prod = sim_routines.production(snap_initial, gpu, epsilonAB, kT, iterations, period, flog=flog)
    return state_prod

### helper FlowProject functions to simplify syntax
def job_build_system_spec(job):
    return build_system_spec(M_A=job.sp.num_A, N_A=job.sp.length_A, M_B=job.sp.num_B, 
                             N_B=job.sp.length_B, M_CP=job.sp.num_CP, N_CP=job.sp.length_CP)

def job_compute_box_dimensions(job):
    return compute_box_dimensions(rho=job.sp.density, M_A=job.sp.num_A, N_A=job.sp.length_A, 
                                  M_B=job.sp.num_B, N_B=job.sp.length_B, M_CP=job.sp.num_CP, 
                                  N_CP=job.sp.length_CP, aspect=job.sp.aspect)

def compute_descriptors_general(job, doc: JSONDict):
    L_x,L_y,L_z =  job_compute_box_dimensions(job)
    
    total_interface_area = (2 * L_y*L_z)
    nbeads = job.sp.num_A*job.sp.length_A + job.sp.num_B*job.sp.length_A + job.sp.num_CP*sum(job.sp.length_CP)

    cp_per_area = job.sp.num_CP / total_interface_area
    cp_beads_per_area = sum(job.sp.length_CP) * cp_per_area
    cp_wt_frac = job.sp.num_CP * sum(job.sp.length_CP) / nbeads # assumes they all have same mass! This is the assumption for now.
    
    # compute number of junctions
    junctions_per_cp = len(job.sp.length_CP)-1
    junctions_per_area = junctions_per_cp*cp_per_area
    
    doc.junctions_per_area = junctions_per_area
    doc.cp_per_area = cp_per_area
    doc.cp_beads_per_area = cp_beads_per_area
    doc.cp_wt_frac = cp_wt_frac

    doc.Lx = L_x
    doc.Ly = L_y
    doc.Lz = L_z
    doc.total_area = total_interface_area
    return

### FlowProject class
class BlendMD(FlowProject):
    pass

### labels and pre/post conditions
@BlendMD.label
def generated(job):
    return job.isfile("struct/random.gsd")

@BlendMD.label
def relaxed(job):
    return job.isfile("struct/relax.gsd")

@BlendMD.label
def equilibrated(job):
    return job.isfile("struct/equil.gsd")

@BlendMD.label
def simulated(job):
    return job.isfile("struct/prod.gsd")

#@BlendMD.label
def gamma_computed(job):
    return "interfacial_tension_average" in job.doc

@BlendMD.label
def checked_micelles(job):
    return "junction_peaks" in job.doc

@BlendMD.label
def micellized(job):
    if "junction_peaks" in job.doc:
        return job.doc.junction_peaks > 2
    else:
        return False # can't label as micellized if it hasn't been checked!

@BlendMD.label
def bulk_gamma_computed(job):
    cond1 = "bulk_interfacial_tension_average" in job.doc
    cond2 = job.isfile("bulk_gamma_independent_samples.txt")
    return cond1 and cond2

@BlendMD.label 
def descriptors_computed(job):
    cond1 = "cp_per_area" in job.doc 
    cond2 = "cp_beads_per_area" in job.doc
    cond3 = "cp_wt_frac" in job.doc
    cond4 = "junctions_per_area" in job.doc
    cond5 = ("Lx" in job.doc) and ("Ly" in job.doc) and ("Lz" in job.doc)
    cond6 = "total_area" in job.doc
    return cond1 and cond2 and cond3 and cond4 and cond5 and cond6

@BlendMD.label
def density_profile_computed(job):
    return job.isfile("density_1D_monomers.pkl") and job.isfile("density_1D_species.pkl")

@BlendMD.label
def internal_distances_computed(job):
    return job.isfile("internaldistances_all.pkl") and job.isfile("internaldistances_species.pkl")

@BlendMD.label
def junction_rdf_computed(job):
    return job.isfile("junction_rdf.pkl")


### operation groups
simulation_group = BlendMD.make_group(name="simulation")
analysis_group = BlendMD.make_group(name="analysis")

### job operations 
@simulation_group
@BlendMD.post(generated)
@BlendMD.operation
def generate_initial(job):
    with job:
        if not os.path.exists("struct"):
            os.makedirs("struct")
        print("Generating initial structure for job {:s}...".format(job.id))
        snapshot = build_phaseseparated_blend(rho=job.sp.density, M_A=job.sp.num_A, N_A=job.sp.length_A,
                                        M_B=job.sp.num_B, N_B=job.sp.length_B, 
                                        M_CP=job.sp.num_CP, N_CP=job.sp.length_CP, aspect=job.sp.aspect)
        with gsd.hoomd.open(name="struct/random.gsd", mode='xb') as f:
            f.append(snapshot)
    return

@simulation_group
@BlendMD.pre.after(generate_initial)
@BlendMD.post(relaxed)
@BlendMD.operation
def relax(job):
    with job:
        print("Relaxing random initial guess for job {:s}...".format(job.id))
        snap_random = gsd.hoomd.open("struct/random.gsd", mode='rb')[0]
        state_relax = _relax(snap_random)
        hoomd.write.GSD.write(state=state_relax, filename="struct/relax.gsd", mode='xb')
    return

@simulation_group
@BlendMD.pre.after(relax)
@BlendMD.post(equilibrated)
@BlendMD.operation
def equilibrate(job):
    with job:
        print("Equilibrating relaxed structure for job {:s}...".format(job.id))
        snap_relax = gsd.hoomd.open("struct/relax.gsd", mode='rb')[0]
        state_equil = _equilibrate(snap_relax, kT=job.sp.kT, epsilonAB=job.sp.epsilon_AB,iterations=40000000)
        hoomd.write.GSD.write(state=state_equil, filename="struct/equil.gsd", mode='xb')
    return

#@BlendMD.pre.after(equilibrate)
#@BlendMD.post(simulated)
#@BlendMD.operation
def productionIK(job):
    with job:
        print("Running production simulation with IK spatial pressure for job {:s}...".format(job.id))
        snap_equil = gsd.hoomd.open("struct/equil.gsd", mode='rb')[0]
        state_prod = _production_IK(snap_equil, kT=job.sp.kT, epsilonAB=job.sp.epsilon_AB, 
                                        flog="prod.log.gsd", fthermo="pressure.gsd", fedge="edges.txt", nbins=200)
        hoomd.write.GSD.write(state=state_prod, filename="struct/prod.gsd", mode='xb')
    return

@simulation_group
@BlendMD.pre.after(equilibrate)
@BlendMD.post(simulated)
@BlendMD.operation
def production(job):
    with job:
        print("Running production simulation for job {:s}...".format(job.id))
        snap_equil = gsd.hoomd.open("struct/equil.gsd", mode='rb')[0]
        state_prod = _production(snap_equil, kT=job.sp.kT, epsilonAB=job.sp.epsilon_AB, flog="prod.log.gsd")
        hoomd.write.GSD.write(state=state_prod, filename="struct/prod.gsd", mode='xb')
    return

#@BlendMD.pre.after(production)
#@BlendMD.post(gamma_computed)
#@BlendMD.operation
def interfacial_tension(job):
    with job:
        print("Computing interfacial tension for job {:s}...".format(job.id))
        # load spatial pressure data
        dat = gsd.hoomd.open("pressure.gsd",'rb')
        # load edges, determine what axis binning was done on, and remove others
        edges = np.loadtxt("edges.txt")
        axis = int(np.where(edges[0,:]!=0)[0])
        edges = edges[:,axis]
        # compute interfacial tension for each frame, determine average and variance
        t,gammas = trajtools.interfacial_tension_IK(dat, edges, axis)
        t = np.squeeze(t)
        var_gamma = statistics.estimator_variance(np.array(gammas))
        avg_gamma = np.average(gammas)
        # store results in job doc
        job.doc.interfacial_tension_average = avg_gamma
        job.doc.interfacial_tension_variance = var_gamma
    return

@BlendMD.post(descriptors_computed)
@BlendMD.operation
def compute_descriptors(job):
    print("Computing descriptors for job {:s}...".format(job.id))
    compute_descriptors_general(job,job.doc)
    return

@analysis_group
@BlendMD.pre.after(production)
@BlendMD.post(checked_micelles)
@BlendMD.operation
def check_for_micelles(job):
    # systems with no copolymers have no micelles :)
    if job.sp.num_CP == 0:
        job.doc.junction_peaks = 0
        return

    # load production snapshot
    with job:
        snap = gsd.hoomd.open("struct/prod.gsd", mode='rb')[0]
    
    # get particle positions
    pos = snap.particles.position

    # get system topology details
    system = job_build_system_spec(job)
    junctionbonds = system.junctions()

    # get junction positions in x direction and make histogram 
    nbins=40
    juncpos = np.array([1/2*np.sum(pos[junc,:],axis=0) for junc in junctionbonds])
    juncpos_x = juncpos[:,0]
    Lx = snap.configuration.box[0]
    hist = np.histogram(juncpos_x, bins=nbins,range=(-Lx/2,Lx/2))

    # count peaks and store in job doc
    peaks,props = find_peaks(hist[0])
    job.doc.junction_peaks = len(peaks)

    return

@analysis_group
@BlendMD.pre.after(production)
@BlendMD.post(bulk_gamma_computed)
@BlendMD.operation
def bulk_interfacial_tension(job):
    with job:
        print("Computing interfacial tension for job {:s}...".format(job.id))
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
        job.doc.bulk_interfacial_tension_average = avg_gamma

        # compute variance using estimated autocorrelation time
        var_gamma = statistics.estimator_variance(gammas)
        job.doc.bulk_interfacial_tension_variance = var_gamma
        
        # compute the number of independent samples and the average for each sample
        samples = statistics.get_independent_samples(gammas,factor=2)
        nsamples = np.shape(samples)[0]
        job.doc.bulk_interfacial_tension_samples = nsamples
        np.savetxt("bulk_gamma_independent_samples.txt", samples)
        
    return

@analysis_group
@BlendMD.pre.after(production)
@BlendMD.post(density_profile_computed)
@BlendMD.operation
def density_profiles(job):
    with job:
        print("Computing density profiles for job {:s}...".format(job.id))
        # density profiles of monomers
        # in the future we should average this over many frames! but I haven't been recording trajectories...
        snap = gsd.hoomd.open("struct/prod.gsd", 'rb')[0] 
        profiles = trajtools.density_1D_monomers(snap)
        with open("density_1D_monomers.pkl", 'wb') as f:
            pickle.dump(profiles, f)
        
        # density profile of species, defined as number of beads belonging to that species in a region
        system = job_build_system_spec(job)
        profiles = trajtools.density_1D_species(snap,system,nBins=100)
        with open("density_1D_species.pkl", 'wb') as f:
            pickle.dump(profiles, f)
    return

@analysis_group
@BlendMD.pre.after(equilibrate)
@BlendMD.post(internal_distances_computed)
@BlendMD.operation
def internal_distances(job):
    with job:
        print("Computing internal distances for job {:s}...".format(job.id))
        # internal distance curve for all molecules
        snap = gsd.hoomd.open("struct/equil.gsd", 'rb')[0] 
        n,avgRsq = trajtools.internaldistances_all(snap)
        with open("internaldistances_all.pkl", 'wb') as f:
            pickle.dump((n,avgRsq),f)
        
        # internal distance curves split out by species
        system = job_build_system_spec(job)
        speciesRsq = trajtools.internaldistances_species(snap,system)
        with open("internaldistances_species.pkl", 'wb') as f:
            pickle.dump(speciesRsq, f)

    return

@analysis_group
@BlendMD.pre.after(production)
@BlendMD.pre(lambda job: job.sp.num_CP!=0)
@BlendMD.post(junction_rdf_computed)
@BlendMD.operation
def junction_rdf(job):
    with job:
        print("Computing junction RDF for job {:s}...".format(job.id))
        snap = gsd.hoomd.open("struct/prod.gsd", 'rb')[0] 
        system = job_build_system_spec(job)

        r,g = trajtools.junction_RDF(snap, system, axis=0)    
        with open("junction_rdf.pkl", 'wb') as f:
            pickle.dump((r,g),f)
            
    return


if __name__ == "__main__":
    BlendMD().main()
