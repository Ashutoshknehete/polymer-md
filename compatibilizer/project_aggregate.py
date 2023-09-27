import os
import pickle
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from flow import FlowProject, aggregator, get_aggregate_id
import gsd.hoomd
from signac import JSONDict
import freud

from polymerMD.structure import systemspec, systemgen
from polymerMD.analysis import trajtools

#### THIS IS A WAY TO TOGGLE IGNORING MICELLIZED SYSTEMS
ignore_micellized = True

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
def simulated(job):
    return job.isfile("struct/prod.gsd")

@BlendMD.label
def bulk_gamma_computed(job):
    cond1 = "bulk_interfacial_tension_average" in job.doc
    cond2 = job.isfile("bulk_gamma_independent_samples.txt")
    return cond1 and cond2

@BlendMD.label
def internal_distances_computed(job):
    return job.isfile("internaldistances_all.pkl") and job.isfile("internaldistances_species.pkl")

### job aggregation functions and classes
class openpath(object):
    def __init__(self, path, create=False):
        
        self.path = path
        self.create = create
        return
    
    def __enter__(self):
        if not os.path.exists(self.path):
            if self.create:
                os.mkdir(self.path)
            else:
                ValueError("Path does not exist and can not be opened.")
        self.origpath = os.getcwd()
        os.chdir(self.path)
        return

    def __exit__(self, excep_type, excep_value, excep_traceback):
        os.chdir(self.origpath)
        return

class replicaAggregate(object):
    def __init__(self,jobs):
        self.jobs = jobs
        self.spdict = {k:v for k,v in self.jobs[0].sp.items() if k!='replica'}
        # check all statepoints identical! 
        for job in self.jobs:
            jobsp = {k:v for k,v in job.sp.items() if k!='replica'}
            if jobsp != self.spdict:
                ValueError("Attempted to create replicaAggregate with non-replica jobs.")
        # get aggregate id
        self.id = get_aggregate_id(jobs)
        
        # get aggregate path and context manager
        self.path = os.getcwd()+"/aggregate/"+self.id+"/"
        self.contextmanager = openpath(self.path, create=False)

        # get filenames for statepoint and document. Initialize to none.
        self.fn_sp = "aggregate_statepoint.json"
        self.fn_doc = "aggregate_document.json"
        self._doc = None
        self._sp = None
        
        return
   
    def init(self):
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        if self._sp == None:
            self._sp = JSONDict(filename=self.path+self.fn_sp, write_concern=True)
            self._sp.reset(self.spdict)
        if not self.isfile("aggregate_jobs.txt"):
            joblist = '\n'.join([j.id for j in self.jobs])
            with open(self.path+"aggregate_jobs.txt","w") as f:
                f.write(joblist)
        return
    
    @property
    def doc(self):
        if self._doc is None:
            self.init()
            self._doc = JSONDict(filename=self.path+self.fn_doc, write_concern=True)
        return self._doc
    
    @property
    def sp(self):
        return self.spdict

    def indoc(self,key):
        # don't create doc if it doesn't exist already
        # if it doesn't exist, nothing is in it. return false! 
        if self._doc==None:
            if self.isfile(self.fn_doc):
                return key in self.doc
            else:
                return False
        else:
            return key in self.doc

    def isfile(self,path):
        if not os.path.exists(self.path):
            return False
        with self.contextmanager:
            return os.path.isfile(path)

    def isdir(self,path):
        if not os.path.exists(self.path):
            return False
        with self.contextmanager:
            return os.path.isdir(path)
    
    def __enter__(self):
        self.init()
        self.contextmanager.__enter__()
        return
    
    def __exit__(self, excep_type, excep_value, excep_traceback):
        self.contextmanager.__exit__(excep_type,excep_value,excep_traceback)
        return

def replica_key(job):
    return [v for k,v in job.sp.items() if k!="replica"]

def replica_aggregator():
    return aggregator.groupby(key=replica_key)

# aggregate pre- and post-conditions
def aggcondition(jobs,condition): 
    # convert a single job condition into an aggregate condition that is true if true for all jobs
    # lambda *jobs: aggconditions(jobs, condition)
    jobconditions = [condition(job) for job in jobs]
    return all(jobconditions)

def agg_junction_rdf_computed(*jobs):
    return replicaAggregate(jobs).isfile("junction_rdf.pkl")

def agg_gamma_computed(*jobs):
    return replicaAggregate(jobs).indoc("bulk_interfacial_tension_average")

def agg_internal_distances_computed(*jobs):
    return replicaAggregate(jobs).isfile("internaldistances_species.png")

def agg_density_1D_computed(*jobs):
    return replicaAggregate(jobs).isfile("density_1D_species.pkl")

def agg_smeared_density_computed(*jobs):
    return replicaAggregate(jobs).isfile("junction_coverage_histogram.png")

def agg_smeared_density_varied_computed(*jobs):
    return replicaAggregate(jobs).isdir("variedsmear")

def agg_junction_bonds_propertied(*jobs):
    aggregate = replicaAggregate(jobs)
    return aggregate.indoc("junction_bondlength_x_avg")

aggregate_group = BlendMD.make_group(name="aggregated", group_aggregator=replica_aggregator())

### aggregate operations
@aggregate_group
@BlendMD.post(lambda *jobs: replicaAggregate(jobs).indoc("Lx"))
@BlendMD.operation(aggregator=replica_aggregator())
def agg_compute_descriptors(*jobs):
    aggregate = replicaAggregate(jobs)
    print("Computing descriptors for aggregate {:s}...".format(aggregate.id))
    compute_descriptors_general(jobs[0],aggregate.doc)
    return

@aggregate_group
@BlendMD.pre(lambda *jobs: aggcondition(jobs, simulated))
@BlendMD.pre(lambda *jobs: aggcondition(jobs, lambda job: job.sp.num_CP!=0))
@BlendMD.post(agg_junction_rdf_computed) 
@BlendMD.operation(aggregator=replica_aggregator())
def agg_junction_rdf(*jobs):
    aggregate = replicaAggregate(jobs)
    print("Computing aggregated junction RDF for {:s}...".format(aggregate.id))
    
    # open production final snapshots and generate system specifications
    snapshots = []
    systems = []
    for job in aggregate.jobs:
        if job.doc.junction_peaks > 2 and ignore_micellized: # ignore micellized replicas
            continue
        with job:
            snapshots.append(gsd.hoomd.open("struct/prod.gsd", 'rb')[0])
            systems.append(job_build_system_spec(job))
    
    if len(snapshots)==0:
        print("All systems micellized, can't compute aggregate properties.")

    nbins = 40
    rmax = 10
    rdf = trajtools.junction_RDF_accumulate(snapshots,systems,axis=0,nBins=nbins,rmax=rmax)
    r,g = (rdf.bin_centers,rdf.rdf)

    with aggregate:
        with open("junction_rdf.pkl", 'wb') as f:
            pickle.dump((r,g),f)

    sp = aggregate.sp
    fig,ax = plt.subplots()
    ax.plot(r,g,color="black")
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$g_{\mathrm{junc}}(r)$")
    fig.tight_layout()
    fig.savefig("plots/rdf/rdf_nblock_{:d}_num_{:d}_len_{:d}.png".format(len(sp['length_CP']),sp['num_CP'],sum(sp['length_CP'])),dpi=300)
    with aggregate:
        fig.savefig("junction_rdf.png", dpi = 300)
    plt.close(fig)
    
    return

@aggregate_group
@BlendMD.pre(lambda *jobs: aggcondition(jobs, bulk_gamma_computed))
@BlendMD.post(agg_gamma_computed) 
@BlendMD.operation(aggregator=replica_aggregator())
def agg_interfacial_tension(*jobs):
    aggregate = replicaAggregate(jobs)
    print("Aggregating interfacial tension for {:s}...".format(aggregate.id))
    exclude_outliers = False

    samples = []
    for job in aggregate.jobs:
        if job.doc.junction_peaks > 2 and ignore_micellized: 
            continue
        with job:
            samples += np.loadtxt("bulk_gamma_independent_samples.txt").tolist()
    samples = np.array(samples)

    if exclude_outliers:
        samples_z = stats.zscore(samples)
        samples = samples[np.abs(samples_z) < 3]
    
    # mean and standard error of mean
    gamma_avg = np.mean(samples)
    gamma_var = np.var(samples)/len(samples) # standard error of mean

    aggregate.doc.bulk_interfacial_tension_average = gamma_avg
    aggregate.doc.bulk_interfacial_tension_variance = gamma_var

@aggregate_group
@BlendMD.pre(lambda *jobs: aggcondition(jobs, internal_distances_computed))
@BlendMD.post(agg_internal_distances_computed)
@BlendMD.operation(aggregator=replica_aggregator())
def agg_internal_distances(*jobs):
    aggregate = replicaAggregate(jobs)

    # empty arrays to hold data to average
    ns = [ [] for i in range(2) ]
    avgRsqs = [ [] for i in range(2) ]

    # figure to plot on as we go
    fig,axs = plt.subplots(1,2,sharey=True,figsize=(6,4))
    
    for job in aggregate.jobs:
        if job.doc.junction_peaks > 2 and ignore_micellized: 
            continue
        with job:
            with open("internaldistances_species.pkl", "rb") as f:
                speciesRsq = pickle.load(f)

            for name,(n,Rsq) in speciesRsq.items():
                if len(name)==1: #homopolymer
                    axs[0].plot(n,Rsq/n,color="#D3d3d3")
                    avgRsqs[0].append(Rsq)
                    ns[0] = n
                else:
                    axs[1].plot(n,Rsq/n,color="#D3d3d3")
                    avgRsqs[1].append(Rsq)
                    ns[1] = n
    
    # make plot 
    for i in range(2):
        avgRsq = np.average(avgRsqs[i],axis=0)
        axs[i].plot(ns[i],avgRsq/ns[i],color="blue")
        axs[i].set_xlabel(r"$N$")
        axs[i].set_xscale('log')
    axs[0].set_ylabel(r"$\langle R^2(N) \rangle / N$")
    axs[0].set_title("homopolymer")
    axs[1].set_title("copolymer")
    fig.tight_layout()
    length_CP = aggregate.sp["length_CP"]
    num_CP = aggregate.sp["num_CP"]
    print(aggregate.sp)
    fig.savefig("plots/segdist/segdist_nblock_{:d}_num_{:d}_len_{:d}.png".format(len(length_CP),num_CP,sum(length_CP)),dpi=300)
    with aggregate:
        fig.savefig("internaldistances_species.png",dpi=300)
    plt.close(fig)
    return

@aggregate_group
@BlendMD.pre(lambda *jobs: aggcondition(jobs, simulated))
@BlendMD.post(agg_density_1D_computed)
@BlendMD.operation(aggregator=replica_aggregator())
def agg_density_1D(*jobs):
    aggregate = replicaAggregate(jobs)
    print("Computing aggregate density profiles for {:s}...".format(aggregate.id))

    # dicts to store aggregate density counts
    profiles_mnr = {}
    profiles_spe = {}

    # system spec for getting indices
    system = job_build_system_spec(aggregate.jobs[0])
    idx_B = [particletype == 'B' for particletype in system.particleTypes()]

    bins = 150

    # loop over all jobs
    for job in aggregate.jobs:
        if job.doc.junction_peaks > 2 and ignore_micellized: 
            continue
        with job:
            snap = gsd.hoomd.open("struct/prod.gsd", 'rb')[0]
        
        # shift positions of snapshot so that the center of mass of all B monomers is centered on 0 
        com_B = np.average(snap.particles.position[idx_B,:],axis=0)
        shiftedpos = snap.particles.position-com_B
        snap.particles.position=systemgen.wrap_coords(shiftedpos,snap.configuration.box)

        # compute and aggregate for monomer case
        temp_mnr = trajtools.density_1D_monomers(snap,nBins=bins,method='binned')
        if not profiles_mnr:
            profiles_mnr = {k: (np.zeros_like(v[0]),v[1]) for k,v in temp_mnr.items()}
        for k,v in temp_mnr.items():
            profiles_mnr[k] = (profiles_mnr[k][0]+v[0], profiles_mnr[k][1])

        # compute and aggregate for species case
        temp_spe = trajtools.density_1D_species(snap,system,nBins=bins,method='binned')
        if not profiles_spe:
            profiles_spe = {k: (np.zeros_like(v[0]),v[1]) for k,v in temp_spe.items()}
        for k,v in temp_spe.items():
            profiles_spe[k] = (profiles_spe[k][0]+v[0], profiles_spe[k][1])
        
    # normalize by dividing by number of jobs, since each bin will independently sum to 1
    njobs = len([job for job in aggregate.jobs if job.doc.junction_peaks==2])
    profiles_mnr = {k: (v[0]/njobs, v[1]) for k,v in profiles_mnr.items()}
    profiles_spe = {k: (v[0]/njobs, v[1]) for k,v in profiles_spe.items()}

    # write densities
    with aggregate:
        with open("density_1D_monomers.pkl", 'wb') as f:
            pickle.dump(profiles_mnr, f)
    
        with open("density_1D_species.pkl", 'wb') as f:
            pickle.dump(profiles_spe, f)

    # make density mnr plot
    fig,ax = plt.subplots(figsize=(5,3))
    for k,v in profiles_mnr.items():
        ax.plot(v[1][:-1],v[0], label=k)
    ax.legend()
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\phi(x)$")
    fig.tight_layout()

    # save plot
    length_CP = aggregate.sp["length_CP"]
    num_CP = aggregate.sp["num_CP"]
    fig.savefig("plots/density/density_mnr_nblock_{:d}_num_{:d}_len_{:d}.png".format(len(length_CP),num_CP,sum(length_CP)),dpi=300)
    with aggregate:
        fig.savefig("density_mnr.png",dpi=300)
    plt.close(fig)

    # make density species plot
    fig,ax = plt.subplots(figsize=(5,3))
    for k,v in profiles_spe.items():
        ax.plot(v[1][:-1],v[0], label=k)
    ax.legend()
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\phi(x)$")
    fig.tight_layout()

    # save plot
    length_CP = aggregate.sp["length_CP"]
    num_CP = aggregate.sp["num_CP"]
    fig.savefig("plots/density/density_spe_nblock_{:d}_num_{:d}_len_{:d}.png".format(len(length_CP),num_CP,sum(length_CP)),dpi=300)
    with aggregate:
        fig.savefig("density_spe.png",dpi=300)
    plt.close(fig)

    return

@aggregate_group
@BlendMD.pre(lambda *jobs: aggcondition(jobs, simulated))
@BlendMD.pre(lambda *jobs: aggcondition(jobs, lambda job: job.sp.num_CP!=0))
@BlendMD.post(agg_smeared_density_computed)
@BlendMD.operation(aggregator=replica_aggregator())
def agg_junction_density_smeared(*jobs):
    aggregate = replicaAggregate(jobs)
    print("Computing aggregate smeared density profiles for {:s}...".format(aggregate.id))

    densities = []
    njobs = len(aggregate.jobs)
    for i,job in enumerate(aggregate.jobs):
        if not i%5:
            print("job {:d}/{:d}".format(i,njobs))
        if job.doc.junction_peaks > 2 and ignore_micellized: 
            continue
        with job:
            snap = gsd.hoomd.open("struct/prod.gsd", 'rb')[0]
        system = job_build_system_spec(job)
        gd_left, gd_right = trajtools.junction_density_smeared(snap,system,axis=0,nBins=500,sigma=5.5)
        
        # store densities in a single flattened array
        densities += gd_left.density.reshape(-1).tolist()
        densities += gd_right.density.reshape(-1).tolist()
    
    # save density arrays, calculate basic statistics
    densities = np.array(densities)
    #with aggregate:
    #    np.savetxt("junction_densities.txt",densities)
    aggregate.doc.interfacial_density_variance = np.var(densities)
    
    # compute and save junction cumulative coverage
    totalbins = len(densities)
    steps = 1500
    maxdens = 0.20
    cutoffs = np.linspace(0,maxdens,steps)
    coverage = np.array([np.sum(densities < cutoff)/(totalbins) for cutoff in cutoffs])
    with aggregate:
        np.savetxt("junction_coverage_cumulative.txt", np.array([cutoffs,coverage]).T)
    
    # compute and save junction density histogram
    histbins = np.linspace(0,0.2,200)
    hist = np.histogram(densities,bins=histbins,density=True)
    with aggregate:
        with open("junction_coverage_histogram.pkl",'wb') as f:
            pickle.dump(hist,f)
    
    # plot histogram and cumulative coverage
    length_CP = aggregate.sp["length_CP"]
    num_CP = aggregate.sp["num_CP"]
    sysstr = "nblock_{:d}_num_{:d}_len_{:d}".format(len(length_CP),num_CP,sum(length_CP))

    fig,ax = plt.subplots(figsize=(5,3))
    ax.plot(cutoffs,coverage)
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"coverage$(\rho)$")
    fig.savefig("plots/junctiondensity/junction_coverage_cumulative_{:s}.png".format(sysstr),dpi=300)
    with aggregate:
        fig.savefig("junction_coverage_cumulative.png",dpi=300)

    fig,ax = plt.subplots(figsize=(5,3))
    ax.bar(hist[1][:-1], hist[0], width=np.diff(hist[1]), edgecolor="black", align="edge")
    ax.set_xlabel(r"$\rho$")
    fig.savefig("plots/junctiondensity/junction_coverage_histogram_{:s}.png".format(sysstr),dpi=300)
    with aggregate:
        fig.savefig("junction_coverage_histogram.png",dpi=300)


    return

@aggregate_group
@BlendMD.pre(lambda *jobs: aggcondition(jobs, simulated))
@BlendMD.pre(lambda *jobs: aggcondition(jobs, lambda job: job.sp.num_CP!=0))
@BlendMD.post(agg_smeared_density_varied_computed)
@BlendMD.operation(aggregator=replica_aggregator())
def agg_junction_density_smeared_vary(*jobs):
    aggregate = replicaAggregate(jobs)
    print("Computing varied-width aggregate smeared density profiles for {:s}...".format(aggregate.id))

    with aggregate:
        if not os.path.exists("variedsmear"):
            os.mkdir("variedsmear")
    
    njobs = len(aggregate.jobs)
    sigmas = np.arange(0.5,10.1,0.25)
    variances = []
    for sigma in sigmas:
        print("For sigma = {:f}".format(sigma))
        densities = []
        for i,job in enumerate(aggregate.jobs):
            if not i%5:
                print("job {:d}/{:d}".format(i,njobs))
            if job.doc.junction_peaks > 2 and ignore_micellized: 
                continue
            with job:
                snap = gsd.hoomd.open("struct/prod.gsd", 'rb')[0]
            system = job_build_system_spec(job)
            gd_left, gd_right = trajtools.junction_density_smeared(snap,system,axis=0,nBins=750,sigma=sigma)
            
            # store densities in a single flattened array
            densities += gd_left.density.reshape(-1).tolist()
            densities += gd_right.density.reshape(-1).tolist()
        
        # save density arrays
        densities = np.array(densities)
        variances.append(np.var(densities))
        
        # compute and save junction cumulative coverage
        totalbins = len(densities)
        steps = 1500
        maxdens = 0.20
        cutoffs = np.linspace(0,maxdens,steps)
        coverage = np.array([np.sum(densities < cutoff)/(totalbins) for cutoff in cutoffs])
        with aggregate:
            np.savetxt("variedsmear/junction_coverage_cumulative_sigma_{:.2f}.txt".format(sigma), np.array([cutoffs,coverage]).T)
        
        # # compute and save junction density histogram
        # histbins = np.linspace(0,0.2,200)
        # hist = np.histogram(densities,bins=histbins,density=True)
        # with aggregate:
        #     with open("variedsmear/junction_coverage_histogram_sigma_{:.2f}.pkl".format(sigma),'wb') as f:
        #         pickle.dump(hist,f)
        
        # # plot histogram and cumulative coverage
        # length_CP = aggregate.sp["length_CP"]
        # num_CP = aggregate.sp["num_CP"]
        # sysstr = "sigma_{:.2f}_nblock_{:d}_num_{:d}_len_{:d}".format(sigma,len(length_CP),num_CP,sum(length_CP))

        # fig,ax = plt.subplots(figsize=(5,3))
        # ax.plot(cutoffs,coverage)
        # ax.set_xlabel(r"$\rho$")
        # ax.set_ylabel(r"coverage$(\rho)$")
        # fig.savefig("plots/junctiondensityvaried/junction_coverage_cumulative_{:s}.png".format(sysstr),dpi=300)
        # with aggregate:
        #     fig.savefig("variedsmear/junction_coverage_cumulative_sigma_{:.2f}.png".format(sigma),dpi=300)

        # fig,ax = plt.subplots(figsize=(5,3))
        # ax.bar(hist[1][:-1], hist[0], width=np.diff(hist[1]), edgecolor="black", align="edge")
        # ax.set_xlabel(r"$\rho$")
        # fig.savefig("plots/junctiondensityvaried/junction_coverage_histogram_{:s}.png".format(sysstr),dpi=300)
        # with aggregate:
        #     fig.savefig("variedsmear/junction_coverage_histogram_sigma_{:.2f}.png".format(sigma),dpi=300)

    #with aggregate:
    #    with open("variance_scaling.pkl",'wb') as f:
    #        pickle.dump((sigmas, np.array(variances)),f)
    
    return

@aggregate_group
@BlendMD.pre(lambda *jobs: aggcondition(jobs, simulated))
@BlendMD.pre(lambda *jobs: aggcondition(jobs, lambda job: job.sp.num_CP!=0))
@BlendMD.post(agg_junction_bonds_propertied)
@BlendMD.operation(aggregator=replica_aggregator())
def agg_junction_bondproperties(*jobs):
    # compute bond properties of the copolymer junctions
    aggregate = replicaAggregate(jobs)
    print("Computing bond properties for aggregate {:s}...".format(aggregate.id))

    # get junction indices, which will be identical for each replica 
    system = job_build_system_spec(aggregate.jobs[0])
    junctionbonds = system.junctions()
    bond0 = [bond[0] for bond in junctionbonds]
    bond1 = [bond[1] for bond in junctionbonds]

    # get a freud box, which will be used to correctly compute distances
    box = freud.box.Box.from_box(job_compute_box_dimensions(aggregate.jobs[0]))

    # construct a list of all junction bond lengths for each junction in each replica
    bondxs = []
    bondcosth = []
    for job in aggregate.jobs:
        if job.doc.junction_peaks > 2 and ignore_micellized: 
            continue
        with job:
            snap = gsd.hoomd.open("struct/prod.gsd", 'rb')[0]
        pos = snap.particles.position 
        posx = np.copy(pos)
        posx[:,[1,2]] = 0
        # set all y,z coordinates to 0 to only compute distance in x direction
        d = box.compute_distances(pos[bond0], pos[bond1])
        dx = box.compute_distances(posx[bond0], posx[bond1])
        bondxs += dx.tolist()
        costh = dx/d
        bondcosth += costh.tolist() 
 
    # compute average bondlength and bondlength variance
    bondxs = np.array(bondxs)
    bondcosth = np.array(bondcosth)
    aggregate.doc.junction_bondlength_x_avg = np.mean(bondxs)
    aggregate.doc.junction_bondlength_x_var = np.var(bondxs)/bondxs.size
    aggregate.doc.junction_bondcosth_avg = np.mean(bondcosth)
    aggregate.doc.junction_bondcosth_var = np.var(bondcosth)/bondcosth.size

    return

if __name__ == "__main__":
    BlendMD().main()
