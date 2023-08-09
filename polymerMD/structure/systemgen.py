import numpy as np
from . import systemspec
import gsd.hoomd

def wrap_coords(coords,boxsize):
    # wrap coordinates into a rectangular box with side lengths given by boxsize

    dims = len(boxsize)
    if dims > 3:
        dims = 3
        boxsize = boxsize[0:3]
    
    wrapped = np.zeros_like(coords)
    for i in range(dims):
        if boxsize[i] == 0:
            wrapped[:,i] = 0
        else:
            wrapped[:,i] = coords[:,i] - boxsize[i] * np.rint(coords[:,i]/boxsize[i])

    return wrapped 

def mc_chain_walk(N, l):
    # montecarlo chain walk with an acceptance condition based on whether the randoomly generated
    # step results in too much backtracking
    PolymerCoords = np.zeros((N,3))
    twostepcutoff = 1.02/0.97 * l
    # "Monte carlo" loop where a random step is taken and only accepted if the distance isn't too far
    idx_mnr = 1
    while idx_mnr < N:
        randstep = np.random.rand(1,3) - 0.5
        newcoord = PolymerCoords[idx_mnr-1,:] + randstep * l/np.linalg.norm(randstep)
        if idx_mnr >= 2:
            twostepdist = np.linalg.norm(newcoord - PolymerCoords[idx_mnr-2])
            if twostepdist < twostepcutoff:
                continue
        # This "update"/"acceptance" code is only reached if twostepdist is greater 
        # than the cut off or if this is the first step being taken along the chain
        PolymerCoords[idx_mnr,:] = newcoord
        idx_mnr += 1

    return PolymerCoords

def connect_chains_linear(chains, l):
    # uses the same montecarlo technique to connect already generated set of coordinates.
    # only checks chain bending in the backward direction, not in the forward direction!
    # monte carlo cutoff 
    twostepcutoff = 1.02/0.97 * l
    # chains should be a list of numpy arrays of coordinates of polymers
    # chain is the numpy array of coordinates of the final chain
    chain = chains[0]
    for i in range(1,len(chains)):
        chainend = chain[-1,:]
        # get a valid starting point for the next block
        while True:
            randstep = np.random.rand(1,3) - 0.5
            chainstart = chainend + l * randstep/np.linalg.norm(randstep)
            twostepdist = np.linalg.norm(chainstart - chain[-2,:])
            if twostepdist > twostepcutoff:
                break
        combinedchaincoords = chains[i] - chains[i][0,:] + chainstart
        chain = np.append(chain, combinedchaincoords, axis = 0)

    return chain

def connect_chains_star(chains, l):
    # uses the same montecarlo technique to connect already generated set of coordinates.
    # only checks chain bending in the backward direction, not in the forward direction!
    # monte carlo cutoff 
    twostepcutoff = 1.02/0.97 * l
    # chains should be a list of numpy arrays of coordinates of polymers
    # block is the numpy array of coordinates of the each chain in chains        
    coords = []
    junction_coords = [0,0,0]
    for i in range(len(chains)):
        block = chains[i]
        block_start = block[0,:]
        randstep = np.random.rand(1,3) - 0.5
        chainstart = junction_coords + l * randstep/np.linalg.norm(randstep)
        shiftedblockcoords = block - block_start + chainstart
        for row in shiftedblockcoords:
            coords.append(row.tolist())      
    coords.append(junction_coords)
    return coords

def connect_chains_graft(chains, n_graft_blocks, n_backbone_blocks, l):
    # uses the same montecarlo technique to connect already generated set of coordinates.
    # only checks chain bending in the backward direction, not in the forward direction!
    # monte carlo cutoff 
    twostepcutoff = 1.02/0.97 * l
    # chains should be a list of numpy arrays of coordinates of polymers
    # block is the numpy array of coordinates of the each chain in chains        
    coords = []
    vertices = []
    vertex_temp = [0,0,0]
    vertices.append(vertex_temp)
    # loop over each backbone block
    for i in range(n_backbone_blocks):
        # take a random step from a vertex (for first block, it is the origin)
        if i == 0: # no need to check anything since its the first block of backbone
            randstep = np.random.rand(1,3) - 0.5
            chainstart = vertices[-1] + l * randstep/np.linalg.norm(randstep)
        else: # need to check distance from previous block's second last monomer to prevent backfolding
            while True:
                randstep = np.random.rand(1,3) - 0.5
                chainstart = vertices[-1] + l * randstep/np.linalg.norm(randstep)
                twostepdist = np.linalg.norm(chainstart - shiftedblockcoords[-1,:])
                if twostepdist > twostepcutoff:
                    break
        block = chains[i]
        block_start = block[0,:]
        # shift the coordinates of the block
        shiftedblockcoords = block - block_start + chainstart
        for row in shiftedblockcoords:
            coords.append(row.tolist())
        # take a random step starting from last monomer of previous block (this is the creation of a vertex)
        while True:
            randstep = np.random.rand(1,3) - 0.5
            chainstart = coords[-1] + l * randstep/np.linalg.norm(randstep)
            twostepdist = np.linalg.norm(chainstart - block[-2,:]) # check the distance from second last monomer of backbone block from this created vertex
            if twostepdist > twostepcutoff:
                break
        # use this vertex as the starting point for next block's random walk
        vertex_temp = chainstart
        vertices.append(vertex_temp.tolist())
    # remove the first and last element of vertices array (first = origin, last = end of backbone)
    junction_coords = vertices[1:-1]
    # loop over each graft block
    for i in range(n_backbone_blocks, n_backbone_blocks + n_graft_blocks):
        block = chains[i] # this is a graft block
        block_start = block[0,:]
        # following two blocks are the backbone blocks connected to the graft block, we need these to check distances during random walk of graft block
        backbone_block_prev = chains[i - n_backbone_blocks]
        backbone_block_next = chains[i - n_backbone_blocks + 1]
        # take a random step from a junction node
        while True:
            randstep = np.random.rand(1,3) - 0.5
            chainstart = junction_coords[i-n_backbone_blocks] + l * randstep/np.linalg.norm(randstep)
            twostepdist_prev = np.linalg.norm(chainstart - backbone_block_prev[-1,:])
            twostepdist_next = np.linalg.norm(chainstart - backbone_block_next[0,:])
            if twostepdist_prev > twostepcutoff and twostepdist_next > twostepcutoff:
                break
        # shift the coordinates of the block
        shiftedblockcoords = block - block_start + chainstart
        for row in shiftedblockcoords:
            coords.append(row.tolist())
    # add the junction nodes to the coords list
    for row in junction_coords:
            coords.append(row[0])
    return coords

def walk_linearPolymer(polymer: systemspec.LinearPolymerSpec):

    # walk each block of the polymer
    blockcoords = []
    for block in polymer.blocks:
        blockcoords.append(mc_chain_walk(block.length, block.monomer.l))

    # connect them
    # NOTE: this is an escapist solution. 
    # I need to find a way to account for different bond lengths l cleanly
    l = polymer.blocks[0].monomer.l
    chain = connect_chains_linear(blockcoords, l)

    return chain

def walk_starPolymer(polymer: systemspec.BranchedPolymerSpec):

    # walk each block of the polymer
    blockcoords = []
    for block in polymer.blocks:
        blockcoords.append(mc_chain_walk(block.length, block.monomer.l))
    # connect them
    # NOTE: this is an escapist solution. 
    # I need to find a way to account for different bond lengths l cleanly
    l = polymer.blocks[0].monomer.l
    coords = connect_chains_star(blockcoords, l)

    return coords

def walk_graftPolymer(polymer: systemspec.BranchedPolymerSpec):

    # walk each block of the polymer
    blockcoords = []
    for block in polymer.blocks:
        blockcoords.append(mc_chain_walk(block.length, block.monomer.l))
    # connect them
    # NOTE: this is an escapist solution. 
    # I need to find a way to account for different bond lengths l cleanly
    l = polymer.blocks[0].monomer.l
    n_graft_blocks = polymer.n_graft_blocks
    n_backbone_blocks = polymer.n_backbone_blocks
    chain = connect_chains_graft(blockcoords, n_graft_blocks, n_backbone_blocks, l)

    return chain

def walkComponent(component):
    # generate a set of random walk coordinates for this component that are the right length
    num = component.N
    coordlist = []
    for i in range(num):
        if component.species.shape == 'linear':
            coordlist.append(walk_linearPolymer(component.species))  
        elif component.species.shape == 'star':
            coordlist.append(walk_starPolymer(component.species))
        elif component.species.shape == 'graft':
            coordlist.append(walk_graftPolymer(component.species))
        elif component.species.shape == 'monoatomic':
            coordlist.append(np.array([0,0,0])) # will be placed randomly in placecomponent, called by systemCoords
    # coordlist is a list of numpy arrays
    return coordlist

def placeComponent(coordlist, region, regioncenter=[0,0,0], COM=True):
    # take a list of random walk coordinates and randomly place COM of each one within the specified region
    # for now, region is just a rectangular prism centered on the origin of a certain size. In future, region should be able to be
    # any geometric region with some common descriptor/interface (i.e.: a sphere or cylinder!)
    # if COM is true, place center of mass of chain at random point. Otherwise, place first point
    comlist = [np.average(coord,axis=0) for coord in coordlist]
    randcomlist = np.multiply(region, np.random.rand(len(comlist),3)-0.5) + np.array(regioncenter)
    newcoordlist = []
    for i,coord in enumerate(coordlist):
        newcoord = (coord - comlist[i] + randcomlist[i,:]).reshape([-1,3]) # make sure correct dimensions even for single-atom molecules
        newcoordlist.append(newcoord)
    return newcoordlist

def systemCoordsRandom(system):

    box = system.box[0:3]
    syscoords = np.zeros((system.numparticles,3))
    totaladded = 0
    for component in system.components:
        # a list of coordinates for each instance of this component
        coordlist = walkComponent(component)
        # place center of mass of each chain at a random point
        coordlist = placeComponent(coordlist, box, COM=True)
        # sequentially add each 
        for coord in coordlist:
            nchain = coord.shape[0]
            syscoords[totaladded:totaladded+nchain,:] = coord
            totaladded += nchain
    syscoords = wrap_coords(syscoords, box)
    return syscoords

def systemCoordsBoxRegions(system, regions, regioncenters):

    # regions is a list of regions with length equal to the number of components
    # the intention is that these should be used to seed components as phase-separated in advance
     
    box = system.box[0:3]
    
    syscoords = np.zeros((system.numparticles,3))

    totaladded = 0        
    for i,component in enumerate(system.components):
        # a list of coordinates for each instance of this component
        coordlist = walkComponent(component)
        # place center of mass of each chain at a random point
        coordlist = placeComponent(coordlist, regions[i], regioncenters[i], COM=True)
        # sequentially add each 
        for coord in coordlist:
            nchain = coord.shape[0]
            syscoords[totaladded:totaladded+nchain,:] = coord
            totaladded += nchain

    syscoords = wrap_coords(syscoords, box)
    return syscoords

def getParticleTypes(system):

    types = system.monomerlabels
    N = system.numparticles # number of particles in system
    alltypes = system.particleTypes()
    typeid = [types.index(label) for label in alltypes]
    return types, typeid

def getBondTypes(system):
    bonds, allbondtypes = system.bonds()
    bondtypes = list(set(allbondtypes))
    bondtypeid = [bondtypes.index(bondlabel) for bondlabel in allbondtypes]

    return bonds, bondtypes, bondtypeid

def build_snapshot(system, type='random', regions=[], regioncenters=[],verbose=True):

    # get system box size, total number of particles, 
    box = system.box
    N = system.numparticles
    # get system coords 
    if type == 'random':
        pos = systemCoordsRandom(system)
    elif type == 'boxregions':
        if len(regions)==0 or len(regioncenters)==0:
            raise ValueError("No regions specified.")
        if len(regions) != len(regioncenters):
            raise ValueError("Lengths of regions and region centers don't match.")
        if len(regions) != system.nComponents:
            raise ValueError("Number of regions do not match number of components.") 
        pos = systemCoordsBoxRegions(system, regions, regioncenters)

    # get particle indices, types, and type ids
    types, typeid = getParticleTypes(system)
    
    # get bond indices, types, and type ids
    bondgroup, bondtypes, bondtypeid = getBondTypes(system)
    nBonds = len(bondgroup)

    # generate snapshot!!
    frame = gsd.hoomd.Snapshot() #.Frame replaced by .Snapshot for version compatibility
    frame.configuration.box = box
    frame.particles.N = N
    frame.particles.position = pos
    frame.particles.types = types
    frame.particles.typeid = typeid 
    frame.bonds.N = nBonds
    frame.bonds.types = bondtypes
    frame.bonds.typeid = bondtypeid
    frame.bonds.group = bondgroup
    if verbose:
        print("Number of particles:             N = {:d}".format(N))
        print("Number of positions:             pos.shape[0] = {:d}".format(pos.shape[0]))
        print("Number of particle types:        len(types) = {:d}".format(len(types)))
        print("Number of particle type ids:     len(typeid) = {:d}".format(len(typeid)))
        print("Number of bonds:                 nBonds = {:d}".format(nBonds))
        print("Number of bond types:            len(bondtypes) = {:d}".format(len(bondtypes)))
        print("Number of bond type ids:         len(bondtypeid) = {:d}".format(len(bondtypeid)))
        print("Number of bond groups:           len(bondgroup) = {:d}".format(len(bondgroup)))
        print("bond types:                      bond types = ",bondtypes)
        print("bond typeID:                     bond typeID = ",bondtypeid)
        print("bond groups:                     bond groups = ",bondgroup)
              
    return frame
