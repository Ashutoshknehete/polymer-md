import numpy as np
import systemspec
import gsd.hoomd

def wrap_coords(coords,boxsize):

    # wrap coordinates into a rectangular box with side lengths given by boxsize

    dims = len(boxsize)
    wrapped = np.zeros_like(coords)
    for i in range(dims):
        wrapped[:,i] = coords[:,i] - boxsize[i] * np.rint(coords[:,i]/boxsize[i])

    return wrapped 

def mc_chain_walk(N, l):

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

def connect_chains(chains, l):
    
    # monte carlo cutoff 
    twostepcutoff = 1.02/0.97 * l

    # chains should be a list of numpy arrays of coordinates of polymers
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
        addedchaincoords = chains[i] - chains[i][0,:] + chainstart
        chain.append(chain, addedchaincoords, axis = 0)

    return chain

def walk_linearPolymer():

    # walk each block of the polymer
    
    # connect them

    return

def walkComponent(component):
    # generate a set of random walk coordinates for this component that are the right length
    num = component.N
    coordlist = []
    for i in range(num):
        if isinstance(component.species, systemspec.LinearPolymerSpec):
            coordlist.append(walk_linearPolymer(component.species))  
    return

def placeComponent():
    # take a list of random walk coordinates and randomly place each one within the specified region

    return 

def systemCoordsRandom(system):

    # for each component:
    #   generate set of random walks
    #   place where you want them 
    #   wrap into box

    return

def build_snapshot(system):

    # get system coords 

    # get particle indices

    # get particle types and type ids

    # get bond indices

    # get bond types and type ids

    # 

    return