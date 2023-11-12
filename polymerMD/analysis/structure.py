import freud
import numpy as np
from polymerMD.analysis.utility import wrap_coords

def getAllPairs(maxIdx, minIdx=0):

    pairs = []
    for i in range(minIdx,maxIdx+1):
        for j in range(minIdx,i):
            pairs.append([i,j])
    return pairs

def meanEndToEnd(coord, molecules, box, power=2):
    '''
    Args:
        coord (np.ndarray):             Nx3 array for the coordinates of N particles
        molecules (List[List[int]]):    list of indices of particles in each molecule
        box (freud.box.Box):            box used to compute distances
        power (float or int):           power to raise distances to inside average         
    Returns:
        avgDistToPower (float): average of end to end distances to inputted power
    '''

    # loop over molecules and identify indices of distances to compute
    # this way we only make one call to compute distances.. much faster!
    points1 = []
    points2 = []
    for mol in molecules:
        points1.append(min(mol))
        points2.append(max(mol))

    # use box object to compute distances
    distances = box.compute_distances(coord[points1], coord[points2])

    # average the distances
    distToPower = np.power(distances,power)
    avgDistToPower = np.mean(distToPower,axis=0)

    return avgDistToPower, distances

def meanRadiusGyration(coord, molecules, box, power=2):
    '''
    Args:
        coord (np.ndarray):             Nx3 array for the coordinates of N particles
        molecules (List[List[int]]):    list of indices of particles in each molecule
        box (freud.box.Box):            box used to compute distances
        power (float or int):           power to raise distances to inside average         
    Returns:
        avgRgToPower (float): average of end to end distances to inputted power
    '''

    # loop over molecules and identify indices of distances to compute
    # this way we only make one call to compute distances.. much faster!
    points1 = []
    points2 = []
    pos = coord
    for mol in molecules:
        # unwrap coordinates in molecule to be continuous based on first particle
        r0 = pos[mol[0],:]
        pos[mol,:] = r0 + wrap_coords(pos[mol,:] - r0, box.L)
        for i in mol:
            points1.append(i)
            points2.append(pos.shape[0]) # the eventual location of the com
        com = np.mean(pos[mol,:],axis=0).reshape(1,-1)
        pos = np.append(pos,com,axis=0)
            
    # use box object to compute distances from com
    distances = box.compute_distances(pos[points1], pos[points2])

    # square and average the distances
    distancesSquared = np.square(distances)
    RgSquared = np.array([np.mean(
        distancesSquared[[points1.index(i) for i in mol]], axis=0
    ) for mol in molecules])
    avgRgToPower = np.mean(np.power(RgSquared,power/2),axis=0) # already raised to 2nd power

    return avgRgToPower, RgSquared

def meanSqInternalDist(coord, molecules, box):
    '''
    Args:
        coord (np.ndarray):             Nx3 array for the coordinates of N particles
        molecules (List[List[int]]):    list of indices of particles in each molecule
        box (freud.box.Box):            
    Returns:
        n (np.ndarray):         1 x max(molecule lengths)-1 containing the corresponding segment lengths
        avgRsq (np.ndarray):    1 x max(molecule lengths)-1 array containing average internal distances 
                                along the chains. Entry i corresponds with segments of length i+2 
    '''

    # find max molecule length and initialize
    molSize = [len(mol) for mol in molecules]
    maxLength = max(molSize)
    avgRsq = np.zeros((maxLength))
    count = np.zeros((maxLength))

    # loop over molecules and identify indices of distances to compute
    # this way we only make one call to compute distances.. much faster!
    points1 = []
    points2 = []
    for mol in molecules:
        minidx = min(mol)
        maxidx = max(mol)
        idxrange = list(range(minidx,maxidx+1))
        for i in idxrange:
            for j in range(minidx,i):
                points1.append(i)
                points2.append(j)

    # use box object to compute distances
    distances = box.compute_distances(coord[points1], coord[points2])

    # average squared segment distances
    distancesSquared = np.square(distances)
    for dsq,(i,j) in zip(distancesSquared,zip(points1,points2)):
        avgRsq[i-j] += dsq
        count[i-j] += 1
    
    # compute average, and remove the 0 element
    avgRsq[1:] = avgRsq[1:]/count[1:]
    avgRsq = avgRsq[1:]
    n = np.arange(2,len(avgRsq)+2)

    return n, avgRsq

def meanSqRadiusGyrationComponents(coord, molecules, box, power=2):
    '''
    Args:
        coord (np.ndarray):             Nx3 array for the coordinates of N particles
        molecules (List[List[int]]):    list of indices of particles in each molecule
        box (freud.box.Box):            box used to compute distances
        power (float or int):           power to raise distances to inside average         
    Returns:
        avgSqRg_x, avgSqRg_y, avgSqRg_z (float): average mean squared radius of gyration of molecule x, y, and z component respectively
    '''

    # loop over molecules and identify indices of distances to compute
    # this way we only make one call to compute distances.. much faster!
    points1 = []
    points2 = []
    pos = coord
    for mol in molecules:
        # unwrap coordinates in molecule to be continuous based on first particle
        r0 = pos[mol[0],:]
        pos[mol,:] = r0 + wrap_coords(pos[mol,:] - r0, box.L)
        for i in mol:
            points1.append(i)
            points2.append(pos.shape[0]) # the eventual location of the com
        com = np.mean(pos[mol,:],axis=0).reshape(1,-1)
        pos = np.append(pos,com,axis=0)
    
    x_coords = pos[points1][:,0]
    y_coords = pos[points1][:,1]
    z_coords = pos[points1][:,2]
    x_coords_COM = pos[points2][:,0]
    y_coords_COM = pos[points2][:,1]
    z_coords_COM = pos[points2][:,2]

    distances_sq_x = np.array([(x_coords[i]-x_coords_COM[i])**2 for i in range(len(x_coords))])
    distances_sq_y = np.array([(y_coords[i]-y_coords_COM[i])**2 for i in range(len(y_coords))])
    distances_sq_z = np.array([(z_coords[i]-z_coords_COM[i])**2 for i in range(len(z_coords))])

    RgSquared_x = np.array([np.mean(
        distances_sq_x[[points1.index(i) for i in mol]], axis=0
    ) for mol in molecules])
    RgSquared_y = np.array([np.mean(
        distances_sq_y[[points1.index(i) for i in mol]], axis=0
    ) for mol in molecules])
    RgSquared_z = np.array([np.mean(
        distances_sq_z[[points1.index(i) for i in mol]], axis=0
    ) for mol in molecules])
    
    mean_RgSquared_x = np.mean(RgSquared_x)
    mean_RgSquared_y = np.mean(RgSquared_y)
    mean_RgSquared_z = np.mean(RgSquared_z)

    std_RgSquared_x = np.std(RgSquared_x)/np.sqrt(len(RgSquared_x))
    std_RgSquared_y = np.std(RgSquared_y)/np.sqrt(len(RgSquared_y))
    std_RgSquared_z = np.std(RgSquared_z)/np.sqrt(len(RgSquared_z))
    
    mean_RgSquared = [mean_RgSquared_x, mean_RgSquared_y, mean_RgSquared_z]
    std_RgSquared = [std_RgSquared_x, std_RgSquared_y, std_RgSquared_z]

    return mean_RgSquared, std_RgSquared

def meanSqRadiusGyrationComponents_monomer(system, BCP_params, coord, particle_types_list, molecules, box, power=2):
    '''
    Args:
        coord (np.ndarray):             Nx3 array for the coordinates of N particles
        molecules (List[List[int]]):    list of indices of particles in each molecule
        box (freud.box.Box):            box used to compute distances
        power (float or int):           power to raise distances to inside average         
    Returns:
        avgSqRg_x, avgSqRg_y, avgSqRg_z (float): average mean squared radius of gyration of molecule x, y, and z component respectively
    '''

    # loop over molecules and identify indices of distances to compute
    # this way we only make one call to compute distances.. much faster!
    points1 = []
    points2 = []
    particle_types = []
    pos = coord
    n_molecules = len(molecules)
    for mol in molecules:
        # unwrap coordinates in molecule to be continuous based on first particle
        r0 = pos[mol[0],:]
        pos[mol,:] = r0 + wrap_coords(pos[mol,:] - r0, box.L)
        for i in mol:
            points1.append(i)
            points2.append(pos.shape[0]) # the eventual location of the com
            particle_types.append(particle_types_list[i])
        com = np.mean(pos[mol,:],axis=0).reshape(1,-1)
        pos = np.append(pos,com,axis=0)
        
    x_coords = pos[points1][:,0]
    y_coords = pos[points1][:,1]
    z_coords = pos[points1][:,2]
    x_coords_COM = pos[points2][:,0]
    y_coords_COM = pos[points2][:,1]
    z_coords_COM = pos[points2][:,2]
    
    # average values for a particular monomer over all the chains is directly calculated, 
    # instead of separately for each chain and then again taking average since its easier this way

    M_CP=BCP_params[0]
    N_CP=BCP_params[1]
    NCP_A = N_CP[0]
    NCP_B = N_CP[1]
    n_arms=BCP_params[2]
    architecture=BCP_params[3]

    def split_arrays_by_monomer(coords):
        coords_A = np.array([coords[i] for i in range(len(coords)) if particle_types[i]==0])
        coords_B = np.array([coords[i] for i in range(len(coords)) if particle_types[i]==1])
        return coords_A, coords_B
    
    def split_arrays_by_molecule_arms(coords_B):
        # for arms of branched BCP
        split_coords = []
        subarray_size = NCP_B
        sub_subarray_size = int(NCP_B/n_arms)
        for i in range(0, len(coords_B), subarray_size):
            subarray = coords_B[i:i + subarray_size]
            sub_subarrays = [subarray[j:j + sub_subarray_size] for j in range(0, len(subarray), sub_subarray_size)]
            split_coords.append(sub_subarrays)
        return split_coords
    
    def split_arrays_by_molecule_backbone(coords_A):
        # for backbone of branched BCP
        split_coords = []
        subarray_size = NCP_A
        for i in range(0, len(coords_A), subarray_size):
            subarray = coords_A[i:i + subarray_size]
            split_coords.append(subarray)
        return split_coords
    
    def mean_squared_Rg_A(split_coords):
        temp1 = []
        for i, subarray in enumerate(split_coords): # loop over each molecule (BCP)
            mean_backbone = np.mean(np.array(subarray)) # i coordinate of the COM of the backbone
            sq_dist_mean_backbone = np.mean((np.array(subarray) - mean_backbone) ** 2) # one value for one BCP
            temp1.append(sq_dist_mean_backbone)
        return temp1 # len(temp1) should be = #BCPs
    
    def mean_squared_Rg_B(split_coords):
        temp1 = []
        for i, subarray in enumerate(split_coords): # loop over each molecule (BCP)
            temp2 = []
            for j, sub_subarray in enumerate(subarray): # loop over each arm of BCP
                mean_single_arm = np.mean(np.array(sub_subarray)) # i coordinate of the COM of the arm
                sq_dist_mean_single_arm = np.mean((np.array(sub_subarray) - mean_single_arm) ** 2)
                temp2.append(sq_dist_mean_single_arm)
            sq_dist_mean_all_arms = np.mean(np.array(temp2)) # one value for one BCP
            temp1.append(sq_dist_mean_all_arms)
        return temp1 # len(temp1) should be = #BCPs
    
    x_coords_A, x_coords_B = split_arrays_by_monomer(x_coords)
    y_coords_A, y_coords_B = split_arrays_by_monomer(y_coords)
    z_coords_A, z_coords_B = split_arrays_by_monomer(z_coords)

    x_coords_A_split = split_arrays_by_molecule_backbone(x_coords_A)
    y_coords_A_split = split_arrays_by_molecule_backbone(y_coords_A)
    z_coords_A_split = split_arrays_by_molecule_backbone(z_coords_A)
    x_coords_B_split = split_arrays_by_molecule_arms(x_coords_B)
    y_coords_B_split = split_arrays_by_molecule_arms(y_coords_B)
    z_coords_B_split = split_arrays_by_molecule_arms(z_coords_B)

    mean_sq_Rg_A_x = mean_squared_Rg_A(x_coords_A_split)
    mean_sq_Rg_A_y = mean_squared_Rg_A(y_coords_A_split)
    mean_sq_Rg_A_z = mean_squared_Rg_A(z_coords_A_split)
    mean_sq_Rg_B_x = mean_squared_Rg_B(x_coords_B_split)
    mean_sq_Rg_B_y = mean_squared_Rg_B(y_coords_B_split)
    mean_sq_Rg_B_z = mean_squared_Rg_B(z_coords_B_split)
        
    #RgSquared_A = [mean_sq_Rg_A_x, mean_sq_Rg_A_y, mean_sq_Rg_A_z]
    #RgSquared_B = [mean_sq_Rg_B_x, mean_sq_Rg_B_y, mean_sq_Rg_B_z]
    
    mean_RgSquared_A_x = np.mean(np.array(mean_sq_Rg_A_x))
    mean_RgSquared_A_y = np.mean(np.array(mean_sq_Rg_A_y))
    mean_RgSquared_A_z = np.mean(np.array(mean_sq_Rg_A_z))
    mean_RgSquared_B_x = np.mean(np.array(mean_sq_Rg_B_x))
    mean_RgSquared_B_y = np.mean(np.array(mean_sq_Rg_B_y))
    mean_RgSquared_B_z = np.mean(np.array(mean_sq_Rg_B_z))
    
    std_RgSquared_A_x = np.std(np.array(mean_sq_Rg_A_x))/np.sqrt(M_CP)
    std_RgSquared_A_y = np.std(np.array(mean_sq_Rg_A_y))/np.sqrt(M_CP)
    std_RgSquared_A_z = np.std(np.array(mean_sq_Rg_A_z))/np.sqrt(M_CP)
    std_RgSquared_B_x = np.std(np.array(mean_sq_Rg_B_x))/np.sqrt(M_CP)
    std_RgSquared_B_y = np.std(np.array(mean_sq_Rg_B_y))/np.sqrt(M_CP)
    std_RgSquared_B_z = np.std(np.array(mean_sq_Rg_B_z))/np.sqrt(M_CP) 

    mean_RgSquared_A = [mean_RgSquared_A_x, mean_RgSquared_A_y, mean_RgSquared_A_z]
    mean_RgSquared_B = [mean_RgSquared_B_x, mean_RgSquared_B_y, mean_RgSquared_B_z]
    mean_RgSquared_A = np.array(mean_RgSquared_A, dtype=np.float64).tolist()
    mean_RgSquared_B = np.array(mean_RgSquared_B, dtype=np.float64).tolist()

    std_RgSquared_A = [std_RgSquared_A_x, std_RgSquared_A_y, std_RgSquared_A_z]
    std_RgSquared_B = [std_RgSquared_B_x, std_RgSquared_B_y, std_RgSquared_B_z]

    return mean_RgSquared_A, mean_RgSquared_B, std_RgSquared_A, std_RgSquared_B

def meanSqDistanceFromJunction(coord, blocks, junctions, box):
    

    '''
    Args:
        coord (np.ndarray):             Nx3 array for the coordinates of N particles
        blocks (List[List[int]]):       list of indices of particles in each block, ordered from junction to "end" (might be middle for a midblock)
        junctions (np.ndarray):         Nbx3 array with the coordinate of the terminal junction for each of the Nb blocks or block segments
        box (freud.box.Box):            
    Returns:
        n (np.ndarray):         1 x max(molecule lengths)-1 containing the corresponding distance from the block junction. 
                                starts at "0" which is actually a half segment away from the juntion
        avgRsq (np.ndarray):    1 x max(molecule lengths)-1 array containing average internal distances 
                                along the chains. Entry i corresponds with segments of length i+0.5
    
    NOTE that all inputted blocks should be exactly the same. 
    This is because results will be averaged together and returned in a single array

    '''
    # find block length and initialize arrays
    blockSize = len(blocks[0])
    for block in blocks:
        if len(block) != blockSize:
            raise ValueError("Inputted blocks are not the same length and thus probably shouldn't be treated as identical.")
    coordWithJunction = coord
    points1 = []
    points2 = []
    segmentlength = []
    for idx_block,block in enumerate(blocks):
        coordWithJunction = np.append(coordWithJunction, junctions[[idx_block],:],axis=0) # can do this cleaner for sure
        idx_junction = coordWithJunction.shape[0]-1 #index of last coordinate, which is the junction that was just appended
        
        # loop over coordinates in the block
        idx_next_to_junction = block[0]
        for idx_segment in block:
            points1.append(idx_junction)
            points2.append(idx_segment)
            segmentlength.append(abs(idx_segment-idx_next_to_junction) + 0.5) # corrected with +0.5 because junction is 0.5 away from last block

    # use box object to compute distances
    distances = box.compute_distances(coordWithJunction[points1], coordWithJunction[points2])        

    # sum up the squared segment distances
    distancesSquared = np.square(distances)
    avgRsq = np.zeros(blockSize)
    count = np.zeros(blockSize)
    for dsq, length in zip(distancesSquared,segmentlength):
        idx = int(length-0.5)
        avgRsq[idx] += dsq
        count[idx] += 1

    # compute average
    avgRsq = avgRsq/count
    n = np.arange(0,len(avgRsq))+0.5

    return avgRsq, n