import numpy as np
from abc import ABCMeta, abstractmethod
from collections import Counter

class Species(metaclass=ABCMeta):

    isPolymer: bool

    # (potentially) non-unique label for molecule 
    @property
    @abstractmethod
    def label(self):
        pass

    # total number of atoms in molecule, length name inspired by linear polymers
    @property
    @abstractmethod
    def length(self):
        pass

    @property
    @abstractmethod
    def particletypes(self):
        pass

class MonomerSpec:

    def __init__(self, label, l, uniqueid):
        self.l = l
        self.label = label
        self._uniqueid = uniqueid

    @property
    def uniqueid(self):
        return self._uniqueid

class BlockSpec:

    def __init__(self):
        return
    
    @property
    def monomer(self):
        return self._monomer

    @monomer.setter
    def monomer(self,mnr):
        self._monomer = mnr
        return
    
    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self._length = value
        return

    @property
    def particletypes(self):
        return [self.monomer.label for i in range(self.length)]

class LinearPolymerSpec(Species):

    def __init__(self, monomers, lengths, shape):
        self.isPolymer = True
        self.nBlocks = len(monomers)
        self.shape = shape
        for i in range(self.nBlocks):
            self._blocks[i].monomer = monomers[i]
            self._blocks[i].length = lengths[i]

    @classmethod
    def linear(cls, monomers, lengths):
        shape = 'star'
        poly = cls(monomers, lengths, shape)
        return poly

    @property
    def shape(self):
        return self._shape
    
    @shape.setter
    def shape(self, value):
        self._shape = value
        return    
    
    @property
    def nBlocks(self):
        return self._nBlocks
    
    @nBlocks.setter
    def nBlocks(self, value):
        self._nBlocks = value
        self._blocks = [BlockSpec() for i in range(value)]
        return    
    
    @property
    def blocks(self):
        return self._blocks
    
    @property
    def length(self):
        return np.sum([block.length for block in self.blocks])

    @property
    def label(self):
        return ''.join([block.monomer.label for block in self.blocks])

    @property
    def bonds(self):
        # this is specific to a linear polymer
        # all the bonds, assuming that the first particle is indexed at 0 
        bonds = []
        Ntot = 0
        for idxblock,block in enumerate(self.blocks):
            # within the block
            for i in range(1,block.length):
                bonds.append([Ntot + (i-1), Ntot + i])
            # connect the blocks
            if idxblock < (self.nBlocks-1):
                bonds.append([Ntot + block.length - 1, Ntot + block.length])
            # chain length so far
            Ntot += block.length
        
        return bonds

    @property
    def bondtypes(self):
        # this is specific to a linear polymer
        # all the bond types
        bondtypes = []
        for idxblock, block in enumerate(self.blocks):
            # within the block
            for i in range(1,block.length):
                bondtypes.append('{:s}-{:s}'.format(block.monomer.label, block.monomer.label))
            # connect the blocks
            if idxblock < (self.nBlocks-1):
                uniqueids = [block.monomer.uniqueid, self.blocks[idxblock+1].monomer.uniqueid]
                labels = [block.monomer.label, self.blocks[idxblock+1].monomer.label]
                minID = np.argmin(uniqueids)
                maxID = np.argmax(uniqueids)
                bondtypes.append('{:s}-{:s}'.format(labels[minID], labels[maxID]))

        return bondtypes

    @property
    def particletypes(self):
        types = []
        for block in self.blocks:
            types += block.particletypes
        return types

class BranchedPolymerSpec(Species):

    def __init__(self, monomers, lengths, vertexID0, vertexID1, vertex, shape):
        self.isPolymer = True
        self.nBlocks = len(monomers)
        self.vertexID0 = vertexID0
        self.vertexID1 = vertexID1
        self.lengths = lengths
        self.vertex = vertex
        self.shape = shape
        self.monomers = monomers

        for i in range(self.nBlocks):
            self._blocks[i].monomer = monomers[i]
            self._blocks[i].length = lengths[i]
        return
    

    @classmethod
    def star(cls, monomers, lengths, vertex):
        shape = 'star'
        vertexID0 = list(range(len(monomers)))
        vertexID1 = [len(monomers)]*len(monomers)
        poly = cls(monomers, lengths, vertexID0, vertexID1, vertex, shape)
        return poly

    @classmethod
    def mikto_arm(cls, chain_monomer, chain_length, arm_monomer, total_arm_length, n_arms, vertex):
        shape = 'star'
        vertexID0 = []
        vertexID1 = []
        monomers = []
        lengths = []
        monomers.append(chain_monomer)
        lengths.append(chain_length)
        arm_length = int(total_arm_length/n_arms)
        for i in range(n_arms):
            monomers.append(arm_monomer)
            lengths.append(arm_length)
        vertexID0 = list(range(len(monomers)))
        vertexID1 = [len(monomers)]*len(monomers)
        poly = cls(monomers, lengths, vertexID0, vertexID1, vertex, shape)
        return poly

    @classmethod
    def customgraft(cls, monomers, lengths, vertex):
        shape = 'graft'
        # specify your custom topology here
        vertexID0 = [0,1,2,1,2] 
        vertexID1 = [1,2,3,4,5]
        poly = cls(monomers, lengths, vertexID0, vertexID1, vertex, shape)
        return poly
    
    @classmethod
    def regulargraft(cls, backbone, backbone_length, sidechain, total_sidechain_length, n_sidechain, vertex):
        # backbone and sidechain are monomer objects A and B
        shape = 'graft'
        sidechain_length = int(total_sidechain_length/n_sidechain)
        n_backbone_blocks = n_sidechain + 1
        n_junctions = n_sidechain
        backbone_block_length = int((backbone_length - n_junctions)/n_backbone_blocks)
        monomers = []
        lengths = []
        for i in range(n_backbone_blocks-1):
            monomers.append(backbone)
            lengths.append(backbone_block_length)
        monomers.append(backbone)
        lengths.append(backbone_length - (n_backbone_blocks-1)*backbone_block_length - n_sidechain)

        for i in range(n_sidechain):
            monomers.append(sidechain)
            lengths.append(sidechain_length)
        vertexID0 = []
        vertexID1 = []
        for i in range(n_backbone_blocks):
            vertexID0.append(i)
            vertexID1.append(i+1)
        for i in range(n_sidechain):
            vertexID0.append(i+1)
            vertexID1.append(n_backbone_blocks+i+1)
        poly = cls(monomers, lengths, vertexID0, vertexID1, vertex, shape)
        return poly
    
    @property
    def length(self):
        return np.sum([block.length for block in self.blocks]) + len(self.junction_nodes)

    @property
    def shape(self):
        return self._shape
    
    @shape.setter
    def shape(self, value):
        self._shape = value
        return    
        
    @property
    def nBlocks(self):
        return self._nBlocks
        
    @nBlocks.setter
    def nBlocks(self, value):
        self._nBlocks = value
        self._blocks = [BlockSpec() for i in range(value)]
        return    

    @property
    def blocks(self):
        return self._blocks
    
    @property
    def label(self):
        return ''.join([block.monomer.label for block in self.blocks]) # labels of the elements in each block

    @property
    def particletypes(self):
        types = []
        for block in self.blocks:
            types += block.particletypes
        for i in range(len(self.junction_nodes)):
            types += self.vertex.label
        return types
    
    @property
    def n_backbone_blocks(self):
        element_counts = Counter(self.monomers)
        element_counts_dict = dict(element_counts)
        first_element_key = list(element_counts_dict.keys())[0]
        self._n_backbone_blocks = element_counts_dict[first_element_key]
        return self._n_backbone_blocks

    @property
    def n_graft_blocks(self):
        element_counts = Counter(self.monomers)
        element_counts_dict = dict(element_counts)
        second_element_key = list(element_counts_dict.keys())[1]
        self._n_graft_blocks = element_counts_dict[second_element_key]
        return self._n_graft_blocks

    @property
    def total_vertices(self):
        # total number of distinct vertices
        combined_array = self.vertexID0 + self.vertexID1
        self._total_vertices = len(set(combined_array))
        return self._total_vertices

    @property
    def vertices(self):
        self._vertices = []
        for i in range(self.total_vertices):
            self._vertices.append(self.vertex)
        return self._vertices

    @property
    def blocksID(self):
        self._blocksID = []
        for i in range(self.nBlocks):
            self._blocksID.append([self.vertexID0[i],self.vertexID1[i]])
        return self._blocksID

    @property
    def connectivity_count(self):
        # to how many vertices is each vertex connected?
        self._connectivity_count = []
        for vertex_index in range(self.total_vertices):
            combined_array = self.vertexID0 + self.vertexID1
            count_element = combined_array.count(vertex_index)
            self._connectivity_count.append(count_element)
        return self._connectivity_count

    @property
    def connectivity_list(self):
        # which vertices are connected to each vertex?
        self._connectivity_list = []
        for vertex_index in range(self.total_vertices):
            vertex_index_connectivity_list = []
            for i in range(self.nBlocks): # note, len(vertexID0) = len(vertexID0) = nBlocks
                if self.vertexID0[i] == vertex_index:
                    vertex_index_connectivity_list.append(self.vertexID1[i])
                elif self.vertexID1[i] == vertex_index:
                    vertex_index_connectivity_list.append(self.vertexID0[i])
            self._connectivity_list.append(vertex_index_connectivity_list)
        self._connectivity_list = [sorted(sublist) for sublist in self._connectivity_list]
        return self._connectivity_list

    @property
    def connectivity_at_start_list(self):
        self._connectivity_at_start_list = [] # list of all vertices connected to a vertex V, when V acts as the start of the block
        for vertex_index in range(self.total_vertices):
            vertex_index_connectivity_at_start_list = []
            for i in range(self.nBlocks): # note, len(vertexID0) = len(vertexID0) = nBlocks
                if self.vertexID0[i] == vertex_index:
                    vertex_index_connectivity_at_start_list.append(self.vertexID1[i])
            self._connectivity_at_start_list.append(vertex_index_connectivity_at_start_list)
        self._connectivity_at_start_list = [sorted(sublist) for sublist in self._connectivity_at_start_list]

        return self._connectivity_at_start_list
    
    @property
    def connectivity_at_end_list(self):
        self._connectivity_at_end_list = [] # list of all vertices connected to a vertex V, when V acts as the end of the block
        for vertex_index in range(self.total_vertices):
            vertex_index_connectivity_at_end_list = []
            for i in range(self.nBlocks): # note, len(vertexID0) = len(vertexID0) = nBlocks
                if self.vertexID1[i] == vertex_index:
                    vertex_index_connectivity_at_end_list.append(self.vertexID0[i])
            self._connectivity_at_end_list.append(vertex_index_connectivity_at_end_list)
        self._connectivity_at_end_list = [sorted(sublist) for sublist in self._connectivity_at_end_list]
        return self._connectivity_at_end_list

    @property
    def free_chain_ends(self):
        # which vertices are free chain ends?
        self._free_chain_ends = []
        for vertex_index, vertex_index_connectivity_list in enumerate(self.connectivity_list):
            if len(vertex_index_connectivity_list) == 1:
                self._free_chain_ends.append(vertex_index)
        self._free_chain_ends.sort()
        return self._free_chain_ends
    
    @property
    def junction_nodes(self):
        # which vertices are junction nodes?
        self._junction_nodes = []
        for vertex_index, vertex_index_connectivity_list in enumerate(self.connectivity_list):
            if len(vertex_index_connectivity_list) > 1:
                self._junction_nodes.append(vertex_index)
        self._junction_nodes.sort()
        return self._junction_nodes

    @property
    def bonds(self):

        # this is specific to a branched polymer
        # all the bonds, assuming that the first particle is indexed at 0 
        bonds = []
        Ntot = 0

        # this array store [0,l1,l1+l2,l1+l2+l3...] where li is the length of each block, these elements act as IDs for connecting bonds
        cumulative_sum_start = []
        cumulative_sum = 0
        for block_length in self.lengths:
            cumulative_sum_start.append(cumulative_sum)
            cumulative_sum += block_length

        # this array store [l1-1,l1+l2-1,l1+l2+l3-1...] where li is the length of each block, these elements act as IDs for connecting bonds
        cumulative_sum_end = []
        cumulative_sum = 0
        for block_length in self.lengths:
            cumulative_sum += block_length
            cumulative_sum_end.append(cumulative_sum-1)
        
        blocks_ID = self.blocksID

        # connect all bonds in the branched polymer
        for block in self.blocks:

            # within the block
            for i in range(1,block.length):
                bonds.append([Ntot + (i-1), Ntot + i])

            # chain length so far
            Ntot += block.length
        
        # iterate over all vertices that are not chain free ends, to connect blocks to vertices
        for junction_idx, junction_node in enumerate(self.junction_nodes):

            # get the list of all vertices connected to each junction node
            connected_vertices_list = self.connectivity_list[junction_node]
            
            for neighbour in connected_vertices_list:
                if any(element == neighbour for element in self.connectivity_at_start_list[junction_node]):
                    bonds.append([Ntot+junction_idx, cumulative_sum_start[neighbour-1]])
                elif any(element == neighbour for element in self.connectivity_at_end_list[junction_node]):
                    bonds.append([Ntot+junction_idx, cumulative_sum_end[neighbour]])

            '''
            # get the list of all vertices connected to each junction node when the junction node acts as the start of block
            for neighbour in self.connectivity_at_start_list[junction_node]:
                bonds.append([Ntot+junction_idx, cumulative_sum_start[neighbour-1]])

            # get the list of all vertices connected to each junction node when the junction node acts as the end of block
            for neighbour in self.connectivity_at_end_list[junction_node]:
                bonds.append([Ntot+junction_idx, cumulative_sum_end[neighbour-1]])
            '''

        return bonds

    @property
    def bondtypes(self):
        # this is specific to a branched polymer
        # all the bond types
        bondtypes = []
        for idxblock, block in enumerate(self.blocks):
            # within the block
            for i in range(1,block.length):
                bondtypes.append('{:s}-{:s}'.format(block.monomer.label, block.monomer.label))
            # connect the blocks
            
        blocks_ID = self.blocksID
        # iterate over all vertices that are not chain free ends, to define the bond types of junction nodes' bonds
        for junction_idx, junction_node in enumerate(self.junction_nodes):

            '''
            # fix this label
            # get the list of all vertices connected to each junction node when the junction node acts as the start of block
            for neighbour in self.connectivity_at_start_list[junction_node]:
                bondtypes.append('{:s}-{:s}'.format(self.vertex.label, self.vertex.label))

            # get the list of all vertices connected to each junction node when the junction node acts as the end of block
            for neighbour in self.connectivity_at_end_list[junction_node]:
                bondtypes.append('{:s}-{:s}'.format(self.vertex.label, self.vertex.label))
                
            # get the list of all vertices connected to each junction node
            connected_vertices_list = self.connectivity_list[junction_node]
            '''

            # get the list of all vertices connected to each junction node
            connected_vertices_list = self.connectivity_list[junction_node]
            
            for neighbour in connected_vertices_list:
                if any(element == neighbour for element in self.connectivity_at_start_list[junction_node]):
                    # find which element of block ID has first element == junction_node
                    for index,block_ID in enumerate(blocks_ID):
                        if block_ID[0] == junction_node and block_ID[1] == neighbour:
                            bondtypes.append('{:s}-{:s}'.format(self.vertex.label, self.blocks[index].monomer.label))
                elif any(element == neighbour for element in self.connectivity_at_end_list[junction_node]):
                    # find which element of block ID has second element == junction_node
                    for index,block_ID in enumerate(blocks_ID):
                        if block_ID[1] == junction_node and block_ID[0] == neighbour:
                            bondtypes.append('{:s}-{:s}'.format(self.vertex.label, self.blocks[index].monomer.label))

        return bondtypes

class MonatomicMoleculeSpec(Species):

    def __init__(self, monomer: MonomerSpec, shape):
        self.isPolymer = False
        self._monomer = monomer
        self.shape = shape
        return

    @classmethod
    def monoatomic(cls, monomer):
        shape = 'monoatomic'
        poly = cls(monomer, shape)
        return poly

    @property
    def label(self):
        return self._monomer.label

    @property
    def length(self):
        return int(1)

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value
        return
    
    @property
    def particletypes(self):
        types = [self.label]
        return types

class Component:

    def __init__(self, species: Species, N: int):
        self.species = species # uses setter! wow, fancy
        self.N = N
        return

    @property 
    def species(self):
        return self._species
    
    @species.setter
    def species(self, molspec):
        self._species = molspec
        return
    
    @property
    def label(self):
        return self._species.label

    @property
    def shape(self):
        return self._species.shape
    
    @property 
    def N(self):
        return self._N
    
    @N.setter
    def N(self, value):
        self._N = value
        return
    
    @property
    def numparticles(self):
        return self.N*self.species.length

    @property
    def particletypes(self):
        types = []
        for i in range(self.N):
            types += self.species.particletypes
        return types
    
class Box:

    def __init__(self, lengths, tilts=[0,0,0]):

        self.lengths = lengths
        self.tilts = tilts

        return

class System:

    def __init__(self):
        self._components = []
        self._componentlabels = []
        self._monomers = []
        self._monomerlabels = []
        return
    
    @property
    def box(self):
        return self._box.lengths + self._box.tilts
    
    @box.setter
    def box(self, size):
        if len(size) == 6:
            self._box = Box(size[0:3], size[3:])
        else:
            self._box = Box(size)
        return
    
    @property
    def nComponents(self):
        return len(self._components)

    @property
    def components(self):
        return self._components
    
    @property
    def componentlabels(self):
        return self._componentlabels
    
    def addComponent(self, species: Species, N: int):
        self.components.append(Component(species, N))
        self._componentlabels.append(species.label)
        return self.components[-1]
    
    # This should not be used because components can have identical labels! Labels are not unique.
    # def componentByLabel(self,label):
    #     return self.components[self._componentlabels.index(label)]
    
    @property
    def nMonomers(self):
        return len(self._monomers)

    @property
    def monomers(self):
        return self._monomers
    
    @property
    def monomerlabels(self):
        return self._monomerlabels
    
    def addMonomer(self, label, l):
        uniqueid = self.nMonomers
        self.monomers.append(MonomerSpec(label, l, uniqueid))
        self._monomerlabels.append(label)
        return self.monomers[-1]

    def monomerByLabel(self,label):
        return self.monomers[self._monomerlabels.index(label)]
    
    @property
    def numparticles(self):
        N = 0
        for component in self.components:
            N += component.numparticles
        return N
    
    # removed particleType(). will probably never be used and is too dependent on the type of species of the component!

    def particleTypes(self):
        # returns a list of all particle types in order
        # faster than running particleType for each particle
        # len(types) = number of particles

        types = []
        for component in self.components:
            types += component.particletypes

        return types
    
    def particleSpeciesTypes(self):
        # returns a list of the type of species a particle is a part of, for all particles
        # len(types) = number of particles

        types = []
        for component in self.components:
            for i in range(component.numparticles):
                types.append(component.label)
        
        return types
    
    def speciesTypes(self):
        # returns a list of all species types (defined by their labels)
        # len(types) = number of molecules
        types = []
        for component in self.components:
            for i in range(component.N):
                types.append(component.label)
        
        return types
    
    def indicesByMolecule(self):
        # returns a list of lists where each list corresponds with a given species
        # contains the indices of all particles in that polymer
        
        indices = []
        idx_current = 0
        for component in self.components:
            specieslength = component.species.length
            for i in range(component.N):
                molindices = list(range(idx_current,idx_current+specieslength))
                indices.append(molindices)
                idx_current += specieslength

        return indices

    def bonds(self):

        bonds = []
        bondtypes = []
        idx_start = 0
        for component in self.components:
            if not component.species.isPolymer: # assuming only polymers have bonds! 
                idx_start += component.numparticles
                continue
            for i in range(component.N):
                bonds += ( np.array(component.species.bonds) + idx_start ).tolist()
                bondtypes += component.species.bondtypes
                idx_start += component.species.length
        
        return bonds, bondtypes
    
    def junctions(self):
        junctions = []
        # junctiontypes = [] could add if needed in future for 3 monomer systems
        bonds,bondtypes = self.bonds()
        for bond,bondtype in zip(bonds,bondtypes):
            monomertypes = bondtype.split("-")
            if monomertypes[0] != monomertypes[1]:
                junctions.append(bond)
        return junctions


    
# Workflow:
# Make a system
# Set the box size
# Set the number of components
# For each component, set the number of that component and its molecule spec