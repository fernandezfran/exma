import numpy as np
from . import boundary

class rdf:
    """
    radial distribution function
    
    this module is extremely slow and requires some functions to be written in C
    """

class monoatomic(rdf):
    """
    rdf of a monoatomic system 
    """

    def __init__(self, natoms, box_size, nbin):
        """
        Parameters
        ----------
        natoms : integer
            number of atoms

        box_size : numpy array with three floats
            the box size in x, y, z

        nbin : integer
            number of bins in the histogram
        """
        self.natoms = natoms
        self.box_size = box_size
        self.nbin = nbin
        
        self.minbox = np.min(self.box_size)
        self.gr = np.zeros(self.nbin)
        self.dg = 0.5 * self.minbox / self.nbin
        self.ngr = 0

        self.bound = boundary.apply(self.box_size)


    def accumulate(self, positions):
        """
        accumulates the information of each frame in self.gr

        Parameters
        ----------
        positions : numpy array with float32 data
            the positions in the SoA convention
            i.e. first all the x, then y and then z
        """

        for i in range(0, self.natoms - 1):

            ri = [positions[i + k*self.natoms] for k in range(0,3)]

            for j in range(i+1, self.natoms):
                
                rj = [positions[j + k*self.natoms] for k in range(0,3)]
                rij = self.bound.minimum_image(ri, rj)
                rij = np.linalg.norm(rij)

                if (rij >= 0.5 * self.minbox): continue
               
                ig = np.intc(rij / self.dg)
                self.gr[ig] += 2
        
        self.ngr += 1


    def end(self, writes=True, file_rdf='rdf.dat'):
        """
        Parameters
        ----------
        writes : True (or False)
            if you want (or don't want) to write an output

        file_rdf : filname
            the file were the g(r) is going to be written

        Returns
        -------
        r : numpy array
            x of the histogram

        self.gr : numpy array
            y of the histogram 
        """
        volume = np.prod(self.box_size)
        rho = self.natoms / volume

        r = np.zeros(self.nbin)
        for i in range(0, self.nbin):
            vb = (np.power(i+1,3) - np.power(i,3)) * np.power(self.dg,3)
            nid = 4.0 * np.pi * vb * rho / 3.0
            
            r[i] = (i + 0.5) * self.dg
            self.gr[i] /= (self.natoms * self.ngr * nid)

        if (writes == True):
            file_rdf = open(file_rdf, 'w')
            file_rdf.write("# r, g(r)\n")
            for i in range(0, self.nbin):
                file_rdf.write("%g %g\n" % (r[i], self.gr[i]))
            file_rdf.close()

        return r, self.gr 


class diatomic(rdf):
    """
    rdf of diatomic systems
    """
    
    def __init__(self, natoms, box_size, nbin, atom_type_a, atom_type_b):
        """
        Parameters
        ----------
        natoms : integer
            number of atoms

        box_size : numpy array with three floats
            the box size in x, y, z

        nbin : integer
            number of bins in the histogram
        
        atom_type_a : integer (or char)
            type of central atoms

        atom_type_a : integer (or char)
            type of interacting atoms
        """
        self.natoms = natoms
        self.box_size = box_size
        self.nbin = nbin
        self.atom_type_a = atom_type_a
        self.atom_type_b = atom_type_b

        self.minbox = np.min(self.box_size)
        self.gr = np.zeros(self.nbin)
        self.dg = 0.5 * self.minbox / self.nbin
        self.ngr = 0

        self.bound = boundary.apply(self.box_size)
    

    def accumulate(self, atom_type, positions):
        """
        accumulates the information of each frame in self.gr

        Parameters
        ----------
        atom_type : numpy array with integers (could be char)
            type of atoms

        positions : numpy array with float32 data
            the positions in the SoA convention
            i.e. first all the x, then y and then z
        """

        for i in range(0, self.natoms):

            if (atom_type[i] != self.atom_type_a): continue
            ri = [positions[i + k*self.natoms] for k in range(0,3)]

            for j in range(0, self.natoms):
                
                if (j == i): continue
                
                if (atom_type[j] != self.atom_type_b): continue
                rj = [positions[j + k*self.natoms] for k in range(0,3)]

                rij = self.bound.minimum_image(ri, rj)
                rij = np.linalg.norm(rij)

                if (rij < 0.5 * self.minbox):
                    ig = np.intc(rij / self.dg)
                    self.gr[ig] += 1
        
        self.ngr += 1


    def end(self, atom_type, writes=True, file_rdf='rdf.dat'):
        """
        Parameters
        ----------
        writes : True (or False)
            if you want (or don't want) to write an output

        file_rdf : filname
            the file were the g(r) is going to be written

        Returns
        -------
        r : numpy array
            x of the histogram

        self.gr : numpy array
            y of the histogram 
        """
        volume = np.prod(self.box_size)

        N_x = np.count_nonzero(atom_type == self.atom_type_a)
        rho_x = N_x / volume

        if (self.atom_type_a != self.atom_type_b): rho_x = (self.natoms - N_x) \
                                                           / volume

        r = np.zeros(self.nbin)
        for i in range(0, self.nbin):
            vb = (np.power(i+1,3) - np.power(i,3)) * np.power(self.dg,3)
            nid = 4.0 * np.pi * vb * rho_x / 3.0
            
            r[i] = (i + 0.5) * self.dg
            self.gr[i] /= (N_x * self.ngr * nid)

        if (writes == True):
            file_rdf = open(file_rdf, 'w')
            file_rdf.write("# r, g(r)\n")
            for i in range(0, self.nbin):
                file_rdf.write("%g %g\n" % (r[i], self.gr[i]))
            file_rdf.close()

        return r, self.gr 
