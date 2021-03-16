import os
import ctypes as ct
import numpy as np

rdf_dir = os.path.dirname(__file__)
librdf = os.path.abspath(os.path.join(rdf_dir, "lib_rdf.so"))
lib_rdf = ct.CDLL(librdf)

class rdf:
    """
    radial distribution function
    """

class monoatomic(rdf):
    """
    rdf of a monoatomic system 
    
    Parameters
    ----------
    natoms : integer
        number of atoms

    box_size : numpy array with three floats
        the box size in x, y, z

    nbin : integer
        number of bins in the histogram
    """

    def __init__(self, natoms, box_size, nbin):

        box_size = box_size.astype(np.float32)

        self.natoms = natoms
        self.box_size = box_size.ctypes.data_as(ct.POINTER(ct.c_void_p))
        self.nbin = nbin
        
        minbox = np.min(box_size)
        self.volume = np.prod(box_size)
        self.dg = 0.5 * minbox / self.nbin
        self.gr = np.zeros(self.nbin, dtype=np.float32)
        self.ngr = 0
        
        self.rdf_c = lib_rdf.monoatomic
        self.rdf_c.argtypes = [ct.c_int, ct.c_void_p, ct.c_void_p,
                               ct.c_float, ct.c_int, ct.c_void_p]
        self.gr_C = (ct.c_int * nbin)()


    def accumulate(self, positions):
        """
        accumulates the information of each frame in self.gr

        Parameters
        ----------
        positions : numpy array with float32 data
            the positions in the SoA convention
            i.e. first all the x, then y and then z
        """
        #  ------------------------------------------------
        # |The C lib works like the previous python module|
        # ------------------------------------------------
        #for i in range(0, self.natoms - 1):
        #
        #    ri = [positions[i + k*self.natoms] for k in range(0,3)]
        #
        #    for j in range(i+1, self.natoms):
        #       
        #        rj = [positions[j + k*self.natoms] for k in range(0,3)]
        #        rij = self.bound.minimum_image(ri, rj)
        #        rij = np.linalg.norm(rij)
        #
        #        if (rij >= 0.5 * self.minbox): continue
        #       
        #        ig = np.intc(rij / self.dg)
        #        self.gr[ig] += 2
        
        # got to be sure that the positions type is np.float32 because that is
        # the pointer type in C
        positions = positions.astype(np.float32)
        x_C = positions.ctypes.data_as(ct.POINTER(ct.c_void_p))

        self.rdf_c(self.natoms, self.box_size, x_C, self.dg, self.nbin, self.gr_C)

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
        rho = self.natoms / self.volume

        r = np.zeros(self.nbin)
        gofr = np.asarray(np.frombuffer(self.gr_C,dtype=np.intc,count=self.nbin))
        for i in range(self.nbin):
            vb = (np.power(i+1,3) - np.power(i,3)) * np.power(self.dg,3)
            nid = 4.0 * np.pi * vb * rho / 3.0
            
            r[i] = (i + 0.5) * self.dg
            self.gr[i] = np.float32(gofr[i]) / (self.natoms * self.ngr * nid)

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
    
    def __init__(self, natoms, box_size, nbin, atom_type_a, atom_type_b):
    
        box_size = box_size.astype(np.float32)

        self.natoms = natoms
        self.box_size = box_size.ctypes.data_as(ct.POINTER(ct.c_void_p))
        self.nbin = nbin
        self.atom_type_a = atom_type_a
        self.atom_type_b = atom_type_b

        minbox = np.min(box_size)
        self.volume = np.prod(box_size)
        self.gr = np.zeros(self.nbin, dtype=np.float32)
        self.dg = 0.5 * minbox / self.nbin
        self.ngr = 0

        self.rdf_c = lib_rdf.diatomic
        self.rdf_c.argtypes = [ct.c_int, ct.c_void_p, ct.c_void_p,
                               ct.c_int, ct.c_int, ct.c_void_p,
                               ct.c_float, ct.c_int, ct.c_void_p]
        self.gr_C = (ct.c_int * nbin)()
    

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

        #  ------------------------------------------------
        # |The C lib works like the previous python module|
        # ------------------------------------------------
        #for i in range(0, self.natoms):
        #
        #    if (atom_type[i] != self.atom_type_a): continue
        #    ri = [positions[i + k*self.natoms] for k in range(0,3)]
        #
        #    for j in range(0, self.natoms):
        #        
        #        if (j == i): continue
        #        
        #        if (atom_type[j] != self.atom_type_b): continue
        #        rj = [positions[j + k*self.natoms] for k in range(0,3)]
        #
        #        rij = self.bound.minimum_image(ri, rj)
        #        rij = np.linalg.norm(rij)
        #
        #        if (rij < 0.5 * self.minbox):
        #            ig = np.intc(rij / self.dg)
        #            self.gr[ig] += 1
        
        # got to be sure that the positions type is np.float32 and atom_type is
        # an array of np.intc because those are the pointers types in C
        atom_type = atom_type.astype(np.intc)
        atom_C = atom_type.ctypes.data_as(ct.POINTER(ct.c_void_p))

        positions = positions.astype(np.float32)
        x_C = positions.ctypes.data_as(ct.POINTER(ct.c_void_p))

        self.rdf_c(self.natoms, self.box_size, atom_C, self.atom_type_a, 
                   self.atom_type_b, x_C, self.dg, self.nbin, self.gr_C)
        
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
        
        N_a = np.count_nonzero(atom_type == self.atom_type_a)
        N_b = np.count_nonzero(atom_type == self.atom_type_b)
        rho = N_a * N_b / self.volume

        gofr = np.asarray(np.frombuffer(self.gr_C,dtype=np.intc,count=self.nbin))
        r = np.zeros(self.nbin)
        for i in range(0, self.nbin):
            vb = (np.power(i+1,3) - np.power(i,3)) * np.power(self.dg,3)
            nid = 4.0 * np.pi * vb * rho / 3.0
            
            r[i] = (i + 0.5) * self.dg
            self.gr[i] = np.float32(gofr[i]) / (self.ngr * nid)

        if (writes == True):
            file_rdf = open(file_rdf, 'w')
            file_rdf.write("# r, g(r)\n")
            for i in range(0, self.nbin):
                file_rdf.write("%g %g\n" % (r[i], self.gr[i]))
            file_rdf.close()

        return r, self.gr 
