import os
import sysconfig
import ctypes as ct
import numpy as np

suffix = sysconfig.get_config_var('EXT_SUFFIX')
if suffix is None: suffix = ".so"

rdf_dir = os.path.dirname(__file__)
rdf_name = "lib_rdf" + suffix
librdf = os.path.abspath(os.path.join(rdf_dir, rdf_name))
lib_rdf = ct.CDLL(librdf)

class gofr:
    """
    radial distribution function, g(r)
    """

class monoatomic(gofr):
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

    pbc : boolean
        True if pbc must be considered
        False if not
    """

    def __init__(self, natoms, box_size, nbin, pbc=True):

        box_size = box_size.astype(np.float32)

        self.natoms = natoms
        self.nbin = nbin
        self.pbc = 1 if pbc else 0
        
        minbox = np.min(box_size)
        self.volume = 0.0
        self.dg = 0.5 * minbox / self.nbin
        self.gr = np.zeros(self.nbin, dtype=np.float32)
        self.ngr = 0
        
        self.rdf_c = lib_rdf.monoatomic
        self.rdf_c.argtypes = [ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_int,
                               ct.c_float, ct.c_int, ct.c_void_p]
        self.gr_C = (ct.c_int * nbin)()


    def accumulate(self, box_size, positions):
        """
        accumulates the information of each frame in self.gr

        Parameters
        ----------
        box_size : numpy array with three floats
            the box size in x, y, z
        
        positions : numpy array with float32 data
            the positions in the SoA convention
            i.e. first all the x, then y and then z
        """
        
        # got to be sure that the box_size and positions type is np.float32 
        # because that is the pointer type in C
        box_size = box_size.astype(np.float32)
        self.volume += np.prod(box_size)
        box_size = box_size.ctypes.data_as(ct.POINTER(ct.c_void_p))
        
        positions = positions.astype(np.float32)
        x_C = positions.ctypes.data_as(ct.POINTER(ct.c_void_p))

        self.rdf_c(self.natoms, box_size, x_C, self.pbc, self.dg, self.nbin,
                   self.gr_C)

        self.ngr += 1


    def end(self, r_mean=None, writes=True, file_rdf='rdf.dat'):
        """
        Parameters
        ----------
        r_mean : float
            the mean radius of the simulated cluster (mainly oriented to spherical
            nanoparticles)

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
        V = self.volume / self.ngr
        if r_mean is not None: V = 4.0 * np.pi * np.power(r_mean, 3) / 3.0
        rho = self.natoms / V

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
            for i in range(self.nbin):
                file_rdf.write('{:.4e}\t{:.6e}\n'.format(r[i], self.gr[i]))
            file_rdf.close()

        return r, self.gr 


class diatomic(gofr):
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
    
    pbc : boolean
        True if pbc must be considered
        False if not
    """
    
    def __init__(self, natoms, box_size, nbin, atom_type_a, atom_type_b,
                 pbc=True):
    
        self.natoms = natoms
        self.nbin = nbin
        self.atom_type_a = atom_type_a
        self.atom_type_b = atom_type_b
        self.pbc = 1 if pbc else 0

        minbox = np.min(box_size)
        self.volume = 0.0
        self.gr = np.zeros(self.nbin, dtype=np.float32)
        self.dg = 0.5 * minbox / self.nbin
        self.ngr = 0

        self.rdf_c = lib_rdf.diatomic
        self.rdf_c.argtypes = [ct.c_int, ct.c_void_p, ct.c_void_p,
                               ct.c_int, ct.c_int, ct.c_void_p, ct.c_int,
                               ct.c_float, ct.c_int, ct.c_void_p]
        self.gr_C = (ct.c_int * nbin)()
    

    def accumulate(self, box_size, atom_type, positions):
        """
        accumulates the information of each frame in self.gr

        Parameters
        ----------
        box_size : numpy array with three floats
            the box size in x, y, z
        
        atom_type : numpy array with integers (could be char)
            type of atoms

        positions : numpy array with float32 data
            the positions in the SoA convention
            i.e. first all the x, then y and then z
        """
        
        # got to be sure that the box_size and the positions types are np.float32
        # and atom_type is an array of np.intc because those are the pointers 
        # types in C
        box_size = box_size.astype(np.float32)
        self.volume += np.prod(box_size)
        box_size = box_size.ctypes.data_as(ct.POINTER(ct.c_void_p))
        
        atom_type = atom_type.astype(np.intc)
        atom_C = atom_type.ctypes.data_as(ct.POINTER(ct.c_void_p))

        positions = positions.astype(np.float32)
        x_C = positions.ctypes.data_as(ct.POINTER(ct.c_void_p))

        self.rdf_c(self.natoms, box_size, atom_C, self.atom_type_a, 
                   self.atom_type_b, x_C, self.pbc, self.dg, self.nbin, self.gr_C)
        
        self.ngr += 1


    def end(self, atom_type, r_mean=None, writes=True, file_rdf='rdf.dat'):
        """
        Parameters
        ----------
        atom_type : numpy array with integers (could be char)
            type of atoms
        
        r_mean : float
            the mean radius of the simulated cluster (mainly oriented to spherical
            nanoparticles)
        
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
        V = self.volume / self.ngr
        if r_mean is not None: vol = 4.0 * np.pi * np.power(r_mean, 3) / 3.0
        rho = N_a * N_b / V

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
            for i in range(self.nbin):
                file_rdf.write('{:.4e}\t{:.6e}\n'.format(r[i], self.gr[i]))
            file_rdf.close()

        return r, self.gr 
