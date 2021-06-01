import numpy as np
import itertools as it

class atoms:
    """
    atoms module
    """


class positions(atoms):
    """
    define the positions of the atoms in an orthogonal lattice
    
    the density is defined by the parameters rho = natoms / (box_size^3)
    
    Parameters
    ----------
    natoms : int
        number of atoms

    box_size : float
        box size in each direction (cubic)
    """
    
    def __init__(self, natoms, box_size):

        self.natoms = natoms
        self.box_size = box_size


    def sc(self):
        """
        simple cubic

        Returns
        -------
        positions : numpy array
            of the atoms in an sc crystal
        """
        nside = np.cbrt(self.natoms, dtype=np.float32)
        tmp = np.intc(nside)
        if (nside % tmp != 0): raise ValueError(
            "Number of atoms must be a power of three"
        )

        s_range = range(int(nside))
        positions = list(it.product(s_range, repeat=3))

        positions = np.array(positions)
        positions = np.transpose(positions) * (self.box_size / nside)

        return np.ravel(positions)


    def bcc(self):
        """
        body-centered cubic 
        
        Returns
        -------
        positions : numpy array
            of the atoms in a bcc crystal
        """
        nside = np.cbrt(self.natoms / 2, dtype=np.float32)
        tmp = np.intc(nside)
        if (nside % tmp != 0): raise ValueError(
            "Number of atoms must be a power of three multiplied by two"
        )
        
        s_range = range(int(nside))
        p0 = list(it.product(s_range, repeat=3))

        # bcc lattice vectors: (0, 0, 0) and (0.5, 0.5, 0.5)
        p0 = np.array(p0)
        p1 = p0 + np.full((len(p0),3), [0.5, 0.5, 0.5])

        positions = np.concatenate((p0, p1))
        positions = np.transpose(positions) * (self.box_size / nside)

        return np.ravel(positions)


    def fcc(self):
        """
        face-centered cubic
        
        Returns
        -------
        positions : numpy array
            of the atoms in an fcc crystal
        """
        nside = np.cbrt(self.natoms / 4, dtype=np.float32)
        tmp = np.intc(nside)
        if (nside % tmp != 0): raise ValueError(
            "Number of atoms must be a power of three multiplied by four"
        )
        
        s_range = range(int(nside))
        p0 = list(it.product(s_range, repeat=3))

        # fcc lattice vectors: (0, 0, 0) (0.5, 0.5, 0) (0.5, 0, 0.5) (0, 0.5, 0.5)
        p0 = np.array(p0)
        p1 = p0 + np.full((len(p0),3), [0.5, 0.5, 0.0])
        p2 = p0 + np.full((len(p0),3), [0.5, 0.0, 0.5])
        p3 = p0 + np.full((len(p0),3), [0.0, 0.5, 0.5])

        positions = np.concatenate((p0, p1, p2, p3))
        positions = np.transpose(positions) * (self.box_size / nside)

        return np.ravel(positions)
    

    def dcc(self):
        """
        diamond cubic crystal
        
        Returns
        -------
        positions : numpy array
            of the atoms in an diamond cubic crystal
        """
        nside = np.cbrt(self.natoms / 8, dtype=np.float32)
        tmp = np.intc(nside)
        if (nside % tmp != 0): raise ValueError(
            "Number of atoms not valid"
        )
        
        s_range = range(int(nside))
        p0 = list(it.product(s_range, repeat=3))

        p0 = np.array(p0)
        p1 = p0 + np.full((len(p0),3), [0.25, 0.75, 0.25])
        p2 = p0 + np.full((len(p0),3), [0.00, 0.00, 0.50])
        p3 = p0 + np.full((len(p0),3), [0.25, 0.25, 0.75])
        p4 = p0 + np.full((len(p0),3), [0.00, 0.50, 0.00])
        p5 = p0 + np.full((len(p0),3), [0.75, 0.75, 0.75])
        p6 = p0 + np.full((len(p0),3), [0.50, 0.00, 0.00])
        p7 = p0 + np.full((len(p0),3), [0.75, 0.25, 0.25])
        p8 = p0 + np.full((len(p0),3), [0.50, 0.50, 0.50])

        positions = np.concatenate((p1, p2, p3, p4, p5, p6, p7, p8))
        positions = np.transpose(positions) * (self.box_size / nside)

        return np.ravel(positions)
