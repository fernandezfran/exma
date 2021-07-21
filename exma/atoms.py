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


class nanoparticle(atoms):
    """
    define a nanoparticle 
    
    Parameters
    ----------
    positions : array
        the positions of the atoms in a lattice that wants to be replicated

    box_size : array
        box size in each direction x, y, z
    """

    def __init__(self, positions, box_size):
        self.positions = positions
        self.box_size  = box_size


    def spherical(self, rcut):
        """
        spherical nanoparticle

        Parameters
        ----------
        rcut : float
            the radius of the nanoparticle
        
        Returns
        -------
        positions : numpy array
            of the atoms in the nanoparticle
        """
        x, y, z = np.split(self.positions, 3)

        n = np.intc(np.ceil(rcut / np.max(self.box_size)))
        boxes = list(it.product(range(-n,n+1), repeat=3))

        npx, npy, npz = [], [], []
        for box in boxes:
            for i in range(len(x)):
                xx = self.box_size[0] * (x[i] + box[0])
                yy = self.box_size[1] * (y[i] + box[1])
                zz = self.box_size[2] * (z[i] + box[2])

                if (np.linalg.norm([xx, yy, zz]) <= rcut):
                    npx.append(xx)
                    npy.append(yy)
                    npz.append(zz)

        positions = np.concatenate((npx, npy, npz))

        return positions


class replicate(atoms):
    """
    replicate a crystalline system in each direction

    Parameters
    ----------
    natoms : integer
        the number of atoms in the crystalographic structure
    
    box_size : numpy array
        with the box lenght in x, y, z

    atom_type : list of integers
        the type of the atoms

    positions : numpy array with float32 data
        the positions in the SoA convention (i.e. first all the x, then y and 
        then z) and in fractions of the box_size (i.e. numbers between 0 and 1)
    """

    def __init__(self, natoms, box_size, atom_type, positions):
        self.natoms = natoms
        self.atom_type = atom_type
        self.box_size = box_size
        self.positions = positions

    def crystal(self, nx, ny, nz):
        """
        n_i : integer >= 1
            replication factor in the i direction.
            n_i = 1 means that only the actual box is consired. 
        """
        x, y, z = np.split(self.positions, 3)
        boxes = list(it.product(range(np.max([nx, ny, nz])), repeat=3))
        newx, newy, newz = [], [], []
        for box in boxes:
            if (box[0] >= nx) or (box[1] >= ny) or (box[2] >= nz): continue

            for i in range(self.natoms):
                newx.append(self.box_size[0] * (x[i] + box[0]))
                newy.append(self.box_size[1] * (y[i] + box[1]))
                newz.append(self.box_size[2] * (z[i] + box[2]))

        N = len(newx)
        box_size = np.array([nx * self.box_size[0], ny * self.box_size[1],
                             nz * self.box_size[2]])
        atom_type = self.atom_type * nx * ny * nz
        positions = np.concatenate((newx, newy, newz))

        return N, box_size, atom_type, positions
