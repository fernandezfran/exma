import numpy as np

class msd:
    """
    mean square displacement

    remember that trajectories must be sorted with the same order as reference
    positions (no problem with .xyz files, but with .lammpstrj file a np.sort /
    np.argsort must be used before the calculation)
    """

class monoatomic(msd):
    """
    msd of a monoatomic system
    """
    
    def __init__(self, natoms, box_size, x_ref):
        """
        Parameters
        ----------
        natoms : integer
            the number of atoms in the frame
       
        box_size : numpy array
            with the box lenght in x, y, z
        
        x_ref : numpy array with float32 data
            the reference positions in the SoA convention
            i.e. first all the x, then y and then z
        """
        self.natoms = natoms
        self.box_size = box_size
        self.x_ref = x_ref

        self.frame = 0

    
    def wrapped(self, positions, image):
        """
        to use if the trajectory is wrapped inside the simulation box and you
        have the image of each particle in the different directions

        Parameters
        ----------
        positions : numpy array with float32 data
            the positions in the SoA convention
            i.e. first all the x, then y and then z
        
        image : numpy array with integer data
            same as positions

        Returns
        -------
        numpy array : floats
            [0]: frame
            [1]: msd
        """

        msd = 0.0
        for i in range(0, self.natoms):
            xx = np.zeros(3)
            for j in range(0, 3):
                xx[j] = positions[j*self.natoms + i] \
                      + image[j*self.natoms + i]*self.box_size[j] \
                      - self.x_ref[j*self.natoms + i]

            rr = np.linalg.norm(xx)
            r2 = rr * rr

            msd += r2
        
        msd /= self.natoms

        self.frame += 1
        return np.array([self.frame, msd], dtype=np.float32)

    
    def unwrapped(self, positions):
        """
        to use if the trajectory is unwrapped outside of the simulation box
        
        Parameters
        ----------
        positions : numpy array with float32 data
            the positions in the SoA convention
            i.e. first all the x, then y and then z

        Returns
        -------
        numpy array : floats
            [0]: frame
            [1]: msd
        """

        msd = 0.0
        for i in range(0, self.natoms):
            xx = np.zeros(3)
            for j in range(0, 3):
                xx[j] = positions[j*self.natoms + i] \
                      - self.x_ref[j*self.natoms + i]

            rr = np.linalg.norm(xx)
            r2 = rr * rr

            msd += r2
        
        msd /= self.natoms

        self.frame += 1
        return np.array([self.frame, msd], dtype=np.float32)


class diatomic(msd):
    """
    msd of a diatomic system
    """

    def __init__(self, natoms, box_size, atom_type, x_ref, atom_type_a,
            atom_type_b):
        """
        Parameters
        ----------
        natoms : integer
            the number of atoms in the frame
       
        box_size : numpy array
            with the box lenght in x, y, z
        
        atom_type : list of integers
            the type of the atoms
        
        x_ref : numpy array with float32 data
            the reference positions in the SoA convention
            i.e. first all the x, then y and then z
        
        atom_type_a : integer
            one type of atom

        atom_type_a : integer
            another type of atom
        """
        self.natoms = natoms
        self.box_size = box_size
        self.x_ref = x_ref
        self.atom_type_a = atom_type_a
        self.atom_type_b = atom_type_b

        self.frame = 0
        self.N_a = np.count_nonzero(atom_type == atom_type_a)
        self.N_b = np.count_nonzero(atom_type == atom_type_b)


    def wrapped(self, atom_type, positions, image):
        """
        to use if the trajectory is wrapped inside the simulation box and you
        have the image of each particle in the different directions

        Parameters
        ----------
        atom_type : list of integers
            the type of the atoms
        
        positions : numpy array with float32 data
            the positions in the SoA convention
            i.e. first all the x, then y and then z
        
        image : numpy array with integer data
            same as positions

        Returns
        -------
        numpy array : floats
            [0]: frame
            [1]: msd of atom type a
            [2]: msd of atom type b
            [3]: total msd
        """

        msd_a, msd_b, msd_t = 0.0, 0.0, 0.0
        for i in range(0, self.natoms):
            xx = np.zeros(3)
            for j in range(0, 3):
                xx[j] = positions[j*self.natoms + i] \
                      + image[j*self.natoms + i]*self.box_size[j] \
                      - self.x_ref[j*self.natoms + i]

            rr = np.linalg.norm(xx)
            r2 = rr * rr

            msd_t += r2
            if (atom_type[i] == self.atom_type_a):
                msd_a += r2
            else: # i.e.: atom_type[i] == self.atom_type_b
                msd_b += r2
        
        msd_t /= self.natoms
        msd_a /= self.N_a
        msd_b /= self.N_b

        self.frame += 1
        return np.array([self.frame, msd_a, msd_b, msd_t], dtype=np.float32)


    def unwrapped(self, atom_type, positions):
        """
        to use if the trajectory is unwrapped outside of the simulation box
        
        Parameters
        ----------
        atom_type : list of integers
            the type of the atoms
        
        positions : numpy array with float32 data
            the positions in the SoA convention
            i.e. first all the x, then y and then z

        Returns
        -------
        numpy array : floats
            [0]: frame
            [1]: msd of atom type a
            [2]: msd of atom type b
            [3]: total msd
        """

        msd_a, msd_b, msd_t = 0.0, 0.0, 0.0
        for i in range(0, self.natoms):
            xx = np.zeros(3)
            for j in range(0, 3):
                xx[j] = positions[j*self.natoms + i] \
                      - self.x_ref[j*self.natoms + i]

            rr = np.linalg.norm(xx)
            r2 = rr * rr

            msd_t += r2
            if (atom_type[i] == self.atom_type_a):
                msd_a += r2
            else: # i.e.: atom_type[i] == self.atom_type_b
                msd_b += r2
        
        msd_t /= self.natoms
        msd_a /= self.N_a
        msd_b /= self.N_b

        self.frame += 1
        return np.array([self.frame, msd_a, msd_b, msd_t], dtype=np.float32)
