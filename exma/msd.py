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
    
    Parameters
    ----------
    natoms : integer
        the number of atoms in the frame
    
    box_size : numpy array
        with the box lenght in x, y, z
    
    x_ref : numpy array with float32 data
        the reference positions in the SoA convention
        i.e. first all the x, then y and then z
        
    image_ref : numpy array with integer data
        reference image, same as positions
    """
    
    def __init__(self, natoms, box_size, x_ref, image_ref=None):

        self.natoms = natoms

        if image_ref is not None:
            x, y, z = np.split(x_ref, 3)
            ix, iy, iz = np.split(image_ref, 3)
            x += box_size[0] * ix
            y += box_size[1] * iy
            z += box_size[2] * iz
            x_ref = np.concatenate((x, y, z))
        
        self.ref = np.split(x_ref,3)
        self.frame = 0

    
    def wrapped(self, box_size, positions, image):
        """
        to use if the trajectory is wrapped inside the simulation box and you
        have the image of each particle in the different directions

        Parameters
        ----------
        box_size : numpy array
            with the box lenght in x, y, z
        
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
        positions = np.split(positions,3)
        image = np.split(image,3)
        MSD = np.zeros(self.natoms, dtype=np.float32)
        for i in range(3):
            xx = positions[i] + image[i]*box_size[i] - self.ref[i]
            MSD += xx * xx
        msd = np.sum(MSD) / self.natoms

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
        positions = np.split(positions,3)
        MSD = np.zeros(self.natoms, dtype=np.float32)
        for i in range(3):
            xx = positions[i] - self.ref[i]
            MSD += xx * xx
        msd = np.sum(MSD) / self.natoms

        self.frame += 1
        
        return np.array([self.frame, msd], dtype=np.float32)


class diatomic(msd):
    """
    msd of a diatomic system
    
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
    
    image_ref : numpy array with integer data
        reference image, same as positions
    """

    def __init__(self, natoms, box_size, atom_type, x_ref, atom_type_a, 
                 atom_type_b, image_ref=None):
  
        self.natoms = natoms
        
        if image_ref is not None:
            x, y, z = np.split(x_ref, 3)
            ix, iy, iz = np.split(image_ref, 3)
            x += box_size[0] * ix
            y += box_size[1] * iy
            z += box_size[2] * iz
            x_ref = np.concatenate((x, y, z))
        
        self.ref = np.split(x_ref,3)
        self.atom_type_a = atom_type_a
        self.atom_type_b = atom_type_b

        self.frame = 0
        self.N_a = np.count_nonzero(atom_type == atom_type_a)
        self.N_b = np.count_nonzero(atom_type == atom_type_b)


    def wrapped(self, box_size, atom_type, positions, image):
        """
        to use if the trajectory is wrapped inside the simulation box and you
        have the image of each particle in the different directions

        Parameters
        ----------
        box_size : numpy array
            with the box lenght in x, y, z
    
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
        
        positions = np.split(positions,3)
        image = np.split(image,3)
        MSD = np.zeros(self.natoms, dtype=np.float32)
        for i in range(3):
            xx = positions[i] + image[i]*box_size[i] - self.ref[i]
            MSD += xx * xx
        
        msd_t = np.sum(MSD) / self.natoms
        msd_a = np.sum(MSD[atom_type == self.atom_type_a]) / self.N_a
        msd_b = np.sum(MSD[atom_type == self.atom_type_b]) / self.N_b

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

        positions = np.split(positions,3)
        MSD = np.zeros(self.natoms, dtype=np.float32)
        for i in range(3):
            xx = positions[i] - self.ref[i]
            MSD += xx * xx
        
        msd_t = np.sum(MSD) / self.natoms
        msd_a = np.sum(MSD[atom_type == self.atom_type_a]) / self.N_a
        msd_b = np.sum(MSD[atom_type == self.atom_type_b]) / self.N_b

        self.frame += 1
        return np.array([self.frame, msd_a, msd_b, msd_t], dtype=np.float32)
