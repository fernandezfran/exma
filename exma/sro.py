import numpy as np
from . import boundary

class sro:
    """
    short range order
    """

class warren_cowley(sro):
    """
    warren cowley parameters
    """

    def __init__(self, natoms, box_size, atom_type, atom_type_a, atom_type_b, \
                 rcut):
        """
        Parameters
        ----------
        natoms : integer
            number of atoms
        
        box_size : numpy array with three floats
            the box size in x, y, z
        
        atom_type : numpy array with integers
            type of atoms
        
        atom_type_a : integer
            type of central atoms

        atom_type_b : integer
            type of interacting atoms

        rcut : float
            cut radius    
        """
        self.natoms = natoms
        self.box_size = box_size
        self.rcut = rcut
        self.atom_type_a = atom_type_a
        self.atom_type_b = atom_type_b

        self.N = np.zeros(2)
        self.m = np.zeros(2)
        self.alpha = np.zeros(2)

        self.N[0] = np.count_nonzero(atom_type == atom_type_a)
        self.N[1] = np.count_nonzero(atom_type == atom_type_b)
        self.m[0] = self.N[0] / natoms
        self.m[1] = self.N[1] / natoms
        self.nalpha = 0
        self.bound = boundary.apply(self.box_size)


    def accumulate(self, atom_type, positions):
        """
        accumulates the information of each frame in self.alpha

        Parameters
        ----------
        atom_type : numpy array with integers
            type of atoms

        positions : numpy array with float32 data
            the positions in the SoA convention
            i.e. first all the x, then y and then z
        """

        for i in range(0, self.natoms):

            ri = [positions[i + k*self.natoms] for k in range(0,3)]

            n_t, n_a, n_b = 0, 0, 0
            for j in range(0, self.natoms):

                if (i == j): continue

                rj = [positions[j + k*self.natoms] for k in range(0,3)]

                rij = self.bound.minimum_image(ri, rj)
                rij = np.linalg.norm(rij)

                if (rij >= self.rcut): continue

                n_t +=1
                if (atom_type[j] == self.atom_type_a):
                    n_a += 1
                elif (atom_type[j] == self.atom_type_b):
                    n_b += 1

            if (atom_type[i] == self.atom_type_a):
                self.alpha[0] += (1.0 - ( (n_a / n_t) / self.m[0] ))
            elif (atom_type[i] == self.atom_type_b):
                self.alpha[1] += (1.0 - ( (n_b / n_t) / self.m[1] ))
        
        self.nalpha += 1


    def end(self):
        """
        Returns
        -------
        self.alpha : numpy array
            two values, alpha_A and alpha_B
        """

        return np.divide(self.alpha, self.N) / self.nalpha
