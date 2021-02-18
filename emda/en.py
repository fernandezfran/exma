import numpy as np
import boundary

class effective_neighbors:
    """
    the empirical effective coordination model, used to calculate the effective
    neighbors, assumes that the interact atoms donate more of its electron to
    the closest central atoms. Then, fractions of the interact atom can be
    assigned to the various central neighbors atoms

    (V. L. Chevrier and J. R. Dahn 2010 J. Electrochem. Soc. 157 A392)
    (R. Hoppe et al., J. Less Common Met., 156, 105 (1989))
    """

    def __init__(self, natoms, box_size, atom_type_central, atom_type_interact):
        """
        Parameters
        ----------
        natoms : integer
            number of atoms
        
        box_size : numpy array with three floats
            the box size in x, y, z
        
        atom_type_central : integer
            type of central atoms

        atom_type_interact : integer
            type of interacting atoms
        """
        self.natoms = natoms
        self.box_size = box_size
        self.atom_type_central = atom_type_central
        self.atom_type_interact = atom_type_interact

        self.bound = boundary.apply(self.box_size)


    def of_this_frame(self, atom_type, positions):
        """
        obtain the efective (interact) neighbors of the actual frame

        Parameters
        ----------
        atom_type : numpy array with integers
            type of atoms

        positions : numpy array with float32 data
            the positions in the SoA convention
            i.e. first all the x, then y and then z

        Returns
        -------
        effnei : numpy array of float32
            effective (interact) neighbor of the central atoms in the same order
            that are in the positions vector
        """
        
        # calculates the distance matrix between interact and central atoms
        N_central = np.count_nonzero(atom_type == self.atom_type_central)
        N_interact = np.count_nonzero(atom_type == self.atom_type_interact)

        distrix = np.zeros(N_central * N_interact, dtype=np.float32)
        weitrix = distrix

        idx = 0
        for i in range(0, self.natoms):

            if (atom_type[i] != self.atom_type_interact): continue
            ri = [positions[i + k*self.natoms] for k in range(0,3)]

            for j in range(0, self.natoms):

                if (j == i): continue

                if (atom_type[j] != self.atom_type_central): continue
                rj = [positions[j + k*self.natoms] for k in range(0,3)]

                rij = self.bound.minimum_image(ri, rj)
                distrix[idx] = np.linalg.norm(rij)
                idx += 1

        # calculate the weigth of the ith neighbor of the interact atom 
        bondmin = np.min(distrix)   # the smallest bond lenght
        A = np.exp(1.0 - np.power(distrix / bondmin, 6))
        bondavg = np.sum( distrix * A ) / np.sum( A )   # average bond length
        for i in range(0, N_central*N_interact):
            weitrix[i] = np.exp(1.0 - np.power(distrix[i] / bondavg, 6))

        # split the weight matrix to obtain an interact atom in every row and
        #   normalize the weigths
        weitrix = np.split(weitrix, N_interact)
        for i in range(0, N_interact):
            weitrix[i] = weitrix[i] / np.sum(weitrix[i])
        
        # the matrix is transpose so now we have central atoms in each row and
        #   each fraction of every interact neighbor is added to obtain the
        #   effective (interact) neighbor
        weitrix = np.transpose(weitrix)
        effnei = np.zeros(N_central, dtype=np.float32)
        for i in range(0, N_central):
            effnei[i] = np.sum(weitrix[i])

        return effnei
