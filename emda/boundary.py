class boundary:
    """
    boundary conditions
    """
    pass


class apply(boundary):
    """
    apply some boundary condition
    """


    def __init__(self, box_size):
        """
        Parameters
        ----------
        box_size : numpy array of three floats
            box size in x, y, z
        """
        self.box_size = box_size


    def pbc(self, natoms, positions):
        """
        applies periodic boundary conditions to the particles

        Parameters
        ----------
        natoms : integer
            number of atoms

        positions : numpy array with float32 data
            unwrapped positions in the SoA convention

        Returns
        -------
        positions : numpy array with float32 dat
            wrapped positions
        """
        for i in range(0,3):
            for j in range(0, natoms):
                while (positions[i*natoms + j] < 0.0):
                    positions[i*natoms + j] += self.box_size[i]
                while (positions[i*natoms + j] > self.box_size[i]):
                    positions[i*natoms + j] -= self.box_size[i]

        return positions


    def minimum_image(self, x_central, x_interact):
        """
        nearest particle in minimum image convention

        Parameters
        ----------
        x_central : numpy array with three floats
            coordinates of the central atom

        x_interact : numpy array with three floats
            coordinates of the interact atom

        Returns
        -------
        x_ci : numpy array with three float32
            components of the vector that separates the atom interact from the
            atom central
        """
        x_ci = np.zeros(3, dtype=np.float32)

        for i in range(0,3):
            x_ci[i] = x_central[i] - x_interact[i]
            while (x_ci[i] > 0.5 * self.box_size[i]): x_ci[i] -= self.box_size[i]
            while (x_ci[i] < -0.5 * self.box_size[i]): x_ci[i] += self.box_size[i]
        
        return x_ci
